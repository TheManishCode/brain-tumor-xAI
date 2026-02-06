"""
MidLens Flask Application
=========================
HTTP API for brain tumor classification, explainability, and AI chatbot.

Routes
------
GET  /                  → Service information
GET  /api/health        → Health check
POST /api/predict       → Classify an MRI image
POST /api/explain       → Generate Grad-CAM explanation
POST /api/analyze       → Full analysis with multi-method XAI
POST /api/chat          → AI chatbot (agentic RAG)
POST /api/chat/suggestions → Suggested questions
POST /api/chat/clear    → Clear chat session
GET  /api/chat/status   → Chatbot status
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory  # type: ignore[import-not-found]
from flask_cors import CORS  # type: ignore[import-not-found]
from PIL import Image

from .classifier import TumorClassifier
from .config import CLASS_NAMES, SETTINGS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask app with all routes registered.
    """
    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = SETTINGS.max_upload_bytes

    # ------------------------------------------------------------------
    # Initialise core services
    # ------------------------------------------------------------------
    classifier = TumorClassifier(models_dir=SETTINGS.models_dir)

    # Explainability (optional)
    try:
        from .explainability import ExplainabilityEngine
        explainability_available = True
    except ImportError:
        explainability_available = False
        logger.warning("Explainability module not available")

    # Chatbot (primary → fallback)
    chatbot = None
    chatbot_available = False
    knowledge_base_path = str(
        Path(__file__).parent.parent / SETTINGS.knowledge_base_path
    )

    try:
        from .chatbot import create_chatbot
        chatbot = create_chatbot(knowledge_base_path)
        chatbot_available = True
        logger.info("Agentic RAG Chatbot initialised")
    except Exception as exc:
        logger.warning("Chatbot V2 not available: %s", exc)
        try:
            from .chatbot import AIChatbot
            chatbot = AIChatbot(knowledge_base_path=knowledge_base_path)
            chatbot_available = True
            logger.info("Fallback chatbot initialised")
        except Exception as exc2:
            logger.warning("Fallback chatbot also unavailable: %s", exc2)

    # ------------------------------------------------------------------
    # Routes: General
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        return jsonify({
            "service": f"{SETTINGS.service_name} API",
            "version": SETTINGS.version,
            "frontend": "Run 'npm run dev' in frontend/ (http://localhost:5173)",
            "api_docs": {
                "/api/health": "Health check",
                "/api/predict": "POST — Upload MRI image for classification",
                "/api/chat": "POST — AI chatbot for brain tumor questions",
            },
            "models_loaded": classifier.is_ready,
            "chatbot_available": chatbot_available,
        })

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        return send_from_directory("static", filename)

    # ------------------------------------------------------------------
    # Routes: Health
    # ------------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": SETTINGS.service_name,
            "version": SETTINGS.version,
            "models_loaded": classifier.is_ready,
            "num_models": len(classifier.models),
            "device": str(classifier.device),
        })

    # ------------------------------------------------------------------
    # Routes: Prediction
    # ------------------------------------------------------------------
    @app.route("/api/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            image = Image.open(file.stream)
            use_tta = request.form.get("use_tta", "true").lower() == "true"
            include_gradcam = request.form.get("include_gradcam", "true").lower() == "true"

            result = classifier.predict(image, use_tta=use_tta)
            response = {"success": True, **result}

            if include_gradcam:
                try:
                    gradcam = classifier.generate_gradcam(image)
                    if gradcam:
                        response["gradcam"] = gradcam
                except Exception as grad_err:
                    logger.warning("GradCAM generation failed: %s", grad_err)

            return jsonify(response)
        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Routes: Explain (simple Grad-CAM)
    # ------------------------------------------------------------------
    @app.route("/api/explain", methods=["POST"])
    def explain():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]

        try:
            image = Image.open(file.stream)
            result = classifier.predict(image, use_tta=True)
            heatmap = classifier.generate_gradcam(image)
            return jsonify({
                "success": True,
                "prediction": result["predicted_class"],
                "confidence": result["confidence"],
                "heatmap": heatmap,
            })
        except Exception as exc:
            logger.error("Explanation error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Routes: Advanced Explainability
    # ------------------------------------------------------------------
    @app.route("/api/analyze", methods=["POST"])
    def analyze_with_explainability():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            start_time = time.time()
            image = Image.open(file.stream)

            use_tta = request.form.get("use_tta", "true").lower() == "true"
            result = classifier.predict(image, use_tta=use_tta)

            response = {
                "success": True,
                "predicted_class": result["predicted_class"],
                "metadata": result["metadata"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "uncertainty": result.get("uncertainty", 0),
                "ensemble_info": result.get("ensemble_info", {}),
                "processing_time": time.time() - start_time,
            }

            include_xai = (
                request.form.get("include_explainability", "true").lower() == "true"
            )

            if include_xai and explainability_available and classifier.is_ready:
                try:
                    from .explainability import ExplainabilityEngine

                    engine = ExplainabilityEngine(
                        models=classifier.models,
                        class_names=CLASS_NAMES,
                        device=str(classifier.device),
                        transform=classifier.transform,
                    )

                    processed_image = classifier.preprocessor.preprocess(image)

                    import torch
                    model_predictions = {}
                    for arch_name, model in classifier.models.items():
                        tensor = (
                            classifier.transform(processed_image)
                            .unsqueeze(0)  # type: ignore[union-attr]
                            .to(classifier.device)
                        )
                        with torch.no_grad():
                            logits = model(tensor)
                            probs = (
                                F.softmax(logits, dim=1)[0].cpu().numpy()
                            )
                            model_predictions[arch_name] = probs

                    final_probs = np.array([
                        result["probabilities"][cls] / 100
                        for cls in CLASS_NAMES
                    ])

                    explanation = engine.explain(
                        image=image,
                        processed_image=processed_image,
                        model_predictions=model_predictions,
                        final_probs=final_probs,
                        methods=[
                            "gradcam",
                            "integrated_gradients",
                            "saliency",
                            "lime",
                        ],
                    )
                    response["explainability"] = explanation
                except Exception as xai_err:
                    logger.error("Explainability failed: %s", xai_err)
                    response["explainability_error"] = str(xai_err)

            return jsonify(response)
        except Exception as exc:
            logger.error("Analysis error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Routes: AI Chatbot
    # ------------------------------------------------------------------
    @app.route("/api/chat", methods=["POST"])
    def chat():
        if not chatbot_available:
            return jsonify({"error": "AI Chatbot is not available"}), 503

        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Message is required"}), 400

        message = data["message"]
        session_id = data.get("session_id", "default")
        analysis_context = data.get("analysis", None)

        try:
            response = chatbot.chat(message, session_id, analysis_context)  # type: ignore[union-attr]
            if hasattr(response, "to_dict"):
                return jsonify(response.to_dict())  # type: ignore[union-attr]
            return jsonify(response)
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/chat/suggestions", methods=["POST"])
    def chat_suggestions():
        if not chatbot_available:
            return jsonify({"error": "AI Chatbot is not available"}), 503

        data = request.get_json() or {}
        analysis = data.get("analysis", None)

        if hasattr(chatbot, "get_suggestions"):
            return jsonify({"suggestions": chatbot.get_suggestions(analysis)})  # type: ignore[union-attr]

        suggestions = [
            "What is a brain tumor?",
            "What are the symptoms of brain tumors?",
            "How are brain tumors diagnosed?",
            "What treatment options are available?",
        ]

        if analysis:
            prediction = analysis.get("prediction", {})
            display_name = prediction.get(
                "display_name", prediction.get("class", "")
            )
            if display_name and display_name.lower() not in ("notumor", "no tumor"):
                suggestions = [
                    f"What is {display_name}?",
                    f"What are the symptoms of {display_name}?",
                    f"What are the treatment options for {display_name}?",
                    "Explain my analysis results",
                ]

        return jsonify({"suggestions": suggestions})

    @app.route("/api/chat/clear", methods=["POST"])
    def clear_chat():
        if not chatbot_available:
            return jsonify({"error": "AI Chatbot is not available"}), 503
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        chatbot.clear_session(session_id)  # type: ignore[union-attr]
        return jsonify({"success": True, "message": "Session cleared"})

    @app.route("/api/chat/status", methods=["GET"])
    def chat_status():
        if not chatbot_available:
            return jsonify({
                "available": False,
                "message": "AI Chatbot is not available",
            })

        providers = getattr(
            getattr(chatbot, "llm", None), "providers", ["local"]
        )
        return jsonify({
            "available": True,
            "llm_provider": providers[0] if providers else "local",
            "llm_providers": providers,
            "has_llm": any(p in providers for p in ("gemini", "groq")),
            "features": {
                "web_search": True,
                "pubmed_search": True,
                "knowledge_base": True,
                "source_citations": True,
            },
        })

    return app
