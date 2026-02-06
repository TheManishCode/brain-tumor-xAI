"""
MidLens Tumor Classifier
=========================
Weighted ensemble classifier for brain tumor detection from MRI scans.

Features:
    - Multi-model ensemble with performance-based weighting
    - Test-Time Augmentation (TTA) for robust predictions
    - Temperature scaling for calibrated probabilities
    - Grad-CAM visualization for model interpretability
    - CLAHE preprocessing for enhanced contrast
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # type: ignore[import-unresolved]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import (
    CLASS_METADATA,
    CLASS_NAMES,
    MODEL_WEIGHTS,
    TEMPERATURE_SCALES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
def create_model(architecture: str, num_classes: int = 4) -> nn.Module:
    """
    Create a model with architecture matching the trained weights.

    Args:
        architecture: One of ``efficientnet_b3``, ``resnet50``, ``densenet121``.
        num_classes: Number of output classes.

    Returns:
        PyTorch model instance.
    """
    dropout_rate = 0.3

    if architecture == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features  # type: ignore[index]
        model.classifier = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),  # type: ignore[arg-type]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
    elif architecture == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
class ImagePreprocessor:
    """Handles MRI image preprocessing for optimal model performance."""

    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        """
        Apply preprocessing pipeline to MRI image.

        Steps:
            1. Convert to RGB
            2. Apply CLAHE for contrast enhancement
            3. Normalize intensity range

        Args:
            image: Input PIL Image.

        Returns:
            Preprocessed PIL Image.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image)

        # Convert to grayscale for processing
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Normalize intensity range
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # Back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)


# =============================================================================
# ENSEMBLE CLASSIFIER
# =============================================================================
class TumorClassifier:
    """
    Weighted ensemble classifier for brain tumor detection.

    Combines predictions from DenseNet-121, EfficientNet-B3 and ResNet-50
    using validation-F1-derived weights, with optional TTA and
    temperature-scaled calibration.
    """

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.preprocessor = ImagePreprocessor()

        # Standard transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # TTA transforms
        self.tta_transforms = self._create_tta_transforms()

        self._load_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_tta_transforms(self) -> List[transforms.Compose]:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.ToTensor(),
                normalize,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=(270, 270)),
                transforms.ToTensor(),
                normalize,
            ]),
        ]

    def _load_models(self) -> None:
        """Load all available model weights from *models_dir*."""
        if not self.models_dir.exists():
            logger.warning("Models directory not found: %s", self.models_dir)
            return

        for weight_file in self.models_dir.glob("best_*.pth"):
            architecture = weight_file.stem.replace("best_", "")
            try:
                checkpoint = torch.load(
                    weight_file,
                    map_location=self.device,
                    weights_only=False,
                )
                model = create_model(architecture)
                model.load_state_dict(checkpoint["model_state"])
                model.to(self.device)
                model.eval()
                self.models[architecture] = model
                logger.info("✓ Loaded %s", architecture)
            except Exception as e:
                logger.error("✗ Failed to load %s: %s", architecture, e)

        logger.info(
            "Ensemble initialised: %d models on %s",
            len(self.models),
            self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """``True`` if at least one model is loaded."""
        return len(self.models) > 0

    @torch.no_grad()
    def predict(self, image: Image.Image, use_tta: bool = True) -> Dict[str, Any]:
        """
        Classify an MRI image.

        Args:
            image: Input PIL Image.
            use_tta: Whether to use Test-Time Augmentation.

        Returns:
            Dict with ``predicted_class``, ``confidence``, ``probabilities``,
            ``uncertainty`` and ``ensemble_info``.
        """
        processed_image = self.preprocessor.preprocess(image)

        if not self.is_ready:
            return self._create_demo_result()

        model_predictions: Dict[str, np.ndarray] = {}

        for arch_name, model in self.models.items():
            predictions = []
            temperature = TEMPERATURE_SCALES.get(arch_name, 1.0)
            transforms_to_use = self.tta_transforms if use_tta else [self.transform]

            for t in transforms_to_use:
                tensor = t(processed_image).unsqueeze(0).to(self.device)  # type: ignore[union-attr]
                logits = model(tensor) / temperature
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                predictions.append(probs)

            model_predictions[arch_name] = np.mean(predictions, axis=0)

        # Weighted ensemble aggregation
        final_probs = np.zeros(len(CLASS_NAMES))
        total_weight = sum(
            MODEL_WEIGHTS.get(n, 1 / len(self.models))
            for n in model_predictions
        )

        for arch_name, probs in model_predictions.items():
            weight = MODEL_WEIGHTS.get(arch_name, 1 / len(self.models)) / total_weight
            final_probs += weight * probs

        predicted_idx = int(np.argmax(final_probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(final_probs[predicted_idx])

        # Entropy-based uncertainty
        entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))
        max_entropy = np.log(len(CLASS_NAMES))
        uncertainty = entropy / max_entropy

        return {
            "predicted_class": predicted_class,
            "metadata": CLASS_METADATA[predicted_class],
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                CLASS_NAMES[i]: round(float(final_probs[i]) * 100, 1)
                for i in range(len(CLASS_NAMES))
            },
            "uncertainty": round(uncertainty, 3),
            "ensemble_info": {
                "num_models": len(self.models),
                "tta_augmentations": len(self.tta_transforms) if use_tta else 1,
                "total_predictions": (
                    len(self.models) * (len(self.tta_transforms) if use_tta else 1)
                ),
            },
        }

    def generate_gradcam(self, image: Image.Image) -> Optional[str]:
        """
        Generate Grad-CAM visualisation for the primary model.

        Returns:
            Base64-encoded PNG heatmap overlay, or ``None``.
        """
        if not self.is_ready:
            return None

        processed_image = self.preprocessor.preprocess(image)
        original_array = np.array(processed_image)

        arch_name = next(iter(self.models.keys()))
        model = self.models[arch_name]

        # Select target layer
        if "efficientnet" in arch_name:
            target_layer = model.features[-1]  # type: ignore[index]
        elif "resnet" in arch_name:
            target_layer = model.layer4[-1]  # type: ignore[index]
        elif "densenet" in arch_name:
            target_layer = model.features.denseblock4  # type: ignore[union-attr]
        else:
            return None

        gradients: list = []
        activations: list = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        handle = target_layer.register_forward_hook(forward_hook)  # type: ignore[union-attr]

        tensor = self.transform(processed_image).unsqueeze(0).to(self.device)  # type: ignore[union-attr]
        tensor.requires_grad_(True)

        output = model(tensor)
        predicted_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, predicted_class].backward()
        handle.remove()

        grads = gradients[0].cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original_resized = cv2.resize(original_array, (224, 224))
        overlay = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)

        pil_image = Image.fromarray(overlay)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    # ------------------------------------------------------------------
    def _create_demo_result(self) -> Dict[str, Any]:
        return {
            "predicted_class": "glioma",
            "metadata": CLASS_METADATA["glioma"],
            "confidence": 0.0,
            "probabilities": {c: 25.0 for c in CLASS_NAMES},
            "demo_mode": True,
            "message": "Models not loaded — showing demo result",
        }
