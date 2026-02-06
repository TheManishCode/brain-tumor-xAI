"""
Advanced Explainability Module for MidLens
==========================================
Provides comprehensive model interpretability using multiple techniques:
- Grad-CAM and Grad-CAM++
- Integrated Gradients
- Saliency Maps
- Layer-wise Relevance Propagation (LRP)
- Attention-based explanations
- Ensemble uncertainty analysis

Author: Senior Data Scientist
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageFilter
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Optional OpenCV import with PIL fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Using PIL-based image processing.")

logger = logging.getLogger(__name__)


# =============================================================================
# Image Processing Utilities (with cv2 fallback)
# =============================================================================

def resize_array(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize numpy array using cv2 or PIL fallback."""
    if HAS_CV2:
        return cv2.resize(arr, size)
    else:
        # PIL fallback
        if arr.ndim == 2:
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr.astype(np.uint8))
        img = img.resize(size, Image.Resampling.BILINEAR)
        return np.array(img)


def apply_colormap(gray: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """Apply colormap to grayscale array using cv2 or matplotlib fallback."""
    if HAS_CV2:
        colormap_cv2 = {
            'jet': cv2.COLORMAP_JET,
            'inferno': cv2.COLORMAP_INFERNO,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'hot': cv2.COLORMAP_HOT,
            'plasma': cv2.COLORMAP_PLASMA
        }.get(colormap, cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(np.uint8(255 * gray), colormap_cv2)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        # PIL/matplotlib fallback
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap)
            colored = cmap(gray)[:, :, :3]  # Remove alpha
            return (colored * 255).astype(np.uint8)
        except ImportError:
            # Simple fallback - just create RGB from gray
            gray_uint8 = np.uint8(255 * gray)
            return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend two images together."""
    if HAS_CV2:
        return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    else:
        # NumPy fallback
        return ((1 - alpha) * img1 + alpha * img2).astype(np.uint8)


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    method: str
    heatmap: np.ndarray
    overlay: np.ndarray
    base64_heatmap: str
    base64_overlay: str
    importance_scores: Dict[str, float]
    metadata: Dict[str, Any]


class GradCAMExplainer:
    """
    Advanced Grad-CAM implementation with Grad-CAM++ support.
    
    Grad-CAM++ provides better localization for multiple instances
    and improved visual explanations for fine-grained features.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = []
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __call__(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        use_gradcam_pp: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM or Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Preprocessed input image tensor
            target_class: Target class for explanation (None = predicted class)
            use_gradcam_pp: Use Grad-CAM++ algorithm
            
        Returns:
            Tuple of (heatmap array, predicted class index)
        """
        self.gradients = []
        self.activations = []
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()[0]  # [C, H, W]
        activations = self.activations[0].cpu().numpy()[0]  # [C, H, W]
        
        if use_gradcam_pp:
            # Grad-CAM++ weights computation
            # α = ReLU(∂²y/∂A²) / (2·∂²y/∂A² + Σ(A·∂³y/∂A³))
            grad_2 = gradients ** 2
            grad_3 = gradients ** 3
            
            sum_activations = np.sum(activations, axis=(1, 2), keepdims=True)
            alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-10)
            alpha = np.where(gradients != 0, alpha, 0)
            
            weights = np.sum(alpha * np.maximum(gradients, 0), axis=(1, 2))
        else:
            # Standard Grad-CAM weights
            weights = np.mean(gradients, axis=(1, 2))
        
        # Compute weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = resize_array(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class


class IntegratedGradientsExplainer:
    """
    Integrated Gradients implementation for pixel-level attribution.
    
    This method satisfies key axioms:
    - Sensitivity: If input and baseline differ only in one feature, 
      the attribution is non-zero for that feature
    - Implementation Invariance: Attributions are identical for 
      functionally equivalent networks
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class (None = predicted)
            baseline: Baseline tensor (None = black image)
            steps: Number of interpolation steps
            
        Returns:
            Tuple of (attribution map, predicted class)
        """
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device)
        
        # Get prediction if target not specified
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        # Create interpolated inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(steps + 1)
        ]
        
        # Compute gradients at each step
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.requires_grad_(True)
            output = self.model(scaled_input)
            
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
            
            gradients.append(scaled_input.grad.detach().cpu().numpy()[0])
        
        # Approximate integral using trapezoidal rule
        gradients = np.array(gradients)
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_grads = np.mean(avg_gradients, axis=0)
        
        # Scale by input difference
        input_diff = (input_tensor - baseline).cpu().numpy()[0]
        attributions = integrated_grads * input_diff
        
        # Aggregate across channels and normalize
        attribution_map = np.mean(np.abs(attributions), axis=0)
        attribution_map = (attribution_map - attribution_map.min()) / \
                         (attribution_map.max() - attribution_map.min() + 1e-8)
        
        return attribution_map, target_class


class SaliencyMapExplainer:
    """
    Vanilla gradient saliency map computation.
    
    Also includes SmoothGrad variant for noise-reduced attributions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        smooth: bool = True,
        n_samples: int = 25,
        noise_level: float = 0.15
    ) -> Tuple[np.ndarray, int]:
        """
        Compute saliency map with optional SmoothGrad.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class (None = predicted)
            smooth: Use SmoothGrad
            n_samples: Number of noise samples for SmoothGrad
            noise_level: Standard deviation of noise
            
        Returns:
            Tuple of (saliency map, predicted class)
        """
        self.model.eval()
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        if smooth:
            # SmoothGrad: Average gradients over noisy samples
            gradients = []
            stdev = noise_level * (input_tensor.max() - input_tensor.min()).item()
            
            for _ in range(n_samples):
                noisy_input = input_tensor + torch.randn_like(input_tensor) * stdev
                noisy_input = noisy_input.requires_grad_(True)
                
                output = self.model(noisy_input)
                self.model.zero_grad()
                output[0, target_class].backward()
                
                gradients.append(noisy_input.grad.detach().cpu().numpy()[0])
            
            saliency = np.mean(np.array(gradients), axis=0)
        else:
            # Vanilla gradient
            input_tensor = input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            self.model.zero_grad()
            output[0, target_class].backward()
            
            saliency = input_tensor.grad.detach().cpu().numpy()[0]
        
        # Take absolute value and aggregate channels
        saliency = np.max(np.abs(saliency), axis=0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency, target_class


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for image classification.
    
    LIME explains predictions by:
    1. Creating superpixel segments of the image
    2. Generating perturbed versions by masking segments
    3. Getting predictions for perturbed images
    4. Training a linear model to approximate behavior locally
    5. Using linear model weights as feature importance
    
    This provides intuitive, region-based explanations that are
    model-agnostic and easy to interpret.
    """
    
    def __init__(self, model: nn.Module, transform, device: str = 'cuda'):
        self.model = model
        self.transform = transform
        self.device = device
    
    def _create_superpixels(self, image: np.ndarray, n_segments: int = 50) -> np.ndarray:
        """
        Create superpixel segmentation using SLIC-like algorithm.
        
        Falls back to grid-based segmentation if skimage not available.
        """
        try:
            from skimage.segmentation import slic
            segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
            return segments
        except ImportError:
            # Fallback to simple grid segmentation
            h, w = image.shape[:2]
            grid_size = int(np.sqrt(n_segments))
            segments = np.zeros((h, w), dtype=np.int32)
            
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, (i + 1) * cell_h if i < grid_size - 1 else h
                    x1, x2 = j * cell_w, (j + 1) * cell_w if j < grid_size - 1 else w
                    segments[y1:y2, x1:x2] = i * grid_size + j
            
            return segments
    
    def _perturb_image(
        self, 
        image: np.ndarray, 
        segments: np.ndarray, 
        mask: np.ndarray,
        background: str = 'gray'
    ) -> np.ndarray:
        """Apply mask to image, hiding certain segments."""
        perturbed = image.copy()
        
        # Determine background color
        if background == 'gray':
            bg_color = [128, 128, 128]
        elif background == 'black':
            bg_color = [0, 0, 0]
        elif background == 'blur':
            # Use blurred version as background
            if HAS_CV2:
                blurred = cv2.GaussianBlur(image, (21, 21), 0)
            else:
                pil_img = Image.fromarray(image)
                blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(10)))
            for i, active in enumerate(mask):
                if not active:
                    perturbed[segments == i] = blurred[segments == i]
            return perturbed
        else:
            bg_color = [128, 128, 128]
        
        # Apply mask
        for i, active in enumerate(mask):
            if not active:
                perturbed[segments == i] = bg_color
        
        return perturbed
    
    def __call__(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        n_segments: int = 50,
        n_samples: int = 500,
        background: str = 'gray'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate LIME explanation.
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 255]
            target_class: Class to explain (None = predicted class)
            n_segments: Number of superpixels
            n_samples: Number of perturbation samples
            background: Background type for masked regions ('gray', 'black', 'blur')
            
        Returns:
            Tuple of (importance heatmap, metadata dict)
        """
        self.model.eval()
        
        # Ensure image is RGB and correct size
        if image.shape[:2] != (224, 224):
            image = resize_array(image, (224, 224))
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Create superpixels
        segments = self._create_superpixels(image, n_segments)
        n_features = segments.max() + 1
        
        # Get original prediction
        pil_image = Image.fromarray(image.astype(np.uint8))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            original_output = self.model(input_tensor)
            original_probs = F.softmax(original_output, dim=1)[0].cpu().numpy()
            if target_class is None:
                target_class = np.argmax(original_probs)
        
        # Generate perturbations and collect data
        perturbation_masks = []
        predictions = []
        
        for _ in range(n_samples):
            # Random binary mask for segments
            mask = np.random.randint(0, 2, n_features).astype(bool)
            perturbation_masks.append(mask)
            
            # Create perturbed image
            perturbed = self._perturb_image(image, segments, mask, background)
            
            # Get prediction
            pil_perturbed = Image.fromarray(perturbed.astype(np.uint8))
            perturbed_tensor = self.transform(pil_perturbed).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(perturbed_tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
                predictions.append(probs[target_class])
        
        # Fit linear model to find segment importance
        X = np.array(perturbation_masks).astype(float)
        y = np.array(predictions)
        
        # Simple linear regression with regularization
        # weights = (X^T X + λI)^{-1} X^T y
        lambda_reg = 0.1
        XtX = X.T @ X + lambda_reg * np.eye(n_features)
        Xty = X.T @ y
        
        try:
            weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        
        # Create importance heatmap
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for i in range(n_features):
            heatmap[segments == i] = weights[i]
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Resize to standard size
        heatmap = resize_array(heatmap, (224, 224))
        
        # Compute metadata
        top_segments = np.argsort(weights)[::-1][:5]
        
        metadata = {
            'target_class': int(target_class),
            'original_confidence': float(original_probs[target_class]),
            'n_segments': int(n_features),
            'n_samples': n_samples,
            'top_segments': top_segments.tolist(),
            'segment_weights': {int(i): float(weights[i]) for i in top_segments}
        }
        
        return heatmap, metadata


class EnsembleUncertaintyAnalyzer:
    """
    Analyze prediction uncertainty across ensemble models.
    
    Provides metrics for:
    - Epistemic uncertainty (model disagreement)
    - Aleatoric uncertainty (data inherent noise)
    - Predictive entropy
    - Mutual information
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def analyze(
        self,
        model_predictions: Dict[str, np.ndarray],
        final_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute comprehensive uncertainty metrics.
        
        Args:
            model_predictions: Dict of model_name -> probability array
            final_probs: Ensemble averaged probabilities
            
        Returns:
            Dictionary of uncertainty metrics
        """
        n_models = len(model_predictions)
        all_probs = np.array(list(model_predictions.values()))  # [M, C]
        
        # Predictive entropy: H[y|x, D]
        predictive_entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))
        max_entropy = np.log(len(self.class_names))
        normalized_entropy = predictive_entropy / max_entropy
        
        # Average entropy across models: E[H[y|x, θ]]
        model_entropies = [-np.sum(p * np.log(p + 1e-10)) for p in all_probs]
        avg_model_entropy = np.mean(model_entropies)
        
        # Mutual information (epistemic uncertainty): I[y, θ|x, D]
        mutual_information = predictive_entropy - avg_model_entropy
        
        # Model agreement analysis
        predicted_classes = [np.argmax(p) for p in all_probs]
        agreement_score = sum(1 for c in predicted_classes if c == predicted_classes[0]) / n_models
        
        # Variance in predictions (per class)
        class_variances = np.var(all_probs, axis=0)
        
        # Confidence intervals (using model spread)
        confidence_intervals = {}
        for i, cls in enumerate(self.class_names):
            probs_for_class = all_probs[:, i]
            ci_lower = np.percentile(probs_for_class, 2.5)
            ci_upper = np.percentile(probs_for_class, 97.5)
            confidence_intervals[cls] = {
                "mean": float(final_probs[i]),
                "std": float(np.std(probs_for_class)),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper)
            }
        
        # Per-model votes with confidence
        model_votes = []
        for model_name, probs in model_predictions.items():
            pred_class = np.argmax(probs)
            model_votes.append({
                "model": model_name,
                "prediction": self.class_names[pred_class],
                "confidence": float(probs[pred_class]),
                "all_probabilities": {
                    self.class_names[i]: float(probs[i]) 
                    for i in range(len(self.class_names))
                }
            })
        
        # Reliability score (higher = more reliable)
        reliability = agreement_score * (1 - normalized_entropy) * np.max(final_probs)
        
        return {
            "uncertainty_metrics": {
                "predictive_entropy": float(predictive_entropy),
                "normalized_entropy": float(normalized_entropy),
                "mutual_information": float(mutual_information),
                "epistemic_uncertainty": float(mutual_information / max_entropy),
                "aleatoric_uncertainty": float(avg_model_entropy / max_entropy)
            },
            "agreement": {
                "score": float(agreement_score),
                "all_agree": agreement_score == 1.0,
                "models_agreeing": int(agreement_score * n_models)
            },
            "confidence_intervals": confidence_intervals,
            "class_variances": {
                self.class_names[i]: float(class_variances[i])
                for i in range(len(self.class_names))
            },
            "model_votes": model_votes,
            "reliability_score": float(reliability),
            "interpretation": self._interpret_uncertainty(
                normalized_entropy, agreement_score, reliability
            )
        }
    
    def _interpret_uncertainty(
        self, 
        entropy: float, 
        agreement: float, 
        reliability: float
    ) -> Dict[str, str]:
        """Generate human-readable uncertainty interpretation."""
        # Confidence level
        if reliability > 0.8:
            confidence_level = "Very High"
            confidence_desc = "The model is highly confident in this prediction."
        elif reliability > 0.6:
            confidence_level = "High"
            confidence_desc = "The model shows good confidence with minor uncertainty."
        elif reliability > 0.4:
            confidence_level = "Moderate"
            confidence_desc = "There is some uncertainty. Consider additional review."
        elif reliability > 0.2:
            confidence_level = "Low"
            confidence_desc = "Significant uncertainty detected. Medical review recommended."
        else:
            confidence_level = "Very Low"
            confidence_desc = "High uncertainty. This case requires expert evaluation."
        
        # Agreement interpretation
        if agreement == 1.0:
            agreement_desc = "All models agree on the classification."
        elif agreement >= 0.67:
            agreement_desc = "Most models agree, with some variation."
        else:
            agreement_desc = "Models show significant disagreement."
        
        return {
            "confidence_level": confidence_level,
            "confidence_description": confidence_desc,
            "agreement_description": agreement_desc,
            "recommendation": self._get_recommendation(reliability, agreement)
        }
    
    def _get_recommendation(self, reliability: float, agreement: float) -> str:
        """Generate recommendation based on uncertainty analysis."""
        if reliability > 0.7 and agreement >= 0.67:
            return "Prediction appears reliable. Standard clinical workflow applies."
        elif reliability > 0.5:
            return "Recommend verification by radiologist before proceeding."
        else:
            return "High uncertainty case. Recommend immediate expert review and possibly additional imaging."


class ExplainabilityEngine:
    """
    Unified explainability engine combining multiple methods.
    
    Provides comprehensive analysis including:
    - Multiple visualization techniques
    - Uncertainty quantification
    - Feature importance rankings
    - Human-readable explanations
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        class_names: List[str],
        device: str = 'cuda',
        transform=None
    ):
        self.models = models
        self.class_names = class_names
        self.device = device
        self.transform = transform
        self.uncertainty_analyzer = EnsembleUncertaintyAnalyzer(class_names)
    
    def _get_target_layer(self, model: nn.Module, arch_name: str) -> nn.Module:
        """Get the appropriate target layer for Grad-CAM based on architecture."""
        if "efficientnet" in arch_name:
            return model.features[-1]
        elif "resnet" in arch_name:
            return model.layer4[-1]
        elif "densenet" in arch_name:
            return model.features.denseblock4
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")
    
    def _to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 encoded PNG."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _create_overlay(
        self, 
        original: np.ndarray, 
        heatmap: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create overlay of heatmap on original image."""
        heatmap_colored = apply_colormap(heatmap, colormap)
        original_resized = resize_array(original, (224, 224))
        overlay = blend_images(original_resized, heatmap_colored, alpha)
        return overlay
    
    def explain(
        self,
        image: Image.Image,
        processed_image: Image.Image,
        model_predictions: Dict[str, np.ndarray],
        final_probs: np.ndarray,
        methods: List[str] = ['gradcam', 'integrated_gradients', 'saliency', 'lime']
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            image: Original input image
            processed_image: Preprocessed image
            model_predictions: Per-model probability predictions
            final_probs: Ensemble-averaged probabilities
            methods: List of explanation methods to apply
            
        Returns:
            Complete explanation dictionary
        """
        original_array = np.array(processed_image)
        input_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)
        
        predicted_class_idx = int(np.argmax(final_probs))
        predicted_class = self.class_names[predicted_class_idx]
        
        explanations = {}
        
        # Use primary model for visualizations
        primary_model_name = max(
            self.models.keys(),
            key=lambda k: model_predictions.get(k, np.zeros(len(self.class_names)))[predicted_class_idx]
        )
        primary_model = self.models[primary_model_name]
        
        # Grad-CAM / Grad-CAM++
        if 'gradcam' in methods:
            try:
                target_layer = self._get_target_layer(primary_model, primary_model_name)
                gradcam = GradCAMExplainer(primary_model, target_layer, self.device)
                
                # Standard Grad-CAM
                cam, _ = gradcam(input_tensor, predicted_class_idx, use_gradcam_pp=False)
                overlay_std = self._create_overlay(original_array, cam)
                
                # Grad-CAM++
                cam_pp, _ = gradcam(input_tensor, predicted_class_idx, use_gradcam_pp=True)
                overlay_pp = self._create_overlay(original_array, cam_pp)
                
                gradcam.remove_hooks()
                
                explanations['gradcam'] = {
                    'heatmap': self._to_base64(cam),
                    'overlay': self._to_base64(overlay_std),
                    'description': 'Grad-CAM shows which regions most influenced the classification.'
                }
                
                explanations['gradcam_pp'] = {
                    'heatmap': self._to_base64(cam_pp),
                    'overlay': self._to_base64(overlay_pp),
                    'description': 'Grad-CAM++ provides improved localization for complex patterns.'
                }
            except Exception as e:
                logger.error(f"Grad-CAM failed: {e}")
        
        # Integrated Gradients
        if 'integrated_gradients' in methods:
            try:
                ig = IntegratedGradientsExplainer(primary_model, self.device)
                ig_map, _ = ig(input_tensor, predicted_class_idx, steps=30)
                ig_overlay = self._create_overlay(
                    original_array, ig_map, 
                    colormap='inferno'
                )
                
                explanations['integrated_gradients'] = {
                    'heatmap': self._to_base64(ig_map),
                    'overlay': self._to_base64(ig_overlay),
                    'description': 'Integrated Gradients shows pixel-level contribution to the prediction.'
                }
            except Exception as e:
                logger.error(f"Integrated Gradients failed: {e}")
        
        # Saliency Maps (SmoothGrad)
        if 'saliency' in methods:
            try:
                saliency = SaliencyMapExplainer(primary_model, self.device)
                sal_map, _ = saliency(input_tensor, predicted_class_idx, smooth=True)
                sal_overlay = self._create_overlay(
                    original_array, sal_map,
                    colormap='viridis'
                )
                
                explanations['saliency'] = {
                    'heatmap': self._to_base64(sal_map),
                    'overlay': self._to_base64(sal_overlay),
                    'description': 'SmoothGrad saliency map highlights sensitive input regions.'
                }
            except Exception as e:
                logger.error(f"Saliency map failed: {e}")
        
        # LIME (Local Interpretable Model-agnostic Explanations)
        if 'lime' in methods:
            try:
                lime_explainer = LIMEExplainer(
                    primary_model, 
                    self.transform, 
                    self.device
                )
                lime_map, lime_meta = lime_explainer(
                    original_array, 
                    target_class=predicted_class_idx,
                    n_segments=50,
                    n_samples=300  # Balanced for speed vs accuracy
                )
                lime_overlay = self._create_overlay(
                    original_array, lime_map,
                    colormap='RdBu_r'  # Red-Blue diverging for positive/negative contributions
                )
                
                # Compute model fit score from weights variance
                weights = list(lime_meta.get('segment_weights', {}).values())
                model_score = 1.0 - (np.std(weights) / (np.max(weights) - np.min(weights) + 1e-8)) if weights else 0.5
                
                explanations['lime'] = {
                    'heatmap': self._to_base64(lime_map),
                    'overlay': self._to_base64(lime_overlay),
                    'description': 'LIME shows which image regions support or oppose the prediction.',
                    'metadata': {
                        'num_segments': lime_meta.get('n_segments', 0),
                        'model_score': float(np.clip(model_score, 0, 1)),
                        'top_features': lime_meta.get('top_segments', [])[:5],
                        'bottom_features': []
                    }
                }
            except Exception as e:
                logger.error(f"LIME failed: {e}")
        
        # Uncertainty analysis
        uncertainty = self.uncertainty_analyzer.analyze(model_predictions, final_probs)
        
        # Feature importance summary
        feature_importance = self._compute_feature_importance(
            explanations, predicted_class
        )
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(final_probs[predicted_class_idx]),
            'visualizations': explanations,
            'uncertainty_analysis': uncertainty,
            'feature_importance': feature_importance,
            'explanation_summary': self._generate_summary(
                predicted_class, 
                float(final_probs[predicted_class_idx]),
                uncertainty
            )
        }
    
    def _compute_feature_importance(
        self, 
        explanations: Dict, 
        predicted_class: str
    ) -> Dict[str, Any]:
        """Compute aggregate feature importance from multiple methods."""
        regions = {
            'top_left': (0, 0, 112, 112),
            'top_right': (112, 0, 224, 112),
            'bottom_left': (0, 112, 112, 224),
            'bottom_right': (112, 112, 224, 224),
            'center': (56, 56, 168, 168)
        }
        
        importance = {region: [] for region in regions}
        
        for method, data in explanations.items():
            if 'heatmap' in data:
                # Decode and analyze heatmap
                try:
                    heatmap_bytes = base64.b64decode(data['heatmap'])
                    heatmap_img = Image.open(BytesIO(heatmap_bytes))
                    heatmap_array = np.array(heatmap_img.convert('L')) / 255.0
                    
                    for region, (x1, y1, x2, y2) in regions.items():
                        region_importance = np.mean(heatmap_array[y1:y2, x1:x2])
                        importance[region].append(region_importance)
                except:
                    pass
        
        # Average across methods
        avg_importance = {
            region: float(np.mean(scores)) if scores else 0.0
            for region, scores in importance.items()
        }
        
        # Rank regions
        ranked = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'regional_importance': avg_importance,
            'most_important_regions': [r[0] for r in ranked[:3]],
            'interpretation': f"The model focused primarily on the {ranked[0][0].replace('_', ' ')} region of the image."
        }
    
    def _generate_summary(
        self, 
        predicted_class: str, 
        confidence: float,
        uncertainty: Dict
    ) -> Dict[str, str]:
        """Generate human-readable explanation summary."""
        interp = uncertainty.get('interpretation', {})
        
        return {
            'headline': f"Classification: {predicted_class.title()} ({confidence*100:.1f}% confidence)",
            'confidence_assessment': interp.get('confidence_description', ''),
            'model_agreement': interp.get('agreement_description', ''),
            'clinical_recommendation': interp.get('recommendation', ''),
            'technical_note': (
                f"This analysis used {len(self.models)} ensemble models with "
                f"reliability score of {uncertainty.get('reliability_score', 0):.2f}. "
                f"Epistemic uncertainty: {uncertainty.get('uncertainty_metrics', {}).get('epistemic_uncertainty', 0):.3f}"
            )
        }
