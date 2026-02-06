"""
MidLens Configuration
=====================
Centralized configuration for model architecture, class metadata,
ensemble weights, and application settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# TUMOR CLASS CONFIGURATION
# =============================================================================
CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_METADATA: Dict[str, dict] = {
    "glioma": {
        "display_name": "Glioma",
        "description": (
            "A tumor that originates in the glial cells of the brain. "
            "Can range from low-grade (slow-growing) to high-grade "
            "(aggressive, such as glioblastoma)."
        ),
        "severity": "high",
        "recommendations": [
            "Immediate consultation with a neuro-oncologist recommended",
            "Additional imaging (MRI with contrast) may be needed",
            "Biopsy may be required for definitive diagnosis",
        ],
    },
    "meningioma": {
        "display_name": "Meningioma",
        "description": (
            "A tumor arising from the meninges, the membranes surrounding "
            "the brain. Most are benign and slow-growing."
        ),
        "severity": "moderate",
        "recommendations": [
            "Follow-up imaging recommended",
            "Consultation with neurosurgeon if symptomatic",
            "Many cases can be monitored without immediate intervention",
        ],
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "description": (
            "A growth in the pituitary gland at the base of the brain. "
            "Usually benign, but can affect hormone production."
        ),
        "severity": "moderate",
        "recommendations": [
            "Endocrine evaluation recommended",
            "Hormone level testing advised",
            "Treatment depends on tumor type and hormone involvement",
        ],
    },
    "notumor": {
        "display_name": "No Tumor Detected",
        "description": (
            "The scan appears normal with no detectable tumor masses. "
            "Brain tissue shows typical characteristics."
        ),
        "severity": "normal",
        "recommendations": [
            "No immediate action required",
            "Continue routine health monitoring",
            "Consult physician if symptoms persist",
        ],
    },
}


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weights derived from validation F1 scores
MODEL_WEIGHTS: Dict[str, float] = {
    "efficientnet_b3": 0.40,
    "resnet50": 0.35,
    "densenet121": 0.25,
}

# Temperature scaling for probability calibration
TEMPERATURE_SCALES: Dict[str, float] = {
    "efficientnet_b3": 1.15,
    "resnet50": 1.22,
    "densenet121": 1.28,
}


# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
@dataclass(frozen=True)
class AppSettings:
    """Immutable application settings."""

    service_name: str = "MidLens"
    version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 5000
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    models_dir: str = "models"
    knowledge_base_path: str = "knowledge_base/medical_knowledge.json"


SETTINGS = AppSettings()
