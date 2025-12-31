"""
RetinalLayerSegmentation Extension for MedICS.

This extension provides comprehensive retinal layer segmentation capabilities
including automated layer detection, analysis, and quantitative measurements.
"""
from .ExtensionMain import RetinalLayerSegmentationExtension

# Automatically reassemble model file from chunks on package import
# This happens once when the package is first imported
from .model_reassembly import ensure_model_ready
import warnings

try:
    # Attempt to reassemble the model file if needed
    if not ensure_model_ready():
        warnings.warn(
            "Failed to reassemble model file from chunks. "
            "The extension may not function properly. "
            "Please check the installation.",
            RuntimeWarning
        )
except Exception as e:
    warnings.warn(
        f"Error during model file reassembly: {e}. "
        "The extension may not function properly.",
        RuntimeWarning
    )

__all__ = ['RetinalLayerSegmentationExtension']