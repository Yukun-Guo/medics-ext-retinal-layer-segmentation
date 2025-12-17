"""
RetinalLayerSegmentation Extension for MedICS.

This extension provides comprehensive retinal layer segmentation capabilities
including automated layer detection, analysis, and quantitative measurements.
"""
from typing import Optional
from PySide6 import QtWidgets

from medics_extension_sdk import BaseExtension
from .RetinalLayerSegmentation import RetinalLayerSegmentation
import logging

# Module logger
logger = logging.getLogger(__name__)

# Import utils module as referenced in original MedICS.py
try:
    from .utils.utils import utils
except ImportError as e:
    logger.warning("Could not import RetinalLayerSegmentation utils: %s", e)
    utils = None

# Import all sub-modules that are referenced in original MedICS.py namespace
try:
    from .LayerSegmentation import LayerSegmentation
    
    # Additional sub-modules for completeness
    from .ResolutionHelper import CalculateResolution
    from .LoadDataSettingsDlg import LoadDataSettings
    from .LoadMedDataDlg import LoadMedData
except ImportError as e:
    logger.warning("Could not import some RetinalLayerSegmentation sub-modules: %s", e)
    # Set to None for missing modules
    LayerSegmentation = None
    NPASegmentation = None
    RFSegmentation = None
    DrusenSegmentation = None
    GASegmentation = None
    EnfaceViewer = None
    CalculateResolution = None
    BatchProcessing = None
    SaveData = None
    LoadDataSettings = None
    LoadMedData = None


class RetinalLayerSegmentationExtension(BaseExtension):
    """Extension wrapper for RetinalLayerSegmentation toolbox."""

    def __init__(self):
        """Initialize the extension."""
        super().__init__(extension_name="RetinalLayerSegmentation", author_name="MedICS Team")
        # Log the auto-generated extension ID
        logger.info("[RetinalLayerSegmentation] Extension initialized with ID: %s", self.id)
    
    def get_version(self) -> str:
        """Get extension version."""
        return "1.0.0"
    
    def get_description(self) -> str:
        """Get extension description."""
        return "Comprehensive retinal layer segmentation tool with automated analysis and quantitative features"
    
    def get_category(self) -> str:
        """Get extension category."""
        return "Medical Imaging"


    def create_widget(self, parent: Optional[QtWidgets.QWidget] = None, **kwargs) -> QtWidgets.QWidget:
        """Create the extension's main widget."""
        # Create a regular widget instead of a dialog for tab display
        return RetinalLayerSegmentation(parentWindow=parent, app_context=self.app_context, **kwargs)


# Export the extension class for discovery
__all__ = ['RetinalLayerSegmentationExtension']
