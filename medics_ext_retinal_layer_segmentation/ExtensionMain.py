"""
RetinalLayerSegmentation Extension for MedICS.

This extension provides comprehensive retinal layer segmentation capabilities
including automated layer detection, analysis, and quantitative measurements.
"""
from typing import Optional
from PySide6 import QtWidgets
import platform
import os
import subprocess
import numpy as np
import onnxruntime as ort
from medics_extension_sdk import BaseExtension
from .RetinalLayerSegmentation import RetinalLayerSegmentation
from .utils.onnx_provider_utils import create_onnx_session
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

    @staticmethod
    def get_first_available_gpu() -> int:
        """Detect and return the first available GPU device ID.
        
        This method checks for GPU availability across different platforms:
        - CUDA (NVIDIA) on Linux, Windows
        - Metal Performance Shaders (MPS) on macOS
        - DirectML on Windows
        - CoreML on macOS
        
        Returns:
            int: The device ID of the first available GPU (0 if found, -1 if no GPU available).
        """
        system = platform.system().lower()
        available_providers = ort.get_available_providers()
        
        logger.info("Detecting first available GPU. Platform: %s, Available providers: %s", system, available_providers)
        
        # Check for CUDA (NVIDIA GPUs) - works on Linux and Windows
        if 'CUDAExecutionProvider' in available_providers:
            try:
                # Try to detect CUDA devices
                result = subprocess.run(
                    ['nvidia-smi', '-L'], 
                    capture_output=True, 
                    text=True, 
                    timeout=5,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpu_count = len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
                    if gpu_count > 0:
                        logger.info("Found %d CUDA GPU(s). Using device 0.", gpu_count)
                        return 0
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
                logger.debug("Could not query NVIDIA GPUs: %s", e)
                # Even if nvidia-smi fails, CUDA provider might still work
                logger.info("CUDA provider available, defaulting to device 0")
                return 0
        
        # Check for platform-specific GPU providers
        if system == 'darwin':  # macOS
            # Check for CoreML or Metal
            if 'CoreMLExecutionProvider' in available_providers:
                logger.info("CoreML provider available on macOS. Using device 0.")
                return 0
            # Note: Metal (MPS) support in ONNX Runtime typically uses CUDA provider interface
            
        elif system == 'windows':
            # Check for DirectML (Windows GPU acceleration)
            if 'DmlExecutionProvider' in available_providers:
                logger.info("DirectML provider available on Windows. Using device 0.")
                return 0
        
        # No GPU found
        logger.warning("No GPU device detected. Will fall back to CPU.")
        return -1
    
    @staticmethod
    def get_model_session(prefer_gpu: bool = True, device_id: int = -1, input_channels: int = 3) -> ort.InferenceSession:
        """Get the ONNX model session from the extension instance.
        
        This method loads the layer segmentation model and creates an ONNX runtime session.
        The model is loaded from the extension's model directory and decompressed before use.
        
        Args:
            prefer_gpu (bool, optional): Whether to prefer GPU acceleration. Defaults to True.
            device_id (int, optional): GPU device ID to use. Use -1 to auto-detect the first 
                available GPU device. Defaults to -1.
            input_channels (int, optional): Number of input channels (1 for sparse, 3 for standard). Defaults to 3.
            
        Returns:
            ort.InferenceSession: The ONNX runtime inference session for layer segmentation.
            
        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If the model fails to load or session creation fails.
        """
        try:
            # Auto-detect first available GPU if device_id is -1
            if device_id == -1:
                logger.info("device_id=-1 specified. Auto-detecting first available GPU...")
                detected_device_id = RetinalLayerSegmentationExtension.get_first_available_gpu()
                if detected_device_id == -1:
                    logger.warning("No GPU detected. Falling back to CPU execution.")
                    prefer_gpu = False
                    device_id = 0  # Use 0 as default, but prefer_gpu=False will use CPU
                else:
                    device_id = detected_device_id
                    logger.info("Auto-detected GPU device_id: %d", device_id)
            
            # Get extension directory relative to this file
            extension_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Get model file path using reassembly module
            if input_channels == 1:
                # Get extension directory for sparse model
                model_path = os.path.join(extension_dir, "model", "layersegmodel_sparse.enc")
            else:
                # Use reassembly module for main model
                from .model_reassembly import get_model_file_path
                model_path_obj = get_model_file_path()
                if model_path_obj is None:
                    raise RuntimeError(
                        "Failed to load model file. The model may not have been "
                        "properly reassembled from chunks. Please reinstall the package."
                    )
                model_path = str(model_path_obj)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Load and decompress the model
            logger.info("Loading layer segmentation model from: %s", model_path)
            model_buffer = utils.loadDLModel(model_path)
            
            # Create ONNX session with optimal providers
            logger.info("Creating ONNX inference session (prefer_gpu=%s, device_id=%d)", prefer_gpu, device_id)
            ort_session = create_onnx_session(
                model_buffer, 
                device_id=device_id, 
                prefer_gpu=prefer_gpu, 
                optimization_level="basic"
            )
            
            logger.info("Model session created successfully")
            return ort_session
            
        except FileNotFoundError as e:
            logger.error("Model file not found: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to create model session: %s", e)
            raise RuntimeError(f"Failed to create model session: {e}") from e
    
    @staticmethod
    def run_retinal_layer_segmentation(
        session: ort.InferenceSession, 
        input_volume_array: np.ndarray,
        input_channels: int = 3,
        permute: str = "0,1,2",
        flip: str = "None",
        flatten_offset: int = 0,
        flatten_baseline: int = -1,
        is_flatten: bool = False,
        crop_top: int = 0,
        crop_bottom: int = 0
    ) -> tuple:
        """Run retinal layer segmentation on the input volume array.
        
        This method performs AI-based retinal layer segmentation using the provided ONNX session
        and input OCT volume data. It returns curve data and fluid mask.
        
        Args:
            session (ort.InferenceSession): The ONNX runtime inference session.
            input_volume_array (np.ndarray): Input OCT volume data with shape (z, y, x).
            input_channels (int, optional): Number of input channels. Defaults to 3.
            permute (str, optional): Permutation string. Defaults to "0,1,2".
            flip (str, optional): Flip direction. Defaults to "None".
            flatten_offset (int, optional): Flatten offset. Defaults to 0.
            flatten_baseline (int, optional): Flatten baseline. Defaults to -1.
            is_flatten (bool, optional): Whether the data is flattened. Defaults to False.
            crop_top (int, optional): Crop offset at the top. Defaults to 0.
            crop_bottom (int, optional): Crop offset at the bottom. Defaults to 0.
            
        Returns:
            tuple: (curve_data, fluid_mask, layer_seg_mat)
                - curve_data: Dictionary containing curve information
                - fluid_mask: Fluid segmentation mask
                - layer_seg_mat: Layer segmentation matrix
                
        Raises:
            ValueError: If the input volume array is None or has invalid shape.
            RuntimeError: If the segmentation fails.
        """
        if input_volume_array is None:
            raise ValueError("Input volume array cannot be None")
        
        if len(input_volume_array.shape) != 3:
            raise ValueError(f"Input volume must be 3D, got shape: {input_volume_array.shape}")
        
        try:
            logger.info("Running layer segmentation on volume with shape: %s", input_volume_array.shape)
            
            # Store original shape for later resizing if needed
            raw_size = input_volume_array.shape
            octdata_ = input_volume_array.copy()
            
            # Check the size and resize if needed
            resized = False
            if octdata_.shape[0] >= 640 and octdata_.shape[2] >= 640:
                # Resize 3D data
                output_shape = [raw_size[0] // 2, raw_size[1], raw_size[2] // 2]
                octdata_ = utils.resize3D(octdata_, output_shape, order=0)
                # Resize flatten offset if needed
                if not isinstance(flatten_offset, int) and flatten_offset is not None:
                    flatten_offset = utils.resize3D(
                        flatten_offset, [output_shape[0], output_shape[2]], order=0
                    )
                resized = True
            
            # Normalize data
            octdata = utils.mat2gray(octdata_)
            
            # Preprocessing: crop data if needed
            if crop_bottom == 0:
                octdata_crop, [top, bottom] = utils.cropping_data_for_layerSegment(
                    octdata,
                    min_height=416,
                    is_flatten=is_flatten,
                    flatten_baseline=flatten_baseline,
                    flatten_offset=flatten_offset,
                )
                crop_top = top
                crop_bottom = bottom
            else:
                octdata_crop = octdata[:, crop_top:crop_bottom, :]
            
            # Run the model
            logger.info("Running model inference...")
            layerSegMat = utils.runModel(
                session, 
                octdata_crop,
                input_shape=[None, None, input_channels]
            )
            
            # Generate curves from segmentation matrix
            logger.info("Generating layer boundaries...")
            curve_data, fluid_mask, _ = utils.generateCurve(
                layerSegMat,
                permute,
                flip,
                flatten_offset,
                flatten_baseline,
                is_flatten,
                crop_top,
            )
            
            # Pad fluid mask to original size
            fluid_mask = np.pad(
                fluid_mask,
                ((0, 0), (crop_top, raw_size[1] - crop_bottom), (0, 0)),
                mode="constant",
                constant_values=0
            )
            
            # Inverse flatten if needed
            if is_flatten and not isinstance(flatten_offset, int):
                fluid_mask = utils.roll_volume_inverse(fluid_mask, flatten_offset)
            
            # Update curve data with volume size
            curve_data["volumeSize"] = raw_size
            curve_data["flatten_offset"] = flatten_offset
            
            # Resize back to original size if needed
            if resized:
                curve_data = utils.resizeCurveData(curve_data, raw_size)
                fluid_mask = utils.resize3D(fluid_mask, raw_size, order=0)
            
            logger.info("Layer segmentation completed successfully")
            return curve_data, fluid_mask, layerSegMat
            
        except Exception as e:
            logger.error("Layer segmentation failed: %s", e)
            raise RuntimeError(f"Layer segmentation failed: {e}") from e
    
    @staticmethod
    def generate_curve(
        layerSegMat: np.ndarray,
        permute: str = "0,1,2",
        flip: str = "None",
        flatten_offset: int = 0,
        flatten_baseline: int = -1,
        is_flatten: bool = False,
        crop_top: int = 0,
    ) -> tuple:
        """Generate curve data from layer segmentation matrix.
        
        This is a wrapper around utils.generateCurve for convenience.
        
        Args:
            layerSegMat (np.ndarray): Layer segmentation matrix.
            permute (str, optional): Permutation string. Defaults to "0,1,2".
            flip (str, optional): Flip direction. Defaults to "None".
            flatten_offset (int, optional): Flatten offset. Defaults to 0.
            flatten_baseline (int, optional): Flatten baseline. Defaults to -1.
            is_flatten (bool, optional): Whether the data is flattened. Defaults to False.
            crop_top (int, optional): Crop offset at the top. Defaults to 0.
            
        Returns:
            tuple: (curve_data, fluid_mask, curve_array)
                - curve_data: Dictionary containing curve information
                - fluid_mask: Fluid segmentation mask
                - curve_array: Raw curve array
                
        Raises:
            ValueError: If the input segmentation matrix is invalid.
        """
        if layerSegMat is None:
            raise ValueError("Layer segmentation matrix cannot be None")
        
        if len(layerSegMat.shape) != 3:
            raise ValueError(f"Layer segmentation matrix must be 3D, got shape: {layerSegMat.shape}")
        
        try:
            logger.info("Generating curves from segmentation matrix with shape: %s", layerSegMat.shape)
            
            curve_data, fluid_mask, curve_array = utils.generateCurve(
                layerSegMat,
                permute,
                flip,
                flatten_offset,
                flatten_baseline,
                is_flatten,
                crop_top,
            )
            
            logger.info("Curve generation completed successfully")
            return curve_data, fluid_mask, curve_array
            
        except Exception as e:
            logger.error("Curve generation failed: %s", e)
            raise RuntimeError(f"Curve generation failed: {e}") from e

# Export the extension class for discovery
__all__ = ['RetinalLayerSegmentationExtension']
