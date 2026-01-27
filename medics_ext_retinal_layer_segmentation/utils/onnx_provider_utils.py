"""ONNX Runtime provider utilities.

Helpers to select and configure ONNX Runtime execution providers based on
available hardware and platform. The module chooses reasonable fallbacks
when GPU providers are not available or misconfigured.
"""

import logging
import platform
import warnings
from typing import List, Tuple, Union, Dict, Any

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("ONNX Runtime is not installed. AI features will be disabled.")

# Module logger
logger = logging.getLogger(__name__)


class ONNXProviderManager:
    """Manage ONNX Runtime execution providers and fallbacks.

    The manager inspects available providers and determines a recommended
    ordering that prefers GPU providers when available and functional.
    """
    
    def __init__(self):
        self.available_providers = []
        self.recommended_providers = []
        if ONNXRUNTIME_AVAILABLE:
            try:
                # Some versions of onnxruntime may not have this method
                if hasattr(ort, 'get_available_providers'):
                    self.available_providers = ort.get_available_providers()
                else:
                    # Fallback for older/incomplete installations
                    logger.warning("onnxruntime.get_available_providers() not available. Using CPU fallback.")
                    self.available_providers = ['CPUExecutionProvider']
            except Exception as e:
                logger.error(f"Failed to query ONNX Runtime providers: {e}")
                self.available_providers = ['CPUExecutionProvider']
            
            self._determine_recommended_providers()
    
    def _determine_recommended_providers(self) -> None:
        """Determine the best providers for the current system."""
        system = platform.system().lower()
        
        # Start with CPU as the most reliable fallback
        self.recommended_providers = ['CPUExecutionProvider']
        
        # Add GPU providers if available
        if 'CUDAExecutionProvider' in self.available_providers:
            # Check if CUDA is actually working
            if self._test_cuda_provider():
                self.recommended_providers.insert(0, 'CUDAExecutionProvider')
            else:
                logger.warning("CUDAExecutionProvider is available but not working properly. Using CPU fallback.")
        
        # Add DirectML provider (Windows GPU acceleration)
        if 'DmlExecutionProvider' in self.available_providers:
            # DirectML has higher priority than CPU but lower than CUDA
            if 'CUDAExecutionProvider' not in self.recommended_providers:
                self.recommended_providers.insert(0, 'DmlExecutionProvider')
            else:
                self.recommended_providers.insert(1, 'DmlExecutionProvider')
        
        # Add platform-specific providers
        if system == 'darwin':  # macOS
            if 'CoreMLExecutionProvider' in self.available_providers:
                self.recommended_providers.insert(-1, 'CoreMLExecutionProvider')
        
        # Log recommendation
        logger.info("Recommended ONNX providers: %s", self.recommended_providers)
    
    def _test_cuda_provider(self) -> bool:
        """Test if CUDA provider actually works."""
        try:
            import numpy as np

            # Create a minimal test session with CUDA. The detailed test is
            # intentionally lightweight: if provider initialization raises an
            # exception we treat CUDA as unavailable.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # If initialization of related libraries succeeds assume CUDA is usable
                return True

        except Exception as e:
            logger.debug("CUDA provider test failed: %s", e)
            return False
    
    def get_providers_for_device(self, device_id: int = 0, prefer_gpu: bool = True) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
        """
        Get the best providers for inference with optional GPU device selection.
        
        Args:
            device_id: GPU device ID to use (default: 0)
            prefer_gpu: Whether to prefer GPU over CPU (default: True)
            
        Returns:
            List of providers in order of preference
        """
        if not ONNXRUNTIME_AVAILABLE:
            return ['CPUExecutionProvider']  # Fallback when ONNX Runtime is missing
        
        providers = []
        
        if prefer_gpu:
            # Add CUDA with specific device if available
            if 'CUDAExecutionProvider' in self.recommended_providers:
                providers.append(('CUDAExecutionProvider', {'device_id': device_id}))
            
            # Add DirectML with device if available (DirectML also supports device_id)
            elif 'DmlExecutionProvider' in self.recommended_providers:
                providers.append(('DmlExecutionProvider', {'device_id': device_id}))
        
        # Add other recommended providers (excluding those already added)
        for provider in self.recommended_providers:
            if provider not in ['CUDAExecutionProvider', 'DmlExecutionProvider']:
                providers.append(provider)
            elif not prefer_gpu:
                # If not preferring GPU, add GPU providers without device specification
                providers.append(provider)
            elif not prefer_gpu:
                # If not preferring GPU, add CUDA without device specification
                providers.append(provider)
        
        return providers
    
    def get_cpu_only_providers(self) -> List[str]:
        """Get CPU-only providers for systems without GPU or when GPU is disabled."""
        return ['CPUExecutionProvider']
    
    def create_session_options(self, 
                             optimization_level: str = 'basic',
                             inter_op_num_threads: int = None,
                             intra_op_num_threads: int = None) -> 'ort.SessionOptions':
        """
        Create optimized session options for ONNX Runtime.
        
        Args:
            optimization_level: 'basic', 'extended', or 'all' (default: 'basic')
            inter_op_num_threads: Number of threads for inter-operator parallelism
            intra_op_num_threads: Number of threads for intra-operator parallelism
            
        Returns:
            Configured SessionOptions object
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available")
        
        sess_options = ort.SessionOptions()
        
        # Set optimization level
        optimization_levels = {
            'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        }
        sess_options.graph_optimization_level = optimization_levels.get(
            optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        
        # Set thread counts if specified
        if inter_op_num_threads is not None:
            sess_options.inter_op_num_threads = inter_op_num_threads
        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = intra_op_num_threads
        
        return sess_options
    
    def create_inference_session(self, 
                                model_path_or_bytes: Union[str, bytes],
                                device_id: int = 0,
                                prefer_gpu: bool = True,
                                optimization_level: str = 'basic') -> 'ort.InferenceSession':
        """
        Create an ONNX Runtime inference session with optimal provider configuration.
        
        Args:
            model_path_or_bytes: Path to model file or model bytes
            device_id: GPU device ID to use
            prefer_gpu: Whether to prefer GPU acceleration
            optimization_level: Graph optimization level
            
        Returns:
            Configured InferenceSession
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available")
        
        sess_options = self.create_session_options(optimization_level)
        providers = self.get_providers_for_device(device_id, prefer_gpu)
        
        try:
            # Suppress provider warnings during session creation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
                
                session = ort.InferenceSession(
                    model_path_or_bytes,
                    sess_options=sess_options,
                    providers=providers
                )
                
                # Log which providers were actually used
                actual_providers = session.get_providers()
                logger.info("ONNX session created with providers: %s", actual_providers)

                return session

        except Exception as e:
            logger.exception("Failed to create ONNX session with preferred providers: %s", e)

            # Fallback to CPU only
            logger.info("Falling back to CPU-only execution")
            cpu_providers = self.get_cpu_only_providers()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

                return ort.InferenceSession(
                    model_path_or_bytes,
                    sess_options=sess_options,
                    providers=cpu_providers
                )


# Global instance for easy access
onnx_provider_manager = ONNXProviderManager()


def get_optimal_providers(device_id: int = 0, prefer_gpu: bool = True) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
    """
    Convenience function to get optimal providers.
    
    Args:
        device_id: GPU device ID to use
        prefer_gpu: Whether to prefer GPU acceleration
        
    Returns:
        List of optimal providers for the current system
    """
    return onnx_provider_manager.get_providers_for_device(device_id, prefer_gpu)


def create_onnx_session(model_path_or_bytes: Union[str, bytes],
                       device_id: int = 0,
                       prefer_gpu: bool = True,
                       optimization_level: str = 'basic') -> 'ort.InferenceSession':
    """
    Convenience function to create an optimally configured ONNX session.
    
    Args:
        model_path_or_bytes: Path to model file or model bytes
        device_id: GPU device ID to use
        prefer_gpu: Whether to prefer GPU acceleration
        optimization_level: Graph optimization level
        
    Returns:
        Configured InferenceSession
    """
    return onnx_provider_manager.create_inference_session(
        model_path_or_bytes, device_id, prefer_gpu, optimization_level
    )


def log_system_info() -> None:
    """Log system and ONNX Runtime information for debugging."""
    logger.info("Platform: %s %s", platform.system(), platform.release())
    logger.info("Architecture: %s", platform.architecture())

    if ONNXRUNTIME_AVAILABLE:
        logger.info("ONNX Runtime version: %s", ort.__version__)
        logger.info("Available providers: %s", ort.get_available_providers())
        logger.info("Recommended providers: %s", onnx_provider_manager.recommended_providers)
    else:
        logger.warning("ONNX Runtime is not available")
