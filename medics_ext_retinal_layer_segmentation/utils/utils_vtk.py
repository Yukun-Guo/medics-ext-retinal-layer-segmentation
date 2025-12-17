# VTK imports with error handling
import logging
logger = logging.getLogger(__name__)

try:
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
    from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
    from vtkmodules.vtkRenderingCore import (
        vtkColorTransferFunction,
        vtkVolumeProperty,
        vtkVolume,
    )
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
    
    VTK_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning("Warning: VTK modules not available in utils_vtk: %s", e)
    logger.warning("VTK-related functionality will be disabled.")
    # Create placeholder classes to prevent runtime errors
    class vtkImageData: pass
    class vtkPiecewiseFunction: pass
    class vtkGPUVolumeRayCastMapper: pass
    class vtkColorTransferFunction: pass
    class vtkVolumeProperty: pass
    class vtkVolume: pass
    VTK_UNSIGNED_CHAR = None
    numpy_support = None
    VTK_AVAILABLE = False

import numpy as np

def numpy_to_vtk_image_data(numpy_array: np.ndarray) -> vtkImageData:
    """Converts a NumPy array to vtkImageData.

    Args:
        numpy_array (np.ndarray): The NumPy array to convert.

    Returns:
        vtkImageData: The converted vtkImageData object.
    """
    if not VTK_AVAILABLE:
        logger.warning("VTK is not available. Cannot convert numpy array to vtkImageData.")
        return None
        
    # Create vtkImageData
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(numpy_array.shape)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)  # Set voxel spacing (modify as needed)
    
    # Convert NumPy array to vtkArray
    flat_array = numpy_array.ravel(order="F")  # Flatten the array in Fortran order
    vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=VTK_UNSIGNED_CHAR)
    
    # Set vtkArray as the scalars of vtkImageData
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image

def render_volume(
    vtk_image: vtkImageData,
    color_labels: list = None,
    opacity: float = 1.0,
    ambient: float = 0.4,
    diffuse: float = 0.8,
    specular: float = 0.5,
) -> vtkVolume:
    """Sets up the volume rendering pipeline.

    Args:
        vtk_image (vtkImageData): The vtkImageData to render.
        color_labels (list, optional): List of RGB color labels. Defaults to None.
        opacity (float, optional): Opacity value for the volume. Defaults to 1.0.
        ambient (float, optional): Base light level (dull effect). Defaults to 0.4.
        diffuse (float, optional): Light scattering (soft shadows). Defaults to 0.8.
        specular (float, optional): Reflective highlights. Defaults to 0.5.

    Returns:
        vtkVolume: The configured vtkVolume object.
    """
    if not VTK_AVAILABLE:
        logger.warning("VTK is not available. Cannot render volume.")
        return None
        
    # Volume mapper
    volume_mapper = vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image)
    volume_property = vtkVolumeProperty()
    
    # Volume color transfer function
    if color_labels is not None:
        color_func = vtkColorTransferFunction()
        for i, color in enumerate(color_labels):
            color_func.AddRGBPoint(i, color[0]/255.0, color[1]/255.0, color[2]/255.0)
        volume_property.SetColor(color_func)
    
    opacity_func = vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(1, opacity)  # All other values are fully opaque
    opacity_func.AddPoint(255, opacity)  # Adjust the max value if necessary
    volume_property.SetScalarOpacity(opacity_func)
    
    volume_property.ShadeOn()  # Disable shading for categorical data
    volume_property.SetAmbient(ambient)  # Base light level (dull effect)
    volume_property.SetDiffuse(diffuse)  # Light scattering (soft shadows)
    volume_property.SetSpecular(specular)  # Reflective highlights
    volume_property.SetSpecularPower(200.0)  # Sharpness of highlights
    volume_property.SetInterpolationTypeToNearest()
    # Volume actor
    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume
