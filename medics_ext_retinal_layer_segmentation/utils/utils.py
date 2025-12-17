import keyword
import os
import pathlib
import re
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata
from .datadict import dataDict
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           binary_opening, distance_transform_bf,
                           distance_transform_cdt, distance_transform_edt,
                           gaussian_filter, generate_binary_structure, label,
                           median_filter, uniform_filter, zoom)
from scipy.stats import pearsonr
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed
from skimage.transform import resize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import logging
logger = logging.getLogger(__name__)

class utils(object):
    """
    Utility class for image processing and data manipulation in OCT/OCTA analysis.

    This class provides a collection of static methods for 2D/3D image processing, segmentation, curve fitting,
    region growing, normalization, model inference, and file/path utilities. It is designed to support
    OCT/OCTA data workflows, including segmentation, flattening, enface projection, and quantitative analysis.

    Member Functions:
        resize3D(img3d, output_shape, order=0) -> np.ndarray:
            Resize a 3D image to the specified output shape.

        zoom3D(img3d, zoom_factor, order=0) -> np.ndarray:
            Zoom a 3D image by the specified zoom factor.

        calculate_edge(img) -> np.ndarray:
            Calculate the edges of a 2D image using dilation.

        get_files_in_folder(folder, pattern, recursive=True) -> List[str]:
            Retrieve all files in a folder and its subfolders matching a regex pattern.

        auto_range(mat, low_percentile_rg=(0.2, 0.35), high_percentile_rg=(0.99, 1)) -> Tuple[float, float]:
            Calculate the auto range for normalization based on percentiles.

        mat2gray(mat, min_max=None, autoAjust=False) -> np.ndarray:
            Normalize a matrix to the range [0, 1].

        caculate_illumination_map(im) -> np.ndarray:
            Placeholder function to calculate illumination map.

        resize_pow2_size(img, pow2=5, min_size=None) -> np.ndarray:
            Resize an image to the nearest power-of-2 size.

        restore_resized_pow2size(img, raw_size) -> np.ndarray:
            Restore an image to its original size after resizing to power-of-2.

        crop_volume(volume_data, flatten_baseline, flatten_offset, max_height=512) -> Tuple[np.ndarray, List[int]]:
            Crop a volume based on flatten parameters and maximum height.

        cropping_data_for_layerSegment(oct_data, min_height=320, is_flatten=False, flatten_baseline=-1, flatten_offset=[0,0]) -> Tuple[np.ndarray, List[int]]:
            Crop OCT data for layer segmentation.

        restore_cropped_data_for_layerSegment(layerSegMat, top, bottom) -> None:
            Pad the layerSegMat to the original size.

        validate_variable_name(name, default_name) -> str:
            Validate and return a Python variable name.

        roll_volume(volume_data, reference_surface) -> Tuple[np.ndarray, np.ndarray, int]:
            Roll each 2D slice in a 3D volume based on a reference surface.

        roll_volume_offsets(volume_data, reference_offsets) -> np.ndarray:
            Roll each 2D slice in a 3D volume based on reference offsets.

        roll_volume_inverse(volume_data, offsets) -> np.ndarray:
            Inverse roll for each 2D slice in a 3D volume.

        flatten_volume(volume_data, surface=None, degree=2) -> Tuple[np.ndarray, np.ndarray, int]:
            Flatten a 3D volume based on a surface or by fitting a curve.

        calculate_roi(volume_data, method="Fitting", max_height=640, degree=2) -> Tuple[int, int]:
            Calculate region of interest (ROI) in a volume.

        add_flatten_parameters_in_curve_dict(curveDict, flatten_offset, flatten_baseline, flatten_permute) -> Dict[str, Any]:
            Add flatten parameters to each entry in the curve dictionary.

        calculate_flatten_parameters(volume_data, method="fitting", degree=2) -> Tuple[Any, int]:
            Calculate flatten offsets and baseline for a volume.

        flatten_oct_octa(oct_data, octa_data, permute="auto", method="fitting", reference_curve=None, degree=2) -> Tuple:
            Flatten OCT and/or OCTA data using a reference surface or fitted curve.

        transform_volume(volume_data, flatten_offset=None, permute=(0,1,2), flip="None") -> np.ndarray:
            Apply flattening, permutation, and flipping to a volume.

        restore_transform_volume(volume_data, flatten_offset=None, permute=(0,1,2), flip="None") -> np.ndarray:
            Restore volume data from transformed state.

        moving_average(arr, window_size) -> np.ndarray:
            Compute moving average of a 1D array.

        smooth_difference(arr, window_size) -> np.ndarray:
            Compute absolute difference between original and smoothed array.

        auto_permute(volume_data) -> List[int]:
            Automatically determine the best axis permutation for a 3D volume.

        createZeroCurve(nimgs, nrows, ncols, permute="0,1,2") -> Dict[str, Any]:
            Create a zero-initialized curve dictionary for a given shape and permutation.

        createZeroCurveDict(nimgs, nrows, ncols) -> Dict[str, Any]:
            Create a dictionary of zero-initialized curves for all flip and permute combinations.

        generateCurveDictKey(permute, flip) -> str:
            Generate a key for curve dictionary based on permute and flip.

        validateCurve(curve) -> bool:
            Validate if the input is a proper curve dictionary.

        resizeCurveData(curve, output_shape, order=2) -> Dict[str, Any]:
            Resize all curve arrays in the curve dictionary to a new shape.

        generateCurve(layerSegMat, permute="0,1,2", flip="None", flatten_offset=0, flatten_baseline=-1, is_flatten=False, crop_top=0) -> Tuple:
            Generate curve data and fluid mask from a layer segmentation matrix.

        cleanUpFluidMat(fluidMat, curves, flatten=0, crop=0) -> np.ndarray:
            Clean up the fluid matrix by masking with region between NFLGCL and RPEBM.

        generateCurveMultiSegMats(layerSegMats, permute="0,1,2", flip="None", flatten_offset=0, flatten_baseline=-1, is_flatten=False, crop_top=0) -> Tuple:
            Generate curve and fluid mask from multiple segmentation matrices.

        generateRetinalFluid(layerSegMat) -> np.ndarray:
            Generate fluid mask from a layer segmentation matrix.

        keepLargestComponents3D(bw3D) -> np.ndarray:
            Keep only the largest connected component in a 3D binary image.

        binary_area_open(bw, min_size) -> np.ndarray:
            Remove small connected components from a binary image.

        loadDLModel(model_path) -> bytes:
            Load and decompress a deep learning model from file.

        runModel(ortSession, octData, input_shape=(None, None, 5), downsample_level=5, batch_size=1) -> np.ndarray:
            Run the segmentation model on OCT data with multi-batch support.

        run_ga_seg_model(ortSession, octData, input_shape=(None, None, 5), downsample_level=5) -> Tuple[np.ndarray, np.ndarray]:
            Run the GA segmentation model.

        run_drusen_seg_model(ortSession, octData, input_shape=[None, None, 5], downsample_level=5) -> Tuple[np.ndarray, np.ndarray]:
            Run the drusen segmentation model.

        run_npa_seg_model(ort_session, octa_data, oct_data, thk_data, downsample_level=5) -> np.ndarray:
            Run the NPA segmentation model.

        get_slab_mask(curve_data, slab_name) -> np.ndarray:
            Generate a slab mask based on curve data and slab name.

        project_enface_image(oct_data, curve_data, slab_name, proj_type) -> np.ndarray:
            Project an enface image based on slab mask and projection type.

        generateEnfaceImage(oct_data, slab_mask, v_size, proj_type) -> np.ndarray:
            Generate an enface image based on slab mask and projection type.

        interpolate_curve(surfaceData, mask=None, filter_window=55) -> np.ndarray:
            Interpolate a curve surface using grid data.

        fit_curve_surface(surfaceData, mask=None, degree=2) -> np.ndarray:
            Fit a polynomial surface to the curve data.

        generate_ilm_map(im) -> np.ndarray:
            Generate an ILM map using Gaussian filters.

        generate_2D_gaussion_map(shape, center, sigma) -> np.ndarray:
            Generate a 2D Gaussian map.

        gabor_filter(image, theta=0, sigma=0.2, frequency=5) -> np.ndarray:
            Apply a Gabor filter to an image.

        bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75) -> np.ndarray:
            Apply a bilateral filter to an image.

        generate_cRORA_HyperTDs(GAVolume, layerVolume, cRORA_diameter_thres_mm=0.25, scan_width_mm=6, scan_height_mm=6) -> Tuple:
            Generate cRORA and HyperTDs features from layer and GA volumes.

        generate_cRORA_HyperTDs_from_labeled_volume(GAVolume, cRORA_diameter_thres_mm=0.25, scan_width_mm=6, scan_height_mm=6) -> Tuple:
            Generates cRORA and hypertransmission defects (HyperTDs) masks from a labeled volume.

        generate_drusen_props(drusen_volume, pixel_size) -> List[float]:
            Compute volume and count for each drusen type in a labeled 3D volume.

        filter_regions_by_neborhood_label(volume, label_for_check, label_must_adjcent) -> np.ndarray:
            Filter out regions in a labeled volume unless they are adjacent to a specific label.

        generate_drusens(RPEDC, DV, drusen_diameter_thresholds_mm=(0.063, 0.125, 0.35), scan_width_mm=6, scan_height_mm=6, axiel_res_mm=0.002, use_max_diameter=True):
            Generate drusen masks and volumes from segmented RPEDC and drusenoid volumes.

        generate_fluid_props(fluid_volume, pixel_size) -> List[float]:
            Calculate fluid volumes based on pixel count and pixel size.

        fill_holes(image, hole_size=30):
            Fill small holes in a binary image.

        generateRegionGrowMask(image, center, stop_circle_radius, loDiff, upDiff) -> np.ndarray:
            Perform 2D region growing from a seed point with stop mask.

        generateRegionGrowMasks(image, centers, stop_circle_radius, loDiff, upDiff) -> np.ndarray:
            Generate region growing masks for multiple seed points.

        generateRegionGrowMask3D(volumeData, currentFrameIdx, center, stop_circle_radius, loDiff, upDiff) -> Tuple[np.ndarray, List[int]]:
            Generate a 3D binary mask using region growing from a center point in a volumetric image.

        generateAdaptThresMask(image, stopMask, loweroffset=0, upperoffset=0) -> np.ndarray:
            Generate a binary mask by thresholding an image within a specified mask region.

        generateAdaptThresMasks(image, centers, stop_circle_radius, angle_degrees, loweroffset=0, upperoffset=0, shape_generator=None) -> np.ndarray:
            Generate binary masks for multiple seed points using adaptive thresholding.

        generateAdaptThresMask3D(volumeData, currentFrameIdx, center=[0,0], stop_circle_radius=1, loweroffset=0, upperoffset=0, mask=None, propagate_step=1) -> Tuple[np.ndarray, List[int]]:
            Generate a 3D mask by adaptive thresholding around a seed point or region.

        generateAdaptThresMasks3D(volumeData, currentFrameIdx, centers, stop_circle_radius, loweroffset, upperoffset, propagate_step=1, angle_degrees=0, shape_generator=None) -> Tuple[np.ndarray, List[int]]:
            Generate a 3D adaptive threshold mask using multiple centers.

        calc_relative_path(file_paths, common_path=None, calculate_common_path=True) -> List[str]:
            Calculate relative paths from a list of file paths, handling different drive letters.
    """

    @staticmethod
    def resize3D(img3d: np.ndarray, output_shape: Tuple[int, int, int], order: int = 0) -> np.ndarray:
        """Resize a 3D image to the specified output shape.

        Args:
            img3d (np.ndarray): Input 3D image.
            output_shape (Tuple[int, int, int]): Desired output shape.
            order (int, optional): Interpolation order. Defaults to 0.

        Returns:
            np.ndarray: Resized 3D image.
        """
        return resize(img3d, output_shape, order=order)


    @staticmethod
    def zoom3D(img3d: np.ndarray, zoom_factor: Tuple[float, float, float], order: int = 0) -> np.ndarray:
        """Zoom a 3D image by the specified zoom factor.

        Args:
            img3d (np.ndarray): Input 3D image.
            zoom_factor (Tuple[float, float, float]): Zoom factor for each dimension.
            order (int, optional): Interpolation order. Defaults to 0.

        Returns:
            np.ndarray: Zoomed 3D image.
        """
        return zoom(img3d, zoom_factor, order=order)


    @staticmethod
    def calculate_edge(img: np.ndarray) -> np.ndarray:
        """Calculate the edges of a 2D image using dilation.

        Args:
            img (np.ndarray): Input 2D image.

        Returns:
            np.ndarray: Edge-detected image.
        """
        # use dilation to get the edge of the image
        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img, kernel, iterations=1)
        img_edge = cv2.absdiff(img, img_dilated)
        return img_edge 


    @staticmethod
    def get_files_in_folder(folder: str, pattern: str, recursive: bool = True) -> List[str]:
        """Retrieve all files in a folder and its subfolders matching a specific regex pattern.

        Args:
            folder (str): Folder path.
            pattern (str): Regex pattern to match files.
            recursive (bool, optional): Whether to search recursively. Defaults to True.

        Returns:
            List[str]: List of file paths matching the pattern.
        """
        files = []
        if pattern is None or pattern == "":
            pattern = "*"
        if recursive:
            for path in pathlib.Path(folder).rglob(pattern):
                files.append(str(path))
        else:
            for path in pathlib.Path(folder).glob(pattern):
                files.append(str(path))
        return files


    @staticmethod
    def auto_range(
        mat: np.ndarray, 
        low_percentile_rg: Tuple[float, float] = (0.2, 0.35), 
        high_percentile_rg: Tuple[float, float] = (0.99, 1)
    ) -> Tuple[float, float]:
        """Calculate the auto range for normalization based on percentiles.

        Args:
            mat (np.ndarray): Input matrix.
            low_percentile_rg (Tuple[float, float], optional): Range for low percentile. Defaults to (0.2, 0.35).
            high_percentile_rg (Tuple[float, float], optional): Range for high percentile. Defaults to (0.99, 1).

        Returns:
            Tuple[float, float]: Calculated lower and upper bounds.
        """
        # get the mean of first 15% to 35% pixel value in tmpMat
        if mat is None:
            return (0, 1)
        tmpMat = mat.reshape(-1)
        tmpMat = np.sort(tmpMat)
        starti = int(len(tmpMat) * low_percentile_rg[0])
        endi = int(len(tmpMat) * low_percentile_rg[1])
        low_bund = np.round(
            np.mean(tmpMat[starti:endi]).astype(np.float32) / (tmpMat[-1] + 0.0000001), 2
        )
        # get the mean of the last 5% to 20% pixel value
        starti = int(len(tmpMat) * high_percentile_rg[0])
        endi = int(len(tmpMat) * high_percentile_rg[1])
        upp_bound = np.round(
            np.mean(tmpMat[starti:endi]).astype(np.float32) / (tmpMat[-1] + 0.0000001), 2
        )
        return (low_bund, upp_bound)


    @staticmethod
    def mat2gray(mat: np.ndarray, min_max: List[float] = None, autoAjust: bool = False) -> np.ndarray:
        """Normalize a matrix to the range [0, 1].

        Args:
            mat (np.ndarray): Input matrix.
            min_max (List[float], optional): Min and max values for normalization. Defaults to None.
            autoAjust (bool, optional): Whether to automatically adjust min and max. Defaults to False.

        Returns:
            np.ndarray: Normalized matrix.
        """
        # get the first 5 images from mat

        # if min_max is not None and it is a list with 2 elements
        # then use the values in min_max as the min and max value
        # if min_max is None or it is not a list with 2 elements, then
        # check if autoAjust is True, if True, then calculate the min and max value, else use the default value [0,1]
        # mat = mat.astype(np.float32)
        meanval = [0, 1]
        if min_max is not None and len(min_max) == 2:
            meanval = min_max
        elif autoAjust:
            # get the mean of first 15% to 35% pixel value in tmpMat
            tmpMat = mat[::60].reshape(-1)
            tmpMat = np.sort(tmpMat)
            starti = int(len(tmpMat) * 0.20)
            endi = int(len(tmpMat) * 0.35)
            meanval[0] = np.mean(tmpMat[starti:endi]).astype(np.float32)
            # get the mean of the last 5% to 20% pixel value
            starti = int(len(tmpMat) * 0.99)
            endi = int(len(tmpMat) * 1)
            meanval[1] = np.mean(tmpMat[starti:endi]).astype(np.float32)
        else:
            meanval = [np.min(mat), np.max(mat)]
        mat = np.clip(mat, meanval[0], meanval[1])
        outMat = (mat - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)

        return outMat.astype(np.float32)


    @staticmethod
    def caculate_illumination_map(im: np.ndarray) -> np.ndarray:
        """Placeholder function to calculate illumination map.

        Args:
            im (np.ndarray): Input image.

        Returns:
            np.ndarray: Illumination map.
        """
        return im


    @staticmethod
    def resize_pow2_size(img: np.ndarray, pow2: int = 5, min_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize an image to the nearest power-of-2 size.

        Args:
            img (np.ndarray): Input image.
            pow2 (int, optional): Power of 2. Defaults to 5.
            min_size (Tuple[int, int], optional): Minimum size. Defaults to None.

        Returns:
            np.ndarray: Resized image.
        """
        size_pow2 = np.power(2, pow2)
        nh, nw = img.shape[0:2]

        offset_r = nh % size_pow2
        offset_c = nw % size_pow2
        if offset_r > 0:
            offset_r = size_pow2 - offset_r
        if offset_c > 0:
            offset_c = size_pow2 - offset_c
        if min_size is not None:
            new_h = max(nh + offset_r, min_size[0])
            new_w = max(nw + offset_c, min_size[1])
        else:
            new_h = max(nh + offset_r, size_pow2)
            new_w = max(nw + offset_c, size_pow2)

        img_r = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if len(img_r.shape) <  len(img.shape):
            img_r = np.expand_dims(img_r, 2)
        # if len(img.shape) == 2:
        #     img_r = np.expand_dims(np.expand_dims(img_r, 2), 0)
        # else:
        #     img_r = np.expand_dims(img_r, 0)
        return img_r


    @staticmethod
    def restore_resized_pow2size(img: np.ndarray, raw_size: Tuple[int, int]) -> np.ndarray:
        """Restore an image to its original size after resizing to power-of-2.

        Args:
            img (np.ndarray): Input image.
            raw_size (Tuple[int, int]): Original size.

        Returns:
            np.ndarray: Restored image.
        """
        return cv2.resize(
            img, dsize=(raw_size[1], raw_size[0]), interpolation=cv2.INTER_NEAREST
        )


    @staticmethod
    def crop_volume(volume_data: np.ndarray, flatten_baseline: int, flatten_offset: int, max_height: int = 512) -> Tuple[np.ndarray, List[int]]:
        """Crop a volume based on flatten parameters and maximum height.

        Args:
            volume_data (np.ndarray): Input volume data.
            flatten_baseline (int): Flatten baseline value.
            flatten_offset (int): Flatten offset value.
            max_height (int, optional): Maximum height for cropping. Defaults to 512.

        Returns:
            Tuple[np.ndarray, List[int]]: Cropped volume and [top, bottom] indices.
        """
        # get the top idx
        # layerSegMatBW = utils.keepLargestComponents3D(volume_data > np.mean(volume_data) * 0.8)

        offset = flatten_offset + flatten_baseline
        top = max(np.min(offset - 100), 0)
        bottom = min(np.max(offset + 100), volume_data.shape[1])
        if bottom - top > max_height:
            top = max(flatten_baseline - max_height // 2, 0)
            bottom = min(flatten_baseline + max_height // 2, volume_data.shape[1])

        return volume_data[:, top:bottom, :], [top, bottom]


    @staticmethod
    def cropping_data_for_layerSegment(
        oct_data: np.ndarray,
        min_height: int = 320,
        is_flatten: bool = False,
        flatten_baseline: int = -1,
        flatten_offset: List[int] = [0, 0],
    ) -> Tuple[np.ndarray, List[int]]:
        """Crop OCT data for layer segmentation.

        Args:
            oct_data (np.ndarray): Input OCT data.
            min_height (int, optional): Minimum height for cropping. Defaults to 320.
            is_flatten (bool, optional): Whether the data is flattened. Defaults to False.
            flatten_baseline (int, optional): Flatten baseline value. Defaults to -1.
            flatten_offset (List[int], optional): Flatten offset values. Defaults to [0, 0].

        Returns:
            Tuple[np.ndarray, List[int]]: Cropped OCT data and [top, bottom] indices.
        """
        if is_flatten:
            oct_data_croped, [top, bottom] = utils.crop_volume(
                oct_data, flatten_baseline, flatten_offset, max_height=min_height
            )
        else:
            # offset, baseline = calculate_flatten_parameters(oct_data, method="Interpolation", degree=2)
            # offset_ = offset + baseline
            top = 0  # max(np.min(offset_ - 150), 0)
            bottom = oct_data.shape[1]  # min(np.max(offset_ + 150), oct_data.shape[1])
            oct_data_croped = oct_data  # [:, top:bottom, :]

        return oct_data_croped, [top, bottom]


    @staticmethod
    def restore_cropped_data_for_layerSegment(layerSegMat: np.ndarray, top: int, bottom: int) -> None:
        """Pad the layerSegMat to the original size.

        Args:
            layerSegMat (np.ndarray): Cropped layer segmentation matrix.
            top (int): Number of rows to pad at the top.
            bottom (int): Number of rows to pad at the bottom.
        """
        # pad the layerSegMat to the original size, the padding value is 0, top is the number of rows to pad at the top of axis 1, bottom is the number of rows to pad at the bottom of axis 1
        layerSegMat = np.pad(
            layerSegMat, ((0, 0), (top, bottom), (0, 0)), mode="constant", constant_values=0
        )


    @staticmethod
    def validate_variable_name(name: str, default_name: str) -> str:
        """Validate and return a Python variable name.

        Args:
            name (str): Input variable name.
            default_name (str): Default name if input is invalid.

        Returns:
            str: Validated variable name.
        """
        # Check if the string is a valid identifier and not a Python keyword
        if name.isidentifier() and not keyword.iskeyword(name):
            return name
        else:
            # Generate a valid name from the given string
            # Replace invalid characters with an underscore
            valid_name = re.sub(r"\W|^(?=\d)", "_", name)

            # Ensure the name doesn't start with a digit and is not a keyword
            if valid_name.isdigit() or keyword.iskeyword(valid_name):
                valid_name = "_" + valid_name

            # Ensure valid identifier by checking again
            if valid_name.isidentifier():
                return valid_name
            else:
                return default_name


    @staticmethod
    def roll_volume(volume_data: np.ndarray, reference_surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Roll each 2D slice in a 3D volume based on a reference surface.

        Args:
            volume_data (np.ndarray): Input 3D volume.
            reference_surface (np.ndarray): Reference surface for rolling.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Rolled volume, offsets, and mean value.
        """
        meanV = int(np.mean(reference_surface))
        offsets = (meanV - reference_surface).astype(int)

        # Define the function to roll a 2D slice based on the shift array
        def roll_slice(slice_2d, shift_row):
            # Roll each column in the 2D slice by the corresponding shift value
            return np.array(list(map(np.roll, slice_2d.T, shift_row))).T

        # Apply rolling to each 2D slice using map, avoiding explicit for loop
        rolled_slices = list(map(roll_slice, volume_data, offsets))
        rolled_arr_3d = np.array(rolled_slices)
        return rolled_arr_3d, offsets, meanV

    @staticmethod
    def roll_volume_offsets(volume_data: np.ndarray, reference_offsets: np.ndarray) -> np.ndarray:
        """Roll each 2D slice in a 3D volume based on reference offsets.

        Args:
            volume_data (np.ndarray): Input 3D volume.
            reference_offsets (np.ndarray): Offsets for rolling.

        Returns:
            np.ndarray: Rolled volume.
        """
        offsets = reference_offsets.astype(int)

        # Define the function to roll a 2D slice based on the shift array
        def roll_slice(slice_2d, shift_row):
            # Roll each column in the 2D slice by the corresponding shift value
            return np.array(list(map(np.roll, slice_2d.T, shift_row))).T

        # Apply rolling to each 2D slice using map, avoiding explicit for loop
        rolled_slices = list(map(roll_slice, volume_data, offsets))
        rolled_arr_3d = np.array(rolled_slices)
        return rolled_arr_3d


    @staticmethod
    def roll_volume_inverse(volume_data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Inverse roll for each 2D slice in a 3D volume.

        Args:
            volume_data (np.ndarray): Input 3D volume.
            offsets (np.ndarray): Offsets for inverse rolling.

        Returns:
            np.ndarray: Inverse rolled volume.
        """
        def roll_slice(slice_2d, shift_row):
            return np.array(list(map(np.roll, slice_2d.T, -shift_row))).T

        rolled_slices = list(map(roll_slice, volume_data, offsets))
        rolled_arr_3d = np.array(rolled_slices)
        return rolled_arr_3d


    @staticmethod
    def flatten_volume(volume_data: np.ndarray, surface: np.ndarray = None, degree: int = 2) -> Tuple[np.ndarray, np.ndarray, int]:
        """Flatten a 3D volume based on a surface or by fitting a curve.

        Args:
            volume_data (np.ndarray): Input 3D volume.
            surface (np.ndarray, optional): Surface for flattening. Defaults to None.
            degree (int, optional): Degree for curve fitting. Defaults to 2.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Flattened volume, offsets, and mean value.
        """
        if surface is not None:
            return utils.roll_volume(volume_data, surface)
        else:
            surface = np.argmax(volume_data, axis=1)
            val = np.max(volume_data, axis=1)
            z_surf = utils.fit_curve_surface(surface, val > np.max(val) * 0.8, degree)
            return utils.roll_volume(volume_data, z_surf)


    @staticmethod
    def calculate_roi(volume_data: np.ndarray, method: str = "Fitting", max_height: int = 640, degree: int = 2) -> Tuple[int, int]:
        """Calculate region of interest (ROI) in a volume.

        Args:
            volume_data (np.ndarray): Input 3D volume.
            method (str, optional): Method for surface calculation. Defaults to "Fitting".
            max_height (int, optional): Maximum height for ROI. Defaults to 640.
            degree (int, optional): Degree for curve fitting. Defaults to 2.

        Returns:
            Tuple[int, int]: Top and bottom indices for ROI.
        """
        surface = np.argmax(volume_data, axis=1)
        val = np.max(volume_data, axis=1)
        if method == "Fitting":
            z_surf = utils.fit_curve_surface(surface, val > np.max(val) * 0.6, degree)
        else:  # method == "Interpolation":
            z_surf = utils.interpolate_curve(surface, val > np.max(val) * 0.7, 35)
        flatten_baseline = int(np.mean(z_surf))

        top = max(np.min(z_surf - 200), 0)
        bottom = min(np.max(z_surf + 100), volume_data.shape[1])
        if bottom - top > max_height:
            top = max(flatten_baseline - max_height // 2, 0)
            bottom = min(flatten_baseline + max_height // 2, volume_data.shape[1])

        return top, bottom

    @staticmethod
    def add_flatten_parameters_in_curve_dict(
        curveDict: dataDict,
        flatten_offset: Any,
        flatten_baseline: Any,
        flatten_permute: Any
    ) -> dataDict:
        """Add flatten parameters to each entry in the curve dictionary.

        Args:
            curveDict (dataDict): Dictionary containing curve data.
            flatten_offset (Any): Flatten offset to be added.
            flatten_baseline (Any): Flatten baseline to be added.
            flatten_permute (Any): Permutation tuple for flattening.

        Returns:
            dataDict: Updated curve dictionary with flatten parameters.
        """

        for keys in curveDict.keys():
            # get flip and permute from the keys
            if keys.lower() == "volumesize":
                continue
            flip, permute = keys.split("_")
            # get the curve from the curveDict
            if flatten_permute[1] == int(permute[1]):
                p = [1, 0] if int(flatten_permute[0]) > int(permute[2]) else [0, 1]
                tmp_offsets = np.transpose(flatten_offset.copy(), p)
                if flip.lower() == "leftright":
                    tmp_offsets = tmp_offsets[:, ::-1]
                elif flip.lower() == "updown":
                    tmp_offsets = tmp_offsets[::-1, :]
                curveDict[keys]["flatten_offset"] = tmp_offsets
                curveDict[keys]["flatten_baseline"] = flatten_baseline
                # add the flatten_offset from the curve
                # for curve_name in curveDict[keys]["curves"].keys():
                #     curveDict[keys]["curves"][curve_name] = curveDict[keys]["curves"][curve_name] - flatten_offset

            else:
                curveDict[keys]["flatten_offset"] = 0
                curveDict[keys]["flatten_baseline"] = -1
        return curveDict


    @staticmethod
    def calculate_flatten_parameters(
        volume_data: Any, method: str = "fitting", degree: int = 2
    ) -> Tuple[Any, int]:
        """Calculate flatten offsets and baseline for a volume.

        Args:
            volume_data (Any): Input volume data.
            method (str, optional): Method for surface calculation. Defaults to "fitting".
            degree (int, optional): Degree for curve fitting. Defaults to 2.

        Returns:
            Tuple[Any, int]: Offsets and mean value for flattening.
        """
        surface = np.argmax(volume_data, axis=1)
        val = np.max(volume_data, axis=1)
        if method.lower() == "fitting":
            z_surf = utils.fit_curve_surface(surface, val > np.max(val) * 0.8, degree)
        else:  # method == "Interpolation":
            z_surf = utils.interpolate_curve(surface, val > np.max(val) * 0.7, 35)
        meanV = int(np.mean(z_surf))
        offsets = (meanV - z_surf).astype(int)
        return offsets, meanV

    @staticmethod
    def flatten_oct_octa(
        oct_data: np.ndarray,
        octa_data: np.ndarray,
        permute: str = "auto",
        method: str = "fitting",
        reference_curve: np.ndarray = None,
        degree: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, Any, Any, Tuple[int, int, int]]:
        """Flatten OCT and/or OCTA data using a reference surface or fitted curve.

        Args:
            oct_data (np.ndarray): OCT volume data.
            octa_data (np.ndarray): OCTA volume data.
            permute (str, optional): Permutation order or "auto". Defaults to "auto".
            method (str, optional): Flattening method. Defaults to "fitting".
            reference_curve (np.ndarray, optional): Reference surface. Defaults to None.
            degree (int, optional): Degree for curve fitting. Defaults to 2.

        Returns:
            Tuple[np.ndarray, np.ndarray, Any, Any, Tuple[int, int, int]]: 
                Flattened OCT, flattened OCTA, flatten offset, flatten baseline, flatten permute.
        """
        flatten_offset = 0
        flatten_baseline = -1
        flatten_permute = (
            0,
            1,
            2,
        )  # the oct_data and octa_data after this permute get the flatten_offset and flatten_baseline
        # calculate the flatten offsets and baseline
        if method.lower() == "rpe-bm":
            if reference_curve is not None and np.any(reference_curve):
                flatten_baseline = int(np.mean(reference_curve))
                flatten_offset = (flatten_baseline - reference_curve).astype(
                    int
                )  # np.transpose((flatten_baseline - reference_curve).astype(int), (1, 0))
                flatten_permute = (0, 1, 2)
            else:
                return oct_data, octa_data, 0, -1, (0, 1, 2)
        else:
            if oct_data is not None:
                flatten_permute = (
                    utils.auto_permute(oct_data)
                    if permute == "auto"
                    else [int(i) for i in permute.split(",")]
                )
                oct_ = np.transpose(oct_data, flatten_permute)
                flatten_offset, flatten_baseline = utils.calculate_flatten_parameters(
                    oct_, method=method, degree=degree
                )
            elif octa_data is not None:
                flatten_permute = (
                    utils.auto_permute(octa_data)
                    if permute == "auto"
                    else [int(i) for i in permute.split(",")]
                )
                oct_ = np.transpose(octa_data, flatten_permute)
                flatten_offset, flatten_baseline = utils.calculate_flatten_parameters(
                    oct_, method=method, degree=degree
                )
            else:
                return None, None, 0, -1, (0, 1, 2)

        if tuple(flatten_permute) != (0, 1, 2):
            if oct_data is not None:
                oct_data = np.transpose(oct_data, flatten_permute)
                oct_data_flatten, _, _ = utils.flatten_volume(
                    oct_data, flatten_baseline - flatten_offset
                )
                # get the permute from the flatten_permute to restore the original data
                permute_values = [flatten_permute.index(i) for i in range(3)]
                oct_data_flatten = np.transpose(oct_data_flatten, permute_values)
            else:
                oct_data_flatten = None
            if octa_data is not None:
                octa_data = np.transpose(octa_data, flatten_permute)
                octa_data_flatten, _, _ = utils.flatten_volume(
                    octa_data, flatten_baseline - flatten_offset
                )
                permute_values = [flatten_permute.index(i) for i in range(3)]
                octa_data_flatten = np.transpose(octa_data_flatten, permute_values)
            else:
                octa_data_flatten = None
        else:
            if oct_data is not None:
                oct_data_flatten, _, _ = utils.flatten_volume(
                    oct_data, flatten_baseline - flatten_offset
                )
            else:
                oct_data_flatten = None

            if octa_data is not None:
                octa_data_flatten, _, _ = utils.flatten_volume(
                    octa_data, flatten_baseline - flatten_offset
                )
            else:
                octa_data_flatten = None
        return (
            oct_data_flatten,
            octa_data_flatten,
            flatten_offset,
            flatten_baseline,
            tuple(flatten_permute),
        )


    @staticmethod
    def transform_volume(
        volume_data: np.ndarray,
        flatten_offset: Any = None,
        permute: Tuple[int, int, int] = (0, 1, 2),
        flip: str = "None"
    ) -> np.ndarray:
        """Apply flattening, permutation, and flipping to a volume.

        Args:
            volume_data (np.ndarray): Input volume data.
            flatten_offset (Any, optional): Flatten offset. Defaults to None.
            permute (Tuple[int, int, int], optional): Permutation order. Defaults to (0, 1, 2).
            flip (str, optional): Flip direction ("None", "Left-Right", "Up-Down"). Defaults to "None".

        Returns:
            np.ndarray: Transformed volume data.
        """
        # flatten volume data
        if flatten_offset is not None:
            volume_data, _, _ = utils.roll_volume_offsets(volume_data, flatten_offset)
        if permute != (0, 1, 2):
            volume_data = np.transpose(volume_data, permute)
        if flip != "None":
            if flip.lower() == "Left-Right":
                volume_data = volume_data[:, :, ::-1]
            elif flip.lower() == "Up-Down":
                volume_data = volume_data[:, ::-1, :]
        return volume_data


    @staticmethod
    def restore_transform_volume(
        volume_data: np.ndarray,
        flatten_offset: Any = None,
        permute: Tuple[int, int, int] = (0, 1, 2),
        flip: str = "None"
    ) -> np.ndarray:
        """Restore volume data from transformed state.

        Args:
            volume_data (np.ndarray): Transformed volume data.
            flatten_offset (Any, optional): Flatten offset. Defaults to None.
            permute (Tuple[int, int, int], optional): Permutation order. Defaults to (0, 1, 2).
            flip (str, optional): Flip direction. Defaults to "None".

        Returns:
            np.ndarray: Restored volume data.
        """
        if flip != "None":
            if flip.lower() == "Left-Right":
                volume_data = volume_data[:, :, ::-1]
            elif flip.lower() == "Up-Down":
                volume_data = volume_data[:, ::-1, :]
        if permute != (0, 1, 2):
            # calculate the inverse permute
            permute = [permute.index(i) for i in range(3)]
            volume_data = np.transpose(volume_data, permute)
        if flatten_offset is not None:
            volume_data = utils.roll_volume_inverse(volume_data, flatten_offset)
        return volume_data


    # Applying smoothing
    @staticmethod
    def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving average of a 1D array.

        Args:
            arr (np.ndarray): Input array.
            window_size (int): Window size for averaging.

        Returns:
            np.ndarray: Smoothed array.
        """
        return np.convolve(arr, np.ones(window_size) / window_size, mode="same")


    # Calculating differences between smoothed and original
    @staticmethod
    def smooth_difference(arr: np.ndarray, window_size: int) -> np.ndarray:
        """Compute absolute difference between original and smoothed array.

        Args:
            arr (np.ndarray): Input array.
            window_size (int): Window size for smoothing.

        Returns:
            np.ndarray: Absolute difference array.
        """
        smoothed = utils.moving_average(arr, window_size)
        return np.abs(smoothed - arr)


    @staticmethod
    def auto_permute(volume_data: np.ndarray) -> List[int]:
        """Automatically determine the best axis permutation for a 3D volume.

        Args:
            volume_data (np.ndarray): Input 3D volume.

        Returns:
            List[int]: Permutation order for axes.
        """
        # Summing across axes
        sum1 = np.sum(volume_data, axis=(1, 2)).flatten()
        sum2 = np.sum(volume_data, axis=(0, 2)).flatten()
        sum3 = np.sum(volume_data, axis=(0, 1)).flatten()

        window_size = 15
        sum1_smoothed = utils.moving_average(sum1, window_size)
        sum2_smoothed = utils.moving_average(sum2, window_size)
        sum3_smoothed = utils.moving_average(sum3, window_size)
        # Resizimoving_average
        lSum1 = cv2.resize(sum1_smoothed.reshape(-1, 1), (1, 512)).flatten()
        lSum2 = cv2.resize(sum2_smoothed.reshape(-1, 1), (1, 512)).flatten()
        lSum3 = cv2.resize(sum3_smoothed.reshape(-1, 1), (1, 512)).flatten()

        # Correlation coefficients
        c1, _ = pearsonr(lSum2, lSum3)
        c2, _ = pearsonr(lSum1, lSum3)
        c3, _ = pearsonr(lSum1, lSum2)

        # Find maximum correlation and its index
        z_idx = np.argmax([c1, c2, c3])

        smoothL1 = np.sum(np.abs(np.diff(sum1))[20:-20])
        smoothL2 = np.sum(np.abs(np.diff(sum2))[20:-20])
        smoothL3 = np.sum(np.abs(np.diff(sum3))[20:-20])

        smooths = [smoothL1, smoothL2, smoothL3]
        smooths[z_idx] = 0
        nimg_axis = np.argmax(smooths)

        ncol_axis = 3 - nimg_axis - z_idx
        if z_idx == 1:
            return [0, 1, 2]
        return [nimg_axis, z_idx, ncol_axis]


    @staticmethod
    def createZeroCurve(
        nimgs: int, nrows: int, ncols: int, permute: str = "0,1,2"
    ) -> dataDict:
        """Create a zero-initialized curve dictionary for a given shape and permutation.

        Args:
            nimgs (int): Number of images.
            nrows (int): Number of rows.
            ncols (int): Number of columns.
            permute (str, optional): Permutation string. Defaults to "0,1,2".

        Returns:
            dataDict: Curve dictionary with dot notation access.
        """
        # volumeSize is raw data shape, the curve shape is after the permutation by permute
        dim_raw = (nimgs, nrows, ncols)
        curve = dataDict({
            "volumeSize": dim_raw,
            "fazCenter": [0, 0],
            "onhCenter": [0, 0],
            "version": "0.0",
            "permute": permute,
            "flip": "None",
            "flatten_offset": 0,
            "flatten_baseline": -1,
        })
        curve_names = [
            "PVD",
            "ILM",
            "NFLGCL",
            "GCLIPL",
            "IPLINL",
            "INLOPL",
            "OPLONL",
            "ELM",
            "EZ",
            "EZIZ",
            "IZRPE",
            "RPEBM",
            "SATHAL",
            "CHOROID",
        ]

        permutes = permute.split(",")
        dim_after_permute = [
            dim_raw[int(permutes[0])],
            dim_raw[int(permutes[1])],
            dim_raw[int(permutes[2])],
        ]
        curves = dataDict({
            name: np.zeros((dim_after_permute[0], dim_after_permute[2]), dtype="int16")
            for name in curve_names
        })
        curve.curves = curves
        return curve


    @staticmethod
    def createZeroCurveDict(nimgs: int, nrows: int, ncols: int) -> dataDict:
        """Create a dictionary of zero-initialized curves for all flip and permute combinations.

        Args:
            nimgs (int): Number of images.
            nrows (int): Number of rows.
            ncols (int): Number of columns.

        Returns:
            dataDict: Dictionary of curves with dot notation access.
        """
        flip_keys = ["None", "LeftRight", "UpDown"]
        permute_keys = ["0,1,2", "0,2,1", "1,0,2", "1,2,0", "2,0,1", "2,1,0"]
        # generate combinations of flip and permute
        dims = [nimgs, nrows, ncols]
        curveDict = dataDict({"volumeSize": (nimgs, nrows, ncols)})
        for flip in flip_keys:
            for permute in permute_keys:
                # change the dimension of the curve to match the flip and permute
                # add ',' to the permute to make it a string
                permutes = permute.replace(",", "")
                curveDict[f"{flip}_{permutes}"] = utils.createZeroCurve(
                    dims[0], dims[1], dims[2], permute=permute
                )

        return curveDict


    @staticmethod
    def generateCurveDictKey(permute: str, flip: str) -> str:
        """Generate a key for curve dictionary based on permute and flip.

        Args:
            permute (str): Permute string.
            flip (str): Flip string.

        Returns:
            str: Generated key.
        """
        if flip is None:
            flip = "None"
        fp = flip.replace("-", "")
        pt = permute.replace(",", "")
        return f"{fp}_{pt}"


    @staticmethod
    def validateCurve(curve) -> bool:
        """Validate if the input is a proper curve dictionary.

        Args:
            curve: Curve dictionary (dict or dataDict).

        Returns:
            bool: True if valid, False otherwise.
        """
        # if curve is not a dictionary, return False
        if not isinstance(curve, (dict, dataDict)):
            return False
        if (
            "volumeSize" not in curve
            or "fazCenter" not in curve
            or "onhCenter" not in curve
            or "version" not in curve
            or "permute" not in curve
            or "flip" not in curve
            # or "flatten_offset" not in curve
            # or "flatten_baseline" not in curve
            or "curves" not in curve
        ):
            return False
        for key in curve["curves"].keys():
            if not isinstance(curve["curves"][key], np.ndarray):
                return False
        return True

    @staticmethod
    def resizeCurveData(curve: dataDict, output_shape: Tuple[int, int, int], order: int = 2) -> dataDict:
        """Resize all curve arrays in the curve dictionary to a new shape.

        Args:
            curve (dataDict): Curve dictionary.
            output_shape (Tuple[int, int, int]): Desired output shape.
            order (int, optional): Interpolation order. Defaults to 2.

        Returns:
            dataDict: Updated curve dictionary.
        """
        # update the curveArray in the curve
        out_size = [output_shape[0], output_shape[2]]
        for key in curve["curves"].keys():
            curve["curves"][key] = resize(curve["curves"][key], out_size, order=order)
        return curve


    @staticmethod
    def generateCurve(
        layerSegMat: np.ndarray,
        permute: str = "0,1,2",
        flip: str = "None",
        flatten_offset: int = 0,
        flatten_baseline: int = -1,
        is_flatten: bool = False,
        crop_top: int = 0,
    ) -> Tuple[dataDict, np.ndarray, np.ndarray]:
        """Generate curve data and fluid mask from a layer segmentation matrix.

        Args:
            layerSegMat (np.ndarray): Layer segmentation matrix.
            permute (str, optional): Permutation string. Defaults to "0,1,2".
            flip (str, optional): Flip direction. Defaults to "None".
            flatten_offset (int, optional): Flatten offset. Defaults to 0.
            flatten_baseline (int, optional): Flatten baseline. Defaults to -1.
            is_flatten (bool, optional): Whether the data is flattened. Defaults to False.
            crop_top (int, optional): Crop offset at the top. Defaults to 0.

        Returns:
            Tuple[dataDict, np.ndarray, np.ndarray]: Curve dictionary, fluid mask, and curve array.
        """
        # generate the curve
        nimgs, nrows, ncols = layerSegMat.shape

        # layerSegment is after the permutation, get the original shape
        dims_after_permute = [nimgs, nrows, ncols]
        permutes = permute.split(",")
        dims_raw = [
            dims_after_permute[int(permutes[0])],
            dims_after_permute[int(permutes[1])],
            dims_after_permute[int(permutes[2])],
        ]

        curve = dataDict({
            "volumeSize": tuple(dims_raw),
            "fazCenter": [0, 0],
            "onhCenter": [0, 0],
            "version": "1.0",
            "permute": permute,
            "flip": flip,
            "flatten_offset": flatten_offset,
            "flatten_baseline": flatten_baseline,
        })
        curves = {}
        # keep largest connected components 3D
        layerSegMatBW = utils.keepLargestComponents3D(layerSegMat > 0)
        layerSegMat *= layerSegMatBW

        # reassign the fluid region to the corresponding layer
        fluidMat = np.zeros((nimgs, nrows, ncols), dtype="uint8")
        fluidMask = (layerSegMat > 0) & (layerSegMat < 3)
        fluidMask = binary_opening(fluidMask, structure=generate_binary_structure(3, 2))
        fluidMask_dilated = binary_dilation(
            fluidMask, structure=generate_binary_structure(3, 2)
        )
        fluidLabel_dilated, labeled_number = label(fluidMask_dilated)
        if labeled_number > 0:
            for i in range(1, labeled_number + 1):
                region = fluidLabel_dilated == i
                unique, counts = np.unique(layerSegMat[region], return_counts=True)
                layerIdxs = unique[unique > 2]
                layerCtxs = counts[unique > 2]
                if layerCtxs.size > 0:
                    layerIdx = layerIdxs[np.argmax(layerCtxs)]
                else:
                    continue
                layerSegMatMsk = region & fluidMask
                fluidMat[layerSegMatMsk] = layerIdx - 2
                layerSegMat[layerSegMatMsk] = layerIdx

        # generate the curveArray
        ordMat3D = np.tile(np.arange(1, nrows + 1).reshape(1, nrows, 1), (nimgs, 1, ncols))
        curveArray = np.zeros((nimgs, 11, ncols), dtype="float32")

        layerSlabs = np.zeros((nimgs, nrows, ncols), dtype="bool")
        if is_flatten:
            flatten_ = flatten_offset
        else:
            flatten_ = 0

        edg_msk = np.zeros((nimgs, ncols), dtype="bool")
        for i in range(3, 10):
            layerSlab = utils.keepLargestComponents3D(layerSegMat == i)
            layerSlabs = np.logical_or(layerSlabs, layerSlab)
            # layerSlabs = binary_closing(layerSlabs, structure = np.ones((5, 5, 5)),border_value=1) # this slow down the process
            if i == 3:
                curveArray[:, 0, :] = (np.min(ordMat3D, axis=1, initial=nrows, where=layerSegMatBW)- flatten_+ crop_top)
                curveArray[:, 9, :] = (np.max(ordMat3D, axis=1, initial=0, where=layerSegMatBW)- flatten_+ crop_top)
                curveArray[:, 1, :] = (np.max(ordMat3D, axis=1, initial=0, where=layerSlabs)- flatten_+ crop_top)

                curveArray[:, 0, :] = cv2.bilateralFilter(curveArray[:, 0, :], 5, 15, 15)
                curveArray[:, 1, :] = cv2.bilateralFilter(curveArray[:, 1, :], 5, 15, 15)
                curveArray[:, 9, :] = cv2.bilateralFilter(curveArray[:, 9, :], 5, 15, 15)

                curveArray[:, 1, :] = np.where(curveArray[:, 1, :] < curveArray[:, 0, :],curveArray[:, 0, :],curveArray[:, 1, :])
                edg_msk = curveArray[:, 0, :] > curveArray[:, 9, :]
                curveArray[:, 1, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, 1, :])
                curveArray[:, 9, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, 9, :])
            else:
                curveArray[:, i - 2, :] = np.max(ordMat3D, axis=1, initial=0, where=layerSlabs)- flatten_+ crop_top
                curveArray[:, i - 2, :] = cv2.bilateralFilter(curveArray[:, i - 2, :], 5, 15, 15)
                curveArray[:, i - 2, :] = np.where(curveArray[:, i - 2, :] < curveArray[:, i - 3, :],curveArray[:, i - 3, :],curveArray[:, i - 2, :])
                curveArray[:, i - 3, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, i - 3, :])

        curveArray[:, 8, :] = (curveArray[:, 7, :] + curveArray[:, 9, :]) / 2

        # # remove all fluid regions that are not in layerSlabs, and reassign the fluid regions to the corresponding layer
        fluidMat[fluidMat == 8] = 0
        fluidMat[fluidMat == 1] = 0
        fluidMat[(fluidMat > 0) & (fluidMat < 6)] = 1
        fluidMat[fluidMat == 6] = 2
        fluidMat[fluidMat == 7] = 3
        # Precompute constant arrays
        ones_curve = np.ones((nimgs, ncols), dtype="uint16")
        curves = dataDict({
            "PVD": ones_curve,
            "ILM": curveArray[:, 0, :],
            "NFLGCL": curveArray[:, 1, :],
            "GCLIPL": ones_curve,
            "IPLINL": curveArray[:, 2, :],
            "INLOPL": curveArray[:, 3, :],
            "OPLONL": curveArray[:, 4, :],
            "ELM": ones_curve,
            "EZ": curveArray[:, 5, :],
            "EZIZ": curveArray[:, 6, :],
            "IZRPE": curveArray[:, 6, :] + 3,
            "RPEBM": curveArray[:, 7, :],
            "SATHAL": curveArray[:, 8, :],
            "CHOROID": curveArray[:, 9, :],
        })

        curve.curves = curves
        fluidMat=utils.cleanUpFluidMat(fluidMat>0, curves,flatten=flatten_offset, crop=crop_top)
        return curve, fluidMat, curveArray

    @staticmethod
    def cleanUpFluidMat(
        fluidMat: np.ndarray,
        curves: Dict[str, np.ndarray],
        flatten: int = 0,
        crop: int = 0
    ) -> np.ndarray:
        """Clean up the fluid matrix by masking with region between NFLGCL and RPEBM.

        Args:
            fluidMat (np.ndarray): Fluid matrix.
            curves (Dict[str, np.ndarray]): Dictionary of curve arrays.
            flatten (int, optional): Flatten offset. Defaults to 0.
            crop (int, optional): Crop offset. Defaults to 0.

        Returns:
            np.ndarray: Cleaned fluid matrix.
        """
        z = np.arange(fluidMat.shape[1])[:, None, None]
        
        region = ((z >= (curves['NFLGCL']+flatten-crop+1)) & (z < (curves['RPEBM'])+flatten-crop)).astype(np.uint8)
        region = np.transpose(region, (1, 0, 2))
        
        # IRFRegion = ((z >= (curves['NFLGCL']+flatten-crop)) & (z < (curves['EZ'])+flatten-crop)).astype(np.uint8)
        # IRFRegion = np.transpose(IRFRegion, (1, 0, 2))
        
        # oRFRegion = ((z >= (curves['EZ']+flatten-crop)) & (z < (curves['IZRPE'])+flatten-crop)).astype(np.uint8)
        # oRFRegion = np.transpose(oRFRegion, (1, 0, 2))
        
        # sRFRegion = ((z >= (curves['IZRPE']+flatten-crop)) & (z < (curves['RPEBM'])+flatten-crop)).astype(np.uint8)
        # sRFRegion = np.transpose(sRFRegion, (1, 0, 2))
        # fluidMat = fluidMat *IRFRegion + fluidMat * oRFRegion*2 + fluidMat * sRFRegion*3

        return fluidMat * region

    @staticmethod
    def generateCurveMultiSegMats(
        layerSegMats: List[np.ndarray],
        permute: str = "0,1,2",
        flip: str = "None",
        flatten_offset: int = 0,
        flatten_baseline: int = -1,
        is_flatten: bool = False,
        crop_top: int = 0,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Generate curve and fluid mask from multiple segmentation matrices.

        Args:
            layerSegMats (List[np.ndarray]): List of layer segmentation matrices.
            permute (str, optional): Permutation string. Defaults to "0,1,2".
            flip (str, optional): Flip direction. Defaults to "None".
            flatten_offset (int, optional): Flatten offset. Defaults to 0.
            flatten_baseline (int, optional): Flatten baseline. Defaults to -1.
            is_flatten (bool, optional): Whether the data is flattened. Defaults to False.
            crop_top (int, optional): Crop offset at the top. Defaults to 0.

        Returns:
            Tuple[Dict[str, Any], np.ndarray, np.ndarray]: Curve dictionary, fluid mask, and curve array.
        """
        # generate the curve
        nimgs, nrows, ncols = layerSegMats[0].shape

        # layerSegment is after the permutation, get the original shape
        dims_after_permute = [nimgs, nrows, ncols]
        permutes = permute.split(",")
        dims_raw = [
            dims_after_permute[int(permutes[0])],
            dims_after_permute[int(permutes[1])],
            dims_after_permute[int(permutes[2])],
        ]

        curve = {
            "volumeSize": tuple(dims_raw),
            "fazCenter": [0, 0],
            "onhCenter": [0, 0],
            "version": "1.0",
            "permute": permute,
            "flip": flip,
            "flatten_offset": flatten_offset,
            "flatten_baseline": flatten_baseline,
        }
        curves = {}


        # generate the curveArray
        ordMat3D = np.tile(np.arange(1, nrows + 1).reshape(1, nrows, 1), (nimgs, 1, ncols))
        curveArray = np.zeros((nimgs, 11, ncols), dtype="float32")

        if is_flatten:
            flatten_ = flatten_offset
        else:
            flatten_ = 0

        edg_msk = np.zeros((nimgs, ncols), dtype="bool")    


        fluidMat = np.zeros((nimgs, nrows, ncols), dtype="uint8")
        
        for i, layerSegMat in enumerate(layerSegMats[1:], start=0):
            # keep largest connected components 3D
            layerSegMatBW = utils.keepLargestComponents3D(layerSegMat > 0)
            layerSegMat *= layerSegMatBW
            fluidMat += (layerSegMat > 0) & (layerSegMat < 3)
            layerSegMat *= layerSegMatBW
            layerSlab4Min = utils.keepLargestComponents3D(layerSegMat==4)
            # layerSlab4Max = utils.keepLargestComponents3D(layerSegMat==3)
            if i == 0: 
                curveArray[:, 0, :] = (np.min(ordMat3D, axis=1, initial=nrows, where=layerSegMatBW)- flatten_+ crop_top)
                curveArray[:, 9, :] = (np.max(ordMat3D, axis=1, initial=0, where=layerSegMatBW)- flatten_+ crop_top)
                curveArray[:, 1, :] = (np.min(ordMat3D, axis=1, initial=nrows, where=layerSlab4Min)- flatten_+ crop_top)

                curveArray[:, 0, :] = cv2.bilateralFilter(curveArray[:, 0, :], 5, 15, 15)
                curveArray[:, 1, :] = cv2.bilateralFilter(curveArray[:, 1, :], 5, 15, 15)
                curveArray[:, 9, :] = cv2.bilateralFilter(curveArray[:, 9, :], 5, 15, 15)

                curveArray[:, 1, :] = np.where(curveArray[:, 1, :] < curveArray[:, 0, :],curveArray[:, 0, :],curveArray[:, 1, :])
                edg_msk = curveArray[:, 0, :] > curveArray[:, 9, :]
                curveArray[:, 1, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, 1, :])
                curveArray[:, 9, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, 9, :])
            else:
                curveArray[:, i+1, :] = (np.min(ordMat3D, axis=1, initial=nrows, where=layerSlab4Min)- flatten_+ crop_top)
                curveArray[:, i+1, :] = cv2.bilateralFilter(curveArray[:, i+1, :], 5, 15, 15)
                curveArray[:, i+1, :] = np.where(curveArray[:, i+1, :] < curveArray[:, i, :],curveArray[:, i, :],curveArray[:, i+1, :])
                curveArray[:, i, :] = np.where(edg_msk, nrows - flatten_ + crop_top, curveArray[:, i, :])

        curveArray[:, 8, :] = (curveArray[:, 7, :] + curveArray[:, 9, :]) / 2

        
        # # remove all fluid regions that are not in layerSlabs, and reassign the fluid regions to the corresponding layer
        # fluidMat[fluidMat == 8] = 0
        # fluidMat[fluidMat == 1] = 0
        # fluidMat[(fluidMat > 0) & (fluidMat < 6)] = 1
        # fluidMat[fluidMat == 6] = 2
        # fluidMat[fluidMat == 7] = 3
        # Precompute constant arrays
        ones_curve = np.ones((nimgs, ncols), dtype="uint16")
        curves = {
            "PVD": ones_curve,
            "ILM": curveArray[:, 0, :],
            "NFLGCL": curveArray[:, 1, :],
            "GCLIPL": ones_curve,
            "IPLINL": curveArray[:, 2, :],
            "INLOPL": curveArray[:, 3, :],
            "OPLONL": curveArray[:, 4, :],
            "ELM": ones_curve,
            "EZ": curveArray[:, 5, :],
            "EZIZ": curveArray[:, 6, :],
            "IZRPE": curveArray[:, 6, :] + 3,
            "RPEBM": curveArray[:, 7, :],
            "SATHAL": curveArray[:, 8, :],
            "CHOROID": curveArray[:, 9, :],
        }

        curve["curves"] = curves
        return curve, (fluidMat > 4).astype(np.uint8), curveArray


    @staticmethod
    def convertArray2CurveDict(
        curveArray: np.ndarray,
        nimgs: int,
        nrows: int,
        ncols: int,
        permute: str = "0,1,2",
        flip: str = "None",
        flatten_offset: int = 0,
        flatten_baseline: int = -1,
    ) -> dataDict:
        """Convert a curve array to a curve dictionary.

        Args:
            curveArray (np.ndarray): Curve array. the shape is (ncols, nlayers, nimgs)
            nimgs (int): Number of images.
            nrows (int): Number of rows.
            ncols (int): Number of columns.
            permute (str, optional): Permutation string. Defaults to "0,1,2".
            flip (str, optional): Flip direction. Defaults to "None".
            flatten_offset (int, optional): Flatten offset. Defaults to 0.
            flatten_baseline (int, optional): Flatten baseline. Defaults to -1.

        Returns:
            dataDict: Curve dictionary.
        """
        curve = dataDict({
            "volumeSize": (nimgs, nrows, ncols),
            "fazCenter": [0, 0],
            "onhCenter": [0, 0],
            "version": "1.0",
            "permute": permute,
            "flip": flip,
            "flatten_offset": flatten_offset,
            "flatten_baseline": flatten_baseline,
        })
        nlayers = curveArray.shape[1]
        if nlayers == 8:
            curves = dataDict({
                "PVD": np.ones((nimgs, ncols), dtype="uint16"),
                "ILM": curveArray[:, 0, :],
                "NFLGCL": curveArray[:, 1, :],
                "GCLIPL": np.ones((nimgs, ncols), dtype="uint16"),
                "IPLINL": curveArray[:, 2, :],
                "INLOPL": curveArray[:, 3, :],
                "OPLONL": curveArray[:, 4, :],
                "ELM": np.ones((nimgs, ncols), dtype="uint16"),
                "EZ": curveArray[:, 5, :],
                "EZIZ": curveArray[:, 6, :],
                "IZRPE": curveArray[:, 6, :] + 3,
                "RPEBM": curveArray[:, 7, :],
                "SATHAL": np.ones((nimgs, ncols), dtype="uint16"),
                "CHOROID": np.ones((nimgs, ncols), dtype="uint16"),
            })
            curve.curves = curves
            return curve
        elif nlayers == 11:
            curves = dataDict({
                "PVD": np.ones((nimgs, ncols), dtype="uint16"),
                "ILM": curveArray[:, 0, :],
                "NFLGCL": curveArray[:, 1, :],
                "GCLIPL": np.ones((nimgs, ncols), dtype="uint16"),
                "IPLINL": curveArray[:, 2, :],
                "INLOPL": curveArray[:, 3, :],
                "OPLONL": curveArray[:, 4, :],
                "ELM": np.ones((nimgs, ncols), dtype="uint16"),
                "EZ": curveArray[:, 5, :],
                "EZIZ": curveArray[:, 6, :],
                "IZRPE": curveArray[:, 6, :] + 3,
                "RPEBM": curveArray[:, 7, :],
                "SATHAL": curveArray[:, 8, :],
                "CHOROID": curveArray[:, 10, :],
            })
            curve.curves = curves
            return curve
        else:
            logger.warning("Invalid curve array shape.")
            return None
    
    @staticmethod
    def generateRetinalFluid(
        layerSegMat: np.ndarray,
    ) -> np.ndarray:
        """Generate fluid mask from a layer segmentation matrix.

        Args:
            layerSegMat (np.ndarray): Layer segmentation matrix.

        Returns:
            np.ndarray: Fluid mask.
        """
        # generate the curve
        nimgs, nrows, ncols = layerSegMat.shape

        # keep largest connected components 3D
        layerSegMatBW = utils.keepLargestComponents3D(layerSegMat > 0)
        layerSegMat *= layerSegMatBW

        # reassign the fluid region to the corresponding layer
        fluidMat = np.zeros((nimgs, nrows, ncols), dtype="uint8")
        fluidMask = (layerSegMat > 0) & (layerSegMat < 3)
        # sio.savemat("O:/VTDR024_19840509_M_OS_2024-10-04_16-04-53/DICOM/fluid.mat", {'fluid':fluidMask.astype(np.uint8)*255})
        fluidMask = binary_opening(fluidMask, structure=generate_binary_structure(3, 2))
        fluidMask_dilated = binary_dilation(
            fluidMask, structure=generate_binary_structure(3, 2)
        )
        fluidLabel_dilated, labeled_number = label(fluidMask_dilated)
        if labeled_number > 0:
            for i in range(1, labeled_number + 1):
                region = fluidLabel_dilated == i
                unique, counts = np.unique(layerSegMat[region], return_counts=True)
                layerIdxs = unique[unique > 2]
                layerCtxs = counts[unique > 2]
                if layerCtxs.size > 0:
                    layerIdx = layerIdxs[np.argmax(layerCtxs)]
                else:
                    continue

                layerSegMatMsk = region & fluidMask
                fluidMat[layerSegMatMsk] = layerIdx - 2
                layerSegMat[layerSegMatMsk] = layerIdx
        
        fluidMat[fluidMat == 8] = 0
        fluidMat[fluidMat == 1] = 0
        fluidMat[(fluidMat > 0) & (fluidMat < 6)] = 1
        fluidMat[fluidMat == 6] = 2
        fluidMat[fluidMat == 7] = 3
        
        return fluidMat



    @staticmethod
    def keepLargestComponents3D(bw3D: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component in a 3D binary image.

        Args:
            bw3D (np.ndarray): 3D binary image.

        Returns:
            np.ndarray: 3D binary image with only the largest component.
        """
        # Label the connected components in the 3D binary image
        if not np.any(bw3D):
            return bw3D
        labeled_image, _ = label(bw3D)
        # Calculate the volume of each region using NumPy's bincount
        region_volumes = np.bincount(labeled_image.ravel())
        # Get the index of the region with the largest volume (excluding the background)
        largest_volume_index = np.argmax(region_volumes[1:]) + 1
        # Zero out all regions except the one with the largest volume
        # bw3D[labeled_image != largest_volume_index] = 0
        return labeled_image == largest_volume_index


    @staticmethod
    def binary_area_open(bw: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small connected components from a binary image.

        Args:
            bw (np.ndarray): Binary image.
            min_size (int): Minimum size to keep.

        Returns:
            np.ndarray: Cleaned binary image.
        """
        labeled_image, num_labels = label(bw)
        region_sizes = np.bincount(labeled_image.ravel())
        too_small = region_sizes < min_size
        too_small_mask = too_small[labeled_image]
        bw[too_small_mask] = 0
        return bw


    @staticmethod
    def loadDLModel(model_path: str) -> bytes:
        """Load and decompress a deep learning model from file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            bytes: Decompressed model buffer.
        """
        with open(model_path, "rb") as f:
            model_buffer = f.read()
            model_buffer = model_buffer[:-2836]
            model_buffer = zlib.decompress(model_buffer)
            model_buffer = (
                model_buffer[: int(len(model_buffer) / 3)]
                + model_buffer[int(len(model_buffer) / 3) : int(2 * len(model_buffer) / 3)][
                    ::-1
                ]
                + model_buffer[int(2 * len(model_buffer) / 3) :]
            )
        return model_buffer

    @staticmethod
    def runModel(
        ortSession: Any,
        octData: np.ndarray,
        input_shape: Tuple,
        downsample_level: int = 5,
        batch_size: int = 1 
    ) -> np.ndarray:
        """Run the segmentation model on OCT data with multi-batch support.

        Args:
            ortSession (Any): ONNX Runtime session.
            octData (np.ndarray): Input OCT data, shape (z, y, x).
            input_shape (Tuple, optional): Shape of the model input. Defaults to (None, None, 5).
            downsample_level (int, optional): Downsampling level for resizing. Defaults to 5.
            batch_size (int, optional): Number of slices per batch. Defaults to 1.

        Returns:
            np.ndarray: Segmentation result, shape (z, y, x).
        """
        raw_shape = octData.shape
        sparse_mode = raw_shape[2] > 2 * raw_shape[0] and raw_shape[0] < 96
        if sparse_mode:
            nframe = min(raw_shape[2], max(raw_shape[0], 11) * 11)
            octData = utils.resize3D(octData, (nframe, raw_shape[1], raw_shape[2]), order=1)

        octData[1:-1] = (octData[1:-1] + octData[:-2] + octData[2:]) / 3
        octData = np.transpose(octData, [1, 2, 0])
        raw_img_size = octData.shape[0:2]
        in_chn = input_shape[-1]
        pad_width = int(np.floor(in_chn / 2))
        x_in_pad = np.pad(octData, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")

        n_slices = octData.shape[2]
        layerSegMat = np.zeros((n_slices, octData.shape[0], octData.shape[1]), dtype="uint8")

        # Prepare all input slices for batching
        x_parts = []
        for ic in range(n_slices):
            x_part = x_in_pad[:, :, ic : ic + in_chn]
            x_part = utils.resize_pow2_size(x_part, downsample_level)
            x_parts.append(x_part)
        x_parts = np.stack(x_parts, axis=0)  # shape: (n_slices, H, W, in_chn)

        # Run in batches
        for start in range(0, n_slices, batch_size):
            end = min(start + batch_size, n_slices)
            batch = x_parts[start:end]
            batch = np.transpose(batch, (0, 3, 1, 2))  # shape: (batch,C, H, W)
            # batch = np.expand_dims(batch, axis=1) if batch.ndim == 4 else batch  # (batch, 1, H, W, in_chn) if needed
            # Remove the expand_dims if your model expects (batch, H, W, in_chn)
            # batch = batch  # shape: (batch, H, W, in_chn)
            layerPds = ortSession.run(None, {"input": batch})[0]  # shape: (batch, H, W, n_classes)
            layerPds = np.argmax(layerPds, axis=1)  # shape: (batch, H, W)
            for i, layerPd in enumerate(layerPds):
                layerPd = utils.restore_resized_pow2size(layerPd, raw_img_size)
                layerSegMat[start + i] = layerPd

        if sparse_mode:
            layerSegMat = utils.resize3D(layerSegMat, raw_shape, order=0)

        return layerSegMat

    @staticmethod
    def run_ga_seg_model(
        ortSession: Any,
        octData: np.ndarray,
        input_shape: Tuple[int, int, int] = (None, None, 5),
        downsample_level: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the GA segmentation model.

        Args:
            ortSession (Any): ONNX Runtime session or similar inference session.
            octData (np.ndarray): Input OCT data, shape (z, y, x).
            input_shape (Tuple[int, int, int], optional): Shape of the model input. Defaults to (None, None, 5).
            downsample_level (int, optional): Downsampling level for resizing. Defaults to 5.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Layer segmentation matrix and GA segmentation matrix.
        """
        raw_shape = octData.shape
        sparse_mode = raw_shape[2] > 2 * raw_shape[0] and raw_shape[0] < 96
        if sparse_mode:
            nframe = min(raw_shape[2], max(raw_shape[0],5)*5)
            octData = utils.resize3D(octData, (nframe, raw_shape[1], raw_shape[2]), order=1)
            
        octData[1:-1] = (octData[1:-1] + octData[:-2] + octData[2:]) / 3
        octData = np.transpose(octData, [1, 2, 0])

        raw_img_size = octData.shape[0:2]
        in_chn = input_shape[-1]
        gaSegMat = np.zeros((octData.shape[2], octData.shape[0], octData.shape[1]), dtype="uint8" )
        layerSegMat = np.zeros((octData.shape[2], octData.shape[0], octData.shape[1]), dtype="uint8")
        pad_width = int(np.floor(in_chn / 2))
        x_in_pad = np.pad(octData, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")
        for ic in range(octData.shape[2]):
            x_part = x_in_pad[:, :, ic : ic + in_chn]
            x_part = utils.resize_pow2_size(x_part, downsample_level)
            x_part = np.expand_dims(x_part, axis=0)
            segRt = ortSession.run(None, {"input": x_part})
            
            layerPd = np.squeeze(np.argmax(segRt[0], axis=-1))
            layerPd = np.clip(layerPd, 0, 255).astype("uint8")
            layerPd = utils.restore_resized_pow2size(layerPd, raw_img_size)
            layerSegMat[ic] = layerPd
            
            gaPd = np.squeeze(np.argmax(segRt[1], axis=-1))
            gaPd = np.clip(gaPd, 0, 255).astype("uint8")
            gaPd = utils.restore_resized_pow2size(gaPd, raw_img_size)
            gaSegMat[ic] = gaPd
        
        if sparse_mode:
            layerSegMat = utils.resize3D(layerSegMat, raw_shape, order=0)
            gaSegMat = utils.resize3D(gaSegMat, raw_shape, order=0)
            
        return layerSegMat, gaSegMat

    @staticmethod
    def run_drusen_seg_model(
        ortSession: Any,
        octData: np.ndarray,
        input_shape: List[int] = [None, None, 5],
        downsample_level: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the drusen segmentation model.

        Args:
            ortSession (Any): ONNX Runtime session or similar inference session.
            octData (np.ndarray): Input OCT data, shape (z, y, x).
            input_shape (List[int], optional): Shape of the model input. Defaults to [None, None, 5].
            downsample_level (int, optional): Downsampling level for resizing. Defaults to 5.

        Returns:
            Tuple[np.ndarray, np.ndarray]: REPDC segmentation matrix and DV segmentation matrix.
        """
        raw_shape = octData.shape
        sparse_mode = raw_shape[2] > 2 * raw_shape[0] and raw_shape[0] < 96
        if sparse_mode:
            nframe = min(raw_shape[2], max(raw_shape[0],5)*5)
            octData = utils.resize3D(octData, (nframe, raw_shape[1], raw_shape[2]), order=1)
            
        octData[1:-1] = (octData[1:-1] + octData[:-2] + octData[2:]) / 3
        octData = np.transpose(octData, [1, 2, 0])
        raw_img_size = octData.shape[0:2]
        in_chn = input_shape[-1]
        repdcSegMat = np.zeros(
            (octData.shape[2], octData.shape[0], octData.shape[1]), dtype="uint8"
        )
        dvSegMat = np.zeros(
            (octData.shape[2], octData.shape[0], octData.shape[1]), dtype="uint8"
        )
        pad_width = int(np.floor(in_chn / 2))
        x_in_pad = np.pad(octData, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")
        for ic in range(octData.shape[2]):
            x_part = x_in_pad[:, :, ic : ic + in_chn]

            # resize axis 1 of x_part to 1.5 time
            # # raw_x_part_size = x_part.shape
            # x_part = resize3D(x_part, (x_part.shape[0], int(x_part.shape[1] * 1.5), x_part.shape[2]))

            x_part = utils.resize_pow2_size(x_part, downsample_level)
            x_part = np.expand_dims(x_part, axis=0)
            dv, repdc = ortSession.run(None, {"input": x_part})
            repdc = np.squeeze(np.argmax(repdc, axis=-1))
            dv = np.squeeze(np.argmax(dv, axis=-1))
            repdc = np.clip(repdc, 0, 255).astype("uint8")
            repdc = utils.restore_resized_pow2size(repdc, raw_img_size)
            dv = np.clip(dv, 0, 255).astype("uint8")
            dv = utils.restore_resized_pow2size(dv, raw_img_size)

            # repdc = resize3D(repdc, raw_x_part_size[0:2])
            # dv = resize3D(dv, raw_x_part_size[0:2])

            repdcSegMat[ic] = repdc
            dvSegMat[ic] = dv
        if sparse_mode:
            repdcSegMat = utils.resize3D(repdcSegMat, raw_shape, order=0)
            dvSegMat = utils.resize3D(dvSegMat, raw_shape, order=0)
        return repdcSegMat, dvSegMat


    @staticmethod
    def run_npa_seg_model(
        ort_session: Any,
        octa_data: np.ndarray,
        oct_data: np.ndarray,
        thk_data: np.ndarray,
        downsample_level: int = 5
    ) -> np.ndarray:
        """Run the NPA segmentation model.

        Args:
            ort_session (Any): ONNX Runtime session or similar inference session.
            octa_data (np.ndarray): Input OCTA data, shape (y, x).
            oct_data (np.ndarray): Input OCT data, shape (y, x).
            thk_data (np.ndarray): Input thickness data, shape (y, x).
            downsample_level (int, optional): Downsampling level for resizing. Defaults to 5.

        Returns:
            np.ndarray: NPA segmentation result, shape (y, x).
        """
        raw_img_size = octa_data.shape[0:2]
        octa_part = utils.resize_pow2_size(octa_data, downsample_level)
        oct_part = utils.resize_pow2_size(oct_data, downsample_level)
        thk_part = utils.resize_pow2_size(thk_data, downsample_level)
        octa_part = np.expand_dims(octa_part, axis=0)
        octa_part = np.expand_dims(octa_part, axis=-1)
        oct_part = np.expand_dims(oct_part, axis=0)
        oct_part = np.expand_dims(oct_part, axis=-1)
        thk_part = np.expand_dims(thk_part, axis=0)
        thk_part = np.expand_dims(thk_part, axis=-1)
        npaPd = ort_session.run(
            None, {"octa": octa_part, "oct": oct_part, "thk": thk_part}
        )[0]
        npaPd = np.squeeze(np.argmax(npaPd, axis=-1))
        npaPd = np.clip(npaPd, 0, 255).astype("uint8")
        npaPd = utils.restore_resized_pow2size(npaPd, raw_img_size)
        return npaPd


    @staticmethod
    def get_slab_mask(
        curve_data: Dict[str, Any],
        slab_name: str
    ) -> np.ndarray:
        """Generate a slab mask based on curve data and slab name.

        Args:
            curve_data (Dict[str, Any]): Curve data containing volume size and curve boundaries.
            slab_name (str): Name of the slab.

        Returns:
            np.ndarray: Slab mask.
        """
        [nimgs, nrows, ncols] = curve_data["volumeSize"]
        if slab_name.lower() == "whole retina":
            return np.zeros((nimgs, ncols), dtype="uint16"), np.ones(
                (nimgs, ncols) * nrows - 1, dtype="uint16"
            )
        elif slab_name.lower() == "inner retina":
            upper_boundry = curve_data["curves"]["ILM"]
            lower_boundry = curve_data["curves"]["OPLONL"]
        elif slab_name.lower() == "svc":
            upper_boundry = curve_data["curves"]["ILM"]
            lower_boundry = (
                curve_data["curves"]["NFLGCL"] * 0.33
                + curve_data["curves"]["IPLINL"] * 0.67
            )
        elif slab_name.lower() == "dvc":
            upper_boundry = (
                curve_data["curves"]["NFLGCL"] * 0.33
                + curve_data["curves"]["IPLINL"] * 0.67
            )
            lower_boundry = curve_data["curves"]["OPLONL"]
        elif slab_name.lower() == "icp":
            upper_boundry = (
                curve_data["curves"]["NFLGCL"] * 0.33
                + curve_data["curves"]["IPLINL"] * 0.67
            )
            lower_boundry = (
                curve_data["curves"]["IPLINL"] * 0.5 + curve_data["curves"]["INLOPL"] * 0.5
            )
        elif slab_name.lower() == "dcp":
            upper_boundry = (
                curve_data["curves"]["IPLINL"] * 0.5 + curve_data["curves"]["INLOPL"] * 0.5
            )
            lower_boundry = curve_data["curves"]["OPLONL"]
        else:
            logger.warning("slab name is not correct: %s", slab_name)
            return None

        lower_boundry = np.clip(lower_boundry, 0, nrows - 1)

        upper_boundry = np.where(
            upper_boundry > lower_boundry, lower_boundry, upper_boundry
        )
        lower_boundry = lower_boundry + 1
        upper_boundry_min = np.min(upper_boundry).astype(np.uint32)
        lower_boundry_max = np.max(lower_boundry).astype(np.uint32)

        lenx = lower_boundry_max - upper_boundry_min

        z = np.arange(lenx)[:, None, None]
        slab_mask = (
            (z >= upper_boundry - upper_boundry_min)
            & (z <= lower_boundry - upper_boundry_min)
        ).astype(np.float32)
        slab_mask = np.transpose(slab_mask, (1, 0, 2))
        return slab_mask


    @staticmethod
    def project_enface_image(
        oct_data: np.ndarray,
        curve_data: Dict[str, Any],
        slab_name: str,
        proj_type: str
    ) -> np.ndarray:
        """Project an enface image based on slab mask and projection type.

        Args:
            oct_data (np.ndarray): Input OCT data, shape (z, y, x).
            curve_data (Dict[str, Any]): Curve data containing volume size and curve boundaries.
            slab_name (str): Name of the slab.
            proj_type (str): Projection type (e.g., "Mean", "Maximum").

        Returns:
            np.ndarray: Enface image.
        """
        [_, nrows, ncols] = curve_data["volumeSize"]
        slab_mask = utils.get_slab_mask(curve_data, slab_name)
        im = utils.generateEnfaceImage(oct_data, slab_mask, [nrows, ncols], proj_type)
        return im


    @staticmethod
    def generateEnfaceImage(
        oct_data: np.ndarray,
        slab_mask: np.ndarray,
        v_size: List[int],
        proj_type: str
    ) -> np.ndarray:
        """Generate an enface image based on slab mask and projection type.

        Args:
            oct_data (np.ndarray): Input OCT data, shape (z, y, x).
            slab_mask (np.ndarray): Slab mask.
            v_size (List[int]): Volume size.
            proj_type (str): Projection type (e.g., "Mean", "Maximum").

        Returns:
            np.ndarray: Enface image.
        """
        if proj_type == "Mean":
            im = np.mean(oct_data * (slab_mask).astype("uint8"), axis=0)
            meanval = [np.min(im), np.max(im)]
            im = np.clip(im, meanval[0], meanval[1])
            im = (im - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)
            # im = mat2gray(im) * 255
        elif proj_type == "Maximum":
            im = np.max(oct_data, axis=0, initial=0, where=slab_mask)
            meanval = [np.min(im), np.max(im)]
            im = np.clip(im, meanval[0], meanval[1])
            im = (im - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)
            # im = mat2gray(im) * 255
        elif proj_type == "Minimum":
            im = np.min(oct_data, axis=0, initial=65535, where=slab_mask)
            meanval = [np.min(im), np.max(im)]
            im = np.clip(im, meanval[0], meanval[1])
            im = (im - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)
            # im = mat2gray(im) * 255
        elif proj_type == "Thickness":
            im = np.sum(slab_mask, axis=0)
            im = v_size[0] - im
            meanval = [np.min(im), np.max(im)]
            im = np.clip(im, meanval[0], meanval[1])
            im = (im - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)
            # im = mat2gray(v_size[0] - im) * 255
        else:
            im = np.mean(oct_data, axis=0, where=slab_mask)
            meanval = [np.min(im), np.max(im)]
            im = np.clip(im, meanval[0], meanval[1])
            im = (im - meanval[0]) / (meanval[1] - meanval[0] + 1e-6)
            # im = mat2gray(im) * 255
        return im * 255


    @staticmethod
    def interpolate_curve(
        surfaceData: np.ndarray,
        mask: np.ndarray = None,
        filter_window: int = 55
    ) -> np.ndarray:
        """Interpolate a curve surface using grid data.

        Args:
            surfaceData (np.ndarray): Surface data, shape (y, x).
            mask (np.ndarray, optional): Mask for valid points. Defaults to None.
            filter_window (int, optional): Filter window size. Defaults to 55.

        Returns:
            np.ndarray: Interpolated surface data.
        """
        surfaceData = median_filter(surfaceData, size=filter_window, mode="mirror")
        surfaceData = median_filter(surfaceData, size=filter_window, mode="mirror")
        # neighborhood_mean = uniform_filter(surfaceData, size=55)
        # lower_bound = neighborhood_mean - 50
        # upper_bound = neighborhood_mean + 50
        # mask2 = np.where((surfaceData < lower_bound) | (surfaceData > upper_bound), False, True)

        rows, cols = np.indices(surfaceData.shape)
        x = rows.ravel()  # X-coordinates (row indices)
        y = cols.ravel()  # Y-coordinates (column indices)
        z = surfaceData.ravel()  # Z-coordinates (pixel values)

        # Apply mask: keep only points where the mask is 1 (valid pixels)
        valid_points = mask.ravel()  #  mask2.ravel()
        x_valid = x[valid_points]
        y_valid = y[valid_points]
        z_valid = z[valid_points]

        # Define the grid for interpolation (same size as the image)
        xi, yi = np.meshgrid(
            np.arange(surfaceData.shape[0]), np.arange(surfaceData.shape[1])
        )

        # Interpolate using valid points
        z_surface = griddata((y_valid, x_valid), z_valid, (xi, yi), method="nearest")
        return z_surface


    @staticmethod
    def fit_curve_surface(
        surfaceData: np.ndarray,
        mask: np.ndarray = None,
        degree: int = 2
    ) -> np.ndarray:
        """Fit a polynomial surface to the curve data.

        Args:
            surfaceData (np.ndarray): Surface data, shape (y, x).
            mask (np.ndarray, optional): Mask for valid points. Defaults to None.
            degree (int, optional): Polynomial degree. Defaults to 2.

        Returns:
            np.ndarray: Fitted surface data.
        """
        # surfacedata is a image with shape (nrows, ncols), the value is the height of the surface
        # generate the x_data, y_data, z_data from the surfaceData, and ignore the nan value of z_data
        nrows, ncols = surfaceData.shape
        x_data = np.tile(np.arange(0, ncols), (nrows, 1))
        y_data = np.tile(np.arange(0, nrows).reshape(nrows, 1), (1, ncols))
        z_data = surfaceData
        if mask is not None:
            x_data = x_data[mask]
            y_data = y_data[mask]
            z_data = z_data[mask]
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = z_data.flatten()

        # Create the design matrix with polynomial terms
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(np.column_stack((x_data, y_data)))

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X_poly, z_data)

        # Create a meshgrid for plotting the fitted surface
        x_mesh, y_mesh = np.meshgrid(
            np.linspace(
                min(x_data),
                max(x_data),
                ncols,
            ),
            np.linspace(min(y_data), max(y_data), nrows),
        )
        X_mesh = poly.transform(np.column_stack((x_mesh.ravel(), y_mesh.ravel())))
        z_mesh = model.predict(X_mesh).reshape(x_mesh.shape)

        return z_mesh


    @staticmethod
    def generate_ilm_map(im: np.ndarray) -> np.ndarray:
        """Generate an ILM map using Gaussian filters.

        Args:
            im (np.ndarray): Input image.

        Returns:
            np.ndarray: ILM map.
        """
        # gaussian filter
        im = (utils.mat2gray(im) * 255).astype(np.uint8)
        im = im - np.mean(im) + 256 / 3 * 1.1
        im_g1 = gaussian_filter(im, sigma=9)
        im_g2 = gaussian_filter(im, sigma=15)
        im_g3 = gaussian_filter(im, sigma=21)
        im_sum = utils.mat2gray(im_g1 + im_g2 + im_g3, autoAjust=True) * 255
        im_ilm = im_sum.astype(np.uint8)

        return im_ilm


    @staticmethod
    def generate_2D_gaussion_map(
        shape: Tuple[int, int],
        center: Tuple[int, int],
        sigma: float
    ) -> np.ndarray:
        """Generate a 2D Gaussian map.

        Args:
            shape (Tuple[int, int]): Shape of the map (height, width).
            center (Tuple[int, int]): Center of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            np.ndarray: 2D Gaussian map.
        """
        x = np.arange(0, shape[0], 1, float)
        y = np.arange(0, shape[1], 1, float)
        x, y = np.meshgrid(x, y)
        z = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
        z = 1 - np.clip(z, 0, 1)

        return z


    @staticmethod
    def gabor_filter(
        image: np.ndarray,
        theta: float = 0,
        sigma: float = 0.2,
        frequency: float = 5
    ) -> np.ndarray:
        """Apply a Gabor filter to an image.

        Args:
            image (np.ndarray): Input image.
            theta (float, optional): Orientation of the Gabor filter. Defaults to 0.
            sigma (float, optional): Standard deviation of the Gaussian envelope. Defaults to 0.2.
            frequency (float, optional): Frequency of the sinusoidal factor. Defaults to 5.

        Returns:
            np.ndarray: Filtered image.
        """
        # image is a 2D array
        # theta is the orientation of the normal to the parallel stripes of a Gabor function
        # sigma is the sigma/standard deviation of the Gaussian envelope
        # frequency is the frequency of the sinusoidal factor
        # phase offset is the phase offset of the sinusoidal factor
        # ksize is the size of the filter
        # return the filtered image
        g_kernel = cv2.getGaborKernel(
            (5, 5), sigma, theta, frequency, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
        return filtered_img


    @staticmethod
    def bilateral_filter(
        image: np.ndarray,
        d: int = 9,
        sigmaColor: float = 75,
        sigmaSpace: float = 75
    ) -> np.ndarray:
        """Apply a bilateral filter to an image.

        Args:
            image (np.ndarray): Input image.
            d (int, optional): Diameter of each pixel neighborhood. Defaults to 9.
            sigmaColor (float, optional): Filter sigma in the color space. Defaults to 75.
            sigmaSpace (float, optional): Filter sigma in the coordinate space. Defaults to 75.

        Returns:
            np.ndarray: Filtered image.
        """
        # image is a 2D array
        # d is the diameter of each pixel neighborhood that is used during filtering
        # sigmaColor is the filter sigma in the color space
        # sigmaSpace is the filter sigma in the coordinate space
        # return the filtered image
        filtered_img = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        return filtered_img

    @staticmethod
    def generate_cRORA_HyperTDs(
        GAVolume: np.ndarray,
        layerVolume: np.ndarray=None,
        cRORA_diameter_thres_mm: float = 0.25,
        scan_width_mm: float = 6,
        scan_height_mm: float = 6
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float], np.ndarray]:
        """Generate cRORA and HyperTDs features from layer and GA volumes.

        Args:
            GAVolume (np.ndarray): GA volume data.
            layerVolume (np.ndarray): Layer volume data.
            cRORA_diameter_thres_mm (float, optional): Diameter threshold for cRORA in mm. Defaults to 0.25.
            scan_width_mm (float, optional): Scan width in mm. Defaults to 6.
            scan_height_mm (float, optional): Scan height in mm. Defaults to 6.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float], np.ndarray]:
                - cRORA and HyperTDs combined volume.
                - All features enface.
                - Areas of iponl_sub, ezrpe_dis, hyper_td, cRORA, and hyper_td_rest.
                - All Features volume.
        """
        pixel_size = (scan_width_mm * scan_height_mm / (GAVolume.shape[0] * GAVolume.shape[2]))
        
        iponl_sub = GAVolume == 1
        iponl_sub = np.sum(iponl_sub, 1, keepdims=False)
        iponl_sub = median_filter(iponl_sub, 5)
        iponl_sub = utils.binary_area_open(iponl_sub, 15)
        iponl_sub_area = np.count_nonzero(iponl_sub) * pixel_size

        ezrpe_dis = GAVolume == 2
        ezrpe_dis = np.sum(ezrpe_dis, 1, keepdims=False)
        ezrpe_dis = median_filter(ezrpe_dis, 5)
        ezrpe_dis = utils.binary_area_open(ezrpe_dis, 15)
        ezrpe_dis_area = np.count_nonzero(ezrpe_dis) * pixel_size

        hyper_td = GAVolume == 3
        hyper_td = np.sum(hyper_td, 1, keepdims=False)
        if layerVolume is not None:
            choroid_thk = np.sum(layerVolume == 5, 1, keepdims=False)*0.5
            hyper_td[hyper_td < choroid_thk] = 0
        
        hyper_td = median_filter(hyper_td, 5)
        hyper_td = utils.binary_area_open(hyper_td, 15)
        hyper_td_area = np.count_nonzero(hyper_td) * pixel_size

        iponl_ezrpe = np.logical_or(iponl_sub, ezrpe_dis)
        iponl_ezrpe_in_hyper_td = np.logical_and(iponl_ezrpe, hyper_td)

        cRORA_t = 3.14 * np.square(cRORA_diameter_thres_mm / 2) / pixel_size

        cRORA = utils.binary_area_open(iponl_ezrpe_in_hyper_td, cRORA_t)
        hyper_td_rest = np.logical_and(hyper_td, ~cRORA)
        hyper_td_rest_area= np.count_nonzero(hyper_td_rest) * pixel_size
        cRORA_area = np.count_nonzero(cRORA) * pixel_size
        cRORA = np.expand_dims(cRORA, -1)
        hyper_td_rest = np.expand_dims(hyper_td_rest, -1)
        cRORA_hyper_td_rest = np.concatenate((cRORA, hyper_td_rest, hyper_td_rest * 0), axis=-1).astype(np.uint8) * 255

        iponl_sub = np.expand_dims(iponl_sub, -1)
        ezrpe_dis = np.expand_dims(ezrpe_dis, -1)
        hyper_td = np.expand_dims(hyper_td, -1)
        all_enface_features = np.concatenate((iponl_sub, ezrpe_dis, hyper_td), axis=-1).astype(np.uint8) * 255
        all_volume_features = (GAVolume == 1).astype(np.uint8) + (GAVolume == 2).astype(np.uint8)*2 + (GAVolume == 3).astype(np.uint8)*3
        return cRORA_hyper_td_rest, all_enface_features, (iponl_sub_area, ezrpe_dis_area, hyper_td_area, cRORA_area, hyper_td_rest_area), all_volume_features

    @staticmethod
    def generate_cRORA_HyperTDs_from_labeled_volume(
        GAVolume: np.ndarray,
        cRORA_diameter_thres_mm: float = 0.25,
        scan_width_mm: float = 6,
        scan_height_mm: float = 6
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float]]:
        """Generates cRORA and hypertransmission defects (HyperTDs) masks from a labeled volume.

        This function processes a 3D labeled volume where:
        - label 1 indicates iPONL subsidence,
        - label 2 indicates EZ-RPE disruption,
        - label 3 indicates hypertransmission defect (HyperTD).

        It computes area-based masks and returns:
        - a composite image of cRORA and remaining hyperTDs,
        - an image showing all intermediate labeled regions,
        - the area statistics for each label type and derived regions.

        Args:
            GAVolume (np.ndarray): A 3D array of labeled segmentation data with shape (depth, height, width).
            cRORA_diameter_thres_mm (float): Minimum diameter (in mm) for a region to be considered cRORA. Defaults to 0.25.
            scan_width_mm (float): Physical width of the scan in millimeters. Defaults to 6.
            scan_height_mm (float): Physical height of the scan in millimeters. Defaults to 6.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float, float]]: A tuple containing:
                - cRORA_hyper_td_rest (np.ndarray): RGB-style image showing cRORA and hyperTD-rest regions.
                - all_features (np.ndarray): RGB-style image showing the original three input labels.
                - area_stats (Tuple[float, float, float, float, float]):
                    - iponl_sub_area (float): Area of iPONL subsidence.
                    - ezrpe_dis_area (float): Area of EZ-RPE disruption.
                    - hyper_td_area (float): Area of HyperTD regions.
                    - cRORA_size (float): Area of the final cRORA region.
                    - hyper_td_rest_size (float): Area of hyperTD excluding cRORA.
        """
        pixel_size: float = (
            scan_width_mm * scan_height_mm / (GAVolume.shape[0] * GAVolume.shape[2])
        )

        iponl_sub: np.ndarray = GAVolume == 1
        iponl_sub = np.sum(iponl_sub, axis=1)
        iponl_sub_area: float = np.count_nonzero(iponl_sub) * pixel_size

        ezrpe_dis: np.ndarray = GAVolume == 2
        ezrpe_dis = np.sum(ezrpe_dis, axis=1)
        ezrpe_dis_area: float = np.count_nonzero(ezrpe_dis) * pixel_size

        hyper_td: np.ndarray = GAVolume == 3
        hyper_td = np.sum(hyper_td, axis=1)
        hyper_td_area: float = np.count_nonzero(hyper_td) * pixel_size

        iponl_ezrpe: np.ndarray = np.logical_or(iponl_sub, ezrpe_dis)
        iponl_ezrpe_in_hyper_td: np.ndarray = np.logical_and(iponl_ezrpe, hyper_td)

        cRORA_t: float = 3.14 * np.square(cRORA_diameter_thres_mm / 2) / pixel_size
        cRORA: np.ndarray = utils.binary_area_open(iponl_ezrpe_in_hyper_td, cRORA_t)

        hyper_td_rest: np.ndarray = np.logical_and(hyper_td, ~cRORA)
        hyper_td_rest_size: float = np.count_nonzero(hyper_td_rest) * pixel_size
        cRORA_size: float = np.count_nonzero(cRORA) * pixel_size

        cRORA = np.expand_dims(cRORA, axis=-1)
        hyper_td_rest = np.expand_dims(hyper_td_rest, axis=-1)
        cRORA_hyper_td_rest: np.ndarray = (
            np.concatenate((cRORA, hyper_td_rest, hyper_td_rest * 0), axis=-1).astype(np.uint8) * 255
        )

        iponl_sub = np.expand_dims(iponl_sub, axis=-1)
        ezrpe_dis = np.expand_dims(ezrpe_dis, axis=-1)
        hyper_td = np.expand_dims(hyper_td, axis=-1)
        all_features: np.ndarray = (
            np.concatenate((iponl_sub, ezrpe_dis, hyper_td), axis=-1).astype(np.uint8) * 255
        )

        return cRORA_hyper_td_rest, all_features, (
            iponl_sub_area,
            ezrpe_dis_area,
            hyper_td_area,
            cRORA_size,
            hyper_td_rest_size,
        )
        
    @staticmethod
    def generate_drusen_props(
        drusen_volume: np.ndarray,
        pixel_size: float
    ) -> List[float]:
        """Compute volume and count for each drusen type in a labeled 3D volume.

        Each drusen type is assumed to be labeled as integers from 1 to 5.

        Args:
            drusen_volume (np.ndarray): 3D array with integer labels representing different drusen types.
            pixel_size (float): Volume represented by a single voxel (e.g., in mm³).

        Returns:
            List[float]: A list of 10 values:
                - The first 5 entries are the total volumes of drusen types 1–5.
                - The next 5 entries are the counts of connected drusen regions for types 1–5.
        """
        drusen_volumes: List[float] = []
        drusen_numbers: List[int] = []

        for i in range(5):
            drusen_mask = (drusen_volume == i + 1)
            drusen_size = np.sum(drusen_mask) * pixel_size
            drusen_volumes.append(drusen_size)
            _, num_components = label(drusen_mask)
            drusen_numbers.append(num_components)

        return drusen_volumes + drusen_numbers


    @staticmethod
    def filter_regions_by_neborhood_label(
        volume: np.ndarray,
        label_for_check: int,
        label_must_adjcent: int
    ) -> np.ndarray:
        """Filter out regions in a labeled volume unless they are adjacent to a specific label.

        Only keeps regions labeled as `label_for_check` if they are directly adjacent
        (in 6-connectivity) to voxels labeled as `label_must_adjcent`.

        Args:
            volume (np.ndarray): A 3D labeled volume.
            label_for_check (int): Label value of the region to test for adjacency.
            label_must_adjcent (int): Label value that must be adjacent to retain the region.

        Returns:
            np.ndarray: The modified volume where only adjacent-valid regions are kept;
                        all other regions are set to 0.
        """
        binary_mask = (volume == label_for_check)
        adjacent_mask = (volume == label_must_adjcent)

        structure = generate_binary_structure(3, 1)

        labeled_mask, _ = label(binary_mask, structure=structure)
        dilated_mask = binary_dilation(labeled_mask > 0, structure=structure)
        adjacent_regions = binary_dilation(adjacent_mask & dilated_mask, structure=structure)

        valid_labels = np.unique(labeled_mask[adjacent_regions])
        valid_mask = np.isin(labeled_mask, valid_labels)

        volume[~valid_mask] = 0

        return volume

    @staticmethod
    def generate_drusens(
        RPEDC: np.ndarray,
        DV: np.ndarray,
        drusen_diameter_thresholds_mm: Tuple[float,float,float]=(0.063, 0.125, 0.35),
        scan_width_mm:float=6,
        scan_height_mm:float=6,
        axiel_res_mm:float=0.002,
        use_max_diameter:bool=True,
    ):
        """
        Generate drusen masks and volumes from segmented RPEDC and drusenoid volumes.

        Parameters:
        RPEDC (np.ndarray): Segmented RPEDC volume with shape (nimgs, nrows, ncols).
        DV (np.ndarray): Segmented drusenoid volume with shape (nimgs, nrows, ncols).
        drusen_diameter_thresholds_mm (tuple): Diameter thresholds for different drusen types.
        scan_width_mm (float): Width of the scan in millimeters.
        scan_height_mm (float): Height of the scan in millimeters.
        axiel_res_mm (float): Axial resolution in millimeters.
        use_max_diameter (bool): Whether to use the maximum diameter for drusen classification.

        Returns:
        tuple: A tuple containing drusen masks, labels, volumes, and volume masks.
        """
        pixel_spacing_width = scan_width_mm / RPEDC.shape[2]
        pixel_spacing_height = scan_height_mm / RPEDC.shape[0]
        pixel_size = pixel_spacing_width * pixel_spacing_height
        pixel_vol = axiel_res_mm * pixel_size

        small_drusen_d, large_drusen_d, ped_drusen_d = drusen_diameter_thresholds_mm

        RPEDC = utils.filter_regions_by_neborhood_label(RPEDC, label_for_check=5, label_must_adjcent=1)
        DV = utils.filter_regions_by_neborhood_label(DV, label_for_check=6, label_must_adjcent=3)
        strel = generate_binary_structure(2, 2)
        rpedc_t = RPEDC == 5

        thk_ez_rpe = np.sum(np.logical_or(DV == 2, DV == 3), axis=1)
        rpedc_t_thk = np.sum(rpedc_t, axis=1)
        rpedc_t_msk = rpedc_t_thk >= thk_ez_rpe * 0.75
        rpedc_t_vol = np.repeat(rpedc_t_msk[:, np.newaxis, :], rpedc_t.shape[1], axis=1)
        rpedc_t = np.logical_and(rpedc_t, rpedc_t_vol)
  
        rpedc_vol = utils.binary_area_open(rpedc_t, 27)
        dv_vol = utils.binary_area_open(DV == 6, 27)
        outerfluid_vol = utils.binary_area_open(DV == 5, 27)
        dv_vol = np.logical_and(dv_vol, rpedc_vol)

        rpedc_ef = np.clip(np.sum(rpedc_vol, axis=1), 0, 60)
        dv_ef = np.clip(np.sum(dv_vol, axis=1), 0, 50)
        outerfluid_ef = np.sum(outerfluid_vol, axis=1)

        rpedc_raw_mask = median_filter(rpedc_ef > 2, 2)
        dv_raw_mask = median_filter(dv_ef > 2, 2)
        outerfluid_raw_mask = median_filter(outerfluid_ef > 2, 2)

        rpedc_ef = gaussian_filter(rpedc_ef, 0.5)
        dv_ef = gaussian_filter(dv_ef, 0.5)
        outerfluid_ef = gaussian_filter(outerfluid_ef, 0.5)
        rpedc_ef_dist = distance_transform_edt(rpedc_ef) / np.max(distance_transform_edt(rpedc_ef))
        dv_ef_dist = distance_transform_edt(dv_ef) / np.max(distance_transform_edt(dv_ef))
        rpedc_ef = rpedc_ef.astype(np.float32)
        dv_ef = dv_ef.astype(np.float32)
        rpedc_ef /= np.max(rpedc_ef)
        dv_ef /= np.max(dv_ef)

        rpedc_coords = peak_local_max(rpedc_ef + rpedc_ef_dist, min_distance=5, footprint=np.ones((5, 5)))
        dv_coords = peak_local_max(dv_ef + dv_ef_dist, min_distance=5, footprint=np.ones((5, 5)))

        rpedc_mask = np.zeros(rpedc_ef_dist.shape, dtype=bool)
        dv_mask = np.zeros(dv_ef_dist.shape, dtype=bool)
        rpedc_mask[tuple(rpedc_coords.T)] = True
        dv_mask[tuple(dv_coords.T)] = True

        rpedc_makers, _ = label(rpedc_mask, structure=strel)
        dv_makers, _ = label(dv_mask, structure=strel)

        rpedc_labels = watershed(-rpedc_ef, rpedc_makers, mask=rpedc_raw_mask)
        dv_labels = watershed(-rpedc_ef, dv_makers, mask=dv_raw_mask)
        # Merge overlapping large regions
        large_drusen_rg_lbl, count = label(dv_ef == 1)
        for i in range(1, count + 1):
            overlap = dv_labels * (large_drusen_rg_lbl == i)
            labels, _ = np.unique(overlap, return_counts=True)
            for label_ in labels[1:]:
                dv_labels[dv_labels == label_] = labels[-1]

        # Remove small or edge-overlapping regions
        stop_remove = False
        while not stop_remove:
            stop_remove = True
            for i in range(1, np.max(dv_labels) + 1):
                tmp = dv_labels == i
                if tmp.sum() == 0:
                    continue
                edge = np.logical_xor(tmp, binary_dilation(tmp, structure=strel))
                overlap = dv_labels * edge
                labels, counts = np.unique(overlap[overlap != 0], return_counts=True)
                edge_pixels = edge.sum()
                if len(counts) and (max(counts) / edge_pixels > 0.3 or sum(counts) / edge_pixels > 0.7 or tmp.sum() <= 10):
                    dv_labels[tmp] = labels[np.argmax(counts)]
                    stop_remove = False
        pseudo_drusen = rpedc_labels.copy()
        to_remove = np.unique(pseudo_drusen[np.logical_or(dv_raw_mask, outerfluid_raw_mask)])
        for label_ in to_remove:
            pseudo_drusen[pseudo_drusen == label_] = 0

        pseudo_boundaries = morphology.skeletonize(find_boundaries(pseudo_drusen, mode="outer"))
        dv_boundaries = morphology.skeletonize(find_boundaries(dv_labels, mode="outer"))

        pseudo_mask = (pseudo_drusen > 0).astype(np.uint8)
        masks = {k: np.zeros_like(pseudo_drusen, dtype=np.uint8) for k in ['small', 'medium', 'large', 'ped']}
        counts = dict.fromkeys(['small', 'medium', 'large', 'ped'], 0)

        for rg in regionprops(dv_labels, spacing=(pixel_spacing_height, pixel_spacing_width)):
            diameter = rg.feret_diameter_max if use_max_diameter else rg.equivalent_diameter_area
            label_map = dv_labels == rg.label
            if diameter <= small_drusen_d:
                masks['small'][label_map] = 1
                counts['small'] += 1
            elif diameter <= large_drusen_d:
                masks['medium'][label_map] = 1
                counts['medium'] += 1
            elif diameter <= ped_drusen_d:
                masks['large'][label_map] = 1
                counts['large'] += 1
            else:
                masks['ped'][label_map] = 1
                counts['ped'] += 1

        counts.update({'sdd': np.unique(pseudo_drusen).size - 1})
        pseudo_vol_mask = rpedc_vol * pseudo_mask[:, None, :]
        pseudo_vol_mask[binary_fill_holes(np.logical_or(DV == 3, DV == 4))] = 0
        volume_masks = {k: dv_vol * masks[k][:, None, :] for k in masks}

        vols = {k: np.sum(volume_masks[k]) * pixel_vol for k in volume_masks}
        vols.update({'sdd': np.sum(pseudo_vol_mask) * pixel_vol})
        all_vol_mask = (pseudo_vol_mask + sum((i + 2) * v for i, v in enumerate(volume_masks.values()))).astype(np.uint8)
        all_mask = (pseudo_mask + sum((i + 2) * masks[k] for i, k in enumerate(masks))).astype(np.uint8)
        all_mask[pseudo_boundaries + dv_boundaries] = 6

        return (
            all_vol_mask,
            all_mask,
            (vols['sdd'],vols['small'], vols['medium'], vols['large'], vols['ped'], counts['sdd'],counts['small'], counts['medium'], counts['large'], counts['ped']),
            dv_vol.astype(np.uint8),
            rpedc_vol.astype(np.uint8),
        )
         
        
    @staticmethod
    def generate_fluid_props(fluid_volume:np.ndarray, pixel_size:float) -> List[float]:
        """
        Calculate fluid volumes based on pixel count and pixel size.

        Args:
            fluid_volume (np.ndarray): 3D array where each fluid type is labeled with a unique integer.
            pixel_size (float): Physical size represented by each voxel.

        Returns:
            list: Volume (in physical units) of each fluid type in the order:
                [intra-retinal, subretinal, subRPE/PED/Drusen].
        """
        fluid_volumes = []
        fluid_volumes.append(np.sum(fluid_volume == 1) * pixel_size)
        fluid_volumes.append(np.sum(fluid_volume == 2) * pixel_size)
        fluid_volumes.append(np.sum(fluid_volume == 3) * pixel_size)
        return fluid_volumes


    @staticmethod
    def fill_holes(image: np.ndarray, hole_size: Optional[int]=30):
        """
        Fill small holes in a binary image.

        Args:
            image (np.ndarray): Binary image (2D).
            hole_size (int, optional): Maximum size of holes to fill. Defaults to 30.

        Returns:
            np.ndarray: Binary image with small holes filled.
        """
        return np.logical_not(utils.binary_area_open(np.logical_not(image), hole_size))


    @staticmethod
    def generateRegionGrowMask(image: np.ndarray, center: Tuple[int,int], stop_circle_radius: int, loDiff: float, upDiff: float)-> np.ndarray:
        """
        Perform 2D region growing from a seed point with stop mask.

        Args:
            image (np.ndarray): Input grayscale image.
            center (tuple): (x, y) coordinates for the seed point.
            stop_circle_radius (int): Radius around the seed point to stop the region growing.
            loDiff (float): Lower intensity difference threshold.
            upDiff (float): Upper intensity difference threshold.

        Returns:
            np.ndarray: Mask of the grown region.
        """
        stop_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
        stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 1, 2)
        cv2.floodFill(image.copy(), stop_mask, center, 1, loDiff=loDiff, upDiff=upDiff)
        stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 0, 2)
        return stop_mask[1:-1, 1:-1]


    @staticmethod
    def generateRegionGrowMasks(image: np.ndarray, centers: List[Tuple[int,int]], stop_circle_radius: int, loDiff: float, upDiff: float)-> np.ndarray:
        """
        Generate region growing masks for multiple seed points.

        Args:
            image (np.ndarray): Grayscale image.
            centers (list of tuple): List of (x, y) seed points.
            stop_circle_radius (int): Radius for the region-growing stop mask.
            loDiff (float): Lower intensity threshold.
            upDiff (float): Upper intensity threshold.

        Returns:
            np.ndarray: Combined region growing mask.
        """
        totalMask = np.zeros(image.shape, dtype=np.uint8)
        for center in centers:
            stop_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
            stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 1, 2)
            cv2.floodFill(image.copy(), stop_mask, center, 1, loDiff=loDiff, upDiff=upDiff)
            stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 0, 2)
            totalMask = np.logical_or(totalMask, stop_mask[1:-1, 1:-1])
        return totalMask

    @staticmethod
    def generateRegionGrowMask3D(
        volumeData: np.ndarray,
        currentFrameIdx: int,
        center: Tuple[int, int],
        stop_circle_radius: int,
        loDiff: float,
        upDiff: float
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate a 3D binary mask using region growing from a center point in a volumetric image.

        Args:
            volumeData (np.ndarray): 3D image volume of shape (Z, Y, X).
            currentFrameIdx (int): Index of the Z-slice to start growing from.
            center (Tuple[int, int]): (X, Y) coordinates in the slice.
            stop_circle_radius (int): Radius of the circular region of interest.
            loDiff (float): Lower intensity threshold difference.
            upDiff (float): Upper intensity threshold difference.

        Returns:
            Tuple[np.ndarray, List[int]]:
                - A 3D binary mask of the grown region.
                - A list of bounding box indices: [z_start, z_stop, y_start, y_stop, x_start, x_stop].
        """
        start_frame = currentFrameIdx
        stop_frame = min(currentFrameIdx + stop_circle_radius + 1, volumeData.shape[0])
        start_col = max(center[0] - stop_circle_radius, 0)
        stop_col = min(center[0] + stop_circle_radius + 1, volumeData.shape[1])
        start_row = max(center[1] - stop_circle_radius, 0)
        stop_row = min(center[1] + stop_circle_radius + 1, volumeData.shape[2])

        volumeData_croped = volumeData[start_frame:stop_frame, start_row:stop_row, start_col:stop_col]

        img_msk = np.zeros((volumeData.shape[1], volumeData.shape[2]), dtype=np.uint8)
        img_msk = cv2.circle(img_msk, center, stop_circle_radius, 1, -1)
        img_msk_croped = img_msk[start_row:stop_row, start_col:stop_col]
        vol_msk = np.repeat(img_msk_croped[np.newaxis, :, :], stop_frame - start_frame, axis=0)

        scol = max(center[0] - 1, 0)
        ecol = min(center[0] + 2, volumeData.shape[1])
        srow = max(center[1] - 1, 0)
        erow = min(center[1] + 2, volumeData.shape[2])
        meanV = np.mean(volumeData[currentFrameIdx, scol:ecol, srow:erow])

        out = np.logical_and(volumeData_croped > meanV - loDiff, volumeData_croped < meanV + upDiff)
        out = np.logical_and(out, vol_msk)
        out = np.pad(out, ((1, 1), (1, 1), (1, 1)), mode="symmetric")

        stel = generate_binary_structure(3, 2)
        out = binary_closing(out, structure=stel)
        out = binary_fill_holes(out, structure=stel)
        out = utils.keepLargestComponents3D(out)
        out = out[1:-1, 1:-1, 1:-1]

        return out, [start_frame, stop_frame, start_row, stop_row, start_col, stop_col]


    @staticmethod
    def generateAdaptThresMask(
        image: np.ndarray,
        stopMask: np.ndarray,
        loweroffset: float = 0,
        upperoffset: float = 0
    ) -> np.ndarray:
        """
        Generate a binary mask by thresholding an image within a specified mask region.

        Args:
            image (np.ndarray): 2D input image.
            stopMask (np.ndarray): Binary mask specifying the region to consider.
            loweroffset (float): Lower threshold offset from the mean.
            upperoffset (float): Upper threshold offset from the mean.

        Returns:
            np.ndarray: Binary mask after adaptive thresholding and region filtering.
        """
        # check if image is color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        meanV = np.mean(image[stopMask > 0])
        mask = np.logical_and(image > meanV + loweroffset, image < meanV + upperoffset)
        mask[stopMask == 0] = 0
        mask = utils.keepLargestComponents3D(mask)
        return mask


    @staticmethod
    def generateAdaptThresMasks(
        image: np.ndarray,
        centers: List[Tuple[int, int]],
        stop_circle_radius: int,
        angle_degrees: float,
        loweroffset: float = 0,
        upperoffset: float = 0,
        shape_generator: Optional[
            Callable[[Tuple[int, int], Tuple[int, int], int, float], np.ndarray]
        ] = None
    ) -> np.ndarray:
        """
        Generate binary masks for multiple seed points using adaptive thresholding.

        Args:
            image (np.ndarray): 2D input image.
            centers (List[Tuple[int, int]]): List of (X, Y) coordinates to threshold around.
            stop_circle_radius (int): Radius of the region around each center to consider.
            angle_degrees (float): Rotation angle for optional shape generator.
            loweroffset (float): Lower threshold offset from the local mean.
            upperoffset (float): Upper threshold offset from the local mean.
            shape_generator (Optional[Callable]): Function to generate custom shaped masks.

        Returns:
            np.ndarray: Combined binary mask from all regions.
        """
        # check if image is color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        totalMask = np.zeros(image.shape, dtype=np.uint8)

        for center in centers:
            center = (int(center[0]), int(center[1]))

            if shape_generator is not None:
                stop_mask = shape_generator(image.shape, center, stop_circle_radius, angle_degrees)
            else:
                stop_mask = np.zeros(image.shape, dtype=np.uint8)
                stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 1, -1)

            meanV = np.mean(image[stop_mask == 1])
            mask = np.logical_and(image > meanV + loweroffset, image < meanV + upperoffset)
            mask[stop_mask == 0] = 0
            mask = utils.keepLargestComponents3D(mask)
            totalMask = np.logical_or(totalMask, mask)

        return totalMask

    @staticmethod
    def generateRegionGrowMask3D(
        volumeData: np.ndarray,
        currentFrameIdx: int,
        center: Tuple[int, int],
        stop_circle_radius: int,
        loDiff: float,
        upDiff: float
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate a 3D binary mask using region growing from a center point in a volumetric image.

        Args:
            volumeData (np.ndarray): 3D image volume of shape (Z, Y, X).
            currentFrameIdx (int): Index of the Z-slice to start growing from.
            center (Tuple[int, int]): (X, Y) coordinates in the slice.
            stop_circle_radius (int): Radius of the circular region of interest.
            loDiff (float): Lower intensity threshold difference.
            upDiff (float): Upper intensity threshold difference.

        Returns:
            Tuple[np.ndarray, List[int]]:
                - A 3D binary mask of the grown region.
                - A list of bounding box indices: [z_start, z_stop, y_start, y_stop, x_start, x_stop].
        """
        start_frame = currentFrameIdx
        stop_frame = min(currentFrameIdx + stop_circle_radius + 1, volumeData.shape[0])
        start_col = max(center[0] - stop_circle_radius, 0)
        stop_col = min(center[0] + stop_circle_radius + 1, volumeData.shape[1])
        start_row = max(center[1] - stop_circle_radius, 0)
        stop_row = min(center[1] + stop_circle_radius + 1, volumeData.shape[2])

        volumeData_croped = volumeData[start_frame:stop_frame, start_row:stop_row, start_col:stop_col]

        img_msk = np.zeros((volumeData.shape[1], volumeData.shape[2]), dtype=np.uint8)
        img_msk = cv2.circle(img_msk, center, stop_circle_radius, 1, -1)
        img_msk_croped = img_msk[start_row:stop_row, start_col:stop_col]
        vol_msk = np.repeat(img_msk_croped[np.newaxis, :, :], stop_frame - start_frame, axis=0)

        scol = max(center[0] - 1, 0)
        ecol = min(center[0] + 2, volumeData.shape[1])
        srow = max(center[1] - 1, 0)
        erow = min(center[1] + 2, volumeData.shape[2])
        meanV = np.mean(volumeData[currentFrameIdx, scol:ecol, srow:erow])

        out = np.logical_and(volumeData_croped > meanV - loDiff, volumeData_croped < meanV + upDiff)
        out = np.logical_and(out, vol_msk)
        out = np.pad(out, ((1, 1), (1, 1), (1, 1)), mode="symmetric")

        stel = generate_binary_structure(3, 2)
        out = binary_closing(out, structure=stel)
        out = binary_fill_holes(out, structure=stel)
        out = utils.keepLargestComponents3D(out)
        out = out[1:-1, 1:-1, 1:-1]

        return out, [start_frame, stop_frame, start_row, stop_row, start_col, stop_col]


    @staticmethod
    def generateAdaptThresMask(
        image: np.ndarray,
        stopMask: np.ndarray,
        loweroffset: float = 0,
        upperoffset: float = 0
    ) -> np.ndarray:
        """
        Generate a binary mask by thresholding an image within a specified mask region.

        Args:
            image (np.ndarray): 2D input image.
            stopMask (np.ndarray): Binary mask specifying the region to consider.
            loweroffset (float): Lower threshold offset from the mean.
            upperoffset (float): Upper threshold offset from the mean.

        Returns:
            np.ndarray: Binary mask after adaptive thresholding and region filtering.
        """
        # check if image is color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        meanV = np.mean(image[stopMask > 0])
        mask = np.logical_and(image > meanV + loweroffset, image < meanV + upperoffset)
        mask[stopMask == 0] = 0
        mask = utils.keepLargestComponents3D(mask)
        return mask


    @staticmethod
    def generateAdaptThresMasks(
        image: np.ndarray,
        centers: List[Tuple[int, int]],
        stop_circle_radius: int,
        angle_degrees: float,
        loweroffset: float = 0,
        upperoffset: float = 0,
        shape_generator: Optional[
            Callable[[Tuple[int, int], Tuple[int, int], int, float], np.ndarray]
        ] = None
    ) -> np.ndarray:
        """
        Generate binary masks for multiple seed points using adaptive thresholding.

        Args:
            image (np.ndarray): 2D input image.
            centers (List[Tuple[int, int]]): List of (X, Y) coordinates to threshold around.
            stop_circle_radius (int): Radius of the region around each center to consider.
            angle_degrees (float): Rotation angle for optional shape generator.
            loweroffset (float): Lower threshold offset from the local mean.
            upperoffset (float): Upper threshold offset from the local mean.
            shape_generator (Optional[Callable]): Function to generate custom shaped masks.

        Returns:
            np.ndarray: Combined binary mask from all regions.
        """
        # check if image is color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        totalMask = np.zeros(image.shape, dtype=np.uint8)

        for center in centers:
            center = (int(center[0]), int(center[1]))

            if shape_generator is not None:
                stop_mask = shape_generator(image.shape, center, stop_circle_radius, angle_degrees)
            else:
                stop_mask = np.zeros(image.shape, dtype=np.uint8)
                stop_mask = cv2.circle(stop_mask, center, stop_circle_radius, 1, -1)

            meanV = np.mean(image[stop_mask == 1])
            mask = np.logical_and(image > meanV + loweroffset, image < meanV + upperoffset)
            mask[stop_mask == 0] = 0
            mask = utils.keepLargestComponents3D(mask)
            totalMask = np.logical_or(totalMask, mask)

        return totalMask

    @staticmethod
    def generateAdaptThresMask3D(
        volumeData: np.ndarray,
        currentFrameIdx: int,
        center: List[int] = [0, 0],
        stop_circle_radius: int = 1,
        loweroffset: float = 0,
        upperoffset: float = 0,
        mask: Optional[np.ndarray] = None,
        propagate_step: int = 1
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate a 3D mask by adaptive thresholding around a seed point or region.

        Args:
            volumeData (np.ndarray): 3D image volume (Z, Y, X).
            currentFrameIdx (int): Z-slice index to start processing.
            center (List[int]): (X, Y) center coordinate if mask is not given.
            stop_circle_radius (int): Radius of region around the center.
            loweroffset (float): Lower threshold offset from the mean.
            upperoffset (float): Upper threshold offset from the mean.
            mask (Optional[np.ndarray]): Optional 2D binary mask.
            propagate_step (int): Number of slices to propagate.

        Returns:
            Tuple[np.ndarray, List[int]]:
                - A 3D binary mask after adaptive thresholding.
                - Bounding box as [z_start, z_stop, y_start, y_stop, x_start, x_stop].
        """
        start_frame = currentFrameIdx
        stop_frame = min(currentFrameIdx + propagate_step + 1, volumeData.shape[0])

        if mask is None:
            start_col = max(center[0] - stop_circle_radius, 0)
            stop_col = min(center[0] + stop_circle_radius + 1, volumeData.shape[1])
            start_row = max(center[1] - stop_circle_radius, 0)
            stop_row = min(center[1] + stop_circle_radius + 1, volumeData.shape[2])
            
            img_msk = np.zeros((volumeData.shape[1], volumeData.shape[2]), dtype=np.uint8)
            img_msk = cv2.circle(img_msk, tuple(center), stop_circle_radius, 1, -1)
            img_msk_croped = img_msk[start_row:stop_row, start_col:stop_col]
        else:
            start_row, start_col, stop_row, stop_col = regionprops(mask.astype(np.uint8))[0].bbox
            img_msk_croped = mask[start_row:stop_row, start_col:stop_col]

        volumeData_croped = volumeData[start_frame:stop_frame, start_row:stop_row, start_col:stop_col]
        vol_msk = np.repeat(img_msk_croped[np.newaxis, :, :], stop_frame - start_frame, axis=0)

        meanV = np.mean(volumeData_croped[0])
        out = np.logical_and(volumeData_croped > meanV + loweroffset, volumeData_croped < meanV + upperoffset)
        out = np.logical_and(out, vol_msk)
        out = np.pad(out, ((1, 1), (1, 1), (1, 1)), mode="symmetric")

        stel = generate_binary_structure(3, 2)
        out = binary_closing(out, structure=stel)
        out = binary_fill_holes(out, structure=stel)
        out = utils.keepLargestComponents3D(out)
        out = out[1:-1, 1:-1, 1:-1]

        return out, [start_frame, stop_frame, start_row, stop_row, start_col, stop_col]

    @staticmethod
    def generateAdaptThresMasks3D(
        volumeData: np.ndarray,
        currentFrameIdx: int,
        centers: List[Tuple[int, int]],
        stop_circle_radius: int,
        loweroffset: float,
        upperoffset: float,
        propagate_step: int = 1,
        angle_degrees: float = 0,
        shape_generator: Optional[
            Callable[[Tuple[int, int], Tuple[int, int], int, float], np.ndarray]
        ] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate a 3D adaptive threshold mask using multiple centers.

        Args:
            volumeData (np.ndarray): 3D volume (Z, Y, X).
            currentFrameIdx (int): Z-slice to start processing.
            centers (List[Tuple[int, int]]): List of (X, Y) center coordinates.
            stop_circle_radius (int): Radius of region to mask.
            loweroffset (float): Lower threshold offset.
            upperoffset (float): Upper threshold offset.
            propagate_step (int): Number of slices to propagate.
            angle_degrees (float): Rotation angle if shape_generator is used.
            shape_generator (Optional[Callable]): Optional shape generator function.

        Returns:
            Tuple[np.ndarray, List[int]]:
                - The resulting 3D binary mask.
                - Bounding box as [z_start, z_stop, y_start, y_stop, x_start, x_stop].
        """
        stop_masks = np.zeros(volumeData[currentFrameIdx].shape, dtype=np.uint8)
        for center in centers:
            center = (int(center[0]), int(center[1]))
            if shape_generator is not None:
                stop_mask = shape_generator((volumeData.shape[1], volumeData.shape[2]), center, stop_circle_radius, angle_degrees)
            else:
                stop_mask = np.zeros((volumeData.shape[1], volumeData.shape[2]), dtype=np.uint8)
                stop_mask = cv2.circle(stop_mask, center, stop_circle_radius // 2, 1, -1)
            stop_masks = np.logical_or(stop_masks, stop_mask)

        stop_frame = min(currentFrameIdx + propagate_step + 1, volumeData.shape[0])
        min_row, min_col, max_row, max_col = regionprops(stop_masks.astype(np.uint8))[0].bbox

        volumeData_croped = volumeData[currentFrameIdx:stop_frame, min_row:max_row, min_col:max_col]
        img_msk_croped = stop_masks[min_row:max_row, min_col:max_col]
        vol_msk = np.repeat(img_msk_croped[np.newaxis, :, :], stop_frame - currentFrameIdx, axis=0)

        meanV = np.mean(volumeData_croped[0])
        out = np.logical_and(volumeData_croped > meanV + loweroffset, volumeData_croped < meanV + upperoffset)
        out = np.logical_and(out, vol_msk)
        out = np.pad(out, ((1, 1), (1, 1), (1, 1)), mode="symmetric")

        stel = generate_binary_structure(3, 2)
        out = binary_closing(out, structure=stel)
        out = binary_fill_holes(out, structure=stel)
        out = utils.keepLargestComponents3D(out)
        out = out[1:-1, 1:-1, 1:-1]

        return out, [currentFrameIdx, stop_frame, min_row, max_row, min_col, max_col]

    @staticmethod
    def calc_relative_path(
        file_paths: List[str],
        common_path: Optional[str] = None,
        calculate_common_path: bool = True
    ) -> List[str]:
        """
        Calculate relative paths from a list of file paths, handling different drive letters.

        Args:
            file_paths (List[str]): List of full file paths.
            common_path (Optional[str]): Predefined common path (if available).
            calculate_common_path (bool): If True, auto-compute common path.

        Returns:
            List[str]: List of relative paths or drive letters if on different drives.
        """
        if not file_paths:
            return []

        file_paths = [os.path.normpath(path) for path in file_paths]

        if common_path is None and not calculate_common_path:
            return [os.path.dirname(p).replace(':', '') for p in file_paths]

        if common_path is None and calculate_common_path:
            drives = set(os.path.splitdrive(path)[0].lower() for path in file_paths)
            if len(drives) > 1:
                return [os.path.dirname(p).replace(':', '') for p in file_paths]
            else:
                common_path = os.path.commonpath(file_paths)
                if common_path:
                    relative_paths = [os.path.relpath(os.path.dirname(path), common_path) for path in file_paths]
                    return [folder if folder != '.' else '' for folder in relative_paths]
                else:
                    return [os.path.dirname(p).replace(':', '') for p in file_paths]
        else:
            relative_paths = [os.path.relpath(os.path.dirname(path), common_path) for path in file_paths]
            return [folder if folder != '.' else '' for folder in relative_paths]
