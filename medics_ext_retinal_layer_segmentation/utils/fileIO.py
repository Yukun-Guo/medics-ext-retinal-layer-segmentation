"""
utils.fileIO
------------
Utility functions for file input/output operations, including image, DICOM, MAT, HDF5 (MED), and custom file formats.
"""

import os
import json
import glob
import datetime
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import natsort
import tifffile
from .datadict import dataDict
import PIL.Image as Image
import h5py
import mat73
from scipy import io as sio
from scipy import ndimage
import pydicom
from pydicom.filereader import dcmread
from pydicom.dataset import Dataset, FileDataset
import importlib.util
import sys
import traceback
import inspect
import logging
logger = logging.getLogger(__name__)


class FileIO(object):
    """
    A utility class providing static methods for file input/output operations, including image, DICOM, MAT, HDF5 (MED), and custom file formats.

    Attributes:
        None (all methods are static).

    Member Functions:
        list_files(directory: str, regx: str = '*.*', recursive: bool = False) -> List[str]:
            List files in a directory, optionally recursively, with natural sorting.

        read_file_list(list_txt_file: str) -> List[str]:
            Reads a list of file paths from a text file.

        read_image_file(fn: str, color_mode: Optional[str] = None) -> Optional[np.ndarray]:
            Read image from file as numpy array.

        write_image_file(fn: str, img: Any, colormap: Any = None) -> None:
            Write numpy image to file, optionally with colormap for indexed images.

        read_stack_image_file(fn: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            Read multi-page TIFF image and colormap (if present).

        write_stack_image_file(fn: str, stack_img: np.ndarray, colormap: Any = None) -> None:
            Write stack image to TIFF, optionally with colormap.

        write_csv_file(fn: str, data_columns: List[Any], colum_name: Optional[List[str]] = None) -> None:
            Write columns of data to a CSV file.

        read_csv_file(fn: str) -> List[List[float]]:
            Read columns of data from a CSV file.

        read_ioct_file(file_path: str) -> Optional[np.ndarray]:
            Reads an IOCT file and returns the data as a 3D numpy array.

        read_octa_file(file_path: str) -> Optional[np.ndarray]:
            Reads an OCTA file and returns the data as a 3D numpy array.

        read_foct_file(file_path: str, shape: Optional[Tuple[int, int, int]] = None) -> Optional[np.ndarray]:
            Reads an FOCT file and returns the data as a 3D numpy array.

        write_foct_file(file_path: str, data: np.ndarray) -> None:
            Writes the data to an FOCT file.

        read_ssada_file(file_path: str) -> Optional[np.ndarray]:
            Reads an SSADA file and returns the data as a 3D numpy array.

        read_dicom_file(file_path: str, as_uint8: bool = False) -> Optional[np.ndarray]:
            Reads a DICOM file and returns the pixel data as a numpy array.

        write_3d_array_to_dicom(output_file: str, image_3d: np.ndarray) -> None:
            Writes a 3D numpy array to a DICOM file.

        write_dicom_file(output_file: str, image_array: np.ndarray) -> None:
            Writes a numpy array to a DICOM file.

        read_img_file(file_path: str) -> Optional[np.ndarray]:
            Placeholder function for reading an IMG file.

        read_mat_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a MAT file and returns the data as a dictionary.

        read_mat_oct_file(file_path: str, tags: Tuple[str, ...] = ("oct","imgMat","imageMat")) -> Optional[np.ndarray]:
            Read OCT data from a MAT file.

        write_mat_file(file_path: str, data: Any, do_compression: bool = False) -> None:
            Save any type of data to a MAT file with optional compression.

        read_curve_mat_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a curve MAT file and returns the data as a dictionary.

        write_curve_mat_file(file_path: str, curve: Dict[str, Any]) -> None:
            Writes the curve data to a MAT file.

        read_curve_med_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a curve MED file and returns the data under the group 'curve'.

        read_seg_med_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a seg MED file and returns the data under the group 'curve'.

        write_curve_med_file(file_path: str, curve: Dict[str, Any], compression: Optional[str] = None) -> None:
            Writes the curve data to a MED file under the 'curve' group.

        read_curve_json_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a curve JSON file and returns the data as a dictionary.

        write_curve_json_file(file_path: str, curve: Dict[str, Any]) -> None:
            Writes the curve data to a JSON file.

        read_curve_dicom_file(file_path: str) -> Optional[Dict[str, Any]]:
            Reads a curve DICOM file and returns the data as a dictionary.

        write_med_file(file_name: str, data: Any, compression: Optional[str] = None, group_name: Optional[str] = None) -> None:
            Save any type of data to an MED file with optional compression.

        read_med_file(file_name: str) -> Any:
            Read data from an MED file, reconstructing nested structures.

        read_label_map(file_path: str, app_name: str = 'imagelabeler') -> Optional[Dict[str, np.ndarray]]:
            Load a label map from a .med or .tif/.tiff file for the specified application.

        load_volume_data(filePath: str) -> Optional[np.ndarray]:
            Load volumetric data from various supported file types.

        readBinaryFile(file_path: str):
            Read a binary file and return its content.

        isBinaryFile(file_path: str):
            Check if a file is binary by reading a small chunk of it.
    """

    @staticmethod
    def list_files(directory: str, regx: str = "*.*", recursive: bool = False) -> List[str]:
        """
        List files in a directory, optionally recursively, with natural sorting.
        Args:
            directory: Path to the directory.
            regx: Glob pattern to filter files.
            recursive: Whether to search recursively.
        Returns:
            List of file paths.
        """
        pattern = os.path.join(directory, "**", regx) if recursive else os.path.join(directory, regx)
        return natsort.natsorted(list(glob.glob(pattern, recursive=recursive)))

    @staticmethod
    def read_file_list(list_txt_file: str) -> List[str]:
        """
        Reads a list of file paths from a text file.
        Args:
            list_txt_file: Path to the text file containing the list of file paths.
        Returns:
            List of file paths.
        """
        with open(list_txt_file, "r") as fp:
            files = fp.readlines()
        return [item.rstrip() for item in files]

    @staticmethod
    def read_image_file(fn: str, color_mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Read image from file as numpy array.
        Args:
            fn: Full file path.
            color_mode: 'gray', 'rgb', or 'idx'.
        Returns:
            Numpy image array or None if error.
        """
        try:
            img = Image.open(fn.rstrip())
            if color_mode is not None:
                mode = color_mode.lower()
                if mode == "gray":
                    img = img.convert("L")
                elif mode == "rgb":
                    img = img.convert("RGB")
                elif mode == "idx":
                    img = img.convert("P")
            return np.array(img)
        except Exception:
            return None

    @staticmethod
    def write_image_file(fn: str, img: Any, colormap: Any = None) -> None:
        """
        Write numpy image to file, optionally with colormap for indexed images.
        Args:
            fn: Full file path.
            img: Numpy image array.
            colormap: Optional colormap for indexed images.
        """
        if colormap is None:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img, mode="P")
            # Flatten colormap to 768 entries (256*3)
            if isinstance(colormap, list):
                if isinstance(colormap[0], (tuple, list)):
                    colormap = [item for sublist in colormap for item in sublist]
            elif isinstance(colormap, np.ndarray):
                if len(colormap.shape) == 2:
                    colormap = colormap.flatten().tolist()
                elif len(colormap.shape) == 3:
                    colormap = colormap.reshape(-1).tolist()
            if len(colormap) > 768:
                colormap = colormap[:768]
            elif len(colormap) < 768:
                colormap += [0] * (768 - len(colormap))
            img.putpalette(colormap)
        img.save(fn)

    @staticmethod
    def read_stack_image_file(fn: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
        """
        Read a single/multi-page TIFF image along with optional colormap and description.

        Args:
            fn: Path to TIFF file.

        Returns:
            Tuple of (image array, colormap array or None, description string or None).
        """
        with tifffile.TiffFile(fn) as tif:
            first_page = tif.pages[0]
            description = json.loads(first_page.description)
            # Check for colormap
            tag = first_page.tags.get("ColorMap")
            colormap = np.stack(tag.value).T if tag else None

            # Read image stack
            if len(tif.pages) > 1:
                image = tif.asarray()
            else:
                image = first_page.asarray()

        return image, colormap, description

    @staticmethod
    def write_stack_image_file(fn: str, stack_img: np.ndarray, colormap: Any = None, description: dict = None) -> None:
        """
        Write stack image to TIFF, optionally with colormap.
        Args:
            fn: Full file path.
            stack_img: Numpy image array.
            colormap: Optional colormap.
        """
        if colormap is None:
            tifffile.imwrite(fn, stack_img, metadata={"axes": "ZXY"}, compression="zlib", description=json.dumps(description))
        else:
            if isinstance(colormap, list):
                if len(colormap) < 256:
                    colormap += [(0, 0, 0)] * (256 - len(colormap))
                elif len(colormap) > 256:
                    colormap = colormap[:256]
                colormap = np.array(colormap).reshape(3, 256)
            elif isinstance(colormap, np.ndarray):
                if colormap.shape == (3, 256):
                    pass
                elif colormap.shape == (256, 3):
                    colormap = colormap.T
                else:
                    raise ValueError("colormap should be a 2D array with shape (3,256) or (256,3)")
            tifffile.imwrite(fn, stack_img, metadata={"axes": "ZXY"}, colormap=colormap, compression="zlib", description=json.dumps(description))

    @staticmethod
    def write_csv_file(fn: str, data_columns: List[Any], colum_name: Optional[List[str]] = None) -> None:
        """
        Write columns of data to a CSV file.
        Args:
            fn: Full file path.
            data_columns: List of columns.
            colum_name: Optional list of column names.
        """
        with open(fn, "w") as f:
            if colum_name is not None:
                f.write(",".join(colum_name) + "\n")
            for row in zip(*data_columns):
                f.write(",".join([str(x) for x in row]) + "\n")

    @staticmethod
    def read_csv_file(fn: str) -> List[List[float]]:
        """
        Read columns of data from a CSV file.
        Args:
            fn: Full file path.
        Returns:
            List of columns (as lists of floats).
        """
        data = []
        with open(fn, "r") as f:
            for line in f:
                data.append([float(x) for x in line.strip().split(",")])
        return list(zip(*data))

    # --- Custom Binary/Medical Formats ---
    @staticmethod
    def read_ioct_file(file_path: str) -> Optional[np.ndarray]:
        """
        Reads an IOCT file and returns the data as a 3D numpy array.

        Args:
            file_path (str): Path to the IOCT file.

        Returns:
            np.ndarray: 3D numpy array containing the data with shape (frames,rows,cols).
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "rb") as f:
                data = f.read()
            firstImgOffsets = [68, 77]
            firstDims = [496, 640, 768, 960]
            offset = 0
            data_shape = None
            for oft in firstImgOffsets:
                lenData = (len(data) - oft) / 2  # 2 bytes per pixel for uint16
                for dim in firstDims:
                    temp = lenData / dim
                    rootofTemp = np.sqrt(temp)
                    if rootofTemp == int(rootofTemp):
                        data_shape = (dim, int(rootofTemp), int(rootofTemp))
                        offset = oft
                        break

            if data_shape is None:
                return None

            data = np.frombuffer(data[offset:], dtype=np.uint16).copy()
            data[-1] = 0
            data = data.reshape(data_shape[0], data_shape[1], data_shape[2], order="F")
            data = np.transpose(data, (2, 0, 1))

            min_val = np.min(data)
            max_val = np.max(data)
            data = ((data - min_val) / (max_val - min_val)) * 255
            return data.astype(np.uint8)
        except Exception as e:
            logger.exception("Error reading IOCT file %s: %s", file_path, e)
            return None

    @staticmethod
    def read_octa_file(file_path: str) -> Optional[np.ndarray]:
        """
        Reads an OCTA file and returns the data as a 3D numpy array.

        Args:
            file_path (str): Path to the IOCT file.

        Returns:
            np.ndarray: 3D numpy array containing the data.
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            data = f.read()

        try:
            info = np.frombuffer(data[0:100], dtype=np.uint16)
            firstDim = info[11]
            secondDim = info[12]
            thirdDim = info[13]
            data_shape = (firstDim, secondDim, thirdDim)
            if firstDim < 496:
                scale = 4
            else:
                scale = 1
            offset = len(data) - (2 * firstDim * secondDim * thirdDim)
            data = np.frombuffer(data[offset:], dtype=np.uint16).copy()
            data = data.reshape(firstDim, secondDim, thirdDim, order="F")
            data = ndimage.zoom(data, (scale, 1, 1), order=0)
        except Exception as e:
            offsets = [68, 77]
            scale = 1
            firstDims = [496, 640, 768, 960]
            data_shape = None
            offset = 0
            for oft in offsets:
                lenData = (len(data) - oft) / 2
                for dim in firstDims:
                    temp = lenData / dim
                    rootofTemp = np.sqrt(temp)
                    if rootofTemp == int(rootofTemp):
                        data_shape = (dim, int(rootofTemp), int(rootofTemp))
                        offset = oft
                        break
            # check if the data_shape is matched with the data length
            if data_shape is None:
                scale = 4
                firstDims = [124, 160, 192, 240]
                data_shape = None
                offset = 0
                for oft in offsets:
                    lenData = (len(data) - oft) / 2
                    for dim in firstDims:
                        temp = lenData / dim
                        rootofTemp = np.sqrt(temp)
                        if rootofTemp == int(rootofTemp):
                            data_shape = (dim, int(rootofTemp), int(rootofTemp))
                            offset = oft
                            break
            if data_shape is None:
                return None
            data = np.frombuffer(data[offset:], dtype=np.uint16)
            data = data.reshape(data_shape[0], data_shape[1], data_shape[2], order="F")
            data = ndimage.zoom(data, (scale, 1, 1), order=0)

        # data = np.flip(data, axis=0)
        data = np.transpose(data, (2, 0, 1))
        min_val = np.min(data)
        max_val = np.max(data)
        data = ((data - min_val) / (max_val - min_val)) * 255
        return data.astype(np.uint8)

    @staticmethod
    def read_foct_file(file_path: str, shape: Optional[Tuple[int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Reads an FOCT file and returns the data as a 3D numpy array.

        Args:
            file_path (str): Path to the FOCT file.
            shape (tuple): Shape (rows,cols,frames) of the data.

        Returns:
            np.ndarray: 3D numpy array containing the data with shape (frames,rows,cols).
        """
        if not os.path.exists(file_path):
            logger.warning("File not found: %s", file_path)
            return None
        special_type = 0
        with open(file_path, "rb") as f:
            data = f.read()
            data = np.frombuffer(data, dtype=np.float32)

        if shape is not None:
            data = data.reshape(shape[0], shape[1], shape[2], order="F")
        else:
            firstDims = [640, 768, 960]
            data_shape = None
            for dim in firstDims:
                temp = len(data) / dim
                rootofTemp = np.sqrt(temp)
                if rootofTemp == int(rootofTemp):
                    data_shape = (dim, int(rootofTemp), int(rootofTemp))
                    break

            if data_shape is None:
                # try special shape
                if len(data) == 640 * 513 * 206:
                    data_shape = (640, 513, 206)
                    special_type = 1
                elif len(data) == 640 * 512 * 43:
                    data_shape = (640, 512, 43)
                    special_type = 2
                else:
                    return None

            data = data.reshape(data_shape[0], data_shape[1], data_shape[2], order="F")
        if special_type == 1:
            data = data[:, :, :201]
        if special_type == 2:
            data = data[:, :, :13]

        data = np.flip(data, axis=0)
        data = np.transpose(data, (2, 0, 1))
        # normalize the data
        min_val = np.min(data)
        max_val = np.max(data)
        data = ((data - min_val) / (max_val - min_val)) * 255
        return data.astype(np.uint8)

    @staticmethod
    def write_foct_file(file_path: str, data: np.ndarray) -> None:
        """
        Writes the data to an FOCT file.

        Args:
            file_path (str): Path to the output FOCT file.
            data (np.ndarray): 3D numpy array containing the data.
        """
        data = np.transpose(data, (1, 2, 0))
        data = np.flip(data, axis=0)
        data = data.flatten(order="F")
        data = data.tobytes()

        with open(file_path, "wb") as f:
            f.write(data)

    @staticmethod
    def read_ssada_file(file_path: str) -> Optional[np.ndarray]:
        """
        Reads an SSADA file and returns the data as a 3D numpy array.

        Args:
            file_path (str): Path to the SSADA file.

        Returns:
            np.ndarray: 3D numpy array containing the data.
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            data = f.read()
            data = np.frombuffer(data, dtype=np.float32)

        firstDims = [160, 192, 240]
        data_shape = None

        for dim in firstDims:
            temp = len(data) / dim
            rootofTemp = np.sqrt(temp)
            if rootofTemp == int(rootofTemp):
                data_shape = (dim, int(rootofTemp), int(rootofTemp))
                break

        if data_shape is None:
            return None

        data = data.reshape(data_shape[0], data_shape[1], data_shape[2], order="F")
        data = ndimage.zoom(data, (4, 1, 1), order=0)
        data = np.flip(data, axis=0)
        data = np.transpose(data, (2, 0, 1))
        # normalize the data
        min_val = np.min(data)
        max_val = np.max(data)
        data = ((data - min_val) / (max_val - min_val)) * 255
        return data.astype(np.uint8)

    @staticmethod
    def read_dicom_file(file_path: str, as_uint8: bool = False) -> Optional[np.ndarray]:
        """
        Reads a DICOM file and returns the pixel data as a numpy array.

        Args:
            file_path (str): Path to the DICOM file.

        Returns:
            np.ndarray: Numpy array containing the pixel data.
        """
        if not os.path.exists(file_path):
            return None

        ds = dcmread(file_path)
        if as_uint8:
            # check the data type of the pixel data
            if ds.pixel_array.dtype == np.uint8:
                return np.transpose(ds.pixel_array, (0, 2, 1))
            else:
                # normalize the pixel data
                min_val = np.min(ds.pixel_array)
                max_val = np.max(ds.pixel_array)
                data = ((ds.pixel_array - min_val) / (max_val - min_val)) * 255
                return np.transpose(data.astype(np.uint8), (0, 2, 1))
        else:
            return np.transpose(ds.pixel_array, (0, 2, 1))

    @staticmethod
    def write_3d_array_to_dicom(output_file: str, image_3d: np.ndarray) -> None:
        """
        Writes a 3D numpy array to a DICOM file.

        Args:
            output_file (str): Path to the output DICOM file.
            image_3d (np.ndarray): 3D numpy array containing the image data.
        """
        # Create a DICOM dataset
        ds = Dataset()

        # Add patient information
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"

        # Set the dimensions of the image
        num_slices, rows, cols = image_3d.shape
        ds.Rows = rows
        ds.Columns = cols
        ds.NumberOfFrames = num_slices  # Indicate multiple frames in the DICOM
        ds.PhotometricInterpretation = "MONOCHROME2"  # Grayscale image
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0  # Unsigned integers

        # Concatenate all image slices into a single byte stream
        ds.PixelData = image_3d.tobytes()

        # Set other required DICOM tags
        ds.Modality = "OT"  # Other
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

        # Add file meta information
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

        # Create the FileDataset (DICOM file)
        filename = output_file
        file_ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Add the dataset to the FileDataset object
        file_ds.update(ds)

        # Add additional metadata
        file_ds.is_little_endian = True
        file_ds.is_implicit_VR = False
        file_ds.PatientName = "Test^Patient"
        file_ds.PatientID = "123456"
        file_ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        file_ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")

        # Write the DICOM file
        file_ds.save_as(filename)
        logger.info("DICOM file saved as %s", filename)

    @staticmethod
    def write_dicom_file(output_file: str, image_array: np.ndarray) -> None:
        """
        Writes a numpy array to a DICOM file.

        Args:
            output_file (str): Path to the output DICOM file.
            image_array (np.ndarray): Numpy array containing the image data.
        """
        # Create a DICOM dataset
        ds = Dataset()

        # Add patient information
        ds.PatientName = "Anonymous"
        ds.PatientID = "xxxxxxx"

        # Set the dimensions of the image
        rows, cols = image_array.shape
        ds.Rows = rows
        ds.Columns = cols
        ds.PhotometricInterpretation = "MONOCHROME2"  # Grayscale image
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8  # 8 bits for uint8 images
        ds.BitsStored = 8  # 8 bits are actually used to store the pixel data
        ds.HighBit = 7  # The most significant bit is at position 7 (since it's 0-based)
        ds.PixelRepresentation = 0  # Unsigned integers

        # Set the image data
        ds.PixelData = image_array.tobytes()

        # Set other required DICOM tags
        ds.Modality = "OT"  # Other
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

        # Add file meta information
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

        # Create the FileDataset (DICOM file)
        filename = output_file
        file_ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Add the dataset to the FileDataset object
        file_ds.update(ds)

        # Add additional metadata
        file_ds.is_little_endian = True
        file_ds.is_implicit_VR = False
        file_ds.PatientName = "Anonymous"
        file_ds.PatientID = "xxxxxx"
        file_ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        file_ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")

        # Write the DICOM file
        file_ds.save_as(filename)

    @staticmethod
    def read_img_file(file_path: str) -> Optional[np.ndarray]:
        """
        Placeholder function for reading an IMG file.

        Args:
            file_path (str): Path to the IMG file.

        Returns:
            None
        """
        if not os.path.exists(file_path):
            return None
        # Implement the logic to read the IMG file

    @staticmethod
    def read_mat_file(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Reads a MAT file and returns the data as a dictionary.

        Args:
            file_path (str): Path to the MAT file.

        Returns:
            dict: Dictionary containing the data.
        """
        if not os.path.exists(file_path):
            return None
        try:
            seg = sio.loadmat(file_path)
        except:
            seg = mat73.loadmat(file_path)
        # check each key in the seg dictionary, if the key is not in the dictionary, raise an error
        data = {}
        for key in seg.keys():
            if key == "__header__" or key == "__version__" or key == "__globals__":
                continue
            data[key] = seg[key]

        return data

    @staticmethod
    def read_mat_oct_file(file_path: str, tags: Tuple[str, ...] = ("oct", "imgMat", "imageMat")) -> Optional[np.ndarray]:
        """read oct data from mat file

        Args:
            file_path (string): file path
            tags (tuple, optional): tags of oct data in mat file. Defaults to ("oct","imgMat").

        Returns:
            oct data: return oct data with shape (frames,rows,cols)
        """
        d = FileIO.read_mat_file(file_path)
        if d is None:
            return None
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                if tag in d.keys():
                    dat = np.transpose(d[tag], (2, 0, 1))
                    min_val = np.min(dat)
                    max_val = np.max(dat)
                    dat = ((dat - min_val) / (max_val - min_val)) * 255
                    return dat.astype(np.uint8)
        return None

    @staticmethod
    def write_mat_file(file_path: str, data: Any, do_compression: bool = False) -> None:
        """
        Save any type of data (including nested structures) to an MED file with optional compression.

        Args:
            file_path (str): Path to the MED file to save data.
            data (any): Data to save. Can be a scalar, string, NumPy array, dictionary, or nested structure.
            do_compression (bool): Whether to compress the data.
        """
        sio.savemat(file_path, data, do_compression=do_compression)

    @staticmethod
    def read_curve_mat_file(file_path: str) -> Optional[dataDict]:
        """
        Reads a curve MAT file and returns the data as a dataDict.

        Args:
            file_path (str): Path to the curve MAT file.

        Returns:
            dataDict: dataDict containing the curve data.
        """
        if not os.path.exists(file_path):
            return None

        try:
            seg = sio.loadmat(file_path)
        except:
            seg = mat73.loadmat(file_path)

        if "permute" in seg.keys():
            curve = {
                "volumeSize": tuple(seg["volumeSize"][0]),
                "fazCenter": tuple(seg["fazCenter"][0]),
                "onhCenter": tuple(seg["onhCenter"][0]),
                "version": seg["version"][0],
                "permute": seg["permute"][0],
                "flip": seg["flip"][0],
                "flatten_offset": seg["flatten_offset"][0],
                "flatten_baseline": seg["flatten_baseline"][0][0],
                "curves": {
                    "PVD": seg["curves"]["PVD"][0][0],
                    "ILM": seg["curves"]["ILM"][0][0],
                    "NFLGCL": seg["curves"]["NFLGCL"][0][0],
                    "GCLIPL": seg["curves"]["GCLIPL"][0][0],
                    "IPLINL": seg["curves"]["IPLINL"][0][0],
                    "INLOPL": seg["curves"]["INLOPL"][0][0],
                    "OPLONL": seg["curves"]["OPLONL"][0][0],
                    "ELM": seg["curves"]["ELM"][0][0],
                    "EZ": seg["curves"]["EZ"][0][0],
                    "EZIZ": seg["curves"]["EZIZ"][0][0],
                    "IZRPE": seg["curves"]["IZRPE"][0][0],
                    "RPEBM": seg["curves"]["RPEBM"][0][0],
                    "SATHAL": seg["curves"]["SATHAL"][0][0],
                    "CHOROID": seg["curves"]["CHOROID"][0][0],
                },
            }
            return dataDict.from_dict(curve)
        else:
            # check each key in the seg dictionary, if the key is not in the dictionary, raise an error
            curve = {}

            curve["volumeSize"] = tuple([seg["volumeSize"][2], seg["volumeSize"][0], seg["volumeSize"][1]]) if "volumeSize" in seg else (0, 0, 0)
            curve["fazCenter"] = tuple(seg["fazCenter"]) if "fazCenter" in seg else (0, 0)
            curve["onhCenter"] = tuple(seg["onhCenter"]) if "onhCenter" in seg else (0, 0)
            curve["version"] = "1.0"
            curve["permute"] = "0,1,2"
            curve["flip"] = "None"
            curve["flatten_offset"] = seg["flatten_offset"] if "flatten_offset" in seg else 0
            curve["flatten_baseline"] = seg["flatten_baseline"] if "flatten_baseline" in seg else -1
            if "ManualCurveData" in seg:
                curveTag = "ManualCurveData"
            elif "CurveData" in seg:
                curveTag = "CurveData"
            elif "Curve" in seg:
                curveTag = "Curve"
            else:
                return None
            cruves = {
                "PVD": np.transpose(seg[curveTag][:, 0, :]) * 0,
                "ILM": np.transpose(seg[curveTag][:, 0, :]),
                "NFLGCL": np.transpose(seg[curveTag][:, 1, :]),
                "GCLIPL": np.transpose(seg[curveTag][:, 0, :]) * 0,
                "IPLINL": np.transpose(seg[curveTag][:, 2, :]),
                "INLOPL": np.transpose(seg[curveTag][:, 3, :]),
                "OPLONL": np.transpose(seg[curveTag][:, 4, :]),
                "ELM": np.transpose(seg[curveTag][:, 4, :]) * 0,
                "EZ": np.transpose(seg[curveTag][:, 5, :]),
                "EZIZ": np.transpose(seg[curveTag][:, 6, :]),
                "IZRPE": np.transpose(seg[curveTag][:, 6, :]) + 3,
                "RPEBM": np.transpose(seg[curveTag][:, 7, :]),
                "SATHAL": np.transpose(seg[curveTag][:, 9, :]),
                "CHOROID": np.transpose(seg[curveTag][:, 10, :]),
            }
            curve["curves"] = cruves
            return dataDict.from_dict(curve)

    @staticmethod
    def write_curve_mat_file(file_path: str, curve: Dict[str, Any]) -> None:
        """
        Writes the curve data to a MAT file.

        Args:
            file_path (str): Path to the output MAT file.
            curve (dict): Dictionary containing the curve data.
        """
        sio.savemat(file_path, curve)

    @staticmethod
    def read_curve_med_file(file_path: str) -> Optional[dataDict]:
        """
        Reads a curve MED file and returns the data under the group 'curve'.

        Args:
            file_path (str): Path to the curve MED file.

        Returns:
            dataDict: dataDict containing the curve data under 'curve', or None if not found.
        """
        if not os.path.exists(file_path):
            return None

        data = FileIO.read_med_file(file_path)
        if isinstance(data, dict) and "curve" in data:
            # check the vaildity of the curve data
            if (
                "volumeSize" not in data["curve"]
                or "fazCenter" not in data["curve"]
                or "onhCenter" not in data["curve"]
                or "version" not in data["curve"]
                or "permute" not in data["curve"]
                or "flip" not in data["curve"]
                or "curves" not in data["curve"]
            ):
                return None
            # parse the curve data
            try:
                curves = {key: value["data"] for key, value in data["curve"]["curves"].items()}
                data["curve"]["curves"] = curves
            except:
                return None

            return dataDict.from_dict(data["curve"])
        return None

    @staticmethod
    def read_seg_med_file(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Reads a seg MED file and returns the data under the group 'curve'.

        Args:
            file_path (str): Path to the curve MED file.

        Returns:
            dict: Dictionary containing the curve data under 'curve', or None if not found.
        """
        if not os.path.exists(file_path):
            return None

        data = FileIO.read_med_file(file_path)
        if isinstance(data, dict) and "curve" in data:
            # check the vaildity of the curve data
            if "volumeSize" not in data["curve"] or "curves" not in data["curve"]:
                return None
            # parse the curve data
            try:
                curves = {key: value["data"] for key, value in data["curve"]["curves"].items()}
                data["curve"]["curves"] = curves
            except:
                return None

            return data["curve"]
        return None

    @staticmethod
    def write_curve_med_file(file_path: str, curve: Dict[str, Any], compression: Optional[str] = None) -> None:
        """
        Writes the curve data to a MED file under the 'curve' group.

        Args:
            file_path (str): Path to the output MED file.
            curve (dict): Dictionary containing the curve data.
            compression (str or None): Compression algorithm to use ("gzip", "lzf", "szip", or None).
        """
        FileIO.write_med_file(file_path, curve, compression=compression, group_name="curve")

    @staticmethod
    def read_curve_json_file(file_path: str) -> Optional[dataDict]:
        """
        Reads a curve JSON file and returns the data as a dataDict.

        Args:
            file_path (str): Path to the curve JSON file.

        Returns:
            dataDict: dataDict containing the curve data with dot notation access.
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            curve = json.load(f)

        # curve = {"volumeSize": seg["volumeSize"], "fazCenter": seg["fazCenter"], "onhCenter": seg["onhCenter"], "curves": seg["curves"]}
        # check the vaildity of the curve data
        if (
            "volumeSize" not in curve
            or "fazCenter" not in curve
            or "onhCenter" not in curve
            or "version" not in curve
            or "permute" not in curve
            or "flip" not in curve
            or "curves" not in curve
        ):
            raise ValueError("Invalid curve data")
        curve["volumeSize"] = tuple(curve["volumeSize"])
        for key in curve["curves"].keys():
            # curve["curves"][key] = np.array(curve["curves"][key], dtype=np.float32)
            curve["curves"][key] = np.array(curve["curves"][key], dtype=np.float32)

        # Convert to dataDict for dot notation access
        return dataDict.from_dict(curve)

    @staticmethod
    def write_curve_json_file(file_path: str, curve: Dict[str, Any]) -> None:
        """
        Writes the curve data to a JSON file.

        Args:
            file_path (str): Path to the output JSON file.
            curve (dict): Dictionary containing the curve data.
        """
        curves = curve["curves"]
        for key in curves.keys():
            curves[key] = np.array(curves[key]).tolist()
        curve["curves"] = curves

        with open(file_path, "w") as f:
            json.dump(curve, f)

    @staticmethod
    def read_curve_dicom_file(file_path: str) -> Optional[dataDict]:
        """
        Reads a curve DICOM file and returns the data as a dataDict.

        Args:
            file_path (str): Path to the curve DICOM file.

        Returns:
            dataDict: dataDict containing the curve data.
        """
        if not os.path.exists(file_path):
            return None

        ds = dcmread(file_path)
        curve_shape = ds.pixel_array.shape
        curves_array = np.frombuffer(ds.pixel_array, dtype=np.float32)
        curves_array = curves_array.reshape(curve_shape)
        curve = {}
        curve["volumeSize"] = None
        curve["fazCenter"] = (0, 0)
        curve["onhCenter"] = (0, 0)
        curve["version"] = "1.0"
        curve["permute"] = "0,1,2"
        curve["flip"] = "None"
        curve["flatten_offset"] = 0
        curve["flatten_baseline"] = -1
        if curve_shape[0] == 13:
            cruves = {
                "PVD": curves_array[0] * 0,
                "ILM": curves_array[12],
                "NFLGCL": curves_array[9],
                "GCLIPL": curves_array[0] * 0,
                "IPLINL": curves_array[7],
                "INLOPL": curves_array[5],
                "OPLONL": curves_array[0] * 0,
                "ELM": curves_array[0] * 0,
                "EZ": curves_array[8],
                "EZIZ": curves_array[0] * 0,
                "IZRPE": curves_array[0] * 0,
                "RPEBM": curves_array[0],
                "SATHAL": curves_array[0] * 0,
                "CHOROID": curves_array[3],
            }
        else:
            cruves = {
                "PVD": curves_array[0] * 0,
                "ILM": curves_array[11],
                "NFLGCL": curves_array[8],
                "GCLIPL": curves_array[0] * 0,
                "IPLINL": curves_array[7],
                "INLOPL": curves_array[6],
                "OPLONL": curves_array[0] * 0,
                "ELM": curves_array[0] * 0,
                "EZ": curves_array[0] * 0,
                "EZIZ": curves_array[0] * 0,
                "IZRPE": curves_array[0] * 0,
                "RPEBM": curves_array[9],
                "SATHAL": curves_array[0] * 0,
                "CHOROID": curves_array[3],
            }
        curve["curves"] = cruves
        return dataDict.from_dict(curve)

    @staticmethod
    def write_med_file(file_name: str, data: Any, compression: Optional[str] = None, group_name: Optional[str] = None) -> None:
        """
        Save any type of data (including nested structures) to an MED file with optional compression.

        Args:
            file_name (str): Path to the MED file to save data.
            data (any): Data to save. Can be a scalar, string, NumPy array, dictionary, or nested structure.
            compression (str or None): Compression algorithm to use ("gzip", "lzf", "szip", or None).
            group_name (str or None): Name of the group to save the data under. If None, save data at the root level.
        """

        def recursively_save(hdf_group, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    # Create groups for nested dictionaries
                    subgroup = hdf_group.create_group(key)
                    recursively_save(subgroup, value)
            elif isinstance(data, np.ndarray):
                hdf_group.create_dataset("data", data=data, compression=compression)
            elif isinstance(data, (int, float, np.number, bool)):
                hdf_group.attrs["value"] = data
            elif isinstance(data, list):
                hdf_group.attrs["value"] = json.dumps(data)
            elif isinstance(data, tuple):
                hdf_group.attrs["value"] = json.dumps(data)
            elif isinstance(data, str):
                hdf_group.attrs["value"] = data
            elif data is None:
                hdf_group.attrs["value"] = "None"
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

        with h5py.File(file_name, "w") as hdf:
            if group_name:
                # Create a new group with the specified group_name
                group = hdf.create_group(group_name)
                recursively_save(group, data)
            else:
                # Save data at the root level
                recursively_save(hdf, data)

    @staticmethod
    def read_med_file(file_name: str) -> Any:
        """
        Read data from an MED file, reconstructing nested structures.

        Args:
            file_name (str): Path to the MED file to read data from.

        Returns:
            any: Reconstructed data from the MED file.
        """

        def recursively_load(hdf_group):
            if isinstance(hdf_group, h5py.Dataset):
                # Return the dataset as a NumPy array
                return hdf_group[:]
            elif "value" in hdf_group.attrs:
                # If it's an attribute, parse its value
                value = hdf_group.attrs["value"]
                try:
                    # Attempt to parse JSON-encoded lists or tuples
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return value
            elif isinstance(hdf_group, h5py.Group):
                # If it's a group, reconstruct it as a dictionary
                result = {}
                for key, subgroup in hdf_group.items():
                    result[key] = recursively_load(subgroup)
                return result
            else:
                return None

        with h5py.File(file_name, "r") as hdf:
            data = recursively_load(hdf)
            return data

    @staticmethod
    def write_json_file(file_path: str, data: Any) -> None:
        """
        Save data to a JSON file.

        Args:
            file_path (str): Path to the JSON file to save data.
            data (any): Data to save. Can be a scalar, string, NumPy array, dictionary, or nested structure.
        """

        # recursively convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(data):
            if isinstance(data, dict):
                return {key: convert_numpy_to_list(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_numpy_to_list(item) for item in data]
            elif isinstance(data, np.ndarray):
                return data.tolist()
            return data

        data = convert_numpy_to_list(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def read_json_file(file_path: str) -> Optional[Any]:
        """
        Read data from a JSON file.

        Args:
            file_path (str): Path to the JSON file to read data from.

        Returns:
            any: Data read from the JSON file, or None if the file does not exist.
        """

        # recursively convert lists back to numpy arrays
        def convert_list_to_numpy(data):
            if isinstance(data, dict):
                return {key: convert_list_to_numpy(value) for key, value in data.items()}
            elif isinstance(data, list):
                return np.array([convert_list_to_numpy(item) for item in data], dtype=np.uint8)
            elif isinstance(data, (int, float, str, bool)):
                return data
            else:
                return None

        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = convert_list_to_numpy(data)
        if isinstance(data, dict) and "data" in data:
            # If the data is wrapped in a dictionary with a 'data' key, extract it
            data = data["data"]
        elif isinstance(data, list) and len(data) == 1:
            # If the data is a single-item list, return the first item
            data = data[0]
        elif isinstance(data, str):
            # If the data is a string, try to parse it as JSON
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        return data

    @staticmethod
    def read_label_map(file_path: str, app_name: str = "imagelabeler") -> Tuple[Optional[Dict[str, np.ndarray]], str]:
        """
        Load a label map from a .med or .tif/.tiff file for the specified application.
        Args:
            file_path: Path to the label map file.
            app_name: Application name (default 'imagelabeler').
        Returns:
            Dictionary of label maps, or None if invalid or not found.
        """
        app_name = app_name.lower()

        def check_label_map_validity(label_map: Any) -> bool:
            return isinstance(label_map, dict) and app_name in label_map and "maps" in label_map[app_name]
        maps = None
        notes = ""
        if os.path.isfile(file_path):
            if file_path.endswith(".med"):
                med_data = FileIO.read_med_file(file_path)
                if check_label_map_validity(med_data):
                    maps = med_data[app_name]["maps"]
                    maps = {key: value["data"] for key, value in maps.items()}
                    notes = med_data[app_name].get("notes", "")
            elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
                im_data, _ , desc_dict = FileIO.read_stack_image_file(file_path)
                maps = {"Default": im_data}
                notes = desc_dict.get("notes", "") if isinstance(desc_dict, dict) else desc_dict
            elif file_path.endswith(".png"):
                im_data = FileIO.read_image_file(file_path, color_mode="idx")
                maps = {"Default": im_data}
                notes = ""
            elif file_path.endswith(".json"):  # special case for label maps in JSON format
                json_data = FileIO.read_json_file(file_path)
                if isinstance(json_data, dict) and "maps" in json_data:
                    maps = json_data["maps"]
                    notes = json_data.get("notes", "Good@")
                elif isinstance(json_data, list):
                    maps = {"Default": np.array(json_data)}
                    notes = ""
        return maps, notes

    @staticmethod
    def load_volume_data(filePath: str) -> Optional[np.ndarray]:
        """
        Load volumetric data from various supported file types.
        Args:
            filePath: Path to the volume file.
        Returns:
            Numpy array of the volume data, or None if not supported or not found.
        """
        if not os.path.isfile(filePath):
            return None
        data = None
        file_ext = os.path.splitext(filePath)[1].lower()
        if file_ext == ".foct":
            data = FileIO.read_foct_file(filePath)
        elif file_ext == ".oct":
            data = FileIO.read_foct_file(filePath)
        elif file_ext == ".dcm":
            data = FileIO.read_dicom_file(filePath, as_uint8=True)
        elif file_ext == ".ioct":
            data = FileIO.read_ioct_file(filePath)
        elif file_ext == ".img":
            data = FileIO.read_img_file(filePath)
        elif file_ext == ".mat":
            data = FileIO.read_mat_oct_file(filePath)
        return data

    @staticmethod
    def load_functions(folder_path: str) -> Dict[str, object]:
        """
        Loads all functions from single .py file or all .py files in the specified folder.

        Args:
            folder_path (str): Path to the .py file or the folder containing .py files.

        Returns:
            A dictionary mapping module names to dictionaries of {function_name: function_object}.
        """

        def load_functions_from_file(file_path: str, module_name: str) -> Dict[str, object]:
            """
            Load functions from a single Python file.
            """
            functions = {}
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None:
                    logger.warning("Warning: Could not load spec from '%s'. Skipping.", file_path)
                    return functions

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Inspect the module for functions
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    functions[name] = obj
            except SyntaxError as e:
                logger.exception("Syntax Error in '%s': %s", file_path, e)
                traceback.print_exc()
            except Exception as e:
                logger.exception("Error loading module '%s': %s", module_name, e)
                traceback.print_exc()
            return functions

        loaded_functions = {}

        if not os.path.exists(folder_path):
            logger.error("Error: The path '%s' does not exist.", folder_path)
            return loaded_functions

        if os.path.isfile(folder_path):
            # If a single file is provided, load functions from that file
            module_name = os.path.splitext(os.path.basename(folder_path))[0]
            functions = load_functions_from_file(folder_path, module_name)
            if functions:
                loaded_functions[module_name] = functions
        elif os.path.isdir(folder_path):
            # If a folder is provided, iterate through all .py files
            for filename in os.listdir(folder_path):
                if filename.endswith(".py"):
                    file_path = os.path.join(folder_path, filename)
                    module_name = os.path.splitext(filename)[0]
                    functions = load_functions_from_file(file_path, module_name)
                    if functions:
                        loaded_functions[module_name] = functions
        else:
            logger.error("Error: '%s' is neither a valid file nor a directory.", folder_path)

        return loaded_functions

    @staticmethod
    def readBinaryFile(file_path: str):
        """
        Read a binary file and return its content.
        Args:
            file_path (str): The path to the binary file.
        Returns:
            bytes: The content of the binary file, or None if an error occurs.
        """
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            logger.exception("Error: %s", e)
            return None

    @staticmethod
    def isBinaryFile(file_path: str):
        """
        Check if a file is binary by reading a small chunk of it.
        Args:
            file_path (str): The path to the file to check.
        Returns:
            bool: True if the file is binary, False if it is text.
        """
        try:
            with open(file_path, "rb") as file:
                # Read a chunk of data and check for null bytes (indicating binary data)
                chunk = file.read(1024)
                if b"\x00" in chunk:
                    return True
            return False
        except Exception as e:
            logger.exception("Error: %s", e)
            return False
