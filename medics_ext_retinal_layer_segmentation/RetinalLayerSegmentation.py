import os
from pathlib import Path
import re
import configparser

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    Qt,
    QFileInfo,
    QCoreApplication,
    QByteArray,
    Slot,
    QUrl,
    QSize,
    QEvent,
    QTimer,
)
from PySide6.QtGui import QImage, QPixmap, QPalette, QPainter, QAction, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
    QMainWindow,
    QMdiSubWindow,
    QMenu,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QProgressDialog,
    QLabel,
    QDialog
)

import logging

# Module logger
logger = logging.getLogger(__name__)

# VTK imports with error handling
try:
    from vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray, VTK_FLOAT
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.util import numpy_support
    VTK_AVAILABLE = True
except ImportError as e:
    logger.warning("VTK modules not available: %s", e)
    logger.warning("VTK-related functionality will be disabled.")
    # Create placeholder classes/constants to prevent runtime errors
    class vtkPoints: pass
    class vtkFloatArray: pass
    class vtkPolyData: pass
    class vtkCellArray: pass
    VTK_FLOAT = None
    numpy_support = None
    VTK_AVAILABLE = False

from .RetinalLayerSegmentaionUI import Ui_RetinalLayerSegmentaion
from .LayerSegmentation import LayerSegmentation
from .ResolutionHelper import CalculateResolution
from .LoadDataSettingsDlg import LoadDataSettings
from .LoadGroupDataDlg.LoadGroupDataMain import LoadGroupDataClass
from .utils.utils import utils
from .utils.fileIO import FileIO as fio
from .utils.datadict import dataDict

import logging

from typing import Optional, Union, List, Dict, Tuple, Any, Callable

pg.setConfigOption("background", "{}{:02x}".format("#151515", 60))


class UpdateUIItems:
    bframeA_oct = False
    bframeA_curve = False
    bframeA_indicators = False
    bframeB_oct = False
    bframeA_octa = False
    bframeB_curve = False
    bframeB_indicators = False
    enfaceA_img = False
    enfaceA_indicators = False
    enfaceA_interp = False
    enfaceB_img = False
    enfaceB_indicators = False

    def __init__(
        self,
        bframeA_oct=False,
        bframeA_curve=False,
        bframeA_indicators=False,
        bframeB_oct=False,
        bframeA_octa=False,
        bframeB_curve=False,
        bframeB_indicators=False,
        enfaceA_img=False,
        enfaceA_indicators=False,
        enfaceA_interp=False,
        enfaceB_img=False,
        enfaceB_indicators=False,
        all=False,
    ):
        if all:
            bframeA_oct = True
            bframeA_curve = True
            bframeA_indicators = True
            bframeB_oct = True
            bframeA_octa = True
            bframeB_curve = True
            bframeB_indicators = True
            enfaceA_img = True
            enfaceA_indicators = True
            enfaceA_interp = True
            enfaceB_img = True
            enfaceB_indicators = True
        self.bframeA_oct = bframeA_oct
        self.bframeA_curve = bframeA_curve
        self.bframeA_indicators = bframeA_indicators
        self.bframeB_oct = bframeB_oct
        self.bframeA_octa = bframeA_octa
        self.bframeB_curve = bframeB_curve
        self.bframeB_indicators = bframeB_indicators
        self.enfaceA_img = enfaceA_img
        self.enfaceA_indicators = enfaceA_indicators
        self.enfaceA_interp = enfaceA_interp
        self.enfaceB_img = enfaceB_img
        self.enfaceB_indicators = enfaceB_indicators


class RetinalLayerSegmentation(QMainWindow):
    """
    Main window for the RetinalLayerSegmentaion tool, providing a multi-tab interface for OCT/OCTA data visualization,
    segmentation, and analysis, including enface viewing, batch processing, and biomarker extraction.
    """
    ##Enface Viewer variables end ############################################################
    def __init__(self, parentWindow=None, app_context=None, theme="dark", **kwargs):

        self.app_context = app_context
        self.parentWindow = parentWindow
        self.theme = theme
        self.extension_name = "RetinalLayerSegmentation"
        self.oct_data = None
        self.octa_data = None
        self.volume_size = (1, 1, 1)
        self.oct_data_raw = None
        self.octa_data_raw = None
        self.oct_data_flatten = None
        self.octa_data_flatten = None
        self.oct_data_flatten_raw = None
        self.octa_data_flatten_raw = None
        self.oct_data_flattened = None
        self.flatten_offset = 0
        self.flatten_baseline = -1
        self.flatten_permute = (0, 1, 2)
        self.curve_data_dict = None
        self.oct_data_range = [0, 1]
        self.octa_data_range = [0, 1]
        self.indicatorDirection = 1
        self.resolution_width = 1
        self.scan_width_mm = 1
        self.resolution_height = 1
        self.scan_height_mm = 1
        self.resolution_depth = 1

        self.tab_initializers = None
        self.current_tab_index = 0

        # self.bframe_b_type = "Fast"
        self.last_opened_dir = ""
        self.oct_file_filters = "OCT Files (*.foct  *.oct *.dcm *.img *.mat);;All Files (*)"
        self.octa_file_filters = "OCTA Files (*.ssada  *.octa *.dcm *.img *.mat);;All Files (*)"
        self.seg_file_filters = "Segmentation Files (*.json *.dcm *.mat);;All Files (*)"
        self.data_extension_pairs = {".foct": ".ssada"}
        self.oct_file_extensions = [".foct", ".dcm", ".img", ".mat"]
        self.octa_file_extensions = [".ssada", ".dcm", ".img", ".mat"]
        self.seg_file_extensions = [".json"]
        self.current_flatten_method = "None"
        self.data_loader_settings = None
        self.ui = None
        self.seg_data = None
        self.status_data_shape = None
        self.status_mouse_pos = None
        self.status_pixel_value_pos = None
        # tabs
        self.LayerSegmentation = None
        self.gCurve_data = None

        self.setupUI()

    def setupUI(self):
        QMainWindow.__init__(self)
        self.ui = Ui_RetinalLayerSegmentaion()
        self.ui.setupUi(self)
        self.ui.centralwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.ui.toolBar.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        
        # Enable drag and drop for the main window to accept variables from Variable Tree
        self.setAcceptDrops(True)
        self.ui.pushButton_update_resolution.clicked.connect(self.updateResolutionDlg)
        self.ui.comboBox_flatten.wheelEvent = lambda event: event.ignore()
        self.ui.comboBox_flatten.keyPressEvent = lambda event: event.ignore()
        self.ui.comboBox_flatten.keyReleaseEvent = lambda event: event.ignore()
        self.ui.comboBox_flatten.currentTextChanged.connect(self.onTransform_changed)
        self.ui.comboBox_permute.currentTextChanged.connect(self.onTransform_changed)
        self.ui.comboBox_flip.currentTextChanged.connect(self.onTransform_changed)
        self.ui.spinBox_roi_bot.valueChanged.connect(self.onROIChanged)
        self.ui.spinBox_roi_top.valueChanged.connect(self.onROIChanged)
        self.ui.pushButton_auto_roi.clicked.connect(self.onAutoROI)
        self.ui.pushButton_transfer_oct.clicked.connect(lambda checked=False: self.on_transferData("OCT"))
        self.ui.pushButton_transfer_octa.clicked.connect(lambda checked=False: self.on_transferData("OCTA"))
        self.ui.pushButton_transfer_seg.clicked.connect(lambda checked=False: self.on_transferData("Seg"))
        self.ui.lineEdit_transfer_oct_name.textChanged.connect(self.on_octNameChanged)
        self.ui.lineEdit_transfer_octa_name.textChanged.connect(self.on_octaNameChanged)
        self.ui.lineEdit_transfer_seg_name.textChanged.connect(self.on_segNameChanged)
        # self.ui.tabWidget.currentChanged.connect(self.onTabChanged)
        self.ui.pushButton_load_settings.clicked.connect(self._on_Load_Settings)
        # self.ui.statusbar.mouseDoubleClickEvent = self.onStatusBarDoubleClick

        # set up other ui elements
        self.ui.statusbar.setStyleSheet("background-color: #16825d;color: #ffffff;font-size: 9pt;")
        self.setWindowIcon(self.getIcon("windowIcon"))
        self.ui.toolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        # config statusbar
        self.status_data_shape = QLabel(self)
        self.status_mouse_pos = QLabel(self)
        self.status_pixel_value_pos = QLabel(self)
        self.ui.statusbar.addPermanentWidget(self.status_mouse_pos)
        self.ui.statusbar.addPermanentWidget(self.status_pixel_value_pos)
        self.ui.statusbar.addPermanentWidget(self.status_data_shape)

        # load settings.ini
        try:
            settings = configparser.ConfigParser()
            # settings.read(os.path.join(self.app_context.root_path, "extensions/RetinalLayerSegmentation/settings.ini"))
            settings.read(Path(__file__).parent / "settings.ini")
            self.oct_file_extensions = settings.get("DATA_PAIRS", "OCTExtensions").lower().split(",")
            # generate the filters from the extensions
            self.oct_file_filters = "OCT Files (" + " ".join([f"*{ext}" for ext in self.oct_file_extensions]) + ");;All Files (*)"
            self.octa_file_extensions = settings.get( "DATA_PAIRS", "OCTAExtensions").lower().split(",")
            # generate the filters from the extensions
            self.octa_file_filters = "OCTA Files (" + " ".join([f"*{ext}" for ext in self.octa_file_extensions]) + ");;All Files (*)"
            self.seg_file_extensions = settings.get("DATA_PAIRS", "SegExtensions").lower().split(",")
            # generate the filters from the extensions
            self.seg_file_filters = "Segmentation Files (" + " ".join([f"*{ext}" for ext in self.seg_file_extensions]) + ");;All Files (*)"
            # generate the filters from the extensions
            self.data_extension_pairs = dict(zip(self.oct_file_extensions, self.octa_file_extensions))
            #self.enabledTabs = self.parentWindow.enabledToolboxes if self.parentWindow else []
            
        except Exception as e:
            logging.error("Failed to load settings.ini: %s", e)
            self.ui.statusbar.showMessage("Failed to load settings.ini", 5000)

        # Initialize data
        self._update_datasource_dropdown()

        # Connect signals
        self.ui.spinBox_frameIdx.valueChanged.connect(self.onFrameIdx_changed)
        self.ui.spinBox_frameIdx.keyPressEvent = lambda event: event.ignore()
        self.ui.spinBox_frameIdx.keyReleaseEvent = lambda event: event.ignore()

        self.ui.pushButton_refreshDatalist.clicked.connect(self._update_datasource_dropdown)
        self.ui.comboBox_OCT.currentIndexChanged.connect(self._on_OCTData_changed)
        self.ui.comboBox_OCTA.currentIndexChanged.connect(self._on_OCTAData_changed)
        self.ui.comboBox_Seg.currentIndexChanged.connect(self._on_SegData_changed)
        self.ui.lineEdit_OCTFilename.textChanged.connect(self._on_OCTFile_changed)
        self.ui.lineEdit_OCTFilename.dropEvent = self.on_OCTFile_dropEvent
        self.ui.lineEdit_OCTFilename.dragEnterEvent = self.on_OCTFile_dragEnterEvent
        self.ui.lineEdit_OCTFilename.dragLeaveEvent = self.on_OCTFile_dragLeaveEvent
        self.ui.lineEdit_OCTAFilename.dropEvent = self.on_OCTAFile_dropEvent
        self.ui.lineEdit_OCTAFilename.dragEnterEvent = self.on_OCTAFile_dragEnterEvent
        self.ui.lineEdit_OCTAFilename.dragLeaveEvent = self.on_OCTAFile_dragLeaveEvent
        self.ui.lineEdit_SegFilename.dropEvent = self.on_SegFile_dropEvent
        self.ui.lineEdit_SegFilename.dragEnterEvent = self.on_SegFile_dragEnterEvent
        self.ui.lineEdit_SegFilename.dragLeaveEvent = self.on_SegFile_dragLeaveEvent
        self.ui.pushButton_openOCT.clicked.connect(lambda checked=False: self.on_openFileDialog("OCT"))
        self.ui.pushButton_openOCTA.clicked.connect(lambda checked=False: self.on_openFileDialog("OCTA"))
        self.ui.pushButton_openSeg.clicked.connect(lambda checked=False: self.on_openFileDialog("Seg"))
        self.ui.pushButton_loadData.clicked.connect(self.loadData)
        self.ui.pushButton_loadData.dragEnterEvent = self.on_loadData_dragEnterEvent
        self.ui.pushButton_loadData.dragLeaveEvent = self.on_loadData_dragLeaveEvent
        self.ui.pushButton_loadData.dropEvent = self.on_loadData_dropEvent

        self.ui.comboBox_permute.currentTextChanged.connect(self.onPermute_changed)
        self.ui.comboBox_flip.currentTextChanged.connect(self.onFlip_changed)
        self.ui.checkBox_OCTA.stateChanged.connect(self.on_changedOCTAVisibility)
        self.ui.actionSave.triggered.connect(self.on_saveData)

        self.tab_initializers = {
            'LayerSegmentation': self.add_layer_segmentation_tab,
        }
        QTimer.singleShot(0, self.initialize_tabs)
        self.ui.actionCleanUp.triggered.connect(self.cleanup)
        
    def initialize_tabs(self):
        self.current_tab_index = 0
        self.add_next_tab()

    def add_next_tab(self):
        tab_keys = list(self.tab_initializers.keys())
        if self.current_tab_index < len(self.tab_initializers):
            self.tab_initializers[tab_keys[self.current_tab_index]](self.current_tab_index)
            self.current_tab_index += 1
            QTimer.singleShot(0, self.add_next_tab)

    def add_layer_segmentation_tab(self, idx: int):
        self.LayerSegmentation = LayerSegmentation(parentWindow=self, app_context=self.app_context, theme=self.theme)
        self.ui.horizontalLayout_15.addWidget(self.LayerSegmentation)
        self.ui.stackedWidget.insertWidget(idx, self.LayerSegmentation.LayerSegCtrlPanelWidget)
        
    def cleanup(self):
        """Clean up the current LayerSegmentation widget and create a new one.
        
        Args:
            confirm (bool): If True, show confirmation dialog before cleanup. 
                          Set to False when called during application shutdown.
        """
        if self.LayerSegmentation is not None:
            # # Ask for user confirmation before cleanup (only if confirm=True)
            # reply = QMessageBox.question(
            #     self,
            #     "Confirm Cleanup",
            #     "Reset Layer Segmentation? This will remove current data and create a fresh instance.",
            #     QMessageBox.Yes | QMessageBox.No,
            #     QMessageBox.No,
            # )
            # if reply == QMessageBox.No:
            #     return
            
            # Reset all UI elements to initial values
            try:
                # Reset data source dropdowns
                self.ui.comboBox_OCT.setCurrentIndex(0)
                self.ui.comboBox_OCTA.setCurrentIndex(0)
                self.ui.comboBox_Seg.setCurrentIndex(0)
                
                # Reset file path inputs
                self.ui.lineEdit_OCTFilename.clear()
                self.ui.lineEdit_OCTAFilename.clear()
                self.ui.lineEdit_SegFilename.clear()
                
                # Reset transfer name inputs
                self.ui.lineEdit_transfer_oct_name.clear()
                self.ui.lineEdit_transfer_octa_name.clear()
                self.ui.lineEdit_transfer_seg_name.clear()
                
                # Reset frame controls
                self.ui.spinBox_frameIdx.setValue(1)
                self.ui.spinBox_frameIdx.setMaximum(1)
                self.ui.lineEdit_totalFrame.clear()
                
                # Reset transform controls
                self.ui.comboBox_flatten.setCurrentIndex(0)
                self.ui.comboBox_permute.setCurrentIndex(0)
                self.ui.comboBox_flip.setCurrentIndex(0)
                
                # Reset ROI controls
                self.ui.spinBox_roi_top.setValue(0)
                self.ui.spinBox_roi_bot.setValue(0)
                
                # Reset OCTA visibility checkbox
                self.ui.checkBox_OCTA.setChecked(True)
                
                # Clear status bar messages
                self.ui.statusbar.clearMessage()
                if self.status_data_shape:
                    self.status_data_shape.clear()
                if self.status_mouse_pos:
                    self.status_mouse_pos.clear()
                if self.status_pixel_value_pos:
                    self.status_pixel_value_pos.clear()
                    
                # reset the data
                self.oct_data = None
                self.octa_data = None
                self.volume_size = (1, 1, 1)
                self.oct_data_raw = None
                self.octa_data_raw = None
                self.oct_data_flatten = None
                self.octa_data_flatten = None
                self.oct_data_flatten_raw = None
                self.octa_data_flatten_raw = None
                self.oct_data_flattened = None
                self.flatten_offset = 0
                self.flatten_baseline = -1
                self.flatten_permute = (0, 1, 2)
                # curve_data_bk = None
                self.curve_data_dict = None
                self.oct_data_range = [0, 1]
                self.octa_data_range = [0, 1]
                self.indicatorDirection = 1
                self.resolution_width = 1
                self.scan_width_mm = 1
                self.resolution_height = 1
                self.scan_height_mm = 1
                self.resolution_depth = 1

                self.current_tab_index = 0
                self.last_opened_dir = ""
                self.seg_data = None
                self.gCurve_data = None
                    
            except Exception as e:
                logger.warning("Error resetting UI elements: %s", e)
            
            # Remove from layouts first
            self.ui.horizontalLayout_15.removeWidget(self.LayerSegmentation)
            self.ui.stackedWidget.removeWidget(self.LayerSegmentation.LayerSegCtrlPanelWidget)
            
            # Properly clean up VTK resources via closeEvent (pass None for programmatic cleanup)
            try:
                self.LayerSegmentation.closeEvent(None)
            except Exception as e:
                logger.warning("Error during LayerSegmentation closeEvent: %s", e)
            
            # Delete the widget properly
            try:
                self.LayerSegmentation.deleteLater()
            except Exception as e:
                logger.warning("Error deleting LayerSegmentation widget: %s", e)
            
            # Set to None before creating new one
            self.LayerSegmentation = None
            
            # Reset internal data attributes
            self.oct_data = None
            self.octa_data = None
            self.seg_data = None
            self.volume_size = [1, 1, 1]
            self.resolution_width = 1
            self.resolution_height = 1
            self.resolution_depth = 1
            self.scan_width_mm = 1
            self.scan_height_mm = 1
            
            # Create new instance
            self.initialize_tabs()
            
            # Show success message
            self.ui.statusbar.showMessage("Layer Segmentation reset successfully", 3000)
            
    def updateResolution(self, resolution, scan_size):
        self.resolution_width = resolution[0]
        self.resolution_height = resolution[1]
        self.resolution_depth = resolution[2]
        self.scan_width_mm = scan_size[0]
        self.scan_height_mm = scan_size[1]

    def closeEvent(self, event:Any):
        """Handle the close event for the main window.
        
        Args:
            event: The close event from Qt.
        """
        if self.LayerSegmentation is not None:
            # Clean up LayerSegmentation without confirmation during app shutdown
            try:
                self.LayerSegmentation.closeEvent(event)
            except Exception as e:
                logger.warning("Error during LayerSegmentation closeEvent: %s", e)
        return super().closeEvent(event)

    def keyReleaseEvent(self, event:Any):
        if not isinstance(self.focusWidget(), pg.GraphicsLayoutWidget):
            return
        if event.key() == Qt.Key.Key_Down:
            idx = min(
                self.ui.spinBox_frameIdx.value() + 1,
                self.ui.spinBox_frameIdx.maximum(),
            )
            self.ui.spinBox_frameIdx.setValue(idx)
        elif event.key() == Qt.Key.Key_Up:
            idx = max(
                self.ui.spinBox_frameIdx.value() - 1,
                self.ui.spinBox_frameIdx.minimum(),
            )
            self.ui.spinBox_frameIdx.setValue(idx)
        if event.key() == Qt.Key.Key_Right:
            # check if the current frame is the last frame
            idx = min(
                self.ui.spinBox_frameIdx.value()+ 10,
                self.ui.spinBox_frameIdx.maximum(),
            )
            self.ui.spinBox_frameIdx.setValue(idx)
        elif event.key() == Qt.Key.Key_Left:
            idx = max(
                self.ui.spinBox_frameIdx.value() - 10,
                self.ui.spinBox_frameIdx.minimum(),
            )
            self.ui.spinBox_frameIdx.setValue(idx)
        super().keyReleaseEvent(event)
    
    def load_droped_file(self, file_path: str, file_type: str = None, group_data: dict = None):
        # find the extension of the file
        self.ui.statusbar.showMessage("Loading Data...")
        self.ui.statusbar.repaint()
        # get permute
        permute = self.ui.comboBox_permute.currentText()
        permute = [int(x) for x in permute.split(",")]
        nframes, nrows, ncols = self.volume_size

        # OCT file
        if file_type == "oct":
            # if file_extension in self.oct_file_extensions:
            if self.data_loader_settings is not None and self.data_loader_settings['oct_loader']['module'] != "default":
                loaded_functions = self.data_loader_settings['loaded_functions']
                module_name = self.data_loader_settings['oct_loader']['module']
                function_name = self.data_loader_settings['oct_loader']['function']
                oct_loader = loaded_functions[module_name][function_name]
                self.oct_data = oct_loader(self.ui.lineEdit_OCTFilename.text())
            else:
                self.oct_data = self.loadOCTFile(file_path)
            if self.oct_data is not None:
                self.last_opened_dir = os.path.dirname(file_path)
                self.oct_data_range = [np.min(self.oct_data), np.max(self.oct_data)]
                # self.oct_data = utils.mat2gray(self.oct_data, self.oct_data_range)
                self.ui.lineEdit_OCTFilename.blockSignals(True)
                self.ui.lineEdit_OCTFilename.setText(file_path)
                self.ui.lineEdit_OCTFilename.blockSignals(False)
                self.oct_data_raw = self.oct_data
                self.oct_data_flatten = None
                self.oct_data_flatten_raw = None

                if self.octa_data is not None:
                    # if the shape is not the same, then set octa_data to None
                    # reorder self.octa_data.shape with permute
                    reorderd_shape = tuple([self.octa_data.shape[i] for i in permute])
                    if reorderd_shape != self.oct_data.shape:
                        self.octa_data = None
                        self.octa_data_raw = None
                        self.octa_data_flatten_raw = None
                if self.curve_data_dict is not None:
                    # if the the volumeSize in curve_data is not the same as the oct_data, then set curve_data to zeros
                    if self.curve_data_dict["volumeSize"] != self.oct_data.shape:
                        self.curve_data_dict = utils.createZeroCurveDict(
                            *self.oct_data.shape
                        )
                        self.gCurve_data = self.curve_data_dict["None_012"]
                else:
                    self.curve_data_dict = utils.createZeroCurveDict(
                        *self.oct_data.shape
                    )
                    self.gCurve_data = self.curve_data_dict["None_012"]

                [nframes, nrows, ncols] = self.oct_data.shape
                self.volume_size = (nframes, nrows, ncols)
                self.ui.lineEdit_totalFrame.setText(str(nframes))
                self.ui.spinBox_frameIdx.setMaximum(nframes)
                self.ui.comboBox_flatten.setCurrentText("None")

                self.ui.spinBox_roi_top.setMaximum(
                    nrows - 1
                )
                self.ui.spinBox_roi_bot.setMaximum(nrows)
                self.updateResolutionDlg()

        # OCTA file
        elif file_type == "octa":
            if self.data_loader_settings is not None and self.data_loader_settings['octa_loader']['module'] != "default":
                loaded_functions = self.data_loader_settings['loaded_functions']
                module_name = self.data_loader_settings['octa_loader']['module']
                function_name = self.data_loader_settings['octa_loader']['function']
                octa_loader = loaded_functions[module_name][function_name]
                self.octa_data = octa_loader(self.ui.lineEdit_OCTAFilename.text())
            else:
                self.octa_data = self.loadOCTAFile(file_path)
            if self.octa_data is not None:
                self.last_opened_dir = os.path.dirname(file_path)
                self.ui.lineEdit_OCTAFilename.blockSignals(True)
                self.ui.lineEdit_OCTAFilename.setText(file_path)
                self.ui.lineEdit_OCTAFilename.blockSignals(False)
                self.octa_data_raw = self.octa_data
                self.octa_data_flatten = None
                self.octa_data_flatten_raw = None

                if self.oct_data is not None:
                    # if the shape is not the same, then set oct_data to None
                    reorderd_shape = tuple([self.oct_data.shape[i] for i in permute])
                    if reorderd_shape != self.octa_data.shape:
                        self.oct_data = None
                        self.oct_data_raw = None
                        self.oct_data_flatten = None
                        self.oct_data_flatten_raw = None

                if self.curve_data_dict is not None:
                    # if the the volumeSize in curve_data is not the same as the oct_data, then set curve_data to zeros
                    if self.curve_data_dict["volumeSize"] != self.oct_data.shape:
                        self.curve_data_dict = utils.createZeroCurveDict(
                            *self.oct_data.shape
                        )
                        self.gCurve_data = self.curve_data_dict["None_012"]
                else:
                    self.curve_data_dict = utils.createZeroCurveDict(
                        *self.oct_data.shape
                    )
                    self.gCurve_data = self.curve_data_dict["None_012"]

                self.octa_data_range = [np.min(self.octa_data), np.max(self.octa_data)]
                [nframes, nrows, ncols] = self.oct_data.shape
                self.volume_size = (nframes, nrows, ncols)
                self.ui.lineEdit_totalFrame.setText(str(nframes))
                self.ui.spinBox_frameIdx.setMaximum(nframes)
                self.ui.spinBox_roi_top.setMaximum(
                    nrows - 1
                )
                self.ui.spinBox_roi_bot.setMaximum(nrows)

        # Segmentation file
        elif file_type == "seg":
            if self.data_loader_settings is not None and self.data_loader_settings['seg_loader']['module'] != "default":
                loaded_functions = self.data_loader_settings['loaded_functions']
                module_name = self.data_loader_settings['seg_loader']['module']
                function_name = self.data_loader_settings['seg_loader']['function']
                seg_loader = loaded_functions[module_name][function_name]
                curve_data = seg_loader(self.ui.lineEdit_SegFilename.text())
            else:
                curve_data = self.loadCurveFile(file_path)
            if curve_data is not None:
                if (
                    curve_data["volumeSize"] is None
                    or len(curve_data["volumeSize"]) < 3
                ):  # if the volumeSize is not in the curve_data, then set it to the default value
                    curve_data["volumeSize"] = self.volume_size
                self.last_opened_dir = os.path.dirname(file_path)
                self.ui.lineEdit_SegFilename.blockSignals(True)
                self.ui.lineEdit_SegFilename.setText(file_path)
                self.ui.lineEdit_SegFilename.blockSignals(False)

                if self.curve_data_dict is not None:
                    if self.curve_data_dict["volumeSize"] != curve_data["volumeSize"]:
                        self.curve_data_dict = utils.createZeroCurveDict(
                            *curve_data["volumeSize"]
                        )
                        self.gCurve_data = self.curve_data_dict["None_012"]
                    else:
                        curve_dict_key = utils.generateCurveDictKey(
                            curve_data["permute"], curve_data["flip"]
                        )
                        self.curve_data_dict[curve_dict_key] = curve_data
                        self.gCurve_data = self.curve_data_dict[curve_dict_key]
                else:
                    self.curve_data_dict = utils.createZeroCurveDict(
                        *curve_data["volumeSize"]
                    )
                    self.gCurve_data = self.curve_data_dict["None_012"]

                [nframes, nrows, ncols] = self.oct_data.shape
                self.ui.lineEdit_totalFrame.setText(str(nframes))
                self.ui.spinBox_frameIdx.setMaximum(nframes)
                self.ui.comboBox_permute.setCurrentText(self.gCurve_data["permute"])
                self.ui.comboBox_flip.setCurrentText(self.gCurve_data["flip"])

                self.ui.spinBox_roi_top.setMaximum(nrows - 1)
                self.ui.spinBox_roi_bot.setMaximum(nrows)
        
        # Group data
        elif file_type == "group":
            # Load group data containing OCT, OCTA, and/or segmentation data
            self.oct_data = group_data.get("oct", None)
            self.octa_data = group_data.get("octa", None)
            curve_data = group_data.get("seg", None)
            
            # Set last opened directory from file path
            self.last_opened_dir = os.path.dirname(file_path)
            
            # Process OCT data if available
            if self.oct_data is not None:
                self.oct_data_range = [np.min(self.oct_data), np.max(self.oct_data)]
                self.oct_data_raw = self.oct_data
                self.oct_data_flatten = None
                self.oct_data_flatten_raw = None
                self.ui.lineEdit_OCTFilename.blockSignals(True)
                self.ui.lineEdit_OCTFilename.setText(file_path)
                self.ui.lineEdit_OCTFilename.blockSignals(False)
                
            # Process OCTA data if available
            if self.octa_data is not None:
                self.octa_data_range = [np.min(self.octa_data), np.max(self.octa_data)]
                self.octa_data_raw = self.octa_data
                self.octa_data_flatten = None
                self.octa_data_flatten_raw = None
                self.ui.lineEdit_OCTAFilename.blockSignals(True)
                self.ui.lineEdit_OCTAFilename.setText(file_path)
                self.ui.lineEdit_OCTAFilename.blockSignals(False)
            
            # Validate shape consistency between OCT and OCTA
            if self.oct_data is not None and self.octa_data is not None:
                reorderd_shape = tuple([self.octa_data.shape[i] for i in permute])
                if reorderd_shape != self.oct_data.shape:
                    # If shapes don't match, keep only the first loaded data
                    self.octa_data = None
                    self.octa_data_raw = None
                    self.octa_data_flatten = None
                    self.octa_data_flatten_raw = None
            
            # Determine volume size from available data
            if self.oct_data is not None:
                [nframes, nrows, ncols] = self.oct_data.shape
            elif self.octa_data is not None:
                [nframes, nrows, ncols] = self.octa_data.shape
            elif curve_data is not None and curve_data.get("volumeSize") is not None:
                [nframes, nrows, ncols] = curve_data["volumeSize"]
            else:
                [nframes, nrows, ncols] = self.volume_size
                
            self.volume_size = (nframes, nrows, ncols)
            
            # Process segmentation/curve data if available
            if curve_data is not None and utils.validateCurve(curve_data):
                if (
                    curve_data.get("volumeSize") is None
                    or len(curve_data.get("volumeSize", [])) < 3
                ):
                    curve_data["volumeSize"] = self.volume_size
                    
                self.ui.lineEdit_SegFilename.blockSignals(True)
                self.ui.lineEdit_SegFilename.setText(file_path)
                self.ui.lineEdit_SegFilename.blockSignals(False)
                
                if self.curve_data_dict is not None:
                    if self.curve_data_dict["volumeSize"] != curve_data["volumeSize"]:
                        self.curve_data_dict = utils.createZeroCurveDict(
                            *curve_data["volumeSize"]
                        )
                        self.gCurve_data = self.curve_data_dict["None_012"]
                    else:
                        curve_dict_key = utils.generateCurveDictKey(
                            curve_data.get("permute", "0,1,2"), curve_data.get("flip", "None")
                        )
                        self.curve_data_dict[curve_dict_key] = curve_data
                        self.gCurve_data = self.curve_data_dict[curve_dict_key]
                else:
                    self.curve_data_dict = utils.createZeroCurveDict(
                        *curve_data["volumeSize"]
                    )
                    self.gCurve_data = self.curve_data_dict["None_012"]
                    
                # Update UI with curve data settings
                self.ui.comboBox_permute.setCurrentText(self.gCurve_data.get("permute", "0,1,2"))
                self.ui.comboBox_flip.setCurrentText(self.gCurve_data.get("flip", "None"))
            else:
                # Create zero curve dict if no valid curve data
                self.curve_data_dict = utils.createZeroCurveDict(*self.volume_size)
                self.gCurve_data = self.curve_data_dict["None_012"]
            
            # Update UI elements with volume size
            self.ui.lineEdit_totalFrame.setText(str(nframes))
            self.ui.spinBox_frameIdx.setMaximum(nframes)
            self.ui.spinBox_roi_top.setMaximum(nrows - 1)
            self.ui.spinBox_roi_bot.setMaximum(nrows)
            self.ui.comboBox_flatten.setCurrentText("None")
            
            # Update resolution dialog
            if self.oct_data is not None or self.octa_data is not None:
                self.updateResolutionDlg()
                
        else:
            self.ui.statusbar.showMessage("Invalid file type", 5000)
            return

        if (
            self.oct_data is None
            and self.octa_data is None
            and self.gCurve_data is None
        ):
            self.ui.statusbar.showMessage("No data was loaded", 5000)
            return

        self.flatten_offset = 0
        self.flatten_baseline = -1
        self.ui.comboBox_flatten.setCurrentText("None")
        self.onTransform_changed()
        # update Statusbar
        self.ui.statusbar.showMessage("Data Loaded", 5000)
        self.status_data_shape.setText(f"Data Shape: {nframes}x{nrows}x{ncols}")

    def _update_datasource_dropdown(self):
        # ignore the event of comboBox_OCT, comboBox_OCTA, comboBox_Seg
        self.ui.comboBox_OCT.blockSignals(True)
        self.ui.comboBox_OCTA.blockSignals(True)
        self.ui.comboBox_Seg.blockSignals(True)
        # get keys in workspace
        self.ui.comboBox_OCT.clear()
        self.ui.comboBox_OCT.addItem("Select OCT Data")
        self.ui.comboBox_OCTA.clear()
        self.ui.comboBox_OCTA.addItem("Select OCTA Data")
        self.ui.comboBox_Seg.clear()
        self.ui.comboBox_Seg.addItem("Select Segmentation Data")
        
        # Get workspace data from workspace manager through app_context
        workspace_data = {}
        if self.app_context:
            workspace_manager = self.app_context.get_component("workspace_manager")
            if workspace_manager:
                workspace_data = workspace_manager.get_workspace_data()
        elif self.parentWindow:
            workspace_data = self.parentWindow.get_workspace_data()
        if self.parentWindow is not None:
            for key in workspace_data.keys():
                fieldvalue = workspace_data[key]
                if isinstance(fieldvalue, np.ndarray) and len(fieldvalue.shape) == 3:
                    self.ui.comboBox_OCT.addItem(key)
                    self.ui.comboBox_OCTA.addItem(key)
                elif utils.validateCurve(fieldvalue):
                    self.ui.comboBox_Seg.addItem(key)
                # Check if it's a dict with 'curve' wrapper (from .med files)
                elif isinstance(fieldvalue, dict) and 'curve' in fieldvalue:
                    if isinstance(fieldvalue['curve'], dict) and utils.validateCurve(fieldvalue['curve']):
                        self.ui.comboBox_Seg.addItem(key)

        self.ui.comboBox_OCT.blockSignals(False)
        self.ui.comboBox_OCTA.blockSignals(False)
        self.ui.comboBox_Seg.blockSignals(False)

    def _create_progress_dialog(
        self,
        title: str,
        text: str,
        max_value: int = 100,
        cancelable: bool = False,
        cancel_callback: Optional[Callable] = None,
    ):
        msg = QProgressDialog(text, None, 0, max_value, self)
        msg.setWindowTitle(title)
        msg.setWindowModality(Qt.WindowModality.WindowModal)
        if cancelable:
            # Connect to provided callback if callable, otherwise close dialog on cancel
            if callable(cancel_callback):
                msg.canceled.connect(cancel_callback)
            else:
                try:
                    msg.canceled.connect(msg.close)
                except Exception:
                    pass
        msg.show()
        QCoreApplication.processEvents()
        msg.setValue(0)

        return msg

    def _update_progress_dialog(
        self, msg: QProgressDialog, value: int, text: str = None
    ):
        msg.setValue(value)
        if text is not None:
            msg.setLabelText(text)
        QCoreApplication.processEvents()
        if value == 100:
            msg.close()

    @Slot()
    def onROIChanged(self):
        if self.oct_data is None:
            return
        self.onTabChanged(autoRange=True, roi_update=True)

    # @Slot()
    # def onStatusBarDoubleClick(self, event:Any):
    #     self.parentWindow.mdiFocusMode()

    @Slot()
    def updateResolutionDlg(self):
        # check of oct_data is not None
        if self.oct_data is None:
            self.ui.statusbar.showMessage("No data is loaded", 5000)
            return
        dlg = CalculateResolution(parentWindow=self, parent=self)
        # dlg.setStyleSheet(self.parentWindow.styleSheet())
        dlg.exec_()

    @Slot()
    def on_changedOCTAVisibility(self):
        self.onTabChanged(autoRange=True)

    def on_OCTFile_dropEvent(self, event):
        path = event.mimeData().text()
        path = re.sub(r"[\n\r]", "", path)
        self.ui.lineEdit_OCTFilename.setStyleSheet("")  # Reset style
        if path.startswith("file://"):
            path = QUrl(path).toLocalFile()
        self.ui.lineEdit_OCTFilename.setText(path)

    def on_OCTFile_dragEnterEvent(self, event:Any):
        if event.mimeData().hasText():
            self.ui.lineEdit_OCTFilename.setStyleSheet(
                "background-color: rgba(150,200,150,200);"
            )
            event.acceptProposedAction()

    def on_OCTFile_dragLeaveEvent(self, event:Any):
        self.ui.lineEdit_OCTFilename.setStyleSheet("")  # Reset style

    def on_OCTAFile_dropEvent(self, event:Any):
        path = event.mimeData().text()
        path = re.sub(r"[\n\r]", "", path)
        self.ui.lineEdit_OCTAFilename.setStyleSheet("")
        if path.startswith("file://"):
            path = QUrl(path).toLocalFile()
        self.ui.lineEdit_OCTAFilename.setText(path)

    def on_OCTAFile_dragEnterEvent(self, event:Any):
        if event.mimeData().hasText():
            self.ui.lineEdit_OCTAFilename.setStyleSheet(
                "background-color: rgba(150,200,150,200);"
            )
            event.acceptProposedAction()

    def on_OCTAFile_dragLeaveEvent(self, event:Any):
        self.ui.lineEdit_OCTAFilename.setStyleSheet("")

    def on_SegFile_dropEvent(self, event:Any):
        path = event.mimeData().text()
        path = re.sub(r"[\n\r]", "", path)
        self.ui.lineEdit_SegFilename.setStyleSheet("")
        if path.startswith("file://"):
            path = QUrl(path).toLocalFile()
        self.ui.lineEdit_SegFilename.setText(path)

    def on_SegFile_dragEnterEvent(self, event:Any):
        if event.mimeData().hasText():
            self.ui.lineEdit_SegFilename.setStyleSheet(
                "background-color: rgba(150,200,150,200);"
            )
            event.acceptProposedAction()

    def on_SegFile_dragLeaveEvent(self, event:Any):
        self.ui.lineEdit_SegFilename.setStyleSheet("")

    def on_loadData_dragEnterEvent(self, event:Any):
        if event.mimeData().hasText():
            self.ui.pushButton_loadData.setText("Drop Here to Load Data")
            event.acceptProposedAction()

    def on_loadData_dragLeaveEvent(self, event:Any):
        self.ui.pushButton_loadData.setText("Load Data")

    def on_loadData_dropEvent(self, event:Any):
        path = event.mimeData().text()
        # remove all \n \r characters from the path and convert to lower case
        path = re.sub(r"[\n\r]", "", path).lower()
        # convert path to localpath
        if path.startswith("file://"):
            path = QUrl(path).toLocalFile()
        # print(path)
        self.ui.pushButton_loadData.setText("Load Data")

        if path.endswith(tuple(self.oct_file_extensions)):
            self.load_droped_file(path, "oct")
        elif path.endswith(tuple(self.octa_file_extensions)):
            self.load_droped_file(path, "octa")
        elif path.endswith(tuple(self.seg_file_extensions)):
            self.load_droped_file(path, "seg")
        elif path.endswith('.mat') or path.endswith('.med'):
            # open the loadgroupdialog to ask the user to select the data
            dlg = LoadGroupDataClass(parentWindow=self, filePath=path)
            dlg.exec_()
            self.load_droped_file(path, "group", group_data={
                "oct": dlg.selected_oct_data,
                "octa": dlg.selected_octa_data,
                "seg": dlg.selected_seg_data
            })
        else:
            # show a messagebox asking the user to select the data type
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Load file as:")
            msg.setWindowTitle("Select Data Type")
            # add three buttons, oct, octa, seg
            oct_button = msg.addButton("OCT", QMessageBox.ButtonRole.ActionRole)
            octa_button = msg.addButton("OCTA", QMessageBox.ButtonRole.ActionRole)
            seg_button = msg.addButton(
                "Segmentation", QMessageBox.ButtonRole.ActionRole
            )
            # get the button clicked
            oct_button.clicked.connect(lambda checked=False: self.load_droped_file(path, "oct"))
            octa_button.clicked.connect(lambda checked=False: self.load_droped_file(path, "octa"))
            seg_button.clicked.connect(lambda checked=False: self.load_droped_file(path, "seg"))
            msg.exec_()

    def onTransform_changed(self):
        # Return if both oct_data and octa_data are None
        if self.oct_data is None and self.octa_data is None:
            return

        if self.ui.comboBox_flatten.currentText() != "None":
            if (
                self.flatten_baseline == -1
                or self.current_flatten_method != self.ui.comboBox_flatten.currentText()
            ):
                msg = self._create_progress_dialog("Flatten Data", "Flattening Data...")
                self.current_flatten_method = self.ui.comboBox_flatten.currentText()
                (
                    self.oct_data_flatten,
                    self.octa_data_flatten,
                    self.flatten_offset,
                    self.flatten_baseline,
                    self.flatten_permute,
                ) = utils.flatten_oct_octa(
                    self.oct_data_raw,
                    self.octa_data_raw,
                    permute="auto",
                    method=self.ui.comboBox_flatten.currentText(),
                    reference_curve=self.curve_data_dict["None_012"]["curves"]["RPEBM"],
                )
                self.oct_data_flatten_raw = self.oct_data_flatten
                self.octa_data_flatten_raw = self.octa_data_flatten
                self._update_progress_dialog(msg, 90)
                # updated the curve_data_dict to add the flatten data for each curve
                self.curve_data_dict = utils.add_flatten_parameters_in_curve_dict(
                    self.curve_data_dict,
                    self.flatten_offset,
                    self.flatten_baseline,
                    self.flatten_permute,
                )
                self._update_progress_dialog(msg, 100)

        # Permute the data
        permute = [int(x) for x in self.ui.comboBox_permute.currentText().split(",")]
        if tuple(permute) != (0, 1, 2):
            oct_data = (
                np.transpose(self.oct_data_raw, permute)
                if self.oct_data_raw is not None
                else None
            )
            octa_data = (
                np.transpose(self.octa_data_raw, permute)
                if self.octa_data_raw is not None
                else None
            )
            oct_data_flatten = (
                np.transpose(self.oct_data_flatten_raw, permute)
                if self.oct_data_flatten_raw is not None
                else None
            )
            octa_data_flatten = (
                np.transpose(self.octa_data_flatten_raw, permute)
                if self.octa_data_flatten_raw is not None
                else None
            )
        else:
            oct_data = self.oct_data_raw
            octa_data = self.octa_data_raw
            oct_data_flatten = self.oct_data_flatten_raw
            octa_data_flatten = self.octa_data_flatten_raw

        # Flip the data
        flipAxis = self.ui.comboBox_flip.currentText()
        if flipAxis == "Left-Right":
            if oct_data is not None:
                oct_data = np.flip(oct_data, axis=2)
            if octa_data is not None:
                octa_data = np.flip(octa_data, axis=2)
            if oct_data_flatten is not None:
                oct_data_flatten = np.flip(oct_data_flatten, axis=2)
            if octa_data_flatten is not None:
                octa_data_flatten = np.flip(octa_data_flatten, axis=2)

        elif flipAxis == "Up-Down":
            if oct_data is not None:
                oct_data = np.flip(oct_data, axis=1)
            if octa_data is not None:
                octa_data = np.flip(octa_data, axis=1)
            if oct_data_flatten is not None:
                oct_data_flatten = np.flip(oct_data_flatten, axis=1)
            if octa_data_flatten is not None:
                octa_data_flatten = np.flip(octa_data_flatten, axis=1)

        # get curve data from the curve_data_dict
        curve_key = utils.generateCurveDictKey(
            self.ui.comboBox_permute.currentText(), flipAxis
        )
        self.gCurve_data = self.curve_data_dict[curve_key]

        # Set the updated data
        self.oct_data = oct_data
        self.octa_data = octa_data
        self.oct_data_flatten = oct_data_flatten
        self.octa_data_flatten = octa_data_flatten

        # Update the ROI in layer Segmentation
        if oct_data is not None:
            self.ui.lineEdit_totalFrame.setText(str(oct_data.shape[0]))
            self.ui.spinBox_frameIdx.setMaximum(oct_data.shape[0])
            if (
                self.ui.spinBox_roi_top.value()
                > oct_data.shape[1] - 1
            ):
                self.ui.spinBox_roi_top.setValue(
                    oct_data.shape[1] - 1
                )
            if (
                self.ui.spinBox_roi_bot.value()
                > oct_data.shape[1]
            ):
                self.ui.spinBox_roi_bot.setValue(
                    oct_data.shape[1]
                )
            self.ui.spinBox_roi_top.setMaximum(
                oct_data.shape[1] - 1
            )
            self.ui.spinBox_roi_bot.setMaximum(
                oct_data.shape[1]
            )
            self.onAutoROI()
        # Update the GUI
        self.onTabChanged(autoRange=True)

    @Slot()
    def onAutoROI(self):
        if self.oct_data is None and self.octa_data is None:
            self.ui.statusbar.showMessage("No data is loaded", 5000)
            return
        self.ui.statusbar.showMessage("Computing ROI...", 5000)
        self.ui.statusbar.repaint()
        if self.oct_data is not None:
            [top, bot] = utils.calculate_roi(self.oct_data)
        else:
            [top, bot] = utils.calculate_roi(self.octa_data)
        self.ui.spinBox_roi_top.blockSignals(True)
        self.ui.spinBox_roi_bot.blockSignals(True)
        self.ui.spinBox_roi_top.setValue(top)
        self.ui.spinBox_roi_bot.setValue(bot)
        self.ui.spinBox_roi_top.blockSignals(False)
        self.ui.spinBox_roi_bot.blockSignals(False)
        self.onROIChanged()
        self.ui.statusbar.showMessage("ROI updated", 5000)

    # @Slot()
    # def onGPU_changed(self):
    #     # set the GPU index
    #     self.ort_session = None

    @Slot()
    def onTabChanged(self, autoRange: bool=False, roi_update: bool=False):
        tab_keys = list(self.tab_initializers.keys())
        # Since we only have LayerSegmentation now, always set it as current
        self.ui.stackedWidget.setCurrentIndex(tab_keys.index('LayerSegmentation'))
        self.ui.groupBox_4.setVisible(True)
        self.ui.groupBox_frame.setVisible(True)
        self.ui.groupBox_5.setVisible(True)
        if self.LayerSegmentation is not None:
            self.LayerSegmentation.onTabChanged(autoRange=autoRange,roi_update=roi_update)

    @Slot()
    def onFrameIdx_changed(self):
        self.LayerSegmentation._refreshGUI(
                updateItems=UpdateUIItems(
                    bframeA_oct=True,
                    bframeA_curve=True,
                    bframeA_octa=True,
                    bframeB_oct=True,
                    bframeB_curve=True,
                    enfaceA_img=False,
                    enfaceA_indicators=True,
                    enfaceB_img=False,
                    enfaceB_indicators=True,
                )
            )

    @Slot()
    def on_octNameChanged(self):
        # check if the name is valid for a python variable name
        new_name = utils.validate_variable_name(
            self.ui.lineEdit_transfer_oct_name.text(), "oct_data"
        )
        self.ui.lineEdit_transfer_oct_name.setText(new_name)

    @Slot()
    def on_octaNameChanged(self):
        # check if the name is valid for a python variable name
        new_name = utils.validate_variable_name(
            self.ui.lineEdit_transfer_octa_name.text(), "octa_data"
        )
        self.ui.lineEdit_transfer_octa_name.setText(new_name)

    @Slot()
    def on_segNameChanged(self):
        # check if the name is valid for a python variable name
        new_name = utils.validate_variable_name(
            self.ui.lineEdit_transfer_seg_name.text(), "seg_data"
        )
        self.ui.lineEdit_transfer_seg_name.setText(new_name)

    @Slot()
    def on_transferData(self, data_type: str):
        # Get workspace data from workspace manager through app_context
        workspace_data = {}
        workspace_manager = None
        if self.app_context:
            workspace_manager = self.app_context.get_component("workspace_manager")
            if workspace_manager:
                workspace_data = workspace_manager.get_workspace_data()
        elif self.parentWindow:
            workspace_data = self.parentWindow.get_workspace_data()
            
        if data_type == "OCT":
            if self.oct_data_raw is None:
                self.ui.statusbar.showMessage("No OCT data to transfer", 5000)
                return
            workspace_data[self.ui.lineEdit_transfer_oct_name.text()] = (
                self.oct_data_raw.copy()
            )
        elif data_type == "OCTA":
            if self.octa_data_raw is None:
                self.ui.statusbar.showMessage("No OCTA data to transfer", 5000)
                return
            workspace_data[self.ui.lineEdit_transfer_octa_name.text()] = (
                self.octa_data_raw.copy()
            )
        elif data_type == "Seg":
            if self.gCurve_data is None:
                self.ui.statusbar.showMessage("No segmentation data to transfer", 5000)
                return
            workspace_data[self.ui.lineEdit_transfer_seg_name.text()] = (
                self.gCurve_data.copy()
            )
        
        # Update workspace data
        if workspace_manager:
            workspace_manager.update_workspace_data(workspace_data,source="LayerSegmentation")
        elif self.parentWindow:
            self.parentWindow.update_workspace_data(workspace_data,source="LayerSegmentation")
        self.ui.pushButton_refreshDatalist.click()

    @Slot()
    def on_saveData(self):
        # open a file dialog to save the data
        # default file name is the current OCT filename with _seg appended
        if self.oct_data is None and self.octa_data is None and self.gCurve_data is None:
            self.ui.statusbar.showMessage("No data to save", 5000)
            return

        if self.ui.lineEdit_OCTFilename.text():
            default_filename = os.path.splitext(
                os.path.basename(self.ui.lineEdit_OCTFilename.text())
            )[0] + "_seg"
        elif self.ui.lineEdit_OCTAFilename.text():
            default_filename = os.path.splitext(
                os.path.basename(self.ui.lineEdit_OCTAFilename.text())
            )[0] + "_seg"
        else:
            default_filename = "segmentation_data"
        file_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            os.path.join(self.last_opened_dir, default_filename),
            "MED Files (*.med)"
            
        )
        # get the file path with extension
        if not file_path:
            return
        if not file_path.endswith('.med'):
            # if the file path does not have an extension, check the filter
            if _filter.startswith("MED Files"):
                file_path += ".med"
            else:
                self.ui.statusbar.showMessage("Invalid file type", 5000)
                return
            
        # check if the file path is valid
        if not os.path.isdir(os.path.dirname(file_path)):
            self.ui.statusbar.showMessage("Invalid file path", 5000)
            return

        # save the data based on the file extension
        if file_path.endswith(".med") :
            fio.write_curve_med_file(file_path, self.gCurve_data, compression='gzip')
        else:
            self.ui.statusbar.showMessage("Invalid file type", 5000)
            return
    @Slot()
    def onPermute_changed(self, state: bool):
        self.LayerSegmentation._refreshGUI(updateItems=UpdateUIItems())

    @Slot()
    def onFlip_changed(self, state: bool):
        self.LayerSegmentation._refreshGUI(updateItems=UpdateUIItems())

    @Slot()
    def loadData(self):
        # check if the files are valid
        if not (
            os.path.isfile(self.ui.lineEdit_OCTFilename.text())
            or os.path.isfile(self.ui.lineEdit_OCTAFilename.text())
            or os.path.isfile(self.ui.lineEdit_SegFilename.text())
        ):
            self.ui.statusbar.showMessage("Invalid file path", 5000)
            return

        self.ui.statusbar.showMessage("Loading Data...")
        self.ui.statusbar.repaint()
        # show a progress bar
        msg = self._create_progress_dialog("Loading Data", "Loading Data...", 100)
        if self.data_loader_settings is not None and self.data_loader_settings['oct_loader']['module'] != "default":
            loaded_functions = self.data_loader_settings['loaded_functions']
            module_name = self.data_loader_settings['oct_loader']['module']
            function_name = self.data_loader_settings['oct_loader']['function']
            oct_loader = loaded_functions[module_name][function_name]
            self.oct_data = oct_loader(self.ui.lineEdit_OCTFilename.text())
        else:
            self.oct_data = self.loadOCTFile(self.ui.lineEdit_OCTFilename.text())
        self._update_progress_dialog(msg, 30)
        if self.data_loader_settings is not None and self.data_loader_settings['octa_loader']['module'] != "default":
            loaded_functions = self.data_loader_settings['loaded_functions']
            module_name = self.data_loader_settings['octa_loader']['module']
            function_name = self.data_loader_settings['octa_loader']['function']
            octa_loader = loaded_functions[module_name][function_name]
            self.octa_data = octa_loader(self.ui.lineEdit_OCTAFilename.text())
        else:
            self.octa_data = self.loadOCTAFile(self.ui.lineEdit_OCTAFilename.text())
        self._update_progress_dialog(msg, 60)
        if self.data_loader_settings is not None and self.data_loader_settings['seg_loader']['module'] != "default":
            loaded_functions = self.data_loader_settings['loaded_functions']
            module_name = self.data_loader_settings['seg_loader']['module']
            function_name = self.data_loader_settings['seg_loader']['function']
            seg_loader = loaded_functions[module_name][function_name]
            curve_data = seg_loader(self.ui.lineEdit_SegFilename.text())
        else:
            curve_data = self.loadCurveFile(self.ui.lineEdit_SegFilename.text())
        self._update_progress_dialog(msg, 70)
        if self.oct_data is None and self.octa_data is None and curve_data is None:
            self.ui.statusbar.showMessage("No data was loaded", 5000)
            self._update_progress_dialog(msg, 100)
            return

        [nframes, nrows, ncols] = 1, 1, 1
        if self.oct_data is not None:
            # t = time.time()
            self.oct_data_range = [np.min(self.oct_data), np.max(self.oct_data)]
            # print("Time to load OCT data:", time.time() - t)
            # t = time.time()
            # self.oct_data = utils.mat2gray(self.oct_data, self.oct_data_range)
            # print("Time to normalize OCT data:", time.time() - t)
            self.oct_data_raw = self.oct_data
            [nframes, nrows, ncols] = self.oct_data.shape
            self.volume_size = (nframes, nrows, ncols)

            self._update_progress_dialog(msg, 80)
            self.last_opened_dir = os.path.dirname(self.ui.lineEdit_OCTFilename.text())
        if self.octa_data is not None:
            # t = time.time()
            self.octa_data_range = [np.min(self.octa_data), np.max(self.octa_data)]
            # print("Time to load OCTA data:", time.time() - t)
            # t = time.time()
            # self.octa_data = utils.mat2gray(self.octa_data, self.octa_data_range)
            # print("Time to normalize OCTA data:", time.time() - t)
            self.octa_data_raw = self.octa_data
            [nframes, nrows, ncols] = self.octa_data.shape
            self.volume_size = (nframes, nrows, ncols)
            self._update_progress_dialog(msg, 90)
            self.last_opened_dir = os.path.dirname(self.ui.lineEdit_OCTAFilename.text())
        if curve_data is None or not utils.validateCurve(curve_data):
            self.curve_data_dict = utils.createZeroCurveDict(nframes, nrows, ncols)
            self.gCurve_data = self.curve_data_dict["None_012"]
            self.flatten_offset = 0
            self.flatten_baseline = -1
            # self.gNPA_labelImage = None
            # self.gRF_labelVolume = None
            # self.gDrusen_labelVolume = None
            # self.gVD_image = None
            # self.gGA_labelVolume = None
        else:
            if curve_data["volumeSize"] is None or len(curve_data["volumeSize"]) < 3:
                # in case the curve data is not saved with the volume size
                curve_data["volumeSize"] = self.volume_size
            self.curve_data_dict = utils.createZeroCurveDict(nframes, nrows, ncols)
            dict_key = utils.generateCurveDictKey(
                curve_data["permute"], curve_data["flip"]
            )
            self.curve_data_dict[dict_key] = curve_data
            self.gCurve_data = self.curve_data_dict[dict_key]
            self.flatten_offset = 0
            self.flatten_baseline = -1
            self.last_opened_dir = os.path.dirname(self.ui.lineEdit_SegFilename.text())
            # if 'GA' in curve_data.keys():
            #     self.gGA_labelVolume = curve_data['GA']
            # if 'VD' in curve_data.keys():
            #     self.gVD_image = curve_data['VD']
            # if 'NPA' in curve_data.keys():
            #     self.gNPA_labelImage = curve_data['NPA']
            # if 'RF' in curve_data.keys():
            #     self.gRF_labelVolume = curve_data['RF']
            # if 'Drusen' in curve_data.keys():
            #     self.gDrusen_labelVolume = curve_data['Drusen']

        # initialize the interp_key_curve_ranges
        # self.interp_key_curve_ranges = {
        #     key: np.zeros_like(self.gCurve_data["curves"][key], dtype="uint8")
        #     for key in self.gCurve_data["curves"].keys()
        # }

        self.flatten_offset = 0
        self.flatten_baseline = -1
        # update the data source dropdown
        self.ui.lineEdit_totalFrame.setText(str(nframes))
        # self.ui.spinBox_roi_top.setMaximum(nrows - 1) # TODO: set ROI
        # self.ui.spinBox_roi_bot.setMaximum(nrows)
        self.ui.spinBox_frameIdx.setMaximum(nframes)
        self.ui.comboBox_permute.setCurrentText(self.gCurve_data["permute"])
        self.ui.comboBox_flip.setCurrentText(self.gCurve_data["flip"])
        self.ui.comboBox_flatten.setCurrentText("None")
        self.onTransform_changed()
        # update GUI
        # self.onAutoROI() TODO: set ROI

        # check current tab and update the GUI
        # self.onTabChanged(autoRange=True)
        self._update_progress_dialog(msg, 100)
        self.ui.statusbar.showMessage("Data Loaded", 5000)
        self.status_data_shape.setText(f"Data Shape: {nframes}x{nrows}x{ncols}")
        self.updateResolutionDlg()

    @Slot()
    def on_openFileDialog(self, fileType: str):
        if fileType == "OCT":
            fileName = QFileDialog.getOpenFileName(
                self, "Open OCT File", self.last_opened_dir, self.oct_file_filters
            )
            if fileName:
                self.ui.lineEdit_OCTFilename.setText(fileName[0])
                # get the folder of the file
                # self.last_opened_dir = os.path.dirname(fileName[0])
        elif fileType == "OCTA":
            fileName = QFileDialog.getOpenFileName(
                self, "Open OCTA File", self.last_opened_dir, self.octa_file_filters
            )
            if fileName:
                self.ui.lineEdit_OCTAFilename.setText(fileName[0])
                # get the folder of the file
                # self.last_opened_dir = os.path.dirname(fileName[0])
        elif fileType == "Seg":
            fileName = QFileDialog.getOpenFileName(
                self,
                "Open Segmentation File",
                self.last_opened_dir,
                self.seg_file_filters,
            )
            if fileName:
                self.ui.lineEdit_SegFilename.setText(fileName[0])
                # get the folder of the file
                # self.last_opened_dir = os.path.dirname(fileName[0])

    @Slot()
    def _on_OCTData_changed(self):
        if self.ui.comboBox_OCT.currentIndex() == 0:
            self.oct_data = None
        else:
            # Get workspace data from workspace manager through app_context
            workspace_data = {}
            if self.app_context:
                workspace_manager = self.app_context.get_component("workspace_manager")
                if workspace_manager:
                    workspace_data = workspace_manager.get_workspace_data()
            elif self.parentWindow:
                workspace_data = self.parentWindow.get_workspace_data()
                
            self.oct_data = workspace_data[
                self.ui.comboBox_OCT.currentText()
            ]
            self.oct_data_range = [np.min(self.oct_data), np.max(self.oct_data)]
            # self.oct_data = (utils.mat2gray(self.oct_data, self.oct_data_range)*255).astype(np.uint8)
            self.oct_data_raw = self.oct_data
            # update the data source dropdown
            [nframes,nrows,ncols] = self.oct_data.shape
            self.ui.lineEdit_totalFrame.setText(str(nframes))
            self.ui.spinBox_frameIdx.setMaximum(nframes)
            if self.curve_data_dict is None: 
                self.curve_data_dict = utils.createZeroCurveDict(nframes, nrows, ncols)
                
        self.onTransform_changed()
        # check current tab and update the GUI
        # self.onTabChanged(autoRange=True)
        # self._refreshGUI(updateItems=UpdateUIItems(bframeA_oct=True, bframeB_oct=True))

    @Slot()
    def _on_OCTAData_changed(self):
        if self.ui.comboBox_OCTA.currentIndex() == 0:
            self.octa_data = None
        else:
            # Get workspace data from workspace manager through app_context
            workspace_data = {}
            if self.app_context:
                workspace_manager = self.app_context.get_component("workspace_manager")
                if workspace_manager:
                    workspace_data = workspace_manager.get_workspace_data()
            elif self.parentWindow:
                workspace_data = self.parentWindow.get_workspace_data()
                
            self.octa_data = workspace_data[
                self.ui.comboBox_OCTA.currentText()
            ]
            self.octa_data_range = [np.min(self.octa_data), np.max(self.octa_data)]
            # self.octa_data = (utils.mat2gray(self.octa_data, self.octa_data_range)* 255).astype(np.uint8)
            self.octa_data_raw = self.octa_data
            nframes = self.octa_data.shape[0]
            self.ui.lineEdit_totalFrame.setText(str(nframes))
            self.ui.spinBox_frameIdx.setMaximum(nframes)
            if self.curve_data_dict is None: 
                self.curve_data_dict = utils.createZeroCurveDict(
                    nframes, self.octa_data.shape[1], self.octa_data.shape[2]
                )
        self.onTransform_changed()

    @Slot()
    def _on_SegData_changed(self):
        if self.ui.comboBox_Seg.currentIndex() == 0:
            if self.oct_data is not None:
                self.gCurve_data = utils.createZeroCurve(*self.oct_data.shape)
            elif self.octa_data is not None:
                self.gCurve_data = utils.createZeroCurve(*self.octa_data.shape)
            else:
                self.gCurve_data = utils.createZeroCurve(1, 1, 1)
        else:
            # Get workspace data from workspace manager through app_context
            workspace_data = {}
            if self.app_context:
                workspace_manager = self.app_context.get_component("workspace_manager")
                if workspace_manager:
                    workspace_data = workspace_manager.get_workspace_data()
            elif self.parentWindow:
                workspace_data = self.parentWindow.get_workspace_data()
                
            self.gCurve_data = workspace_data[
                self.ui.comboBox_Seg.currentText()
            ]
            
            # Check if data has a 'curve' key (from .med files with root keys)
            # Extract the actual curve data from the wrapper
            if isinstance(self.gCurve_data, dict) and 'curve' in self.gCurve_data and isinstance(self.gCurve_data['curve'], dict):
                self.gCurve_data = self.gCurve_data['curve']
                logger.debug(f"Extracted curve data from 'curve' wrapper in _on_SegData_changed")
            
            nframes = self.gCurve_data["volumeSize"][0]
            self.ui.lineEdit_totalFrame.setText(str(nframes))
            self.ui.spinBox_frameIdx.setMaximum(nframes)
            self.ui.comboBox_permute.setCurrentText(self.gCurve_data["permute"])
            self.ui.comboBox_flip.setCurrentText(self.gCurve_data["flip"])
            if self.curve_data_dict is None:
                self.curve_data_dict = utils.createZeroCurveDict(
                    nframes, self.gCurve_data["volumeSize"][1], self.gCurve_data["volumeSize"][2]
                )
                self.curve_data_dict[utils.generateCurveDictKey(self.gCurve_data["permute"], self.gCurve_data["flip"])] = self.gCurve_data
        self.onTransform_changed()
        # check current tab and update the GUI
    
    @Slot()
    def _on_Load_Settings(self):
        # open LoadSettings dialog
        load_settings_dlg = LoadDataSettings(self, self.data_loader_settings)
        if load_settings_dlg.exec_() == QDialog.DialogCode.Accepted:
            self.data_loader_settings = load_settings_dlg.get_loader_settings()

    @Slot()
    def _on_SegData_Edited(self, idx: int, curve:Any):
        if self.ui.comboBox_flatten.currentText() != "None":
            if isinstance(self.gCurve_data["flatten_offset"], (int, float)):
                offsets = self.gCurve_data["flatten_offset"]
            else:
                offsets = self.gCurve_data["flatten_offset"][
                    self.ui.spinBox_frameIdx.value() - 1
                ]
        else:
            offsets = 0
        keys = list(self.gCurve_data["curves"].keys())
        self.LayerSegmentation.ui.comboBox_enfaceA_Slab.setCurrentText(
            keys[idx]
        )  # optimize this
        self.gCurve_data["curves"][keys[idx]][self.ui.spinBox_frameIdx.value() - 1] = (
            curve - offsets
        )
        self.LayerSegmentation._refreshGUI(
            updateItems=UpdateUIItems(
                bframeA_curve=True,
                enfaceA_img=True,
                enfaceA_interp=True,
                enfaceB_img=True,
            )
        )

    @Slot()
    def _on_OCTFile_changed(self):
        # load file
        # remove " in self.ui.lineEdit_OCTFilename.text()
        oct_fn = self.ui.lineEdit_OCTFilename.text().replace('"','')
        self.ui.lineEdit_OCTFilename.blockSignals(True)
        self.ui.lineEdit_OCTFilename.setText(oct_fn)
        self.ui.lineEdit_OCTFilename.blockSignals(False)
        
        if not os.path.isfile(oct_fn):
            self.ui.statusbar.showMessage("File not found", 5000)
            return
        # oct extension
        # find the longest match of the suffix in the data_extension_pairs.keys() from self.ui.lineEdit_OCTFilename.text()
        oct_ext = max( [
                key
                for key in self.data_extension_pairs.keys()
                if key in self.ui.lineEdit_OCTFilename.text()
            ],
            key=len,
            default="",
        )

        # update the OCTA file name if OCTA is checked
        # if self.ui.checkBox_OCTA.isChecked():
        # update the OCTA file name. use RegEx to replace the extension, if it is in the data_extension_pairs
        for key in self.data_extension_pairs:
            # check if the file name has the key and located at the end of the file name
            if re.search(key + "$", self.ui.lineEdit_OCTFilename.text()):
                octa_file = re.sub(
                    key + "$",
                    self.data_extension_pairs[key],
                    self.ui.lineEdit_OCTFilename.text(),
                )
                if os.path.isfile(octa_file):
                    oct_ext = key
                    self.ui.lineEdit_OCTAFilename.setText(octa_file)
                    break

        # replace the oct_ext with the items in seg_file_extensions, and check if the file exists, if it does, update the seg file name
        candidate_seg_files = [re.sub(oct_ext + "$", ext, self.ui.lineEdit_OCTFilename.text()) for ext in self.seg_file_extensions]
        # check if the candidate seg files exist
        candidate_seg_files = [seg_file for seg_file in candidate_seg_files if os.path.isfile(seg_file)]
        # sort the candidate seg files by modification time
        candidate_seg_files.sort(key=os.path.getmtime, reverse=True)
        # if there are any candidate seg files, set the first one as the seg file name
        if candidate_seg_files:
            self.ui.lineEdit_SegFilename.setText(candidate_seg_files[0])
        else:
            self.ui.lineEdit_SegFilename.setText("")

    def _create_surface_from_image(self, img: np.ndarray)-> Any:
        """
        Create a VTK surface from a 2D image.

        Parameters:
        img (numpy.ndarray): 2D array representing the image.

        Returns:
        vtk.vtkPolyData: VTK PolyData object representing the surface.
        """
        if not VTK_AVAILABLE:
            logger.warning("VTK is not available. Cannot create surface from image.")
            return None
            
        dims = img.shape

        # Step 1: Initialize VTK objects
        points = vtkPoints()
        scalars = vtkFloatArray()
        polyData = vtkPolyData()
        cells = vtkCellArray()

        # Step 2: Convert image data to float32
        image_data_array = img.astype(np.float32)

        # Step 3: Generate grid indices
        i, j = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")

        # Step 4: Flatten the arrays
        pixel_flat = image_data_array.flatten()

        # Step 5: Combine the indices and pixel intensities to create points
        points_array = np.vstack((i.flatten(), j.flatten(), pixel_flat)).T

        # Step 6: Insert the points and scalars into VTK
        points.SetData(numpy_support.numpy_to_vtk(points_array, deep=True))

        # normalize the scalar values to 0-1
        pixel_flat = (pixel_flat - pixel_flat.min()) / (
            pixel_flat.max() - pixel_flat.min() + 1e-6
        )

        scalars = numpy_support.numpy_to_vtk(
            pixel_flat, deep=True, array_type=VTK_FLOAT
        )
        scalars.SetName("Intensity")
        # Step 7: Generate grid indices for cells
        i = i[:-1, :-1]
        j = j[:-1, :-1]

        # Step 8: Compute the cell points
        id1 = i * dims[1] + j
        id2 = (i + 1) * dims[1] + j
        id3 = (i + 1) * dims[1] + (j + 1)
        id4 = i * dims[1] + (j + 1)

        # Step 9: Flatten the arrays
        id1 = id1.flatten()
        id2 = id2.flatten()
        id3 = id3.flatten()
        id4 = id4.flatten()

        # Step 10: Prepare the cell data
        cell_data = np.vstack((id1, id2, id3, id4)).T

        # Step 11: Create a single array for all cell points
        num_cells = cell_data.shape[0]
        cell_array = (
            np.hstack((np.full((num_cells, 1), 4), cell_data))
            .astype(np.int64)
            .flatten()
        )

        # Step 12: Convert the NumPy array to a VTK-compatible format
        vtk_cell_array = numpy_support.numpy_to_vtkIdTypeArray(cell_array, deep=True)

        # Step 13: Insert cells into the VTK cell array
        cells.SetCells(num_cells, vtk_cell_array)

        # Step 14: Set points, cells, and scalars in polyData
        polyData.SetPoints(points)
        polyData.SetPolys(cells)
        polyData.GetPointData().SetScalars(scalars)
        polyData.GetPointData().GetScalars()
        return polyData
    
    @staticmethod
    def loadOCTFile(filePath: str):
        if not os.path.isfile(filePath):
            return None
        data = None
        # get file extension
        file_ext = os.path.splitext(filePath)[1]
        if file_ext == ".foct":
            data = fio.read_foct_file(filePath)
        if file_ext == ".oct":
            data = fio.read_foct_file(filePath)
        elif file_ext == ".dcm":
            data = fio.read_dicom_file(filePath, as_uint8=True)
        elif file_ext == ".ioct":
            data = fio.read_ioct_file(filePath)
        elif file_ext == ".img":
            data = None
        elif file_ext == ".mat":
            data = fio.read_mat_oct_file(filePath)

        return data

    @staticmethod
    def loadOCTAFile(filePath: str):
        if not os.path.isfile(filePath):
            return None
        # get file extension
        file_ext = os.path.splitext(filePath)[1]
        if file_ext == ".ssada":
            data = fio.read_ssada_file(filePath)
        elif file_ext == ".dcm":
            data = fio.read_dicom_file(filePath, as_uint8=True)
        elif file_ext == ".octa":
            data = fio.read_octa_file(filePath)
        elif file_ext == ".img":
            data = None
        elif file_ext == ".mat":
            data = fio.read_mat_octa_file(filePath)
        elif file_ext == ".med":
            data = None
        else:
            data = None

        return data

    @staticmethod
    def loadCurveFile(filePath: str) -> Any:
        try:
            if not os.path.isfile(filePath):
                return None
            # get file extension
            file_ext = os.path.splitext(filePath)[1]
            if file_ext == ".json":
                data = fio.read_curve_json_file(filePath)
            elif file_ext == ".dcm":
                data = fio.read_curve_dicom_file(filePath)
            elif file_ext == ".mat":
                data = fio.read_curve_mat_file(filePath)
            elif file_ext == ".med":
                data = fio.read_curve_med_file(filePath)
            else:
                data = None
            return data
        except Exception as e:
            logger.exception("Error loading curve file: %s", e)
        return None

    @staticmethod
    def loadDataFromFolder(folderPath: str):
        logger.info("Loading Data from Folder %s", folderPath)
        pass

    def dragEnterEvent(self, event):
        """Handle drag enter event - accept variable drops from Variable Tree."""
        if event.mimeData().hasText():
            # Check if it's a variable from workspace (starts with "ws.")
            text = event.mimeData().text()
            if text.startswith("ws."):
                event.acceptProposedAction()
                # Visual feedback
                self.ui.statusbar.showMessage("Drop variable here to load it", 0)
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event - reset visual feedback."""
        self.ui.statusbar.clearMessage()
        event.accept()
    
    def dropEvent(self, event):
        """Handle drop event - load variable from Variable Tree.
        
        Accepts 3D numpy arrays as OCT or OCTA data.
        Shows dialog to ask user which type of data to load.
        """
        text = event.mimeData().text()
        self.ui.statusbar.clearMessage()
        
        if not text.startswith("ws."):
            event.ignore()
            return
        
        # Extract variable name (remove "ws." prefix)
        var_name = text[3:]  # Remove "ws." prefix
        
        # Get the variable data from workspace
        try:
            workspace_data = {}
            if self.app_context:
                workspace_manager = self.app_context.get_component("workspace_manager")
                if workspace_manager:
                    workspace_data = workspace_manager.get_workspace_data()
            elif self.parentWindow:
                workspace_data = self.parentWindow.get_workspace_data()
            
            # Navigate nested structure (e.g., "parent.child.subchild")
            var_data = workspace_data
            path_parts = var_name.split('.')
            
            for part in path_parts:
                if isinstance(var_data, dict) and part in var_data:
                    var_data = var_data[part]
                else:
                    QMessageBox.warning(
                        self,
                        "Variable Not Found",
                        f"Variable '{var_name}' not found in workspace."
                    )
                    event.ignore()
                    return
            
            # Check if it's a dict (segmentation/curve data)
            if isinstance(var_data, dict):
                # Check if data has a 'curve' key (from .med files saved by Save menu)
                # Extract the actual curve data from the wrapper
                if 'curve' in var_data and isinstance(var_data['curve'], dict):
                    var_data = var_data['curve']
                    logger.debug(f"Extracted curve data from 'curve' wrapper for '{var_name}'")
                
                # Check if it's valid segmentation data
                if utils.validateCurve(var_data):
                    # Convert dict to dataDict for dot notation access
                    
                    var_data = dataDict.from_dict(var_data)
                    
                    # It's segmentation data
                    reply = QMessageBox.question(
                        self,
                        "Load Segmentation Data",
                        f"Variable '{var_name}' is segmentation data.\n\n"
                        f"Load as segmentation?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        self.curve_data_dict = dataDict({utils.generateCurveDictKey(var_data.permute, var_data.flip): var_data})
                        self.gCurve_data = var_data
                        self.ui.lineEdit_transfer_seg_name.setText(var_name)
                        self.ui.comboBox_Seg.setCurrentText(var_name)
                        self.ui.statusbar.showMessage(f"Loaded segmentation data from '{var_name}'", 3000)
                        logger.info(f"Loaded segmentation data from variable: {var_name}")
                        event.acceptProposedAction()
                        return
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Segmentation Data",
                        f"Variable '{var_name}' is a dict but not valid segmentation data.\n"
                        f"Missing required fields for segmentation."
                    )
                    event.ignore()
                    return
            
            # Check if it's a 3D numpy array
            elif isinstance(var_data, np.ndarray):
                if len(var_data.shape) != 3:
                    QMessageBox.warning(
                        self,
                        "Invalid Array Dimensions",
                        f"Variable '{var_name}' is not a 3D array.\n"
                        f"Expected: 3D array (frames, rows, cols)\n"
                        f"Got: {len(var_data.shape)}D array with shape {var_data.shape}"
                    )
                    event.ignore()
                    return
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Data Type",
                    f"Variable '{var_name}' has unsupported type.\n"
                    f"Expected: numpy.ndarray (3D) or dict (segmentation)\n"
                    f"Got: {type(var_data).__name__}"
                )
                event.ignore()
                return
            
            # Check if numpy array could be segmentation data
            if isinstance(var_data, np.ndarray) and utils.validateCurve(var_data):
                # Convert to dataDict if it's a dict-like structure
                
                if isinstance(var_data, dict):
                    var_data = dataDict.from_dict(var_data)
                
                # It's segmentation data
                reply = QMessageBox.question(
                    self,
                    "Load Segmentation Data",
                    f"Variable '{var_name}' appears to be segmentation data.\n"
                    f"Shape: {var_data.shape if hasattr(var_data, 'shape') else 'N/A'}\n\n"
                    f"Load as segmentation?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.curve_data_dict = var_data if isinstance(var_data, dataDict) else dataDict.from_dict(var_data) if isinstance(var_data, dict) else var_data
                    self.ui.lineEdit_transfer_seg_name.setText(var_name)
                    self.ui.comboBox_Seg.setCurrentText(var_name)
                    self.ui.statusbar.showMessage(f"Loaded segmentation data from '{var_name}'", 3000)
                    logger.info(f"Loaded segmentation data from variable: {var_name}")
                    event.acceptProposedAction()
                    return
            
            # It's a 3D array - ask user whether it's OCT or OCTA
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Select Data Type")
            dialog.setText(f"Variable '{var_name}' is a 3D numpy array.\n"
                          f"Shape: {var_data.shape}\n"
                          f"Data type: {var_data.dtype}\n\n"
                          f"Which type of data is this?")
            dialog.setIcon(QMessageBox.Icon.Question)
            
            oct_button = dialog.addButton("OCT", QMessageBox.ButtonRole.AcceptRole)
            octa_button = dialog.addButton("OCTA", QMessageBox.ButtonRole.AcceptRole)
            layerseg_button = dialog.addButton("LayerSeg", QMessageBox.ButtonRole.AcceptRole)
            cancel_button = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            dialog.exec()
            clicked_button = dialog.clickedButton()
            
            if clicked_button == oct_button:
                # Load as OCT data
                self.oct_data = var_data
                self.oct_data_raw = var_data.copy()
                self.volume_size = var_data.shape
                self.ui.lineEdit_transfer_oct_name.setText(var_name)
                self.ui.comboBox_OCT.setCurrentText(var_name)
                self.ui.statusbar.showMessage(f"Loaded OCT data from '{var_name}'", 3000)
                logger.info(f"Loaded OCT data from variable: {var_name}, shape: {var_data.shape}")
                
                # Initialize curve_data_dict if needed
                if self.curve_data_dict is None:
                    self.curve_data_dict = utils.createZeroCurveDict(*self.volume_size)
                    self.gCurve_data = self.curve_data_dict["None_012"]
                
                # Update UI
                self.onTransform_changed()
                event.acceptProposedAction()
                
            elif clicked_button == octa_button:
                # Load as OCTA data
                self.octa_data = var_data
                self.octa_data_raw = var_data.copy()
                if self.oct_data is None:
                    self.volume_size = var_data.shape
                self.ui.lineEdit_transfer_octa_name.setText(var_name)
                self.ui.comboBox_OCTA.setCurrentText(var_name)
                self.ui.statusbar.showMessage(f"Loaded OCTA data from '{var_name}'", 3000)
                logger.info(f"Loaded OCTA data from variable: {var_name}, shape: {var_data.shape}")
                
                # Initialize curve_data_dict if needed
                if self.curve_data_dict is None:
                    self.volume_size = var_data.shape if self.oct_data is None else self.oct_data.shape
                    self.curve_data_dict = utils.createZeroCurveDict(*self.volume_size)
                    self.gCurve_data = self.curve_data_dict["None_012"]
                
                # Update UI
                self.onTransform_changed()
                event.acceptProposedAction()
            elif clicked_button == layerseg_button:
                # Attempt to convert the 3D numpy array (layer segmentation) to curve format
                try:
                    # Use current permute/flip settings from UI as defaults for conversion
                    permute = self.ui.comboBox_permute.currentText() if hasattr(self.ui, 'comboBox_permute') else "0,1,2"
                    flip = self.ui.comboBox_flip.currentText() if hasattr(self.ui, 'comboBox_flip') else "None"
                    
                    nimgs, nrows, ncols = self.volume_size
                    var_data = np.transpose(var_data,(2,1,0))
                    curve_data = utils.convertArray2CurveDict(
                        var_data, nimgs, nrows, ncols, permute=permute, flip=flip
                    )

                    # Validate and assign the generated curve data
                    # if curve_data is None or not utils.validateCurve(curve_data):
                    #     raise ValueError("Generated curve data is invalid")

                    # Place the generated curve into the curve_data_dict under the generated key
                    key = utils.generateCurveDictKey(curve_data.get("permute", "0,1,2"), curve_data.get("flip", "None"))
                    # Ensure curve_data_dict exists and store the generated curve
                    if self.curve_data_dict is None:
                        self.curve_data_dict = dataDict({"volumeSize": tuple(curve_data.get("volumeSize", (1, 1, 1)))})
                    self.curve_data_dict[key] = curve_data
                    self.gCurve_data = curve_data

                    # Update UI selections / names
                    self.ui.lineEdit_transfer_seg_name.setText(var_name)
                    # Note: comboBox_Seg lists workspace keys; set the combobox text if the variable exists there
                    try:
                        self.ui.comboBox_Seg.setCurrentText(var_name)
                    except Exception:
                        # ignore if var_name is not in the list
                        pass

                    self.ui.statusbar.showMessage(f"Converted layer segmentation '{var_name}' to curve data", 5000)
                    logger.info(f"Converted variable '{var_name}' (3D array) to curve data and loaded into LayerSegmentation")
                    # Update UI
                    self.onTransform_changed()
                    event.acceptProposedAction()
                except Exception as e:
                    logger.exception("Failed to convert 3D array to curve data: %s", e)
                    QMessageBox.critical(
                        self,
                        "Conversion Error",
                        f"Failed to convert variable '{var_name}' to curve data:\n{str(e)}",
                    )
                    event.ignore()
            else:
                # User cancelled
                event.ignore()
                
        except Exception as e:
            logger.error(f"Failed to load dropped variable '{var_name}': {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load variable '{var_name}':\n{str(e)}"
            )
            event.ignore()

    @staticmethod
    def getIcon(iconName: str)-> QIcon:
        def _iconFromBase64(base64: bytes) -> QIcon:
            return QIcon(
                QPixmap.fromImage(QImage.fromData(QByteArray.fromBase64(base64)))
            )

        if iconName == "windowIcon":
            return _iconFromBase64(
                b"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"
            )
        elif iconName == "hlayout":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAdhwAAHYcBj+XxZQAAAVxQTFRF/////////////////////////////////P3+097ry9joy9foy9jozdno9Pf6////7fH3THevNmelNmelNmelNmelytfn////7PD2SnauNGWkNGWkNGWkNGWkx9Xm7PD2SnauNGWkNGWkx9TmNGWk////7PH2SnauNGWkNGWkNGWkNGWkx9Xm////7vL3Vn6zQm+qQm+qQm+qQW+qzdnp/////////////////////////////////////////////////////////////////////////////////////////f3+4ejx3ubw3uXw3ubw3ubw+Pr8////////7fH3SnauNGWkNGWkNGWkydbn7PD2SnauNGWkNGWkNGWkNGWkx9XmSnauNGWkNGWkx9Xm7vL4WYG1RXKsRXKsRnKsRnKsztrp/f7+5+304uny4uny4uny5Orz+vv9////uQgtCAAAAHR0Uk5TfltcYmNef1o7VW1wb11EVkuPwcnGn1xZWrPy+/bGblu2/8lw+lhXrOjx7L5rVER/rbOwjVRNJTRFSEc6LFBJFAkLFREBAAwOFkpOKDtPUlFBL1FVSIe3upZYWK7r9PDAbK/twm1Ggq+2s5BVNUZZW1pMPH0cCY6RAAAAsElEQVR4nGNkYAQCBiIBI1A1OwkagCq/M/KTpuE9ozhpGp4yinMw/mElUsMvdsaHoxpGNQwNDcz/gBpUScsP1xl1SdNwltGUNA2MjHYkFBoM/5n+MFpykaLh/zeQBqLVA3V8Y7RjYSHFSYyMXqR6OhiIrxKpQYeRcSWjuDNJNiweZIlvVMOoBupqcGZn3EWkhn+ejPMZg4VIykDTGcVDSdIwgbF4ic0RItUzMPz48RMAC9N2VIWDm0wAAAAASUVORK5CYII="
            )
        elif iconName == "vlayout":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAdhwAAHYcBj+XxZQAAAVxQTFRF////////////////////////////////////////////////9Pf6ytfnx9Xmx9Tmx9Xmzdnp////////////////+Pr8ydbnx9Xmx9Xmztrp+vv9////zdnoNmelNGWkNGWkNGWkQW+q////////////////3ubwNGWkNGWkNGWkRnKs5Orz////y9joNmelNGWkNGWkNGWkNGWkQm+q////////////////3ubwNGWkNGWkNGWkRnKs4uny////y9foNmelNGWkNGWkQm+q3uXwNGWkRXKs4unyy9joNmelNGWkQm+q////3ubwNGWkNGWkNGWkRXKs4uny097rTHevSnauSnauSnauVn6z////4ejxSnauSnauSnauWYG15+30/////P3+7fH37PD27PD27PH27vL3/////////f3+7fH37PD27vL4/f7+////////////////////eoErrgAAAHR0Uk5Tf15iY1tQSUhKUVx+RFxucGtULBURFi9YbG1VPFpdn8bJvo06CQEMQZbAwpBMVm/G9vv67LBHCwAOUbrw8rNaWXDJ//GzUvS2W23B6K1FT7fr7a9ZVY+ztqx/NDuHrq+CRlU7S1pbV0QlFChIWEY1WFRNTn1VZHvwAAAAuklEQVR4nO3WMQ4BQRjF8fkvCdmJYkkoaEhUCgpKtcQFdE7gOk7hGDqdXjZaiSwRsYhkzW4039hGo5oppnhvfpn2oVCgxEk8SFPudlFOU40N4AOIZeFnTetHUOjZIHtPVDPXTQKdgTHnwP5h3Tegsuu+ZBE1iB1wwAEHHHDAgf8Bb8LlKvMmHAyow14WbTjiT/Omw8ZMhxE8ZVFKl0Bxnrs1ViqZweMLbNFBUJWxGsASlSxOoVWoTjh8A4UAO/2l82y6AAAAAElFTkSuQmCC"
            )
        elif iconName == "fast":
            return _iconFromBase64(
                b"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"
            )
        elif iconName == "slow":
            return _iconFromBase64(
                b"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"
            )
        elif iconName == "normalSize":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAQVQTFRFAAAAJJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/J4n/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/JJL/Wa3/k8n/mcz/i8X/P6D/crn//v//////9vr/RaP/zeb/gcD/udz/O53/0uj//f7/NJr/TKb/UKj/abT/3u//LZb/0un/PZ7/wOD/3+//v9//YLD/weD/NZr/hsP/O57/Mpn/4fD/U6n/PJ7/otH/rdb/ZLL/ms3/aLT/YbD/4vD/i8b/JJL/isX/JJL/JJL/JJL/JJL/JJL/JJL/JJL/5VWqAQAAAFd0Uk5TAAlYl8ns+//rx1QQZcz6xmYMItP+zyAAW/RcTL79BTiMyv//////////////////////////////////////////////////////////6v8xtR8hvbYyXA9TZAAAAZJJREFUSMeVltdWwkAQQIcSho50ECmCQIYmTYqgINJsqFj+/1N8CTh6srB735LMPbk5cHYXwMBitdk1B5rgcLpsbgv8xeself.p1+fEA/sBJkM+HwhE8QiQa24/HE0mUQDuNG0LCiVJoKaMniZIkQwAAnjBKc+YBAG9EXkh7ASw+VMCXAatfRfBnwYZK5OBc8CRfuDAolthtO2jm82Wd9lSq7LcAh7lQI0ad/RNBUNTgQpE9EAmXzZaagO1O96qnIiD2B0pvwOFALWl4TUREI1nB6BnfSApGz7jflROMntsJdnQimoqEcq1xx3smiLN7In0uEPI6UbPNehARF8vVGgVCgYhaHdZjBhceiIi6rEdKeOQ9MkLvUI+ZcLBHJAh7BMLoqduZqSWR/rxQE4iWIoEtAi9cWJnPO2Dze1GqsKa1ubABF7uq1l8NpnNBUUB9qXSrLcZvYAmoCIGM+oYCwai88B4EAIhpsvPOrdq260ztNvaU3Maeiu/PAh+f6aPf+7VVO5x4g//OM5lsTnT82bhy35nd3A+GoXqDgbGVqQAAAABJRU5ErkJggg=="
            )
        elif iconName == "reset":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAcVQTFRFAAAAJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMRsTUJLnMOcDRxOzxQsPTgtfioeHpn+HofdXgPMHSJLnMsebt+v390PDz/v/+///+x+3xSsXVJLnMJLnMS8bV+/793vX3jdrkbNDdbtDdk93m5/f5+f39d9Tfxe3xddPfKLrNgNfh+v79ac/cJLnMKrvNeNTfh9njjNrkOsDRZc7b/f7+8/v7OMDRJLnMJLnMpePqktzlJLnMNr/Q6vj5JbnMJLnMP8HSftbh1/L1XszZqOPqS8XVVMjXtOftze/zRMPTJLnMLrzP/P79JrrN8fv7K7vOJLnMXMvZuOnuJLnMcNHe3/X3L7zONL7Q6fj5V8nYJLnMw+zxyu7yM77PO8HR2fP2q+XrJLnMMb3P1fL06/n5jtvkO8DRQcLTmd7nLbzOp+Pq9fz8md/nJLnMJLnMJLnMScXUld3m9Pv70fD0JLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMJLnMXhgzbAAAAJd0Uk5TABZxstjx/e/VrmgPNbX5//esJw6l/PuXCeLXJEby7DAj1Kb/hP//////////tv////////+S+v////////////////////9L////////////jNn//8X////q/////////////+f//////8D//4X///////9F////////Cv//////////////+BqW/////3LWwgbrGC8hhnlJJ+sAAAIDSURBVHicY2SAA0YwgFBfEaIwBg9IEkntO1QFwoyMvxiQATsj4wskBTy83xjQATfjU7gCGcZPGPIMDPyMD6EKeASxyQNVMD0AKxBmwzQfAoS+vwApUHoL5Yu8QVchegeogIcT6n4xxpcS4EB4jPAL1wNGBt1HUJ48IyQo/jMx3oapUDvNyGh6E8LWAIYEO9iEL7yM56EKGH8zMvFC2UYfGRgEzgIZJoyMb0RPQkUtGJl5ICxLxkd/lRgZD4PYdkBj9kOEnRiD9mLxn8t9pd0QlhsjKxcWBe6P5Bm3g1lejGycYIY349f9SCp8r/2CeJYNpsDvqs5GJAXiPLobICHByM4BZgRe0l+HpCD4guEaMCOUkYMdzAhj/Ai1FQQiGC8YLgOzohk52SBiikyWS+AKrF5aLYaw4hi5WCGs+CO2jPOh8kmMBxznQJipjNys/yHMtL0ujDPArEzGnR5TIYJ8cYwMgv8g7FzGrT6MjP0MRYzPTn2J7YHaMJmRQUgFGluJImsZGWTN1jL8D30McSFDBWMFMH4722Buq2b8xH/Y7tOKFzCB+iJQipIob2DADhoZ88GJVqGoFqs8V1UuNNlP+YdNBVdNNgMs4yjAcxqS+cy5DHAFDBI8n36iSFewM+czICkAGiJ6+z9cmq/yUwWUicjRDMImXCwsexj92JjZ2VvhogBgzXHH59+tNwAAAABJRU5ErkJggg=="
            )
        elif iconName == "OFF":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAAXNSR0IB2cksfwAAAAlwSFlzAAAN1wAADdcBQiibeAAAAWhQTFRFAAAAsaeoTEFBKyQkFhEQEAsLBwUFCAYGEAwMGBISLSYmVktKwLu+z9jcdmpqLScnCwkJAAAAAAAAAQAADgsLLSUmem1tiYCBKCEhCAYGCggILygopZaXhnx8FBAQDAMDOA8MTxYSThURMw4MCAICGhYWp5aXu8PHXk1MDgsLFAUEeCEbxTYs8EI19EM27kE1vjQqbR4YDQQDDwsLfHFxhXp7DgsKBwICdSAa4j4y2jwwYxsWBAEBDwsLqpqbhXx9Ew8PJgoJuDIppy4lHAgGGBMTp5aWKiMjJgsJyTctujMpGwcGAQEBMCkpZVhZCAYGuTMppi0lAwEBDAoKi4GCJyEheiIbXhoVNy8vn5GSCwkJGAcF5T8z1zswCgMCEQ0Ny9DUQjc3fyMcZBsWXlFRJR0dzTgutjIoLygnEw8PFgYF80M26EA0AgEBHhkYCQcHRxQQJAoIExAQEA4OYBoVPREOEQwMx9DUen6KKAAAAHh0Uk5TAAYvfK/R0tLPqnQrBAAWf93+//3Xaw0Jd+rjZwUevv///////68OACHZ////////////wxMf2f///////8IODL//////qASB//////xiFu7////hDYT//2kH5v/////QAjj//yKG//9kvP////+f0///xO7//9IA6c5jiwAAATZJREFUeJyFk0ErhFEUhp93Z2E2UzILFpIQNQtRZkEisiALREmIhZWFhd/gR0iSIkSx0IjEgjRZWAhJFixGspmxsHPna5r5xnwz592c2z1P97z33nNETvIpnd/NxpC370M/C4Fq6Re/KqR3H1CrlPLnfHuJsJ692PgETUq6ZSRbJBPeXKxTIntCu16gQfrJFaiUHqBFlx7QrXuI6qvAQ5XuoE1xB4Rit6JDHxSqRtfEdOSAYV0Rbn3lv+p1QY/2xNgZ9D0W5aH5FPq3NakTBjMuihTVMUPS7KEYSQTk3e0OiEW0sA+jN4FA5y6Ma3GHCWczSF3aYkpLm0yfB+ahd4MZLa8zFy8BDKwxbwNmCdOkeU3zocynNj/L/m6zYcyWc27TybJNa7d9+cEhRMoePXN4V1YpNf5/0vZpJfiv0TEAAAAASUVORK5CYII="
            )
        elif iconName == "ON":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAAXNSR0IB2cksfwAAAAlwSFlzAAAN1wAADdcBQiibeAAAAo5QTFRFz9jcz9jcz9jcucbFY21qQUhFLDIvICQiHSAfJSsoNTs4UFhVkZ2az9jcz9jcAAAAztfbVV9bHyQhBwkIAAAAAQIBEhUTOUE9lKCdz9jcVV9aFRkXAgICCw0MMTg0pbOvz9jcMjg1BggHAQIBBxAICRUJAwgEAAAAFhoXjJmUz9jcJComAQIBDyIQLWgvP5BCSqpOTK9QTK5QRZ5INXs4HUMfAwcDDxEQYnBoJSonDR4ONn05S61PQ5pGH0ghAQMBDA4MgI2IMTc0AQICAgUCIk4kSKVMN386CxoMERQSg4+LVWBbBgcGKFwqPpBCDB0NAAAAIyglGBsZIUwjO4g+BAgEBQYGX2plYGtmAgMDDBwNR6VLK2ItIickIygmNHk3SalNBgcGhZKOxM/RCQsKS6xOL2sxOD87bXdzKmItR6NKGx8dSVBOPIk/EysUDxEQMzg2R6RLI1AlBAUEKzEuKV8rAAAAJy0qKmEsMjg1JlcoAQIBQEhEQJREGjsbDA4NW2NgMXAzS6xPBQsFFRkXrLi3BAUEFzQYLTMvGRwaGDYZAQIBY25qR09LGTgaOYI8FhkXz9jcqbezDxIQAAEAMXI0RqFJAQEBRk9KOkI+CBIIOoU9SadMGjwcFBcWtsTCxdDRGh4cNnw5RqBJGjwbCAoJUltXtcLAEBMRAgQCI1ElR6NLDBsMBQUFQUhFgpCKDRAOCBEIQ5lGNXo4FTEWBQYFNDs3mamiFxoYDR0NHkUfJVYnJ1kpIk8kFTEXAwYDCQsJQUhFm6mlKzIuBwkIAQEBVF1Zbnp1KC0qAgMCGR0bUFlVusfGnaunR1BLIyglFBcVCQwKAwQDAQIBBggGDxEPGBwaMTg0a3Zyz9jcYWNgSAAAANp0Uk5TGAwNHFCKs7y8vJprKw8HAAM3pu//+sxrEQA0uvrlbwkBcu///////bkbA4b8/////////////9snhf///////9wnbPz//////8sSMOz////+k7j////1KDX6////op7///QYF+j//2BD//+ye///4aL///O8//u8/7P/9Ir//+pX////xiL1/3u9//srV///yAQK2v////1LW//////ABQSi////6jIPxf/////2WRXH//////ZhE7r/////////6VcLfOz8LRuM+cRKBA9Rotju8/Pz4r5wJAKoDjhXAAAB9ElEQVR4nHWTXUiTURjHf3/vnLWwQJDqKojEwN0YBEr0gTHCKMRkhEUfkqLRSGEUhovIaEUxMFmlZGEhJWL0gRhU5AIvrChYXRVdRRF+khTd1NnbNt5tp+fqOc/vx3nOed/nCJEKueIHLFWq6nVhl/ozkzuCx5R+4YpCac4lFEsLznKZEWecbIX0JSOslL6avNTp4vT6ZJZr9MHZ2ktx6cw0rJUW0w2WSAlYr9eO4CnTW3zSd/cZSjQFlXqZbFGlSXyFs4tkxWoDq/RmVipb9wI2fcvhySM8Y4vGpAY9oUbvczmUj4F/Qmp8DDve5XMqfo9TKx3SKDVFUxaByhGoU/Mw1E/aOBsTCRrUNgSBuFWg+g57FRyExud2YfMt9qtjAN+0nbPtBgcV6md50X+E7X3mW524DqtkF/zXOKLOGLQ8tAu1vbQq3ANHR618t6IEdVaXaL9rFUpeQYcKzl2gfK7AJgQihMzwXJyIc3LwTz7f1zNPpxGiOk1YfXm8KT7OGZktvJGF83SrN4e3KgQRBY0QVbuZ61OXs/hxdc0TVYsz1TG1wRWp23WBpma4qgOpsR/QYVPsj1d3ObSiTp/DcFMB0g9nSDHnj9/+9zLrk/mwdpIReCDtyjrD/T33/LgEPCP6eCyDH5lttpIlmHjqfv8b0tW/fYJ3YlmasDgAAAAASUVORK5CYII="
            )
        elif iconName == "run":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAASxQTFRFAAAA4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4LMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMW4bMWqXqn+AAAAGR0Uk5TAAIVJjlAFB92xer7//robxsEYMP9wVkBtqw2AH/zraFa/P6pC4z4fUsrgsqiiuRsGHLmAQmIJCyqBoezRYaOBQy+mIMlmnpqj8bS9lV8E+cZcy2Jkiif+ZPrZa46wDug9dFmN8z3OhsAAAKESURBVHic1ZXfTxNBEMd3ev1xRaJeKNggxeKPRBMINVYbJNJqQGNEHozhmb9NE1+MaGJMNNFYY21ThUQRRKSJTYhWKViwkP446nm7e9fe0VmLbzoPnZnd+dx+d257C+QvDf4PAKgRTZNUbS+Ah9dTTnXDRgtA0WvLllxrA7UgBg6t+7Z3q4B2gC8ioAcKpNmUn5s44Pb/QMqpdeSqGBCEvADoAsgggOLL8aB7pZe21QHZ+mxwHgEIGcjqWz4K8NbIQwCfjPCYcxYBwrC26u94bRmJQIZ32dH+FQF0VSunU7aRIZjjwWBSswEKwTpK7fw7VgODCSsQ9gK8wIFYmW/47HMrcOlzPrIjIEaX2OJysVIHwvvpRoeqAuJykrnhJyagN4e+Mu1CCSdCy+yxwQXNAHQ9fEak6upL+jvytMoAroebQNUJdgSij2sUMPRwE6i68oq58g4F6nr+oGo8zlxFNfZAXB42EFhsro0609YUYgxwyizr+YjIibqTjeMDJzMskdrYOen9gAAk6k3U44uJAgMm4Bl1RxYwoKEKYg+NPYAsUaeNPcAJrkrXUzDP0oEac2P3UcBQpeshJnBQZd4xOi0gnGmmpw6QzhJz/ZASEJI6X7ACXSXWJzl6DweIsmF84QxAkYs8OLUvbiuM+ZY8M9YB86V0S3xFbSBw1zIdWdussOY0AYe3aubiIcnsbqha1Gs7ncsIQMjkzKoRaSPbXhdRS+v9j/hAn0VVA3AHvu/eqWnX0lkEIMflbFMpt/FvcQzQ/yipX0j5dbhjyWzAmVzfXNOFMgGLsyJAT4dd721X1g24bX8EcinG/Hn2FbwJb85Nt7oU+ZirNnVr79duC/sHgd9cksOPQIV+0AAAAABJRU5ErkJggg=="
            )
        elif iconName == "run-disabled":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAASxQTFRFAAAAcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFoLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLcFkLJbpeoQAAAGR0Uk5TAAIVJjlAFB92xer7//robxsEYMP9wVkBtqw2AH/zraFa/P6pC4z4fUsrgsqiiuRsGHLmAQmIJCyqBoezRYaOBQy+mIMlmnpqj8bS9lV8E+cZcy2Jkiif+ZPrZa46wDug9dFmN8z3OhsAAAKESURBVHic1ZXfTxNBEMd3ev1xRaJeKNggxeKPRBMINVYbJNJqQGNEHozhmb9NE1+MaGJMNNFYY21ThUQRRKSJTYhWKViwkP446nm7e9fe0VmLbzoPnZnd+dx+d257C+QvDf4PAKgRTZNUbS+Ah9dTTnXDRgtA0WvLllxrA7UgBg6t+7Z3q4B2gC8ioAcKpNmUn5s44Pb/QMqpdeSqGBCEvADoAsgggOLL8aB7pZe21QHZ+mxwHgEIGcjqWz4K8NbIQwCfjPCYcxYBwrC26u94bRmJQIZ32dH+FQF0VSunU7aRIZjjwWBSswEKwTpK7fw7VgODCSsQ9gK8wIFYmW/47HMrcOlzPrIjIEaX2OJysVIHwvvpRoeqAuJykrnhJyagN4e+Mu1CCSdCy+yxwQXNAHQ9fEak6upL+jvytMoAroebQNUJdgSij2sUMPRwE6i68oq58g4F6nr+oGo8zlxFNfZAXB42EFhsro0609YUYgxwyizr+YjIibqTjeMDJzMskdrYOen9gAAk6k3U44uJAgMm4Bl1RxYwoKEKYg+NPYAsUaeNPcAJrkrXUzDP0oEac2P3UcBQpeshJnBQZd4xOi0gnGmmpw6QzhJz/ZASEJI6X7ACXSXWJzl6DweIsmF84QxAkYs8OLUvbiuM+ZY8M9YB86V0S3xFbSBw1zIdWdussOY0AYe3aubiIcnsbqha1Gs7ncsIQMjkzKoRaSPbXhdRS+v9j/hAn0VVA3AHvu/eqWnX0lkEIMflbFMpt/FvcQzQ/yipX0j5dbhjyWzAmVzfXNOFMgGLsyJAT4dd721X1g24bX8EcinG/Hn2FbwJb85Nt7oU+ZirNnVr79duC/sHgd9cksOPQIV+0AAAAABJRU5ErkJggg=="
            )
        elif iconName == "transpose-on":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAF1QTFRFAAAAKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjwEkwZgAAAB90Uk5TACKs+//2FKOhDAsc/VXaBrufSfw3Nd64BEfsQAlaAWVbsUIAAADgSURBVHic7dRdC4IwFAbgc7YVYoj0JQT9/x9mBF6EYgjmF0uRZMHW2aDyxnOnnEeGe3kRHAfnANiPdrUTtRYwZvo6r7SAG48pOy0Qpn2AdgGvWUnTPjaOFyceWrBmjT4ajJUzpZUGU57HANNA+S1DgGmg5HkIMA3Uq211S0J5v4BvAb92BJsKvMIFBCX4dwcQYgFB5gB2mMvtzR4cMYU9JrYg9LsUIryCFThnjPMc4IQxAd66LOIX5YkE8iB4bA28/kgiASDBh/kPUOp66GcaKL009DMNproe+5kGTvN78ARK100xKsuhfQAAAABJRU5ErkJggg=="
            )
        elif iconName == "transpose-off":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAF1QTFRFAAAAKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjKqXjwEkwZgAAAB90Uk5TACKs+//2FKOhDAsc/VXaBrufSfw3Nd64BEfsQAlaAWVbsUIAAADfSURBVHic7dRdC4IwFAbgc7YVYoj0JQT9/x9mBF6EYgimJssxkF1MzwaVN5475Twy3MuL4Dm4BMBhrKu9aK2Asamv88YK+OQxZW8FAuBtW9fvV6BnI23rarDzvDjxsoIt6+zRYKxeKK00GPOsA0wD47eoANPAyLMKMA3Gq52/4xV8F4StJ9g1EFQ+IKohfHqAGCuICg9wwFLuH+7gjDkcMXMFcdjnkOAdnMC1YJyXABdMCSBMlvCb8UQCeRI8dQbBcCSRAZBgZv4DjLpW/UwDo5dUP9NgrGvdzzTwmt+DD+h3aDHGjlqFAAAAAElFTkSuQmCC"
            )
        elif iconName == "expand":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAABpElEQVRoge2ZyW7DMAxE3SL57l6atL1aXQJ0yT92AZITQ9bOxQ0tDbXYLjSA0IMVcZ6cigzVNFVV6bR2tOFx5EHAkPkbOFZL2+Sx+OEBXPA8Pg0AX8ZYhzEAy4JHNnMLA9je9u+AAFaO3psbukYNRovoijfmORpgEogR88EAq5ZeBxBvRSA4hsQa8xIEcGkXsr+JkZg4gLZgSx9ZILpYL9qG2QBKQXjMxwHkhggwHw+QCyLQfBoALaAVAlwrDYAW2NEegpA1HO2QjQgFOKd2vd4QyVnNAU0Q3Tm/R99ikLe11CdSZIXUNhYIo3nYW7AuQPBwqomWHovkFEh/d/VbmyrPzP83WdWlesc7/MN/77Rp/Py+n+PmY76qqqpqeskxKknKd4w6epjfMTpMZJzutam9+RklMrSUkGSXs5To232S7v2twtLFXIi3/1BO237QoF8Fwxr1J2VS84Y14wCA7oERwtvtsAPkNA9A2ABK9kc9sXCApTd3F99eL7rzQ3WNryfNDwQw6RWTAqF+RiP2jJLXrF6AxV+z2i66y12zmmJVVY3oBDC/ecEZPpBBAAAAAElFTkSuQmCC"
            )
        elif iconName == "collapse":
            return _iconFromBase64(
                b"iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAABpElEQVRoge2ZyU7DMBCGw5Ln5tKyXBOgSGw3DjweBak5DTPZaEM89nhrI80vWb1MPP9np/HYLgqVSqU6SZU13JUV/OBvXVzBuTW+gmuM/ca2DokZhTkx/r73cCO03yajB4HaZQ3vNgiMa/r4XUjMYJ5yDvnRy1YOgCM/dkAQFbxxEPuxITGtecy1H9u+BWIBnCH55gCCmYkoAJSzhifJwMkhDB0GAwhyJYEIAkhmXpDAGyC5+cNEj6ZEXgDz/7PX+OY5iD6hGIDpK435Qd03+mUyE89SgOkz1Gd684NmRk8KkH/kp+q+2Q+hAMnM93VLYxo1rkkAHBp5sNdOMwBbz4RcLbTz7PNLDoDUHjPQIPiKGZRV7D5VKpXqROW5kLGLjvenWReyY5cSSRYypiKVAByvEl1sOb3oDQ2zh5UC2PbYWc3/M+cC8NdnBgiH0wMvgCwQiz7YEpxVBgGYIBxOxc2aHnFbRiUYwAzxUXzChdh/e7kgmNIoAKRYx+tld5PiPJVlygsO9CIHqOBWdMVEtRMVgNym3iFmVLe21N5XTCqVSpVFvx61waSIQGBVAAAAAElFTkSuQmCC"
            )


# if __name__ == "__main__":

#     import sys
#     from PySide6 import QtWidgets, QtCore

#     app = QtWidgets.QApplication(sys.argv)
#     root = QtCore.QFileInfo.path(
#         QtCore.QFileInfo(QtCore.QCoreApplication.arguments()[0])
#     )
#     mainWin = RetinalLayerSegmentaion(None)
#     mainWin.show()
#     app.exec()
