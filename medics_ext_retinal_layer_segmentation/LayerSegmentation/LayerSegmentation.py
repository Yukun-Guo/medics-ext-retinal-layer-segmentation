import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    Qt,
    QFileInfo,
    QCoreApplication,
    QByteArray,
    Slot,
    QSize,
)
from PySide6.QtGui import QImage, QPixmap,QIcon
from PySide6.QtWidgets import (
    QWidget,
    QSizePolicy,
    QMessageBox,
    QHBoxLayout,
    QProgressDialog,
)
from scipy.interpolate import PchipInterpolator
import logging

# Import model reassembly module
from ..model_reassembly import get_model_file_path

logger = logging.getLogger(__name__)

# VTK imports with error handling
try:
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
    from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
    from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints, vtkFloatArray, VTK_FLOAT
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.util import numpy_support
    
    # Load implementations for rendering and interaction factory classes
    import vtkmodules.vtkRenderingOpenGL2
    import vtkmodules.vtkInteractionStyle
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    
    VTK_AVAILABLE = True
except ImportError as e:
    logger.warning("VTK modules not available in LayerSegmentation: %s", e)
    logger.warning("VTK-related functionality will be disabled.")
    # Create placeholder classes to prevent runtime errors
    class vtkActor: pass
    class vtkPolyDataMapper: pass
    class vtkRenderer: pass
    class vtkCameraOrientationWidget: pass
    class vtkLookupTable: pass
    class vtkPoints: pass
    class vtkFloatArray: pass
    class vtkPolyData: pass
    class vtkCellArray: pass
    class QVTKRenderWindowInteractor: pass
    class vtkInteractorStyleTrackballCamera: pass
    VTK_FLOAT = None
    numpy_support = None
    VTK_AVAILABLE = False

import onnxruntime as ort
from ..utils.onnx_provider_utils import create_onnx_session, get_optimal_providers
from .LayerSegmentationUI import (
    Ui_LayerSegmentation,
)
from .LayerSegSideCtrlPanelUI import (
    Ui_LayerSegSideControlPanel,
)
from ..utils.curveeditorplot import CurveEditorPlot
from ..utils.indicatorPlot import IndicatorPlot
from ..utils.utils import utils

from typing import Any

import GPUtil

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
        bframeA_oct: bool = False,
        bframeA_curve: bool = False,
        bframeA_indicators: bool = False,
        bframeB_oct: bool = False,
        bframeA_octa: bool = False,
        bframeB_curve: bool = False,
        bframeB_indicators: bool = False,
        enfaceA_img: bool = False,
        enfaceA_indicators: bool = False,
        enfaceA_interp: bool = False,
        enfaceB_img: bool = False,
        enfaceB_indicators: bool = False,
        all: bool = False,
    ):
        """Initialize UpdateUIItems with flags for which UI elements to update.

        Args:
            bframeA_oct (bool): Update B-frame A OCT image.
            bframeA_curve (bool): Update B-frame A curve.
            bframeA_indicators (bool): Update B-frame A indicators.
            bframeB_oct (bool): Update B-frame B OCT image.
            bframeA_octa (bool): Update B-frame A OCTA image.
            bframeB_curve (bool): Update B-frame B curve.
            bframeB_indicators (bool): Update B-frame B indicators.
            enfaceA_img (bool): Update enface A image.
            enfaceA_indicators (bool): Update enface A indicators.
            enfaceA_interp (bool): Update enface A interpolation.
            enfaceB_img (bool): Update enface B image.
            enfaceB_indicators (bool): Update enface B indicators.
            all (bool): Update all UI items.
        """
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


class LayerSegmentation(QWidget):
    """
    LayerSegmentation is a QWidget-based class for interactive visualization, editing, and AI-based segmentation
    of retinal OCT data and associated layer boundaries.
    """

    def __init__(self, parentWindow=None, app_context=None, theme: str = "dark"):
        """Initialize the LayerSegmentation widget.

        Args:
            parentWindow: The parent window.
            app_context: The application context (optional).
            theme (str, optional): Theme string. Defaults to "dark" (fallback if app_context not available).
        """
        self.parentWindow = parentWindow
        self.app_context = app_context
        
        # Determine theme from app_context if available
        if self.app_context:
            theme_manager = self.app_context.get_component('theme_manager')
            if theme_manager:
                self.theme = theme_manager.get_current_theme()
            else:
                self.theme = theme
        else:
            self.theme = theme
        
        self.extension_name = "LayerSegmentation"
        self.oct_data = None
        self.octa_data = None
        self.curve_data = None
        self.oct_data_raw = None
        self.octa_data_raw = None
        self.oct_data_flatten = None
        self.octa_data_flatten = None
        self.oct_data_flatten_raw = None
        self.octa_data_flatten_raw = None
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

        self.all_fluid_data = None
        self.imgItem_enface_a = None
        self.imgItem_enface_b = None
        self.imgItem_bframe_a = None
        self.imgItem_bframe_b = None
        self.bframe_b_type = "Fast"
        self.last_opened_dir = ""
        self.oct_file_filters = "OCT Files (*.foct  *.dcm *.img *.mat);;All Files (*)"
        self.octa_file_filters = "OCTA Files (*.ssada  *.dcm *.img *.mat);;All Files (*)"
        self.seg_file_filters = "Segmentation Files (*.json *.dcm *.mat);;All Files (*)"
        self.data_extension_pairs = {".foct": ".ssada"}
        self.oct_fle_extensions = [".foct", ".dcm", ".img", ".mat"]
        self.octa_file_extensions = [".ssada", ".dcm", ".img", ".mat"]
        self.seg_file_extensions = [ ".json"]
        self.current_flatten_method = "None"
        self.glView = None
        self.ui = None
        self.cam_orient_manipulator = None
        self.curve_ui_mapping = None
        self.ui_pushButton_choroid_color = None
        self.ui_pushButton_eziz_color = None
        self.ui_pushButton_ez_color = None
        self.ui_pushButton_elm_color = None
        self.ui_pushButton_oplonl_color = None
        self.ui_pushButton_inlopl_color = None
        self.ui_pushButton_iplinl_color = None
        self.ui_pushButton_gclipl_color = None
        self.ui_pushButton_nflgcl_color = None
        self.ui_pushButton_ilm_color = None
        self.ui_pushButton_pvd_color = None
        self.ui_pushButton_rpebm_color = None
        self.ui_pushButton_izrpe_color = None
        self.ui_pushButton_sathal_color = None
        # ui_pushButton_haller_color = None
        self.enface_plot = None
        self.bframeA_cmpbar = None
        self.bframeB_cmpbar = None
        self.enfaceA_cmpbar = None
        # enfaceB_cmpbar = None
        self.imgView_bframe_b = None
        self.imgView_bframe_a = None
        self.imgView_enface = None
        self.bframe_plot_a = None
        self.bframe_plot_b = None
        self.imgItem_bframe_oct_a = None
        self.imgItem_bframe_octa_a = None
        self.imgItem_bframe_oct_b = None
        self.imgItem_enface = None
        self.seg_data = None
        self.vtkRender = None
        self.curve_vtk_mapper = None
        self.curve_vtk_actor = None
        self.enableInterpolation = False
        # interploated_curve = Non
        self.interp_key_curve_ranges = None
        self.isTranspose = False
        self.default_camera_position = [762.55, 16.01, 619.85]  # 570.08, 447.98, 724.61
        self.default_camera_focal_point = [160.41, 141.85, 114.49]  # 160.41, 141.85, 114.49
        self.default_camera_view_up = [-0.65, -0.03, 0.76]  # -0.86, 0.29, 0.43

        self.status_data_shape = None
        self.status_mouse_pos = None
        self.status_pixel_value_pos = None
        self._previous_interp_range = None
        # ai segmentation
        self.ort_session = None

        ##Enface Viewer variables begin ############################################################
        self.upper_boundry = None
        self.lower_boundry = None
        self.slab_mask = None
        self.isTranspose_enfaceA = False
        self.isTranspose_enfaceB = False
        self.enfaceViewer_enfaceImageA = None
        self.enfaceViewer_enfaceImageB = None
        self.indicatorDirection_enfaceA = 1
        self.indicatorDirection_enfaceB = 1
        self.upper_boundry_min = 0
        self.lower_boundry_max = 0
    
        self.setupUI()

    def setupUI(self) -> None:
        """Set up the user interface and connect signals."""
        QWidget.__init__(self)
        self.ui = Ui_LayerSegmentation()
        self.ui.setupUi(self)

        self.controlPanel = Ui_LayerSegSideControlPanel()
        self.LayerSegCtrlPanelWidget = QWidget()
        self.controlPanel.setupUi(self.LayerSegCtrlPanelWidget)

        # set gpu device
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                self.controlPanel.comboBox_gpu.addItem(f"GPU {gpu.id} ({gpu.name})")

        self.controlPanel.comboBox_gpu.currentIndexChanged.connect(self.onGPU_changed)
        # set up Pyqtgraph ui elements
        self.ui_pushButton_pvd_color = pg.ColorButton(color=(0, 255, 0))
        self.controlPanel.horizontalLayout_40.insertWidget(1, self.ui_pushButton_pvd_color)

        self.ui_pushButton_ilm_color = pg.ColorButton(color=(0, 255, 0))
        self.controlPanel.horizontalLayout_2.insertWidget(1, self.ui_pushButton_ilm_color)

        self.ui_pushButton_nflgcl_color = pg.ColorButton(color=(0, 255, 255))
        self.controlPanel.horizontalLayout_4.insertWidget(1, self.ui_pushButton_nflgcl_color)

        self.ui_pushButton_gclipl_color = pg.ColorButton(color=(0, 155, 0))
        self.controlPanel.horizontalLayout_5.insertWidget(1, self.ui_pushButton_gclipl_color)

        self.ui_pushButton_iplinl_color = pg.ColorButton(color=(0, 255, 0))
        self.controlPanel.horizontalLayout_6.insertWidget(1, self.ui_pushButton_iplinl_color)

        self.ui_pushButton_inlopl_color = pg.ColorButton(color=(255, 0, 255))
        self.controlPanel.horizontalLayout_7.insertWidget(1, self.ui_pushButton_inlopl_color)

        self.ui_pushButton_oplonl_color = pg.ColorButton(color=(255, 255, 0))
        self.controlPanel.horizontalLayout_8.insertWidget(1, self.ui_pushButton_oplonl_color)

        self.ui_pushButton_elm_color = pg.ColorButton(color=(0, 155, 155))
        self.controlPanel.horizontalLayout_9.insertWidget(1, self.ui_pushButton_elm_color)

        self.ui_pushButton_ez_color = pg.ColorButton(color=(255, 85, 55))
        self.controlPanel.horizontalLayout_10.insertWidget(1, self.ui_pushButton_ez_color)

        self.ui_pushButton_eziz_color = pg.ColorButton(color=(50, 50, 255))
        self.controlPanel.horizontalLayout_11.insertWidget(1, self.ui_pushButton_eziz_color)

        self.ui_pushButton_izrpe_color = pg.ColorButton(color=(255, 0, 255))
        self.controlPanel.horizontalLayout_12.insertWidget(1, self.ui_pushButton_izrpe_color)

        self.ui_pushButton_rpebm_color = pg.ColorButton(color=(0, 0, 255))
        self.controlPanel.horizontalLayout_13.insertWidget(1, self.ui_pushButton_rpebm_color)

        self.ui_pushButton_sathal_color = pg.ColorButton(color=(255, 50, 150))
        self.controlPanel.horizontalLayout_14.insertWidget(1, self.ui_pushButton_sathal_color)

        self.ui_pushButton_choroid_color = pg.ColorButton(color=(250, 250, 55))
        self.controlPanel.horizontalLayout_21.insertWidget(1, self.ui_pushButton_choroid_color)

        self.controlPanel.comboBox_corrector.currentTextChanged.connect(self.onCorrector_changed)

        self.ui.pushButton_layoutdirection.setIcon(self.getIcon("hlayout"))
        self.ui.pushButton_bframe_type.setIcon(self.getIcon("fast"))
        self.ui.pushButton_bframe_type.setToolTip("Change to Slow Scan View")
        self.ui.pushButton_reset.clicked.connect(self.onResetCamera)
        self.ui.pushButton_reset.setIcon(self.getIcon("reset"))
        self.ui.pushButton_transp.clicked.connect(self.onTranspose)
        self.ui.pushButton_transp.setIcon(self.getIcon("transpose-off"))
        self.ui.pushButton_transp.setToolTip("Transpose OFF")

        self.controlPanel.pushButton_Switch_Interp.setText("Interpolation: OFF")
        self.controlPanel.pushButton_Switch_Interp.setIcon(self.getIcon("OFF"))
        self.controlPanel.pushButton_Switch_Interp.setIconSize(QSize(21, 21))
        self.controlPanel.pushButton_Switch_Interp.clicked.connect(
            self.onSwitchInterpolation
        )
        self.controlPanel.pushButton_Restart_Interp.setEnabled(False)
        self.controlPanel.pushButton_Restart_Interp.setIcon(
            self.getIcon("run-disabled")
        )
        self.controlPanel.pushButton_Restart_Interp.clicked.connect(
            self.onRunInterpolation
        )

        self.ui.pushButton_hide_right.clicked.connect(self.onHideRight)
        self.ui.pushButton_hide_right.setIcon(self.getIcon("expand"))
        self.ui.pushButton_hide_left.clicked.connect(self.onHideLeft)
        self.ui.pushButton_hide_left.setIcon(self.getIcon("expand"))

        self.controlPanel.pushButton_ai_seg.clicked.connect(self.on_AISeg)
        self.controlPanel.checkBox_sparse.setVisible(False)
        self.controlPanel.checkBox_sparse.clicked.connect(self.onSparseCheckBoxChanged)

        self.gr_wid_bf_a = pg.GraphicsLayoutWidget()
        self.gr_wid_bf_a.setMaximumWidth(3000)
        # self.gr_wid_bf_a.setMinimumWidth(20)
        self.gr_wid_bf_a.ci.setContentsMargins(0, 0, 0, 0)
        self.gr_wid_bf_a.setMaximumHeight(22)
        self.bframeA_cmpbar = pg.ColorBarItem(
            values=(0, 255),
            colorMap=pg.colormap.get(name="afmhot", source="matplotlib"),
            limits=(0, 255),
            rounding=1,
            orientation="h",
            interactive=False,
            colorMapMenu=pg.ColorMapMenu(showColorMapSubMenus=True),
        )
        ply = QSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum
        )
        ply.setHorizontalStretch(1)
        ply.setRetainSizeWhenHidden(True)
        self.bframeA_cmpbar.setSizePolicy(ply)
        self.bframeA_cmpbar.hideAxis("bottom")
        self.bframeA_cmpbar.colorMapMenu.sigColorMapTriggered.connect(
            lambda: self.onColormap_changed(self.bframeA_cmpbar)
        )
        # Apply theme-consistent styling to colorbar menu
        if self.theme == "dark":
            self.bframeA_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.bframeA_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        self.gr_wid_bf_a.addItem(self.bframeA_cmpbar)
        self.ui.horizontalLayout_33.insertWidget(3, self.gr_wid_bf_a, 1)

        self.gr_wid_bf_b = pg.GraphicsLayoutWidget()
        self.gr_wid_bf_b.setMaximumWidth(3000)
        # self.gr_wid_bf_b.setMinimumWidth(20)
        self.gr_wid_bf_b.ci.setContentsMargins(0, 0, 0, 0)
        self.gr_wid_bf_b.setMaximumHeight(22)

        self.bframeB_cmpbar = pg.ColorBarItem(
            values=(0, 255),
            colorMap="CET-L1",
            limits=(0, 255),
            rounding=1,
            orientation="h",
            interactive=False,
            colorMapMenu=pg.ColorMapMenu(showColorMapSubMenus=True),
        )
        self.bframeB_cmpbar.setSizePolicy(ply)
        self.bframeB_cmpbar.hideAxis("bottom")
        self.bframeB_cmpbar.colorMapMenu.sigColorMapTriggered.connect(
            lambda: self.onColormap_changed(self.bframeB_cmpbar)
        )
        # Apply theme-consistent styling to colorbar menu
        if self.theme == "dark":
            self.bframeB_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.bframeB_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        self.gr_wid_bf_b.addItem(self.bframeB_cmpbar)
        self.ui.horizontalLayout_36.insertWidget(2, self.gr_wid_bf_b, 1)

        # add color bar to enface A
        gr_wid_a = pg.GraphicsLayoutWidget()
        gr_wid_a.ci.setContentsMargins(0, 0, 0, 0)
        gr_wid_a.setMaximumHeight(22)
        self.enfaceA_cmpbar = pg.ColorBarItem(
            values=(0, 255),
            colorMap="turbo",  # "CET-L1",
            limits=(0, 255),
            rounding=1,
            orientation="h",
            interactive=False,
            colorMapMenu=pg.ColorMapMenu(showColorMapSubMenus=True),
        )
        self.enfaceA_cmpbar.hideAxis("bottom")
        self.enfaceA_cmpbar.colorMapMenu.sigColorMapTriggered.connect(
            lambda: self.onColormap_changed(self.enfaceA_cmpbar)
        )
        # Apply theme-consistent styling to colorbar menu
        if self.theme == "dark":
            self.enfaceA_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.enfaceA_cmpbar.colorMapMenu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        gr_wid_a.addItem(self.enfaceA_cmpbar)
        self.ui.horizontalLayout_31.insertWidget(2, gr_wid_a)

        # # BFrame A
        self.imgView_bframe_a = pg.GraphicsLayoutWidget(self)
        self.imgView_bframe_a.sceneObj.contextMenu[0].setVisible(False)
        self.imgView_bframe_a.ci.setContentsMargins(0, 0, 0, 0)
        self.imgItem_bframe_oct_a = pg.ImageItem()
        self.imgItem_bframe_octa_a = pg.ImageItem()
        self.bframe_plot_a = CurveEditorPlot(enableEditorMenu=False, parentWindow=self)
        self.bframe_plot_a.setAspectLocked(True)
        self.bframe_plot_a.invertY(True)
        self.bframe_plot_a.hideAxis("left")
        self.bframe_plot_a.hideAxis("bottom")
        self.imgView_bframe_a.addItem(self.bframe_plot_a)
        # self.bframe_plot_a.setMenuEnabled(False)
        # self.bframe_plot_a.setMouseEnabled(x=False, y=False)
        self.bframe_plot_a.addItem(self.imgItem_bframe_oct_a)
        self.bframe_plot_a.addItem(self.imgItem_bframe_octa_a)
        self.ui.verticalLayout.insertWidget(1, self.imgView_bframe_a, 1)
        # self.ui.verticalLayout.removeItem(self.ui.verticalSpacer_3)
        # self.ui.verticalLayout.setStretch(0, 1)
        # add colorbar to bframe A
        self.bframeA_cmpbar.setImageItem(self.imgItem_bframe_octa_a)

        # BFrame B
        self.imgView_bframe_b = pg.GraphicsLayoutWidget()
        self.imgView_bframe_b.sceneObj.contextMenu[0].setVisible(False)
        self.imgView_bframe_b.ci.setContentsMargins(0, 0, 0, 0)
        self.imgItem_bframe_oct_b = pg.ImageItem()
        self.bframe_plot_b = CurveEditorPlot(parentWindow=self)
        self.bframe_plot_b.setAspectLocked(True)
        self.bframe_plot_b.invertY(True)
        self.bframe_plot_b.hideAxis("left")
        self.bframe_plot_b.hideAxis("bottom")
        self.bframe_plot_b.sigCurveChanged.connect(self._on_SegData_Edited)
        self.bframe_plot_b.sigControlPointChanged.connect(
            self._on_CurveEdit_range_changed
        )
        self.bframe_plot_b.sigCurveEditorTypeChanged.connect(
            self._on_CurveEditorType_Changed
        )

        # self.bframe_plot_b.setMenuEnabled(False)
        # self.bframe_plot_b.setMouseEnabled(x=False, y=False)
        self.bframe_plot_b.addItem(self.imgItem_bframe_oct_b)
        self.imgView_bframe_b.addItem(self.bframe_plot_b)

        self.ui.verticalLayout_2.insertWidget(1, self.imgView_bframe_b, 1)
        # self.ui.verticalLayout_2.removeItem(self.ui.verticalSpacer_4)
        # self.ui.verticalLayout_2.setStretch(0, 1)
        # add colorbar to bframe B
        self.bframeB_cmpbar.setImageItem(
            [self.imgItem_bframe_oct_a, self.imgItem_bframe_oct_b]
        )

        # Enface A
        self.imgView_enface = pg.GraphicsLayoutWidget()
        self.imgView_enface.sceneObj.contextMenu[0].setVisible(False)
        self.imgView_enface.ci.setContentsMargins(0, 0, 0, 0)
        self.imgItem_enface = pg.ImageItem()
        self.imgItem_enface_interp_line = pg.ImageItem()
        self.enface_plot = IndicatorPlot(parentWindow=self)
        self.enface_plot.sigIndicatorChanged.connect(self.onIndicatorChanged)
        self.enface_plot.setAspectLocked(True)
        self.enface_plot.invertY(True)
        # self.enface_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.enface_plot.hideAxis("left")
        self.enface_plot.hideAxis("bottom")
        # self.enface_plot.mousePressEvent = self.onEnfaceMousePressEvent
        self.enface_plot.addItem(self.imgItem_enface)
        self.enface_plot.addItem(self.imgItem_enface_interp_line)
        self.imgView_enface.addItem(self.enface_plot)
        # add colorbar to bframe B
        self.enfaceA_cmpbar.setImageItem(self.imgItem_enface)
        self.ui.verticalLayout_3.insertWidget(1, self.imgView_enface, 1)
        
        # 3D view
        self.glView = QVTKRenderWindowInteractor(self)
        self.cam_orient_manipulator = vtkCameraOrientationWidget()
        self.glView.GetRenderWindow().GetInteractor().SetInteractorStyle(
            vtkInteractorStyleTrackballCamera()
        )
        self.curve_vtk_actor = vtkActor()
        self.curve_vtk_mapper = vtkPolyDataMapper()
        self.curve_vtk_actor.SetMapper(self.curve_vtk_mapper)

        self.curve_interp_vtk_actor = vtkActor()
        self.curve_interp_vtk_mapper = vtkPolyDataMapper()
        self.curve_interp_vtk_actor.SetMapper(self.curve_interp_vtk_mapper)
        # set transparency
        self.curve_interp_vtk_actor.GetProperty().SetOpacity(0.5)

        self.vtkRender = vtkRenderer()
        self.vtkRender.SetBackground(0.17, 0.17, 0.17)
        self.glView.GetRenderWindow().AddRenderer(self.vtkRender)
        self.cam_orient_manipulator.SetParentRenderer(self.vtkRender)
        self.cam_orient_manipulator.On()

        self.vtkRender.AddActor(self.curve_vtk_actor)
        self.vtkRender.AddActor(self.curve_interp_vtk_actor)
        self.vtkRender.ResetCameraScreenSpace()

        self.glView.Initialize()
        self.glView.Start()
        self.ui.verticalLayout_3.insertWidget(3, self.glView, 1)

        self.root = self.parentWindow.app_context.root_path

        self.ui.doubleSpinBox_bframeA_cmbar_low.valueChanged.connect(self.onDoubleSpinBox_bframeA_cmbar_low_changed)
        self.ui.doubleSpinBox_bframeA_cmbar_high.valueChanged.connect(self.onDoubleSpinBox_bframeA_cmbar_heigh_changed)
        self.ui.doubleSpinBox_bframeB_cmbar_low.valueChanged.connect(self.onDoubleSpinBox_bframeB_cmbar_low_changed)
        self.ui.doubleSpinBox_bframeB_cmbar_high.valueChanged.connect(self.onDoubleSpinBox_bframeB_cmbar_heigh_changed)
        self.ui.doubleSpinBox_enfaceA_cmbar_low.valueChanged.connect(self.onDoubleSpinBox_enfaceA_cmbar_low_changed)
        self.ui.doubleSpinBox_enfaceA_cmbar_high.valueChanged.connect(self.onDoubleSpinBox_enfaceA_cmbar_heigh_changed)
        self.ui.pushButton_layoutdirection.clicked.connect(self.onChangeBframeLayout)
        self.ui.pushButton_bframe_type.clicked.connect(self.onChangeBframeType)

        self.ui_pushButton_pvd_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_ilm_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_nflgcl_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_gclipl_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_iplinl_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_inlopl_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_oplonl_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_elm_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_ez_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_eziz_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_izrpe_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_rpebm_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_sathal_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)
        self.ui_pushButton_choroid_color.sigColorChanged.connect(self.on_changedLayerBoundryStyle)

        self.controlPanel.comboBox_ilm_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_nflgcl_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_gclipl_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_iplinl_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_inlopl_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_oplonl_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_elm_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_ez_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_eziz_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_izrpe_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_rpebm_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_sathal_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)
        self.controlPanel.comboBox_choroid_line.currentTextChanged.connect(self.on_changedLayerBoundryStyle)

        self.controlPanel.checkBox_pvd.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_ilm.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_nflgcl.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_gclipl.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_iplinl.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_inlopl.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_oplonl.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_elm.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_ez.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_eziz.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_izrpe.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_rpebm.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_sathal.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_choroid.stateChanged.connect(self.on_changedLayerVisibility)
        self.controlPanel.checkBox_all.stateChanged.connect(self.on_changedAllLayerVisibility)

        # define the curve_ui_mapping
        self.curve_ui_mapping = {
            "PVD": {
                "color": self.ui_pushButton_pvd_color,
                "line": self.controlPanel.comboBox_pvd_line,
                "visible": self.controlPanel.checkBox_pvd,
            },
            "ILM": {
                "color": self.ui_pushButton_ilm_color,
                "line": self.controlPanel.comboBox_ilm_line,
                "visible": self.controlPanel.checkBox_ilm,
            },
            "NFLGCL": {
                "color": self.ui_pushButton_nflgcl_color,
                "line": self.controlPanel.comboBox_nflgcl_line,
                "visible": self.controlPanel.checkBox_nflgcl,
            },
            "GCLIPL": {
                "color": self.ui_pushButton_gclipl_color,
                "line": self.controlPanel.comboBox_gclipl_line,
                "visible": self.controlPanel.checkBox_gclipl,
            },
            "IPLINL": {
                "color": self.ui_pushButton_iplinl_color,
                "line": self.controlPanel.comboBox_iplinl_line,
                "visible": self.controlPanel.checkBox_iplinl,
            },
            "INLOPL": {
                "color": self.ui_pushButton_inlopl_color,
                "line": self.controlPanel.comboBox_inlopl_line,
                "visible": self.controlPanel.checkBox_inlopl,
            },
            "OPLONL": {
                "color": self.ui_pushButton_oplonl_color,
                "line": self.controlPanel.comboBox_oplonl_line,
                "visible": self.controlPanel.checkBox_oplonl,
            },
            "ELM": {
                "color": self.ui_pushButton_elm_color,
                "line": self.controlPanel.comboBox_elm_line,
                "visible": self.controlPanel.checkBox_elm,
            },
            "EZ": {
                "color": self.ui_pushButton_ez_color,
                "line": self.controlPanel.comboBox_ez_line,
                "visible": self.controlPanel.checkBox_ez,
            },
            "EZIZ": {
                "color": self.ui_pushButton_eziz_color,
                "line": self.controlPanel.comboBox_eziz_line,
                "visible": self.controlPanel.checkBox_eziz,
            },
            "IZRPE": {
                "color": self.ui_pushButton_izrpe_color,
                "line": self.controlPanel.comboBox_izrpe_line,
                "visible": self.controlPanel.checkBox_izrpe,
            },
            "RPEBM": {
                "color": self.ui_pushButton_rpebm_color,
                "line": self.controlPanel.comboBox_rpebm_line,
                "visible": self.controlPanel.checkBox_rpebm,
            },
            "SATHAL": {
                "color": self.ui_pushButton_sathal_color,
                "line": self.controlPanel.comboBox_sathal_line,
                "visible": self.controlPanel.checkBox_sathal,
            },
            # "HALLER": {"color": self.ui_pushButton_haller_color, "line": self.ui.comboBox_haller_line, "visible": self.ui.checkBox_haller},
            "CHOROID": {
                "color": self.ui_pushButton_choroid_color,
                "line": self.controlPanel.comboBox_choroid_line,
                "visible": self.controlPanel.checkBox_choroid,
            },
        }
        self.ui.comboBox_enfaceA_Slab.addItems(self.curve_ui_mapping.keys())
        self.ui.comboBox_enfaceA_Slab.currentTextChanged.connect(
            self.onEnfaceA_Slab_changed
        )
        self.ui.comboBox_enfaceA_Slab.setCurrentText("ILM")
        
        # Subscribe to theme changes
        if self.app_context:
            self.app_context.event_bus.subscribe('theme_changed', self._on_theme_changed)

    def get_closest_within(self, arr, v, d):
        """
        Get up to two closest elements around v within distance D.
        Returns [a, v], [v, b], or [a, v, b].

        Parameters:
            arr (array-like): Input array
            v (float): Target value
            D (float): Max allowed distance

        Returns:
            list[float]: New array [a, v, b] (depending on available neighbors)
        """
        arr = np.asarray(arr)
        diff = arr - v

        # Elements within distance d
        mask = np.abs(diff) <= d
        nearby = arr[mask]

        if nearby.size == 0:
            return [v]  # no close elements

        smaller = nearby[nearby < v]
        larger = nearby[nearby > v]

        a = smaller[np.argmax(smaller)] if smaller.size > 0 else None  # closest smaller
        b = larger[np.argmin(larger)] if larger.size > 0 else None  # closest larger

        if a is not None and b is not None:
            return [a, v, b]
        elif a is not None:
            return [a, v]
        elif b is not None:
            return [v, b]
        else:
            return [v]
    
    def _update_interp_regions(self, interp_range: list) -> None:
        """Update interpolation regions for curve editing.

        Args:
            interp_range (list): [curve_name, frameIdx, low, high]
        """
        # check if the interp_range is the same as previous one
        # if (self._previous_interp_range == interp_range):
        #     print("Same interp range as previous one, skip interpolation.")
        #     return
        
        if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
            curve_ = (
                self.parentWindow.gCurve_data["curves"][interp_range[0]]
                + self.parentWindow.gCurve_data["flatten_offset"]
            )
        else:
            curve_ = self.parentWindow.gCurve_data["curves"][interp_range[0]]
        # interp_range=[curve_name,frameIdx,low,high]
        self.interp_key_curve_ranges[interp_range[0]][
            interp_range[1], interp_range[2] : interp_range[3]
        ] = 1
      
        for i in range(self.interp_key_curve_ranges[interp_range[0]].shape[1]):
            c = self.interp_key_curve_ranges[interp_range[0]][:, i]
            # find the index of non-zero values
            idx = np.where(c > 0)[0]
            idx = self.get_closest_within(idx, interp_range[1], self.controlPanel.spinBox_Interp_step.value())
            if len(idx) > 1:
                # print('i:',i)
                # print('idx:',idx)
                # print("interp_range:",interp_range)
                # print('interp curve curve:',self.parentWindow.gCurve_data['curves'][interp_range[0]][i,idx[0]:idx[1]+1])
                for j in range(len(idx) - 1):
                    if (
                        idx[j + 1] - idx[j]
                        <= self.controlPanel.spinBox_Interp_step.value()
                    ):
                        # interplote between idx[j] and idx[j+1]
                        xs = [0, idx[j + 1] - idx[j]]
                        ys = [curve_[idx[j], i], curve_[idx[j + 1], i]]
                        # print(xs,ys)
                        cs = PchipInterpolator(xs, ys)
                        # self.interploated_curve[interp_range[0]][i,idx[j]:idx[j+1]+1] = cs(range(0,idx[j+1]-idx[j]+1))
                        curve_[idx[j] : idx[j + 1] + 1, i] = cs(
                            range(0, idx[j + 1] - idx[j] + 1)
                        )

        if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
            self.parentWindow.gCurve_data["curves"][interp_range[0]] = (
                curve_ - self.parentWindow.gCurve_data["flatten_offset"]
            )
        else:
            self.parentWindow.gCurve_data["curves"][interp_range[0]] = curve_

        self._refreshGUI(
            updateItems=UpdateUIItems(
                enfaceA_interp=True, enfaceA_img=True, enfaceB_img=True
            )
        )
        
        
    def _create_progress_dialog(
        self, title: str, text: str, cancelable: bool = False, cancle_callback=None
    ) -> QProgressDialog:
        """Create and return a progress dialog.

        Args:
            title (str): Dialog title.
            text (str): Dialog text.
            cancelable (bool, optional): Whether the dialog can be canceled.
            cancle_callback (callable, optional): Callback for cancel event.

        Returns:
            QProgressDialog: The created progress dialog.
        """
        msg = QProgressDialog(text, None, 0, 100, self)
        msg.setWindowTitle(title)
        msg.setWindowModality(Qt.WindowModality.WindowModal)
        msg.canceled.connect(cancle_callback)
        msg.show()
        QCoreApplication.processEvents()
        msg.setValue(0)

        return msg

    def _update_progress_dialog(
        self, msg: QProgressDialog, value: int, text: str = None
    ) -> None:
        """Update the progress dialog with a new value and optional text.

        Args:
            msg (QProgressDialog): The progress dialog.
            value (int): Progress value.
            text (str, optional): Optional label text.
        """
        msg.setValue(value)
        if text is not None:
            msg.setLabelText(text)
        QCoreApplication.processEvents()
        if value == 100:
            msg.close()
    
    # @Slot()
    # def clean_up(self) -> None:
    #     """
    #     Clear and re-initialize any loaded OCT/OCTA data and related visualization
    #     items so the LayerSegmentation returns to a clean state.

    #     This resets internal data attributes, clears image items/plots (if present), and
    #     triggers a UI refresh.
    #     """
    #     # Ask user for confirmation before clearing
    #     reply = QMessageBox.question(
    #         self,
    #         "Confirm Clear",
    #         "Clear loaded data and reset Layer Segmentation? This will remove current segmentation and images.",
    #         QMessageBox.Yes | QMessageBox.No,
    #         QMessageBox.No,
    #     )
    #     if reply == QMessageBox.No:
    #         return

    #     # Core data
    #     self.oct_data = None
    #     self.octa_data = None
    #     self.curve_data = None
    #     self.oct_data_raw = None
    #     self.octa_data_raw = None
    #     self.oct_data_flatten = None
    #     self.octa_data_flatten = None
    #     self.oct_data_flatten_raw = None
    #     self.octa_data_flatten_raw = None
    #     self.curve_data_dict = None
    #     self.all_fluid_data = None

    #     # Flattening / slab state
    #     self.flatten_offset = 0
    #     self.flatten_baseline = -1
    #     self.flatten_permute = (0, 1, 2)

    #     # Ranges / indicators
    #     self.oct_data_range = [0, 1]
    #     self.octa_data_range = [0, 1]
    #     self.indicatorDirection = 1

    #     # Resolution / scan params
    #     self.resolution_width = 1
    #     self.scan_width_mm = 1
    #     self.resolution_height = 1
    #     self.scan_height_mm = 1
    #     self.resolution_depth = 1

    #     # Enface / slab boundaries
    #     self.upper_boundry = None
    #     self.lower_boundry = None
    #     self.slab_mask = None
    #     self.enfaceViewer_enfaceImageA = None
    #     self.enfaceViewer_enfaceImageB = None
    #     self.upper_boundry_min = 0
    #     self.lower_boundry_max = 0

    #     # Try clearing pyqtgraph image items and plots if they exist
    #     def try_clear(item):
    #         try:
    #             if item is None:
    #                 return
    #             # many pyqtgraph ImageItem support setImage(None)
    #             if hasattr(item, 'setImage'):
    #                 try:
    #                     item.setImage(None)
    #                     return
    #                 except Exception:
    #                     pass
    #             if hasattr(item, 'clear'):
    #                 try:
    #                     item.clear()
    #                     return
    #                 except Exception:
    #                     pass
    #         except Exception:
    #             pass

    #     try_clear(getattr(self, 'imgItem_bframe_oct_a', None))
    #     try_clear(getattr(self, 'imgItem_bframe_octa_a', None))
    #     try_clear(getattr(self, 'imgItem_bframe_oct_b', None))
    #     try_clear(getattr(self, 'imgItem_enface', None))
    #     try_clear(getattr(self, 'imgItem_enface_interp_line', None))
    #     try_clear(getattr(self, 'imgItem_enface_a', None))
    #     try_clear(getattr(self, 'imgItem_enface_b', None))

    #     # Clear Indicator/Curve plots where possible
    #     try:
    #         if getattr(self, 'bframe_plot_a', None) is not None:
    #             try:
    #                 self.bframe_plot_a.clear()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     try:
    #         if getattr(self, 'bframe_plot_b', None) is not None:
    #             try:
    #                 self.bframe_plot_b.clear()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     try:
    #         if getattr(self, 'enface_plot', None) is not None:
    #             try:
    #                 self.enface_plot.clear()
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     # Reset VTK actors/mappers (do not destroy render window here)
    #     try:
    #         if getattr(self, 'curve_vtk_mapper', None) is not None:
    #             try:
    #                 # detach mapper from actor
    #                 if getattr(self, 'curve_vtk_actor', None) is not None:
    #                     try:
    #                         self.curve_vtk_actor.SetMapper(None)
    #                     except Exception:
    #                         pass
    #             except Exception:
    #                 pass
    #     except Exception:
    #         pass

    #     # Refresh GUI if available
    #     try:
    #         self._refreshGUI(updateItems=UpdateUIItems(all=True))
    #     except Exception:
    #         pass

    def _on_theme_changed(self, event_data: dict) -> None:
        """Handle theme change events from the event bus.
        
        Args:
            event_data (dict): Event data containing theme information.
        """
        # Update theme attribute
        theme_id = event_data.get('theme_id', 'dark')
        self.theme = theme_id
        
        # Update all ColorBarItem menu stylesheets
        if self.theme == "dark":
            stylesheet = "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
        else:
            stylesheet = "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
        
        # Apply stylesheet to all ColorBarItem menus
        if hasattr(self, 'bframeA_cmpbar') and self.bframeA_cmpbar:
            self.bframeA_cmpbar.colorMapMenu.setStyleSheet(stylesheet)
        if hasattr(self, 'bframeB_cmpbar') and self.bframeB_cmpbar:
            self.bframeB_cmpbar.colorMapMenu.setStyleSheet(stylesheet)
        if hasattr(self, 'enfaceA_cmpbar') and self.enfaceA_cmpbar:
            self.enfaceA_cmpbar.colorMapMenu.setStyleSheet(stylesheet)

    def keyReleaseEvent(self, event) -> None:
        """Handle key release events for navigation.

        Args:
            event: The key event.
        """
        if event.key() == Qt.Key.Key_Down:
            idx = min(
                self.parentWindow.ui.spinBox_frameIdx.value() + 1,
                self.parentWindow.ui.spinBox_frameIdx.maximum(),
            )
            self.parentWindow.ui.spinBox_frameIdx.setValue(idx)
        elif event.key() == Qt.Key.Key_Up:
            idx = max(
                self.parentWindow.ui.spinBox_frameIdx.value() - 1,
                self.parentWindow.ui.spinBox_frameIdx.minimum(),
            )
            self.parentWindow.ui.spinBox_frameIdx.setValue(idx)
        if event.key() == Qt.Key.Key_Right:
            # check if the current frame is the last frame
            idx = min(
                self.parentWindow.ui.spinBox_frameIdx.value()
                + self.controlPanel.spinBox_Interp_step.value(),
                self.parentWindow.ui.spinBox_frameIdx.maximum(),
            )
            self.parentWindow.ui.spinBox_frameIdx.setValue(idx)
        elif event.key() == Qt.Key.Key_Left:
            idx = max(
                self.parentWindow.ui.spinBox_frameIdx.value()
                - self.controlPanel.spinBox_Interp_step.value(),
                self.parentWindow.ui.spinBox_frameIdx.minimum(),
            )
            self.parentWindow.ui.spinBox_frameIdx.setValue(idx)
        # super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        """Handle the close event for the widget.

        Args:
            event: The close event (can be None for programmatic cleanup).
        """
        try:
            if hasattr(self, 'glView') and self.glView:
                # Ensure proper VTK cleanup
                render_window = self.glView.GetRenderWindow()
                if render_window:
                    try:
                        # Make the OpenGL context current before finalization
                        render_window.MakeCurrent()
                    except Exception:
                        pass  # Context might already be invalid
                    
                    # Finalize the render window
                    try:
                        self.glView.Finalize()
                    except Exception as e:
                        logger.warning("VTK finalization error in LayerSegmentation: %s", e)
        except Exception as e:
            logger.exception("Error during LayerSegmentation closeEvent: %s", e)
        finally:
            # Only call super().closeEvent if event is not None
            # When event is None, it's a programmatic cleanup and we shouldn't
            # call the parent's closeEvent as it expects a valid QCloseEvent object
            if event is not None:
                super().closeEvent(event)

    def onTabChanged(self, autoRange: bool = False, roi_update: bool = False) -> None:
        """Handle tab changes and update the UI accordingly.

        Args:
            autoRange (bool, optional): Whether to auto-range the plots.
            roi_update (bool, optional): Whether ROI was updated.
        """
        if roi_update:
            self._refreshGUI(updateItems=UpdateUIItems(bframeA_curve=True))
        else:
            self._refreshGUI(updateItems=UpdateUIItems(all=True))
        if autoRange:
            self.bframe_plot_a.vb.autoRange()
            self.bframe_plot_b.vb.autoRange()
            self.enface_plot.vb.autoRange()
    
    def onResetCamera(self) -> None:
        """Reset the camera in the 3D view."""
        self.vtkRender.ResetCameraScreenSpace()

    @Slot()
    def onHideLeft(self) -> None:
        """Hide or show the left panel in the UI."""
        self.ui.pushButton_hide_right.setHidden(
            not self.ui.pushButton_hide_right.isHidden()
        )
        self.ui.pushButton_layoutdirection.setHidden(
            not self.ui.pushButton_layoutdirection.isHidden()
        )
        self.ui.doubleSpinBox_bframeA_cmbar_low.setHidden(
            not self.ui.doubleSpinBox_bframeA_cmbar_low.isHidden()
        )
        self.ui.doubleSpinBox_bframeA_cmbar_high.setHidden(
            not self.ui.doubleSpinBox_bframeA_cmbar_high.isHidden()
        )
        self.gr_wid_bf_a.setHidden(not self.gr_wid_bf_a.isHidden())
        self.bframeA_cmpbar.setVisible(not self.bframeA_cmpbar.isVisible())
        self.imgView_bframe_a.setHidden(not self.imgView_bframe_a.isHidden())
        if self.ui.pushButton_hide_right.isHidden():
            self.ui.pushButton_hide_left.setIcon(self.getIcon("collapse"))
        else:
            self.ui.pushButton_hide_left.setIcon(self.getIcon("expand"))

    @Slot()
    def onHideRight(self) -> None:
        """Hide or show the right panel in the UI."""
        self.ui.pushButton_hide_left.setHidden(
            not self.ui.pushButton_hide_left.isHidden()
        )
        self.ui.pushButton_bframe_type.setHidden(
            not self.ui.pushButton_bframe_type.isHidden()
        )
        self.ui.doubleSpinBox_bframeB_cmbar_low.setHidden(
            not self.ui.doubleSpinBox_bframeB_cmbar_low.isHidden()
        )
        self.ui.doubleSpinBox_bframeB_cmbar_high.setHidden(
            not self.ui.doubleSpinBox_bframeB_cmbar_high.isHidden()
        )
        self.gr_wid_bf_b.setHidden(not self.gr_wid_bf_b.isHidden())
        self.bframeB_cmpbar.setVisible(not self.bframeB_cmpbar.isVisible())
        self.imgView_bframe_b.setHidden(not self.imgView_bframe_b.isHidden())

        if self.ui.pushButton_hide_left.isHidden():
            self.ui.pushButton_hide_right.setIcon(self.getIcon("collapse"))
        else:
            self.ui.pushButton_hide_right.setIcon(self.getIcon("expand"))

    @Slot()
    def onGPU_changed(self) -> None:
        """Handle GPU selection changes."""
        # set the GPU index
        self.ort_session = None

    @Slot()
    def onSparseCheckBoxChanged(self) -> None:
        """Handle changes to the sparse checkbox."""
        self.ort_session = None
    
    @Slot()
    def on_AISeg(self) -> None:
        """Perform automatic segmentation using the AI model."""
        if self.parentWindow.oct_data is None:
            self.parentWindow.ui.statusbar.showMessage("No OCT data to segment", 5000)
            return
        if self.parentWindow.gCurve_data is not None and (
            np.any(self.parentWindow.gCurve_data["curves"]["RPEBM"])
            or np.any(self.parentWindow.gCurve_data["curves"]["ILM"])
        ):
            # ask user if user wants to replace the current curve data
            reply = QMessageBox.question(
                self,
                "AI Segmentation",
                "Do you want to cover the current curve data?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        input_channels = 3 if not self.controlPanel.checkBox_sparse.isChecked() else 1
        msg = self._create_progress_dialog(
            "AI Layer Segmentation", "Loading AI model..."
        )
        if self.ort_session is None:
            self.parentWindow.ui.statusbar.showMessage("Loading AI model...", 5000)
            self.parentWindow.ui.statusbar.repaint()
            try:
                self._update_progress_dialog(msg, 10, "Loading AI model...")
                # Get model file path using reassembly module
                if input_channels == 1:
                    # Get extension directory for sparse model
                    extension_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    model_path = os.path.join(extension_dir, "model", "layersegmodel_sparse.enc")
                else:
                    # Use reassembly module for main model
                    model_path_obj = get_model_file_path()
                    if model_path_obj is None:
                        raise RuntimeError(
                            "Failed to load model file. The model may not have been "
                            "properly reassembled from chunks. Please reinstall the package."
                        )
                    model_path = str(model_path_obj)
                
                model_buffer = utils.loadDLModel(model_path)
                self.ort_session = create_onnx_session(model_buffer, device_id=self.controlPanel.comboBox_gpu.currentIndex(), prefer_gpu=True, optimization_level="basic")
                
            except:
                self._update_progress_dialog(msg, 99, "Load AI model failed!")
                self.parentWindow.ui.statusbar.showMessage(
                    "Failed to load AI Model", 5000
                )
                self.controlPanel.pushButton_ai_seg.setEnabled(False)
                return
            self.parentWindow.ui.statusbar.showMessage("AI model Loaded", 5000)

        if self.parentWindow.oct_data is None:
            self.parentWindow.ui.statusbar.showMessage("No OCT data to segment", 5000)
        else:
            if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
                octdata_ = self.parentWindow.oct_data_flatten
                is_flatten = True
            else:
                octdata_ = self.parentWindow.oct_data
                is_flatten = False
            self._update_progress_dialog(msg, 30, "Preparing data...")
            self.parentWindow.ui.statusbar.showMessage("Runing...")
            self.parentWindow.ui.statusbar.repaint()
            
            ##################################################################################
            # check the size of the octdata, if it is larger than 1000, resize it to half
            raw_size = octdata_.shape
            flatten_offset = (
                self.parentWindow.flatten_offset.copy()
                if isinstance(self.parentWindow.flatten_offset, np.ndarray)
                else self.parentWindow.flatten_offset
            )
            resized = False
            if octdata_.shape[0] >= 640 and octdata_.shape[2] >= 640:
                # reshze 3D data
                output_shape = [raw_size[0] // 2, raw_size[1], raw_size[2] // 2]
                octdata_ = utils.resize3D(octdata_, output_shape, order=0)
                # check if the flatten offset is a integer
                if not isinstance(flatten_offset, int):
                    flatten_offset = utils.resize3D(
                        flatten_offset, [output_shape[0], output_shape[2]], order=0
                    )
                resized = True
            octdata = utils.mat2gray(octdata_)
            # preprocessing the data
            if self.parentWindow.ui.spinBox_roi_bot.value() == 0:
                octdata_crop, [top, bottom] = utils.cropping_data_for_layerSegment(
                    octdata,
                    min_height=416,
                    is_flatten=is_flatten,
                    flatten_baseline=self.flatten_baseline,
                    flatten_offset=flatten_offset,
                )
                self.parentWindow.ui.spinBox_roi_top.setValue(top)
                self.parentWindow.ui.spinBox_roi_bot.setValue(bottom)
            else:
                top = self.parentWindow.ui.spinBox_roi_top.value()
                bottom = self.parentWindow.ui.spinBox_roi_bot.value()
                octdata_crop = octdata[:, top:bottom, :]

            self._update_progress_dialog(msg, 40, " Runing segmentation...")

            layerSegMat = utils.runModel(self.ort_session, octdata_crop,input_shape=[None,None, input_channels])
            permute = self.parentWindow.ui.comboBox_permute.currentText()
            flip = self.parentWindow.ui.comboBox_flip.currentText()

            self._update_progress_dialog(msg, 90, " Generating layer boundaries...")
            keys = utils.generateCurveDictKey(permute, flip)

            curve_data, self.all_fluid_data, _ = utils.generateCurve(
                layerSegMat,
                permute,
                flip,
                flatten_offset,
                self.parentWindow.curve_data_dict[keys]["flatten_baseline"],
                is_flatten,
                top,
            )
            # curve_data, self.all_fluid_data, _ = utils.generateCurveMultiSegMats(
            #     layerSegMat,
            #     permute,
            #     flip,
            #     flatten_offset,
            #     self.parentWindow.curve_data_dict[keys]["flatten_baseline"],
            #     is_flatten,
            #     top,
            # )
            self.all_fluid_data = np.pad(self.all_fluid_data,((0, 0), (top, raw_size[1] - bottom), (0, 0)),mode="constant",constant_values=0)
            
            if is_flatten:
                self.all_fluid_data = utils.roll_volume_inverse(self.all_fluid_data, flatten_offset)
            curve_data["volumeSize"] = self.parentWindow.curve_data_dict[keys]["volumeSize"]
            curve_data["flatten_offset"] = self.parentWindow.flatten_offset
            if resized:
                curve_data = utils.resizeCurveData(curve_data, raw_size)
                self.all_fluid_data = utils.resize3D(self.all_fluid_data, raw_size, order=0)

            ########################################################################
            
            self.parentWindow.curve_data_dict[keys] = curve_data
            self.parentWindow.gCurve_data = curve_data
            self.parentWindow.gRF_labelVolume = self.all_fluid_data
            self._update_progress_dialog(msg, 100, " Runing segmentation...")
            self.parentWindow.ui.statusbar.showMessage("Segmentation Done", 5000)
            self._refreshGUI(updateItems=UpdateUIItems(all=True))

    @Slot()
    def onTranspose(self) -> None:
        """Toggle transpose mode for the view."""
        self.isTranspose = not self.isTranspose
        self.indicatorDirection = 0 if self.isTranspose else 1
        self.ui.pushButton_transp.setIcon(
            self.getIcon("transpose-on" if self.isTranspose else "transpose-off")
        )
        self.ui.pushButton_transp.setToolTip(
            "Transpose ON" if self.isTranspose else "Transpose OFF"
        )
        if self.isTranspose:
            self.default_camera_position = [557.52, 331.38, 777.95]
            self.default_camera_focal_point = [160.41, 141.85, 114.49]
            self.default_camera_view_up = [-0.86, 0.23, 0.45]
        else:
            self.default_camera_position = [
                762.55,
                16.01,
                619.85,
            ]  # 570.08, 447.98, 724.61
            self.default_camera_focal_point = [
                160.41,
                141.85,
                114.49,
            ]  # 160.41, 141.85, 114.49
            self.default_camera_view_up = [-0.65, -0.03, 0.76]  # -0.86, 0.29, 0.43
        self.onResetCamera()
        self._refreshGUI(
            updateItems=UpdateUIItems(
                enfaceA_img=True,
                enfaceA_interp=True,
                enfaceB_img=True,
                enfaceA_indicators=True,
            )
        )
        self.enface_plot.vb.autoRange()

    @Slot()
    def onRunInterpolation(self) -> None:
        """Run interpolation for curve editing."""
        if self.enableInterpolation and self.parentWindow.gCurve_data is not None:
            self.interp_key_curve_ranges = {
                key: np.zeros_like(
                    self.parentWindow.gCurve_data["curves"][key], dtype="uint8"
                )
                for key in self.parentWindow.gCurve_data["curves"].keys()
            }

        self._refreshGUI(updateItems=UpdateUIItems(enfaceA_interp=True))

    @Slot()
    def onSwitchInterpolation(self) -> None:
        """Switch interpolation mode for curve editing."""
        self.enableInterpolation = not self.enableInterpolation
        if self.parentWindow.gCurve_data is not None:
            self.interp_key_curve_ranges = {
                key: np.zeros_like(
                    self.parentWindow.gCurve_data["curves"][key], dtype="uint8"
                )
                for key in self.parentWindow.gCurve_data["curves"].keys()
            }
        self.controlPanel.pushButton_Switch_Interp.setText(
            "Interpolation: " + ("ON" if self.enableInterpolation else "OFF")
        )
        self.controlPanel.pushButton_Switch_Interp.setIcon(
            self.getIcon("ON" if self.enableInterpolation else "OFF")
        )
        self.controlPanel.pushButton_Restart_Interp.setEnabled(self.enableInterpolation)
        self.controlPanel.pushButton_Restart_Interp.setIcon(
            self.getIcon("run" if self.enableInterpolation else "run-disabled")
        )
        self.controlPanel.spinBox_Interp_step.setEnabled(not self.enableInterpolation)

        self._refreshGUI(updateItems=UpdateUIItems(enfaceA_interp=True))

    @Slot()
    def on_changedAllLayerVisibility(self, state) -> None:
        """Set all layers to the same visibility.

        Args:
            state: Visibility state.
        """
        # set all layers to the same visibility
        for key in self.curve_ui_mapping.keys():
            self.curve_ui_mapping[key]["visible"].setChecked(state)

    @Slot()
    def on_changedLayerVisibility(self, state) -> None:
        """Handle changes to individual layer visibility.

        Args:
            state: Visibility state.
        """
        self._refreshGUI(
            updateItems=UpdateUIItems(bframeA_curve=True, bframeB_curve=True)
        )

    @Slot()
    def onCorrector_changed(self, text: str) -> None:
        """Handle changes to the curve corrector.

        Args:
            text (str): Corrector type.
        """
        self.bframe_plot_b.setEditorType(text)

    @Slot()
    def onDoubleSpinBox_enfaceA_cmbar_low_changed(self) -> None:
        """Handle changes to the lower bound of the enface A color map."""
        if (
            self.ui.doubleSpinBox_enfaceA_cmbar_low.value()
            >= self.ui.doubleSpinBox_enfaceA_cmbar_high.value()
        ):
            self.ui.doubleSpinBox_enfaceA_cmbar_low.setValue(
                self.ui.doubleSpinBox_enfaceA_cmbar_high.value() - 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(enfaceA_img=True, enfaceB_img=True))

    @Slot()
    def onDoubleSpinBox_enfaceA_cmbar_heigh_changed(self) -> None:
        """Handle changes to the upper bound of the enface A color map."""
        if (
            self.ui.doubleSpinBox_enfaceA_cmbar_high.value()
            <= self.ui.doubleSpinBox_enfaceA_cmbar_low.value()
        ):
            self.ui.doubleSpinBox_enfaceA_cmbar_high.setValue(
                self.ui.doubleSpinBox_enfaceA_cmbar_low.value() + 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(enfaceA_img=True, enfaceB_img=True))

    @Slot()
    def onDoubleSpinBox_bframeA_cmbar_low_changed(self) -> None:
        """Handle changes to the lower bound of the B-frame A color map."""
        if (
            self.ui.doubleSpinBox_bframeA_cmbar_low.value()
            >= self.ui.doubleSpinBox_bframeA_cmbar_high.value()
        ):
            self.ui.doubleSpinBox_bframeA_cmbar_low.setValue(
                self.ui.doubleSpinBox_bframeA_cmbar_high.value() - 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(bframeA_octa=True))

    @Slot()
    def onDoubleSpinBox_bframeA_cmbar_heigh_changed(self) -> None:
        """Handle changes to the upper bound of the B-frame A color map."""
        if (
            self.ui.doubleSpinBox_bframeA_cmbar_high.value()
            <= self.ui.doubleSpinBox_bframeA_cmbar_low.value()
        ):
            self.ui.doubleSpinBox_bframeA_cmbar_high.setValue(
                self.ui.doubleSpinBox_bframeA_cmbar_low.value() + 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(bframeA_octa=True))

    @Slot()
    def onDoubleSpinBox_bframeB_cmbar_low_changed(self) -> None:
        """Handle changes to the lower bound of the B-frame B color map."""
        if (
            self.ui.doubleSpinBox_bframeB_cmbar_low.value()
            >= self.ui.doubleSpinBox_bframeB_cmbar_high.value()
        ):
            self.ui.doubleSpinBox_bframeB_cmbar_low.setValue(
                self.ui.doubleSpinBox_bframeB_cmbar_high.value() - 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(bframeA_oct=True, bframeB_oct=True))

    @Slot()
    def onDoubleSpinBox_bframeB_cmbar_heigh_changed(self) -> None:
        """Handle changes to the upper bound of the B-frame B color map."""
        if (
            self.ui.doubleSpinBox_bframeB_cmbar_high.value()
            <= self.ui.doubleSpinBox_bframeB_cmbar_low.value()
        ):
            self.ui.doubleSpinBox_bframeB_cmbar_high.setValue(
                self.ui.doubleSpinBox_bframeB_cmbar_low.value() + 0.001
            )
        self._refreshGUI(updateItems=UpdateUIItems(bframeA_oct=True, bframeB_oct=True))

    @Slot()
    def onIndicatorChanged(self, indx: list) -> None:
        """Handle changes to the indicator on the enface plot.

        Args:
            indx (list): Indicator indices.
        """
        self.parentWindow.ui.spinBox_frameIdx.setValue(indx[self.indicatorDirection])
        # print("Indicator Changed to:",indx)

    @Slot()
    def onIndicatorEnfaceAChanged(self, indx: list) -> None:
        """Handle changes to the indicator on enface A.

        Args:
            indx (list): Indicator indices.
        """
        self.parentWindow.ui.spinBox_frameIdx.setValue(
            indx[self.indicatorDirection_enfaceA]
        )
        # print("Indicator Changed to:",indx)

    @Slot()
    def onIndicatorEnfaceBChanged(self, indx: list) -> None:
        """Handle changes to the indicator on enface B.

        Args:
            indx (list): Indicator indices.
        """
        self.parentWindow.ui.spinBox_frameIdx.setValue(
            indx[self.indicatorDirection_enfaceB]
        )
        # print("Indicator Changed to:",indx)

    @Slot()
    def onChangeBframeType(self) -> None:
        """Change the B-frame type between Fast and Slow."""
        if self.bframe_b_type == "Fast":
            self.bframe_b_type = "Slow"
            self.ui.pushButton_bframe_type.setIcon(self.getIcon("slow"))
            self.ui.pushButton_bframe_type.setToolTip("Change to Fast Scan View")

            self.bframe_plot_b.setEnableEditorMenu(False)
            self.controlPanel.comboBox_corrector.setCurrentText("None")
            self.controlPanel.comboBox_corrector.setEnabled(False)
        else:
            self.bframe_b_type = "Fast"
            self.ui.pushButton_bframe_type.setIcon(self.getIcon("fast"))
            self.ui.pushButton_bframe_type.setToolTip("Change to Slow Scan View")
            self.bframe_plot_b.setEnableEditorMenu(True)
            self.controlPanel.comboBox_corrector.setEnabled(True)

        self._refreshGUI(
            updateItems=UpdateUIItems(
                bframeB_oct=True, bframeA_octa=True, bframeB_curve=True
            )
        )

    @Slot()
    def onChangeBframeLayout(self) -> None:
        """Change the layout direction of the B-frame."""
        if self.ui.horizontalLayout_3.direction() == QHBoxLayout.Direction.LeftToRight:
            self.ui.horizontalLayout_3.setDirection(QHBoxLayout.Direction.TopToBottom)
            self.ui.pushButton_layoutdirection.setIcon(self.getIcon("vlayout"))
            self.ui.pushButton_layoutdirection.setToolTip("Change to Horizontal Layout")

        else:
            self.ui.horizontalLayout_3.setDirection(QHBoxLayout.Direction.LeftToRight)
            self.ui.pushButton_layoutdirection.setIcon(self.getIcon("hlayout"))
            self.ui.pushButton_layoutdirection.setToolTip("Change to Vertical Layout")

    @Slot()
    def onColormap_changed(self, cmpbar) -> None:
        """Handle changes to the colormap.

        Args:
            cmpbar: The colorbar item.
        """
        if cmpbar == self.enfaceA_cmpbar:
            self._refreshGUI(updateItems=UpdateUIItems(enfaceB_img=True))
        elif cmpbar == self.bframeA_cmpbar:
            self._refreshGUI(updateItems=UpdateUIItems(bframeA_octa=True))
        elif cmpbar == self.bframeB_cmpbar:
            logger.info("BFrame A: %s", cmpbar.colorMap())

    @Slot()
    def on_changedLayerBoundryStyle(self, btn) -> None:
        """Handle changes to the layer boundary style.

        Args:
            btn: The color button.
        """
        self._refreshGUI(
            updateItems=UpdateUIItems(bframeA_curve=True, bframeB_curve=True)
        )

    @Slot()
    def onEnfaceA_Slab_changed(self) -> None:
        """Handle changes to the selected slab for enface A."""
        self._refreshGUI(
            updateItems=UpdateUIItems(
                enfaceA_img=True, enfaceA_interp=True, enfaceB_img=True
            )
        )

    @Slot()
    def onPermute_changed(self, state) -> None:
        """Handle changes to the permute state.

        Args:
            state: The new state.
        """
        self._refreshGUI(updateItems=UpdateUIItems())

    @Slot()
    def onFlip_changed(self, state) -> None:
        """Handle changes to the flip state.

        Args:
            state: The new state.
        """
        self._refreshGUI(updateItems=UpdateUIItems())

    @Slot()
    def _on_SegData_Edited(self, idx: int, curve: np.ndarray) -> None:
        """Handle edits to segmentation data.

        Args:
            idx (int): Curve index.
            curve (np.ndarray): Edited curve data.
        """
        if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
            if isinstance(self.parentWindow.gCurve_data["flatten_offset"], (int, float)):
                offsets = self.parentWindow.gCurve_data["flatten_offset"]
            else:
                offsets = self.parentWindow.gCurve_data["flatten_offset"][
                    self.parentWindow.ui.spinBox_frameIdx.value() - 1
                ]
        else:
            offsets = 0
        keys = list(self.parentWindow.gCurve_data["curves"].keys())
        self.ui.comboBox_enfaceA_Slab.setCurrentText(keys[idx])
        self.parentWindow.gCurve_data["curves"][keys[idx]][
            self.parentWindow.ui.spinBox_frameIdx.value() - 1
        ] = (curve - offsets)
        self._refreshGUI(
            updateItems=UpdateUIItems(
                bframeA_curve=True,
                enfaceA_img=True,
                enfaceA_interp=True,
                enfaceB_img=True,
            )
        )

    @Slot()
    def _on_CurveEdit_range_changed(self, idx: int, bounds: list) -> None:
        """Handle changes to the curve edit range.

        Args:
            idx (int): Curve index.
            bounds (list): Range bounds.
        """
        if self.enableInterpolation:
            self._update_interp_regions(
                [
                    self.ui.comboBox_enfaceA_Slab.currentText(),
                    self.parentWindow.ui.spinBox_frameIdx.value() - 1,
                    bounds[0],
                    bounds[1],
                ]
            )

    @Slot()
    def _on_CurveEditorType_Changed(self, editor_type: str) -> None:
        """Handle changes to the curve editor type.

        Args:
            editor_type (str): Editor type.
        """
        self.controlPanel.comboBox_corrector.blockSignals(True)
        self.controlPanel.comboBox_corrector.setCurrentText(editor_type.capitalize())
        self.controlPanel.comboBox_corrector.blockSignals(False)

    def _create_surface_from_image(self, img: np.ndarray):
        """Create a VTK surface from a 2D image.

        Args:
            img (np.ndarray): 2D array representing the image.

        Returns:
            vtkPolyData: VTK PolyData object representing the surface.
        """
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

    def _refreshGUI(self, updateItems: UpdateUIItems = UpdateUIItems()) -> None:
        """Refresh the GUI elements based on which items need updating.

        Args:
            updateItems (UpdateUIItems): Specifies which UI elements to update.
        """
        oct_a = None
        oct_cmp_low = self.ui.doubleSpinBox_bframeB_cmbar_low.value() * 255
        oct_cmp_high = self.ui.doubleSpinBox_bframeB_cmbar_high.value() * 255
        octa_cmp_low = self.ui.doubleSpinBox_bframeA_cmbar_low.value() * 255
        octa_cmp_high = self.ui.doubleSpinBox_bframeA_cmbar_high.value() * 255

        if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
            oct_data = self.parentWindow.oct_data_flatten
            octa_data = self.parentWindow.octa_data_flatten
        else:
            oct_data = self.parentWindow.oct_data
            octa_data = self.parentWindow.octa_data

        curve_data = self.parentWindow.gCurve_data

        if updateItems.bframeA_oct:
            if oct_data is not None:
                oct_a = oct_data[self.parentWindow.ui.spinBox_frameIdx.value() - 1]
                oct_a = utils.mat2gray(oct_a, [oct_cmp_low, oct_cmp_high]) * 255
                self.imgItem_bframe_oct_a.setImage(oct_a, levels=[0, 255])
            else:
                self.imgItem_bframe_oct_a.clear()

        if updateItems.bframeA_octa:
            if octa_data is not None and self.parentWindow.ui.checkBox_OCTA.isChecked():
                octa_img = octa_data[self.parentWindow.ui.spinBox_frameIdx.value() - 1]
                octa_img = utils.mat2gray(octa_img, [octa_cmp_low, octa_cmp_high]) * 255
                lut = self.bframeA_cmpbar.colorMap().getLookupTable(nPts=256)
                octa_img_c = lut[octa_img.astype(np.uint8)]
                # add alpha channel to octa_img_c
                rgba = np.concatenate(
                    (octa_img_c, np.expand_dims(octa_img, axis=-1)), axis=-1
                )
                self.imgItem_bframe_octa_a.setImage(rgba, levels=[0, 255])
            else:
                self.imgItem_bframe_octa_a.clear()

        if updateItems.bframeA_curve:
            self.bframe_plot_a.set_added_items(
                [self.imgItem_bframe_oct_a, self.imgItem_bframe_octa_a]
            )
            if self.parentWindow.gCurve_data is not None:
                curves = []
                curve_props = []
                if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
                    if isinstance(curve_data["flatten_offset"], (int, float)):
                        offsets = curve_data["flatten_offset"]
                    else:
                        offsets = curve_data["flatten_offset"][
                            self.parentWindow.ui.spinBox_frameIdx.value() - 1
                        ]
                else:
                    offsets = 0
                for key in curve_data["curves"]:
                    curve = curve_data["curves"][key][
                        self.parentWindow.ui.spinBox_frameIdx.value() - 1
                    ]
                    curve = curve + (offsets if np.any(curve > 1) else 0)
                    curves.append(curve)
                    lineStyle = (
                        Qt.PenStyle.DashLine
                        if self.curve_ui_mapping[key]["line"].currentIndex() == 0
                        else Qt.PenStyle.SolidLine
                    )
                    curve_props.append(
                        {
                            "color": self.curve_ui_mapping[key]["color"]
                            .color()
                            .getRgb(),
                            "width": 3,
                            "style": lineStyle,
                            "connect": "finite",
                            "is_visible": self.curve_ui_mapping[key][
                                "visible"
                            ].isChecked(),
                        }
                    )

                self.bframe_plot_a.set_curve(curves, curve_props)
                self.bframe_plot_a.set_roi(
                    [
                        self.parentWindow.ui.spinBox_roi_top.value(),
                        self.parentWindow.ui.spinBox_roi_bot.value(),
                    ]
                )
                self.bframe_plot_a.plot()
            else:
                self.bframe_plot_a.clear()

        if updateItems.bframeB_oct:
            if oct_data is not None:
                if self.bframe_b_type == "Fast":
                    # draw oct
                    if oct_a is None:
                        oct_b = oct_data[
                            self.parentWindow.ui.spinBox_frameIdx.value() - 1
                        ]
                        oct_b = utils.mat2gray(oct_b, [oct_cmp_low, oct_cmp_high]) * 255
                    else:
                        oct_b = oct_a
                    self.imgItem_bframe_oct_b.setImage(oct_b, levels=[0, 255])
                else:

                    oct_b = np.transpose(
                        oct_data[
                            :, :, self.parentWindow.ui.spinBox_frameIdx.value() - 1
                        ],
                        [1, 0],
                    )
                    oct_b = utils.mat2gray(oct_b, [oct_cmp_low, oct_cmp_high]) * 255
                    self.imgItem_bframe_oct_b.setImage(oct_b, levels=[0, 255])
            else:
                self.imgItem_bframe_oct_b.clear()

        if updateItems.bframeB_curve:
            self.bframe_plot_b.set_added_items(self.imgItem_bframe_oct_b)
            if curve_data is not None:
                curves = []
                curve_props = []
                if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
                    offsets = curve_data["flatten_offset"]
                else:
                    offsets = 0

                for key in curve_data["curves"]:
                    curve = curve_data["curves"][key]
                    curve = curve + (offsets if np.any(curve > 1) else 0)
                    if self.bframe_b_type == "Fast":
                        curve = curve[self.parentWindow.ui.spinBox_frameIdx.value() - 1]
                    else:
                        curve = curve[
                            :, self.parentWindow.ui.spinBox_frameIdx.value() - 1
                        ]
                    curves.append(curve)
                    lineStyle = (
                        Qt.PenStyle.DashLine
                        if self.curve_ui_mapping[key]["line"].currentIndex() == 0
                        else Qt.PenStyle.SolidLine
                    )
                    curve_props.append(
                        {
                            "color": self.curve_ui_mapping[key]["color"]
                            .color()
                            .getRgb(),
                            "width": 3,
                            "style": lineStyle,
                            "connect": "finite",
                            "is_visible": self.curve_ui_mapping[key][
                                "visible"
                            ].isChecked(),
                        }
                    )

                self.bframe_plot_b.set_curve(curves, curve_props)
                self.bframe_plot_b.plot()
            else:
                self.bframe_plot_b.clear()

        if updateItems.enfaceA_img:
            if curve_data is not None:
                permutes = curve_data["permute"].split(",")
                v_size = [
                    curve_data["volumeSize"][int(permutes[0])],
                    curve_data["volumeSize"][int(permutes[1])],
                    curve_data["volumeSize"][int(permutes[2])],
                ]
                if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
                    offsets = curve_data["flatten_offset"]
                else:
                    offsets = 0
                # get the current edited curve
                im = curve_data["curves"][self.ui.comboBox_enfaceA_Slab.currentText()]
                im = im + (offsets if np.any(im > 1) else 0)
                im = utils.mat2gray(v_size[0] - im) * 255
                if self.isTranspose:
                    im = np.transpose(im, [1, 0])
                self.imgItem_enface.setImage(
                    im,
                    levels=[
                        self.ui.doubleSpinBox_enfaceA_cmbar_low.value() * 255,
                        self.ui.doubleSpinBox_enfaceA_cmbar_high.value() * 255,
                    ],
                )
            else:
                self.imgItem_enface.clear()

        if updateItems.enfaceA_interp:
            if self.interp_key_curve_ranges is not None:
                # get the current edited curve
                im = (
                    self.interp_key_curve_ranges[
                        self.ui.comboBox_enfaceA_Slab.currentText()
                    ]
                    * 255
                )
                if self.isTranspose:
                    im = np.transpose(im, [1, 0])
                im_c = np.stack((im, im * 0, im, im), axis=-1)
                self.imgItem_enface_interp_line.setImage(im_c, levels=[0, 255])
            else:
                self.imgItem_enface_interp_line.clear()

        if updateItems.enfaceA_indicators:
            if curve_data is not None:
                permutes = curve_data["permute"].split(",")
                v_size = [
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[0])],
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[1])],
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[2])],
                ]
                self.imgItem_enface.setLevels(
                    [
                        self.ui.doubleSpinBox_enfaceA_cmbar_low.value() * 255,
                        self.ui.doubleSpinBox_enfaceA_cmbar_high.value() * 255,
                    ]
                )
                self.enface_plot.set_added_items(
                    [self.imgItem_enface, self.imgItem_enface_interp_line]
                )
                # self.enface_plot.setPlotSize([v_size[0],v_size[2]])
                self.enface_plot.setPlotSize([v_size[0], v_size[2]])
                self.enface_plot.setIndicatorPos(
                    [
                        self.parentWindow.ui.spinBox_frameIdx.value(),
                        self.parentWindow.ui.spinBox_frameIdx.value(),
                    ]
                )
                self.enface_plot.setTransposed(self.isTranspose)
                self.enface_plot.setDirection(self.indicatorDirection)
                self.enface_plot.plot()
            else:
                self.enface_plot.clear()

        if updateItems.enfaceB_img:
            if self.parentWindow.gCurve_data is not None:
                permutes = self.parentWindow.gCurve_data["permute"].split(",")
                v_size = [
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[0])],
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[1])],
                    self.parentWindow.gCurve_data["volumeSize"][int(permutes[2])],
                ]
                if self.parentWindow.ui.comboBox_flatten.currentText() != "None":
                    offsets = curve_data["flatten_offset"]
                else:
                    offsets = 0
                lut = vtkLookupTable()
                lut.SetNumberOfTableValues(256)  # Number of colors in the table
                lut.Build()

                lut_ = self.enfaceA_cmpbar.colorMap().getLookupTable(nPts=256)
                lut_array = np.asarray(lut_)
                if lut_array.shape[1] == 3:
                    # Add an alpha component (fully opaque)
                    alpha_channel = (
                        np.ones((lut_array.shape[0], 1), dtype=np.uint8) * 255
                    )
                    lut_array = np.hstack((lut_array, alpha_channel))

                lut.SetTable(numpy_support.numpy_to_vtk(lut_array, deep=True))

                # convert color to Hue, Saturation, Value
                im = curve_data["curves"][self.ui.comboBox_enfaceA_Slab.currentText()]
                im = im + (offsets if np.any(im > 1) else 0)
                im = v_size[0] - im
                if self.isTranspose:
                    im = np.transpose(im, [1, 0])
                surface_data = self._create_surface_from_image(utils.mat2gray(im) * 255)
                self.curve_vtk_mapper.SetInputData(surface_data)
                self.curve_vtk_mapper.SetScalarRange(
                    self.ui.doubleSpinBox_enfaceA_cmbar_low.value(),
                    self.ui.doubleSpinBox_enfaceA_cmbar_high.value(),
                )
                self.curve_vtk_mapper.SetLookupTable(lut)

                self.vtkRender.ResetCameraScreenSpace()
                # refresh the 3D view
                self.glView.GetRenderWindow().Render()
            else:
                self.curve_vtk_mapper.SetInputData(None)
                self.glView.GetRenderWindow().Render()

    def update_camera_parameters(self, caller: Any, event:Any) -> None:
        """Print the current camera parameters.

        Args:
            caller: The camera object.
            event: The event object.
        """
        camera = caller
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        view_up = camera.GetViewUp()

        # Format the camera parameters into a string
        camera_params = (
            f"Position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}\n"
            f"Focal Point: {focal_point[0]:.2f}, {focal_point[1]:.2f}, {focal_point[2]:.2f}\n"
            f"View Up: {view_up[0]:.2f}, {view_up[1]:.2f}, {view_up[2]:.2f}"
        )
        logger.debug(camera_params)

    @staticmethod
    def getIcon(iconName: str) -> QIcon:
        """
        Get the icon from the resource file.

        Args:
            iconName (str): Icon name.

        Returns:
            QIcon:  Icon object.
        """
        def _iconFromBase64(base64):
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
