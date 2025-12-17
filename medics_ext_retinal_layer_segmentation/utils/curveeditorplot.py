"""
Various methods of drawing scrolling plots.
"""

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QGraphicsSceneHoverEvent, QGraphicsSceneMouseEvent, QGraphicsSceneWheelEvent, QFileDialog
import numpy as np
import os
import pyqtgraph as pg
from scipy.interpolate import PchipInterpolator
from copy import deepcopy
import cv2


class CurveEditorPlot(pg.PlotItem):
    """A class for editing and plotting curves with control points in a graphical interface."""

    parentWindow: QtWidgets.QWidget = None
    gCurves: list = None
    curveLen: int = 0
    curve_props: list = None
    curve_props_bk: list = None
    ctrl_points: list = []
    highlight_curve: pg.PlotDataItem = None
    current_curve_idx: int = -1
    focus_control_point_plot: pg.PlotDataItem = None
    modified_left_adjcent_plot: pg.PlotDataItem = None
    modified_right_adjcent_plot: pg.PlotDataItem = None
    indicator_plot: pg.PlotDataItem = None
    roi_position: list = [0, 0]
    curves_plot: list = None
    mouse_path: list = []
    modified_x_boundaries: list = None
    mouse_pressed: bool = False
    modified_ctrl_point: list = None  # [curve_index, point_index, x, y]
    modified_left_adjcent_points: list = []
    modified_right_adjcent_points: list = []
    current_mouse_pos: tuple = None
    added_items: list = None
    distance_t: int = 15
    symbolSize: int = 6
    default_view_range: tuple = None
    get_inverse_color = lambda self, color: (255 - color[0], 255 - color[1], 255 - color[2])
    params_for_ctrl_point_changed_signal: list = None
    sigControlPointChanged = QtCore.Signal(object, object)  # Emitted when the control point has changed
    sigCurveChanged = QtCore.Signal(object, object)  # Emitted when the gCurve has changed
    sigCurveChangeFinished = QtCore.Signal(object, object)  # Emitted when the gCurve has changed
    sigCurveEditorTypeChanged = QtCore.Signal(object, object)  # Emitted when the status of enableEditor has changed
    sigMousePosChanged = QtCore.Signal(object)  # Emitted when the mouse position has changed

    def __init__(
        self,
        parentWindow: QtWidgets.QWidget = None,
        parent: QtWidgets.QWidget = None,
        name: str = None,
        labels: dict = None,
        title: str = None,
        viewBox: pg.ViewBox = None,
        axisItems: dict = None,
        enableMenu: bool = True,
        eidtorType: str = "none",
        update_mode: str = "global",
        enableEditorMenu: bool = True,
        **kargs,
    ) -> None:
        """Initializes the CurveEditorPlot with optional parameters for customization.

        Args:
            parentWindow (QtWidgets.QWidget, optional): The parent window. Defaults to None.
            parent (QtWidgets.QWidget, optional): The parent widget. Defaults to None.
            name (str, optional): Name of the plot. Defaults to None.
            labels (dict, optional): Axis labels. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            viewBox (pg.ViewBox, optional): ViewBox for the plot. Defaults to None.
            axisItems (dict, optional): Axis items for the plot. Defaults to None.
            enableMenu (bool, optional): Whether to enable the menu. Defaults to True.
            eidtorType (str, optional): Type of editor ("none", "liner", "livewire", "anchors"). Defaults to "none".
            update_mode (str, optional): Update mode ("global" or "local"). Defaults to "global".
            enableEditorMenu (bool, optional): Whether to enable the editor menu. Defaults to True.
        """
        super().__init__(parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs)
        self.parentWindow = parentWindow
        self.editorType = eidtorType.lower()
        self.target = pg.TargetItem(
            symbol="s", pen=pg.mkPen((19, 234, 201, 0)), brush=pg.mkBrush((250, 0, 0, 0)), size=1
        )
        self.target.wheelEvent = self.targetWhellEvent
        self.target.setParentItem(self)
        self.target.sigPositionChanged.connect(self.targetMoved)
        self.target.sigPositionChangeFinished.connect(self.mouse_release)
        self.target.mouseClickEvent = self.mouseDoubleClickEvent
        self.target.hide()

        self.update_mode = update_mode

        self.liveWire = cv2.segmentation.IntelligentScissorsMB()
        self.liveWire.setEdgeFeatureCannyParameters(32, 100)
        self.liveWire.setGradientMagnitudeMaxLimit(200)

        self.vb.setDefaultPadding(0.001)
        self.setAcceptHoverEvents(True)

        self.menu = self.vb.getMenu(None)
        for i in range(1, len(self.menu.actions())):
            self.menu.removeAction(self.menu.actions()[1])

        self.ctrlMenu.menuAction().setVisible(False)

        if self.parentWindow.theme == "dark":
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )

        if enableEditorMenu:
            corrector_lin = QtGui.QAction(
                "Corrector:Liner",
                self,
                triggered=lambda: self.setEditorType("Liner" if self.editorType != "liner" else "None"),
            )
            corrector_lin.setCheckable(True)
            corrector_lin.setChecked(self.editorType == "liner")
            self.menu.addAction(corrector_lin)

            corrector_liv = QtGui.QAction(
                "Corrector:LiveWire",
                self,
                triggered=lambda: self.setEditorType("Livewire" if self.editorType != "livewire" else "None"),
            )
            corrector_liv.setCheckable(True)
            corrector_liv.setChecked(self.editorType == "livewire")
            self.menu.addAction(corrector_liv)

            corrector_ank = QtGui.QAction(
                "Corrector:Anchors",
                self,
                triggered=lambda: self.setEditorType("Anchors" if self.editorType != "anchors" else "None"),
            )
            corrector_ank.setCheckable(True)
            corrector_ank.setChecked(self.editorType == "anchors")
            self.menu.addAction(corrector_ank)

        self.setEnableEditorMenu(enableEditorMenu)

        self.menu.addAction("Save image...", self.onSaveImageAs)

    def onSaveImageAs(self) -> None:
        """Saves the current plot as an image file."""
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Image", "", "Image (*.png)")[0]
        if filename:
            if self.added_items is not None:
                for i, item in enumerate(self.added_items):
                    if i > 0:
                        filename = filename.replace(".png", f"_{i}.png")
                    item.save(filename)

    def setEnableEditorMenu(self, enableEditorMenu: bool) -> None:
        """Enables or disables the editor menu.

        Args:
            enableEditorMenu (bool): Whether to enable the editor menu.
        """
        self.enableEditorMenu = enableEditorMenu
        if enableEditorMenu:
            # check if the action is already in the menu
            # get all the actions in the menu
            action_texts = [action.text() for action in self.menu.actions()]

            if "Corrector:Liner" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Liner")].setEnabled(True)
                self.menu.actions()[action_texts.index("Corrector:Liner")].setChecked(self.editorType == 'liner')
            else:
                corrector_lin = QtGui.QAction("Corrector:Liner", self, triggered=lambda: self.setEditorType('Liner' if self.editorType != 'liner' else 'None'))
                corrector_lin.setCheckable(True)
                corrector_lin.setChecked(self.editorType == 'liner')
                self.menu.insertAction(self.menu.actions()[1], corrector_lin) 
                
            if "Corrector:LiveWire" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setEnabled(True)
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setChecked(self.editorType == 'livewire')
            else:
                corrector_lin = QtGui.QAction("Corrector:LiveWire", self, triggered=lambda: self.setEditorType('Livewire' if self.editorType != 'livewire' else 'None'))
                corrector_lin.setCheckable(True)
                corrector_lin.setChecked(self.editorType == 'livewire')
                self.menu.insertAction(self.menu.actions()[2], corrector_lin) 
            
            if "Corrector:Anchors" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setEnabled(True)
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setChecked(self.editorType == 'anchors')
            else:
                corrector_ank = QtGui.QAction("Corrector:Anchors", self, triggered=lambda: self.setEditorType('Anchors' if self.editorType != 'anchors' else 'None'))
                corrector_ank.setCheckable(True)
                corrector_ank.setChecked(self.editorType == 'anchors')
                self.menu.insertAction(self.menu.actions()[3], corrector_ank)
        else:
            for action in self.menu.actions():
                if action.text() == "Corrector:Anchors":
                    action.setEnabled(False)
                    action.setChecked(False)
                elif action.text() == "Corrector:Liner":
                    action.setEnabled(False)
                    action.setChecked(False)
                elif action.text() == "Corrector:LiveWire":
                    action.setEnabled(False)
                    action.setChecked(False)
    
    def set_added_items(self, added_items: list) -> None:
        """Sets additional items to the plot.

        Args:
            added_items (list): List of additional items.
        """
        if not isinstance(added_items, list):
            added_items = [added_items]
        self.added_items = added_items
        if self.added_items is not None and len(self.added_items) > 0 and self.added_items[0].image is not None:
            self.liveWire.applyImage((self.added_items[0].image*255).astype(np.uint8))
        else:
            self.liveWire.applyImage(np.zeros((100,100),dtype=np.uint8))
        

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Handles the context menu event to add or remove control points.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        # event.accept()
        event.ignore()
        # https://stackoverflow.com/questions/72386383/how-to-make-widget-receive-mouse-release-event-when-context-menu-appears
        pos = event.scenePos()

        view_ract = self.vb.viewRect()
        self.default_view_range = self.vb.childrenBounds()
        if self.default_view_range is None or self.default_view_range[0] is None:
            return
        ratio = (self.default_view_range[0][1] - self.default_view_range[0][0]) / view_ract.width()
        ratio = max(0.05, min(30, ratio / 2))
        
        if self.editorType == 'anchors':
            for action in self.menu.actions():
                if action.text() in ["addPoint", "removePoint", "resetCurve"]:
                    self.menu.removeAction(action)

            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos).toPoint()
                if self.modified_ctrl_point is not None:
                    act_rm_point = QtGui.QAction("removePoint", self, triggered=self.on_remove_cv_point)
                    if not any(action.text() == act_rm_point.text() for action in self.menu.actions()):
                        self.menu.insertAction(self.menu.actions()[0], act_rm_point)
                        self.menu.insertSeparator(self.menu.actions()[1])
                elif self.gCurves is not None:
                    nearest_distance = self.distance_t / ratio
                    nearest_point = None
                    for i, curve in enumerate(self.gCurves):
                        if mouse_point.x() > 0 and mouse_point.x() < len(curve):
                            distance = np.abs(curve[mouse_point.x()] - mouse_point.y())
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest_point = (i, mouse_point.x(), mouse_point.y())
                    if nearest_point is not None:
                        act_addPoint = QtGui.QAction("addPoint", self, triggered=lambda: self.on_add_cv_point(nearest_point))
                        if not any(action.text() == act_addPoint.text() for action in self.menu.actions()):
                            self.menu.insertAction(self.menu.actions()[0], act_addPoint)
                            self.menu.insertSeparator(self.menu.actions()[1])
            self.menu.exec(event.screenPos())
        elif self.editorType == 'liner':
            self.mouse_path = []
            self.current_curve_idx = -1
        
        elif self.editorType == 'livewire':
            self.mouse_path = []
            self.current_curve_idx = -1


    def on_add_cv_point(self, point: tuple) -> None:
        """Adds a control point to the curve.

        Args:
            point (tuple): The point to add (curve_index, x, y).
        """
        self.add_cv_point(point[1], point[2], point[0])
        self.build_curve(point[0])
        self.plot()

    # def on_reset_curve(self):
    #     # get the curve index
    #     print('reset curve')

    def on_remove_cv_point(self) -> None:
        """Removes the selected control point from the curve."""
        remove_cv_point_x_index = []
        for p in self.modified_left_adjcent_points:
            remove_cv_point_x_index.append(p[1])
        remove_cv_point_x_index.append(self.modified_ctrl_point[1])
        for p in self.modified_right_adjcent_points:
            remove_cv_point_x_index.append(p[1])

        self.remove_cv_points(remove_cv_point_x_index, self.modified_ctrl_point[0])

        self.build_curve(self.modified_ctrl_point[0])
        self.plot()

    def targetWhellEvent(self, event: QGraphicsSceneWheelEvent) -> None:
        """Handles the wheel event for the target.

        Args:
            event (QGraphicsSceneWheelEvent): The wheel event.
        """
        # print(self.modified_ctrl_point)
        # print(self.ctrl_points[self.modified_ctrl_point[0]])
        if self.modified_ctrl_point is None:
            return
        ctp = self.ctrl_points[self.modified_ctrl_point[0]]
        # sort the ctp
        ctp = np.array(ctp)
        ctp = ctp[:, ctp[0].argsort()]
        _, unique_index = np.unique(ctp[0], return_index=True)
        ctp = ctp[:, unique_index]
        curveIdx, pIdx, _, _ = self.modified_ctrl_point
        # select adjcent control point when wheel scrolling up
        if event.delta() > 0:
            if not self.modified_left_adjcent_points:
                if pIdx - 1 >= 0:
                    self.modified_left_adjcent_points.append([curveIdx, pIdx - 1, ctp[0][pIdx - 1], ctp[1][pIdx - 1]])
            else:
                idx = self.modified_left_adjcent_points[-1][1] - 1
                if idx >= 0:
                    self.modified_left_adjcent_points.append([curveIdx, idx, ctp[0][idx], ctp[1][idx]])

            if not self.modified_right_adjcent_points:
                if pIdx + 1 < len(ctp[0]):
                    self.modified_right_adjcent_points.append([curveIdx, pIdx + 1, ctp[0][pIdx + 1], ctp[1][pIdx + 1]])
            else:
                idx = self.modified_right_adjcent_points[-1][1] + 1
                if idx < len(ctp[0]):
                    self.modified_right_adjcent_points.append([curveIdx, idx, ctp[0][idx], ctp[1][idx]])
            # print('scrolling up')
        else:
            if len(self.modified_left_adjcent_points) > 0:
                self.modified_left_adjcent_points.pop()
            if len(self.modified_right_adjcent_points) > 0:
                self.modified_right_adjcent_points.pop()
            # print('down')
        # print('left:',self.modified_left_adjcent_points)
        # print('right:',self.modified_right_adjcent_points)

        self.plot_focus_control_point(self.modified_ctrl_point, eventPos=None)

    def targetMoved(self, target: pg.TargetItem) -> None:
        """Updates the control points when the target is moved.

        Args:
            target (pg.TargetItem): The target item.
        """
        if self.target.isVisible() and self.mouse_pressed:
            pos = self.vb.mapSceneToView(target.scenePos())
            x = pos.x()
            y = pos.y()
            x_round = np.round(x)
            if self.modified_ctrl_point is not None:

                if x_round in self.ctrl_points[self.modified_ctrl_point[0]][0]:
                    # add 1 or -1 to x according to x value bigger or smaller than the np.round(x)
                    x = x_round + 1 if x > x_round else x_round - 1

                x = max(0, min(len(self.gCurves[self.modified_ctrl_point[0]]) - 1, x_round))
                idx_ = self.modified_ctrl_point[1]
                x_s = self.ctrl_points[self.modified_ctrl_point[0]][0].copy()
                x_s = np.delete(x_s, idx_)
                sorted_x = np.unique(np.sort(x_s))
                if idx_ == 0:
                    val_smaller = 0
                else:
                    val_smaller = sorted_x[np.where(sorted_x <= x)[0][-1]]

                if idx_ == len(sorted_x):
                    val_larger = len(self.gCurves[self.modified_ctrl_point[0]]) - 1
                else:
                    val_larger = sorted_x[np.where(sorted_x >= x)[0][0]]
                self.params_for_ctrl_point_changed_signal = [self.modified_ctrl_point[0], val_smaller, val_larger + 1]

                # calculate the difference between the new and old control points
                diffx = x - self.ctrl_points[self.modified_ctrl_point[0]][0][self.modified_ctrl_point[1]]
                diffy = y - self.ctrl_points[self.modified_ctrl_point[0]][1][self.modified_ctrl_point[1]]
                # print('diffx:',diffx,'diffy:',diffy)
                # print(len(self.modified_left_adjcent_points)+len(self.modified_right_adjcent_points),len(self.ctrl_points[self.modified_ctrl_point[0]][0])-1)
                if (
                    len(self.modified_left_adjcent_points) + len(self.modified_right_adjcent_points)
                    == len(self.ctrl_points[self.modified_ctrl_point[0]][0]) - 1
                ):
                    diffx = 0
                    x = 0
                # update the modified adjcent control points and the control points
                for i in range(len(self.modified_left_adjcent_points)):
                    self.modified_left_adjcent_points[i][2] += diffx
                    self.modified_left_adjcent_points[i][3] += diffy
                    self.ctrl_points[self.modified_left_adjcent_points[i][0]][0][self.modified_left_adjcent_points[i][1]] += diffx
                    self.ctrl_points[self.modified_left_adjcent_points[i][0]][1][self.modified_left_adjcent_points[i][1]] += diffy
                for i in range(len(self.modified_right_adjcent_points)):
                    self.modified_right_adjcent_points[i][2] += diffx
                    self.modified_right_adjcent_points[i][3] += diffy
                    self.ctrl_points[self.modified_right_adjcent_points[i][0]][0][self.modified_right_adjcent_points[i][1]] += diffx
                    self.ctrl_points[self.modified_right_adjcent_points[i][0]][1][self.modified_right_adjcent_points[i][1]] += diffy

                self.ctrl_points[self.modified_ctrl_point[0]][0][self.modified_ctrl_point[1]] += diffx
                self.ctrl_points[self.modified_ctrl_point[0]][1][self.modified_ctrl_point[1]] += diffy

                # print(" self.ctrl_points[self.modified_left_adjcent_points[i][0]][1]", self.ctrl_points[self.modified_left_adjcent_points[i][0]][1])

                self.build_curve(self.modified_ctrl_point[0])
                self.plot()
                self.plot_focus_control_point(
                    [
                        0,
                        0,
                        self.ctrl_points[self.modified_ctrl_point[0]][0][self.modified_ctrl_point[1]],
                        self.ctrl_points[self.modified_ctrl_point[0]][1][self.modified_ctrl_point[1]],
                    ],
                    None,
                )

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles mouse press events to enable dragging of control points.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse press event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.editorType=='anchors' and self.modified_ctrl_point is not None:
                self.mouse_pressed = True
            if self.editorType=='liner':
                pos = event.scenePos()
                if self.sceneBoundingRect().contains(pos) and self.current_curve_idx != -1:
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.mouse_path = [(mouse_point.x(), mouse_point.y())]
                    if self.params_for_ctrl_point_changed_signal is None:
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, int(np.round(mouse_point.x())),int(np.round(mouse_point.x()))]
                    else:
                        x0 = self.params_for_ctrl_point_changed_signal[1]
                        x1 = self.params_for_ctrl_point_changed_signal[2]
                        sorted_x = sorted([x0, x1,int(np.round(mouse_point.x()))])
                        val_smaller = sorted_x[0]
                        val_larger = sorted_x[-1]
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, val_smaller, val_larger]
            
            if  self.editorType=='livewire':
                pos = event.scenePos()
                if self.sceneBoundingRect().contains(pos) and self.current_curve_idx != -1:
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.liveWire.buildMap((int(mouse_point.x()), int(mouse_point.y())))
                    self.mouse_path = [(mouse_point.x(), mouse_point.y())]
                    if self.params_for_ctrl_point_changed_signal is None:
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, int(np.round(mouse_point.x())),int(np.round(mouse_point.x()))]
                    else:
                        x0 = self.params_for_ctrl_point_changed_signal[1]
                        x1 = self.params_for_ctrl_point_changed_signal[2]
                        sorted_x = sorted([x0, x1,int(np.round(mouse_point.x()))])
                        val_smaller = sorted_x[0]
                        val_larger = sorted_x[-1]
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, val_smaller, val_larger]

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Handles mouse double-click events to release the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse double-click event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.mouse_release()
            
    def mouse_release(self) -> None:
        """Releases the mouse and hides the target."""
        if self.editorType == 'liner':
            self.ctrl_points = self._generate_control_points()
            self.sigCurveChanged.emit(self.current_curve_idx, self.gCurves[self.current_curve_idx])
            self.sigControlPointChanged.emit(self.params_for_ctrl_point_changed_signal[0], self.params_for_ctrl_point_changed_signal[1:])
            self.plot()
        if self.editorType == 'livewire':
            self.ctrl_points = self._generate_control_points()
            self.sigCurveChanged.emit(self.current_curve_idx, self.gCurves[self.current_curve_idx])
            self.sigControlPointChanged.emit(self.params_for_ctrl_point_changed_signal[0], self.params_for_ctrl_point_changed_signal[1:])
            self.plot()
        self.mouse_pressed = False
        self.modified_ctrl_point = None
        self.target.hide()
        self.mouse_path = []
        self.current_curve_idx = -1
        self.params_for_ctrl_point_changed_signal = None
        
        
        self.sigCurveChangeFinished.emit(None, None)

    def setEditorType(self, editorType: str) -> None:
        """Sets the editor type.

        Args:
            editorType (str): The editor type ("none", "liner", "livewire", "anchors").
        """
        self.editorType = editorType.lower()
        action_texts = [action.text() for action in self.menu.actions()]
        if self.editorType == 'none':
            self.modified_ctrl_point = None
            self.modified_left_adjcent_points = []
            self.modified_right_adjcent_points = []
            if "Corrector:Anchors" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setChecked(False)
            if "Corrector:Liner" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Liner")].setChecked(False)
            if "Corrector:LiveWire" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setChecked(False)
        elif self.editorType == 'anchors':
            if "Corrector:Anchors" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setChecked(True)
            if "Corrector:Liner" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Liner")].setChecked(False)
            if "Corrector:LiveWire" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setChecked(False)
        elif self.editorType == 'liner':
            if "Corrector:Anchors" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setChecked(False)
            if "Corrector:Liner" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Liner")].setChecked(True)
            if "Corrector:LiveWire" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setChecked(False)
            if "addPoint" in action_texts:
                self.menu.removeAction(self.menu.actions()[0])
            if "removePoint" in action_texts:
                self.menu.removeAction(self.menu.actions()[0])
        elif self.editorType == 'livewire':
            if "Corrector:Anchors" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Anchors")].setChecked(False)
            if "Corrector:Liner" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:Liner")].setChecked(False)
            if "Corrector:LiveWire" in action_texts:
                self.menu.actions()[action_texts.index("Corrector:LiveWire")].setChecked(True)
            if "addPoint" in action_texts:
                self.menu.removeAction(self.menu.actions()[0])
            if "removePoint" in action_texts:
                self.menu.removeAction(self.menu.actions()[0])
        self.curve_props = deepcopy(self.curve_props_bk)
        self.plot()
        self.sigCurveEditorTypeChanged.emit(self.editorType, None)
        
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Handles hover move events to highlight the nearest control point.

        Args:
            event (QGraphicsSceneHoverEvent): The hover move event.
        """
        pos = event.scenePos()
        # get the view box range
        view_ract = self.vb.viewRect()
        self.default_view_range = self.vb.childrenBounds()
        if self.default_view_range[0] is None:
            return
        ratio = (self.default_view_range[0][1] - self.default_view_range[0][0]) / view_ract.width()
        # print("gCurves Len:", self.curveLen)
        # print("self.default_view_range:", self.default_view_range)
        # print("view_ract:", view_ract)
        # make ratio in the range of 0.05 to 80
        # ratio = max(0.01, min(100, ratio))
        if ratio ==0:
            return
            
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.vb.mapSceneToView(pos)
            self.sigMousePosChanged.emit([np.int32(mouse_point.x()),np.int32(mouse_point.y())])
            if self.editorType == 'anchors':
                nearest_distance = self.distance_t / ratio
                # print(" self.distance_t / ratio:", self.distance_t / ratio, "ratio:", ratio)
                nearest_point = None
                for i, ctrl_points in enumerate(self.ctrl_points):
                    if self.curve_props[i]["is_visible"]:
                        xs = ctrl_points[0]
                        ys = ctrl_points[1]
                        for j, (x, inner_y) in enumerate(zip(xs, ys)):
                            distance = np.linalg.norm(np.array([x, inner_y]) - np.array([mouse_point.x(), mouse_point.y()]), ord=1)
                            if distance < nearest_distance:
                                # print("distance between mouse and ctl point:", distance, "  nearest_distance:", nearest_distance)
                                nearest_distance = distance
                                nearest_point = (i, j, x, inner_y)
                # check if the nearest point is the same as the modified control point
                if nearest_point != self.modified_ctrl_point:
                    self.modified_left_adjcent_points = []
                    self.modified_right_adjcent_points = []

                self.modified_ctrl_point = nearest_point
                self.plot_focus_control_point(nearest_point, event.pos())
            elif self.editorType == 'liner':
                if len(self.mouse_path)==0:
                    nearest_distance = self.distance_t / ratio
                    if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                        self.current_curve_idx = self.parentWindow.ui.comboBox_enfaceA_Slab.currentIndex()
                    else:
                        self.current_curve_idx=-1
                    x_idx = int(np.round(mouse_point.x()))
                    # x_idx = max(0, min(self.curveLen - 1, x_idx))
                    self.curve_props = deepcopy(self.curve_props_bk)
                    if x_idx < 0 or x_idx >= self.curveLen:
                        self.plot()
                        return
                    for curve_idx, curve in enumerate(self.gCurves):
                        distance = np.abs(curve[x_idx] - mouse_point.y())
                        if distance < nearest_distance:
                            nearest_distance = distance
                            self.current_curve_idx = curve_idx
                    if self.current_curve_idx != -1:
                        self.curve_props[self.current_curve_idx]["style"] = QtCore.Qt.PenStyle.SolidLine
                        self.curve_props[self.current_curve_idx]["width"] = 4
                        
                else:
                    curve = self.gCurves[self.current_curve_idx]
                    x0 = self.mouse_path[0][0]
                    y0 = self.mouse_path[0][1]
                    x1 = mouse_point.x()
                    y1 = mouse_point.y()
                    x0 = max(0, min(len(curve), int(np.round(x0))))
                    x1 = max(0, min(len(curve), int(np.round(x1))))
                    if x0 == x1:
                        return
                    if x0 > x1:
                        x0, x1 = x1, x0
                        y0, y1 = y1, y0
                    x = np.arange(x0, x1)
                    self.gCurves[self.current_curve_idx][x0:x1] = np.interp(x, [x0, x1], [y0, y1])
                    
                    if self.update_mode == 'local':
                        if self.params_for_ctrl_point_changed_signal is None:
                            start_x = x0
                            end_x = x1
                        else:
                            start_x = self.params_for_ctrl_point_changed_signal[1]
                            end_x = self.params_for_ctrl_point_changed_signal[2]
                        sorted_x = sorted([x0, x1,start_x,end_x])
                        val_smaller = sorted_x[0]
                        val_larger = sorted_x[-1]
                        
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, val_smaller, val_larger]
                    else: # global update
                        # update the whole curve
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, 0, len(self.gCurves[self.current_curve_idx]) - 1]
            
                self.plot()
            elif self.editorType == 'livewire':
                if len(self.mouse_path)==0:
                    nearest_distance = self.distance_t / ratio
                    if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                        self.current_curve_idx = self.parentWindow.ui.comboBox_enfaceA_Slab.currentIndex()
                    else:
                        self.current_curve_idx=-1
                    x_idx = int(np.round(mouse_point.x()))
                    # x_idx = max(0, min(self.curveLen - 1, x_idx))
                    self.curve_props = deepcopy(self.curve_props_bk)
                    if x_idx < 0 or x_idx >= self.curveLen:
                        self.plot()
                        return
                    for curve_idx, curve in enumerate(self.gCurves):
                        distance = np.abs(curve[x_idx] - mouse_point.y())
                        if distance < nearest_distance:
                            nearest_distance = distance
                            self.current_curve_idx = curve_idx
                    if self.current_curve_idx != -1:
                        self.curve_props[self.current_curve_idx]["style"] = QtCore.Qt.PenStyle.SolidLine
                        self.curve_props[self.current_curve_idx]["width"] = 4
                else:
                    curve = self.gCurves[self.current_curve_idx]
                    x1 = int(np.round(mouse_point.x()))
                    y1 = int(np.round(mouse_point.y()))
                    x1 = max(0, min(len(curve)-1, x1))
                    
                    contourPonts = self.liveWire.getContour((x1,y1))
                    if contourPonts is not None:
                        contourPonts = [p[0] for p in contourPonts]
                        # clean up the contour points, remove the points that have same x value
                        contourPonts = np.array(contourPonts)
                        contourPonts = contourPonts[contourPonts[:, 0].argsort()]
                        _, unique_index = np.unique(contourPonts[:, 0], return_index=True)
                        contourPonts = contourPonts[unique_index]
                        # replace the curve with the new contour, where x is the index and y is the value
                        # Assuming contourPonts is a NumPy array
                        self.gCurves[self.current_curve_idx][contourPonts[:, 0]] = contourPonts[:, 1]
                        x0 = self.mouse_path[0][0]
                        if self.params_for_ctrl_point_changed_signal is None:
                            start_x = x0
                            end_x = x1
                        else:
                            start_x = self.params_for_ctrl_point_changed_signal[1]
                            end_x = self.params_for_ctrl_point_changed_signal[2]
                        sorted_x = sorted([x0, x1,start_x,end_x])
                        val_smaller = sorted_x[0]
                        val_larger = sorted_x[-1]
                        self.params_for_ctrl_point_changed_signal = [self.current_curve_idx, val_smaller, val_larger]
                        
                self.plot()
                    
        return super().hoverMoveEvent(event)

    def set_curve(self, curve: list, curve_props: dict = None) -> None:
        """Sets the curves and their properties.

        Args:
            curve (list): List of curves.
            curve_props (dict, optional): Properties of the curves. Defaults to None.
        """
        self.gCurves = curve
        self.curveLen = len(curve[0])
        self.curve_props = curve_props
        self.curve_props_bk = deepcopy(curve_props)
        self.ctrl_points = self._generate_control_points()

    def set_roi(self, roi_position: list) -> None:
        """Sets the position of the ROI.

        Args:
            roi_position (list): The ROI position.
        """
        self.roi_position = roi_position

    def plot(self, *args, **kargs) -> None:
        """Plots the curves and control points."""
        if self.gCurves is None or self.curve_props is None:
            return
        self.clear()
        if self.added_items is not None:
            for item in self.added_items:
                self.addItem(item)

        for curve, props in zip(self.gCurves, self.curve_props):
            if props["is_visible"]:
                super().plot(curve, pen=pg.mkPen(props["color"], style=props["style"], width=props["width"], connect=props["connect"]))
        if self.editorType == 'anchors':
            for ctrl_points, props in zip(self.ctrl_points, self.curve_props):
                if props["is_visible"]:
                    super().plot(
                        ctrl_points[0],
                        ctrl_points[1],
                        pen=None,
                        symbol="s",
                        symbolBrush=(77, 77, 77),
                        symbolSize=self.symbolSize,
                        symbolPen=pg.mkPen(color=props["color"], width=props["width"]),
                    )

        if self.roi_position[1] != 0:
            super().plot(
                [0, self.curveLen],
                [self.roi_position[1], self.roi_position[1]],
                pen=pg.mkPen(
                    (200, 200, 200),
                    width=1,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
            )
            super().plot(
                [0, self.curveLen],
                [self.roi_position[0], self.roi_position[0]],
                pen=pg.mkPen(
                    (200, 200, 200),
                    width=1,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
            )

    def plot_indicator(self, pos: list, direction: str = "vertical") -> None:
        """Plots the indicator.

        Args:
            pos (list): Position of the indicator.
            direction (str, optional): Direction of the indicator ("vertical" or "horizontal"). Defaults to "vertical".
        """
        if pos is None:
            if self.indicator_plot is not None:
                self.removeItem(self.indicator_plot)
            return
        if self.indicator_plot is not None:
            self.removeItem(self.indicator_plot)
        
        # get plot range
        # Check if added_items exists and has valid image
        if not self.added_items or self.added_items[0].image is None:
            return
        h,w = self.added_items[0].image.shape
        if direction == 'vertical':
            self.indicator_plot = super().plot(
                [pos[0], pos[0]],
                [0, h],
                pen=pg.mkPen(
                    (200, 200, 0),
                    width=2,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
            )
        else:
            self.indicator_plot = super().plot(
                [0, w],
                [pos[1], pos[1]],
                pen=pg.mkPen(
                    (200, 200, 0),
                    width=2,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
            )
    
    def plot_focus_control_point(self, control_point: list, eventPos: QtCore.QPointF = None) -> None:
        """Plots the focused control point with an inverse color.

        Args:
            control_point (list): The control point to focus on.
            eventPos (QtCore.QPointF, optional): The event position. Defaults to None.
        """
        if self.focus_control_point_plot is not None:
            self.removeItem(self.focus_control_point_plot)

        if control_point is not None:
            self.focus_control_point_plot = super().plot(
                [control_point[2]],
                [control_point[3]],
                pen=None,
                symbol="s",
                symbolBrush=self.get_inverse_color(self.curve_props[control_point[0]]["color"]),
                symbolPen="w",
            )
            if eventPos is not None:
                self.target.setPos(eventPos)
                self.target.show()

        if self.modified_left_adjcent_plot is not None:
            self.removeItem(self.modified_left_adjcent_plot)
        if self.editorType=='anchors' and self.modified_left_adjcent_points:
            cptx = [x[2] for x in self.modified_left_adjcent_points]
            cpty = [x[3] for x in self.modified_left_adjcent_points]
            props = self.curve_props[self.modified_left_adjcent_points[0][0]]
            self.modified_left_adjcent_plot = super().plot(
                cptx,
                cpty,
                pen=None,
                symbol="o",
                symbolBrush=self.get_inverse_color(props["color"]),
                symbolPen="w",
            )

        if self.modified_right_adjcent_plot is not None:
            self.removeItem(self.modified_right_adjcent_plot)
        if self.editorType=='anchors' and self.modified_right_adjcent_points:
            cptx = [x[2] for x in self.modified_right_adjcent_points]
            cpty = [x[3] for x in self.modified_right_adjcent_points]
            props = self.curve_props[self.modified_right_adjcent_points[0][0]]
            self.modified_right_adjcent_plot = super().plot(
                cptx,
                cpty,
                pen=None,
                symbol="o",
                symbolBrush=self.get_inverse_color(props["color"]),
                symbolPen="w",
            )

    def add_cv_point(self, x_value: int, y_value: int, curve_index: int) -> None:
        """Adds a control point to the specified curve.

        Args:
            x_value (int): X-coordinate of the control point.
            y_value (int): Y-coordinate of the control point.
            curve_index (int): Index of the curve.
        """

        # find the position to insert the new control point to keep the x values sorted
        sorted_x = np.sort(self.ctrl_points[curve_index][0])
        idx = np.searchsorted(sorted_x, x_value)
        # insert the new control point
        self.ctrl_points[curve_index][0] = np.insert(self.ctrl_points[curve_index][0], idx, x_value)
        self.ctrl_points[curve_index][1] = np.insert(self.ctrl_points[curve_index][1], idx, y_value)

        sorted_x = np.unique(self.ctrl_points[curve_index][0])
        if x_value == 0:
            val_smaller = 0
        else:
            val_smaller = sorted_x[np.where(sorted_x < x_value)[0][-1]]
        if x_value >= len(self.gCurves[curve_index]) - 1:
            val_larger = len(self.gCurves[curve_index]) - 1
        else:
            val_larger = sorted_x[np.where(sorted_x >= x_value)[0][0]]
        # self.sigControlPointChanged.emit(curve_index, [val_smaller, val_larger])
        self.params_for_ctrl_point_changed_signal = [curve_index, val_smaller, val_larger + 1]

    def remove_cv_points(self, x_indexes: list, curve_index: int) -> None:
        """Removes control points from the specified curve.

        Args:
            x_indexes (list): List of X-coordinates to remove.
            curve_index (int): Index of the curve.
        """
        sorted_x_indexes = np.sort(x_indexes)
        # print('x_indexes before:',sorted_x_indexes)

        if sorted_x_indexes[0] == 0:
            sorted_x_indexes = np.delete(sorted_x_indexes, 0)
        if sorted_x_indexes[-1] == len(self.ctrl_points[curve_index][0]) - 1:
            sorted_x_indexes = np.delete(sorted_x_indexes, -1)

        # print('x_indexes after:',sorted_x_indexes)
        if sorted_x_indexes is None:
            return

        x_val = self.ctrl_points[curve_index][0][sorted_x_indexes[0]]
        sorted_x = np.unique(np.sort(self.ctrl_points[curve_index][0]))
        if sorted_x_indexes[0] < 1:
            val_smaller = 1
        else:
            val_smaller = sorted_x[np.where(sorted_x < x_val)[0][-1]]
        if sorted_x_indexes[-1] == len(self.ctrl_points[curve_index][0]) - 1:
            val_larger = len(self.gCurves[curve_index]) - 2
        else:
            val_larger = sorted_x[np.where(sorted_x > x_val)[0][0]]
        self.params_for_ctrl_point_changed_signal = [curve_index, val_smaller, val_larger + 1]

        self.ctrl_points[curve_index][0] = np.delete(self.ctrl_points[curve_index][0], sorted_x_indexes)
        self.ctrl_points[curve_index][1] = np.delete(self.ctrl_points[curve_index][1], sorted_x_indexes)

    def _generate_control_points(self, t_p: float = 0.2) -> list:
        """Generates control points for the curves based on inflection points.

        Args:
            t_p (float, optional): Threshold for control point generation. Defaults to 0.2.

        Returns:
            list: List of control points.
        """
        ctrl_points = []
        t = t_p * len(self.gCurves[0])
        for curve in self.gCurves:
            dy = np.diff(curve)
            # set abs dy lower than 1 to 0
            # dy = np.where(np.abs(dy) <= 1, 0, dy)
            inflection_points = np.where(np.sign(dy))[0]
            inflection_points = np.insert(inflection_points, 0, 0)
            control_points = [inflection_points[0]]
            # Assuming inflection_points and control_points are NumPy arrays
            control_points = np.append(control_points, inflection_points[1:-1])
            # remove control points that are have similar gradient with adjcent points
            for k in range(10):
                ctrl_points_filtered = []
                ctrl_points_filtered.append(control_points[0])
                for i in range(1, len(control_points) - 1):
                    if t < control_points[i] - ctrl_points_filtered[-1]:
                        ctrl_points_filtered.append(control_points[i])
                        continue
                    deltaY = curve[control_points[i + 1]] - curve[ctrl_points_filtered[-1]]
                    deltaY1 = curve[control_points[i]] - curve[ctrl_points_filtered[-1]]
                    deltaY2 = curve[control_points[i+ 1]] - curve[control_points[i]]
                    if (np.abs(deltaY1) < 2.5 and np.abs(deltaY2) < 2.5 and np.abs(deltaY) < 4.5):
                        continue
                    gradentY = deltaY / (control_points[i + 1] - ctrl_points_filtered[-1]+0.0001)
                    gradentY1 = deltaY1 / (control_points[i] - ctrl_points_filtered[-1]+0.0001)
                    gradentY2 = deltaY2 / (control_points[i + 1] - control_points[i]+0.0001)
                    # if gradientY is similar with adjcent points, remove the control point
                    if np.abs(gradentY - gradentY1) < 0.15 and np.abs(gradentY - gradentY2) < 0.15:
                        continue
                    ctrl_points_filtered.append(control_points[i])
                ctrl_points_filtered.append(control_points[-1])
                if len(ctrl_points_filtered) == len(control_points):
                    break
                control_points = ctrl_points_filtered.copy()
            
            if len(control_points) < 2:
                control_points = np.append(control_points, len(curve) // 2)

            control_points = np.append(control_points, len(curve) - 1)

            ctrl_points.append([control_points, curve[control_points]])
        return ctrl_points

    def build_curve(self, curve_index: int = -1) -> None:
        """Builds the curve using cubic spline interpolation.

        Args:
            curve_index (int, optional): Index of the curve to build. Defaults to -1 (build all curves).
        """
        if curve_index == -1:
            for i in range(len(self.gCurves)):
                self.build_curve(i)
            return
        elif curve_index >= len(self.gCurves):
            return
        else:
            ctrl_points = self.ctrl_points[curve_index]
            ctrl_points = np.array(ctrl_points)
            ctrl_points = ctrl_points[:, ctrl_points[0].argsort()]

            _, unique_index = np.unique(ctrl_points[0], return_index=True)
            ctrl_points = ctrl_points[:, unique_index]

            cs = PchipInterpolator(ctrl_points[0], ctrl_points[1])
            inter_curve = cs(range(0, len(self.gCurves[curve_index])))
            self.gCurves[curve_index][self.params_for_ctrl_point_changed_signal[1]:self.params_for_ctrl_point_changed_signal[2]] = inter_curve[self.params_for_ctrl_point_changed_signal[1]:self.params_for_ctrl_point_changed_signal[2]]
            self.sigCurveChanged.emit(curve_index, self.gCurves[curve_index])
            self.sigControlPointChanged.emit(self.params_for_ctrl_point_changed_signal[0], self.params_for_ctrl_point_changed_signal[1:])
            self.params_for_ctrl_point_changed_signal=None
