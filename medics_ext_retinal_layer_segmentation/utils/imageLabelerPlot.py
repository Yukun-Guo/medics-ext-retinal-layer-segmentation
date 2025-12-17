"""
Various methods of drawing scrolling plots.
"""

from PySide6 import QtCore
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QFileDialog

import numpy as np
import pyqtgraph as pg
import cv2
import os
from .utils import utils


class ImageLabelerPlot(pg.PlotItem):
    sigIndicatorChanged = QtCore.Signal(object)  ## Emitted when the gCurve has changed
    sigLabelImageChanged = QtCore.Signal(
        object, object, object
    )  ## Emitted when the label image has changed
    sigMousePosChanged = QtCore.Signal(
        object
    )  ## Emitted when the mouse position has changed
    sigCenterPointChanged = QtCore.Signal(
        object
    )  ## Emitted when the center point has changed

    def __init__(
        self,
        parentWindow=None,
        parent=None,
        name=None,
        labels=None,
        title=None,
        viewBox=None,
        axisItems=None,
        enableMenu=True,
        disable_label=False,
        **kargs
    ):
        """
        Initialize the IndicatorPlot with optional parameters for customization.
        """
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs
        )
        self.vb.setDefaultPadding(0.001)
        self.setAcceptHoverEvents(True)
        self.parentWindow = parentWindow
        self.plotSize = [1, 1]
        self.indicatorPos = [0, 0]
        self.direction = 0
        self.disable_label = disable_label
        self.hided_indicator = False
        self.added_items = None
        self.transposed = False
        self.draw_center_point = False
        self.show_center_point = True
        self.center_point_plot = None
        self.center_point = None
        self.center_point_indicator_X = None
        self.center_point_indicator_Y = None
        self.imageItem = pg.ImageItem()
        self.imageData = None
        self.labelImageItem = pg.ImageItem()
        self.labelImageData = None
        self.labelImageData_prev = None
        self.labelImageItem_preview = pg.ImageItem()
        self.labelImageData_preview = None
        self.cursorItem = QGraphicsEllipseItem()
        self.cursorImageData = None
        self.labelColors = None
        self.currentLabelIdx = 1
        self.toolSize = 15
        self.labelTool = "off"
        self.cursor_pos = [0, 0]
        self.label_transparency = 0.6
        self.mousePath = []
        self.is_dragging = False
        self.mouseButtonPressed = QtCore.Qt.MouseButton.NoButton
        self.ctrl_pressed = False
        self.fillHole = 15
        self.removeRgn = 15
        self.offset_upper = 0
        self.offset_lower = 0
        self.loDiff = -20
        self.upDiff = 20
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.cursorItem.setPen(pg.mkPen("g", width=2))
        self.cursorItem.setBrush(pg.mkBrush(0, 255, 0, 10))
        self.cursorItem.wheelEvent = self.cursorItem_wheelEvent

        self.setAcceptHoverEvents(True)

        self.menu = self.vb.getMenu(None)
        for i in range(1, len(self.menu.actions())):
            self.menu.removeAction(self.menu.actions()[1])
        # add menu items
        self.menu.addAction("Save image...", self.onSaveImageAs)
        if not self.disable_label:
            self.menu.addAction("Save label...", self.onSaveLabelAs)
        self.ctrlMenu.menuAction().setVisible(False)
        if self.parentWindow.theme == "dark":
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )

    def cursorItem_wheelEvent(self, event):
        # check if the key control is pressed
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.wheelEvent(event)
        else:
            # if scrolling up, increase the tool size by 1, otherwise decrease by 1
            if event.delta() > 0:
                self.toolSize = min(
                    self.toolSize + 1, self.parentWindow.ui.spinBox_toolSize.maximum()
                )
            else:
                self.toolSize = max(
                    self.toolSize - 1, self.parentWindow.ui.spinBox_toolSize.minimum()
                )
            self.parentWindow.ui.spinBox_toolSize.setValue(self.toolSize)
            self.cursorItem.setRect(
                self.cursor_pos[0] - self.toolSize / 2,
                self.cursor_pos[1] - self.toolSize / 2,
                self.toolSize,
                self.toolSize,
            )
            self.plot()

    def setFillHole(self, fillHole):
        self.fillHole = fillHole

    def setRemoveRgn(self, removeRgn):
        self.removeRgn = removeRgn

    def setOffset_upper(self, offset):
        self.offset_upper = offset

    def setOffset_lower(self, offset):
        self.offset_lower = offset

    def setLoDiff(self, loDiff):
        self.loDiff = loDiff

    def setUpDiff(self, upDiff):
        self.upDiff = upDiff

    def set_center_point(self, p):
        self.center_point = p
        self.plot()

    def setCurrentColorIndex(self, idx):
        self.currentLabelIdx = idx

    def setLabelTransparency(self, transparency):
        self.label_transparency = transparency
        self.plot()

    def setLabelTool(self, tool: str):
        self.labelTool = tool.lower()
        # if self.labelTool != 'off':
        #     self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        # else:
        #     self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def setLabelColors(self, colors):
        # colors = np.asarray(colors, dtype=np.float32)
        # self.labelColors = np.column_stack((colors, np.ones(len(colors), dtype=np.float32)*255))
        self.labelColors = np.asarray(colors, dtype=np.uint8)

    def onLabelColorChanged(self, colors):
        self.labelColors = np.asarray(colors, dtype=np.uint8)
        self.plot()

    def setCursorSize(self, size):
        self.toolSize = size

    def setImage(self, data, levels=None):
        self.imageData = np.ascontiguousarray(data, dtype=np.uint8)
        self.labelImageData = np.zeros_like(self.imageData, dtype=np.uint8)
        self.labelImageData_prev = np.zeros_like(self.imageData, dtype=np.uint8)
        self.labelImageData_preview = np.zeros_like(self.imageData, dtype=np.uint8)
        self.cursorImageData = np.zeros_like(self.imageData, dtype=np.uint8)
        self.imageItem.setImage(self.imageData, levels=levels)
        # set center point to the center of the image
        self.center_point = np.array(
            [self.imageData.shape[1] // 2, self.imageData.shape[0] // 2], dtype=np.int32
        )

    def setLabelImageData(self, data):
        if data is not None:
            self.labelImageData = data.copy().astype(np.uint8)
            self.labelImageData_prev = self.labelImageData.copy()

    def setDrawCenterPoint(self, draw_center_point):
        self.draw_center_point = draw_center_point
    
    def setShowCenterPoint(self, show_center_point):
        self.show_center_point = show_center_point

    def setTransposed(self, transposed):
        self.transposed = transposed

    def setPlotSize(self, psize):
        self.plotSize = psize

    def setDirection(self, direction):
        self.direction = direction

    def setIndicatorPos(self, pos):
        self.indicatorPos = pos

    def set_added_items(self, added_items):
        """
        Set the added item to the plot.
        """
        if not isinstance(added_items, list):
            added_items = [added_items]
        self.added_items = added_items

    def onSaveImageAs(self):
        """
        Save the image as a file.
        """
        filename = QFileDialog.getSaveFileName(
            self.parentWindow, "Save Image", "", "Image (*.png)"
        )[0]
        if filename:
            self.imageItem.save(filename)

    def onSaveLabelAs(self):
        """
        Save the label image as a file.
        """
        filename = QFileDialog.getSaveFileName(
            self.parentWindow, "Save Label", "", "Image (*.png)"
        )[0]
        if filename:
            self.labelImageItem.save(filename)

    def hoverMoveEvent(self, event):
        if self.imageData is not None:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                self.cursor_pos = (mouse_point.x(), mouse_point.y())
                self.sigMousePosChanged.emit(
                    [np.int32(mouse_point.x()), np.int32(mouse_point.y())]
                )
                rt_view = QtCore.QRectF(
                    mouse_point.x() - self.toolSize / 2,
                    mouse_point.y() - self.toolSize / 2,
                    self.toolSize,
                    self.toolSize,
                )
                self.cursorItem.setRect(rt_view)
                if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                    self.indicatorPos = [
                        np.floor(mouse_point.x()),
                        np.floor(mouse_point.y()),
                    ]
                    self.sigIndicatorChanged.emit(self.indicatorPos)
                    self.ctrl_pressed = True
                else:
                    self.ctrl_pressed = False
                    if self.labelTool != "off":
                        self.plot()
                if self.draw_center_point:
                    self.plot()

        return super().hoverMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                # refresh cursor position
                self.cursor_pos = (mouse_point.x(), mouse_point.y())
                self.sigMousePosChanged.emit(
                    [np.int32(mouse_point.x()), np.int32(mouse_point.y())]
                )
                rt_view = QtCore.QRectF(
                    mouse_point.x() - self.toolSize / 2,
                    mouse_point.y() - self.toolSize / 2,
                    self.toolSize,
                    self.toolSize,
                )
                self.cursorItem.setRect(rt_view)
                self.mousePath.append(
                    np.array([mouse_point.x(), mouse_point.y()], dtype=np.int32)
                )
            self.handleMouseDrag(event)
        super().mouseMoveEvent(event)

    def handleMouseDrag(self, event):
        if self.mouseButtonPressed == QtCore.Qt.MouseButton.LeftButton:
            if self.labelTool == "brush":
                if len(self.mousePath) > 0:
                    brush_mask = self._generateLineMask(
                        self.labelImageData.shape, self.mousePath, self.toolSize
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.plot()
            elif self.labelTool == "thresholdlower":
                if len(self.mousePath) > 0:
                    brush_mask = utils.generateAdaptThresMasks(
                        self.imageData,
                        self.mousePath,
                        self.toolSize // 2,
                        loweroffset=-255,
                        upperoffset=self.offset_lower,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.plot()
            elif self.labelTool == "thresholdupper":
                if len(self.mousePath) > 0:
                    brush_mask = utils.generateAdaptThresMasks(
                        self.imageData,
                        self.mousePath,
                        self.toolSize // 2,
                        loweroffset=self.offset_upper,
                        upperoffset=255,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.plot()
            elif self.labelTool == "thresholdbetween":
                if len(self.mousePath) > 0:
                    brush_mask = utils.generateAdaptThresMasks(
                        self.imageData,
                        self.mousePath,
                        self.toolSize // 2,
                        loweroffset=self.loDiff,
                        upperoffset=self.upDiff,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.plot()
            else:
                pass
        elif self.mouseButtonPressed == QtCore.Qt.MouseButton.RightButton:
            if self.labelTool != "off":
                if len(self.mousePath) > 0:
                    brush_mask = self._generateLineMask(
                        self.labelImageData.shape, self.mousePath, self.toolSize
                    )
                    self.labelImageData_preview[
                        (brush_mask == 1) & (self.labelImageData > 0)
                    ] = 1
                    self.labelImageData_preview = self._remove_regions(
                        self.labelImageData, self.labelImageData_preview, self.removeRgn
                    )
                    self.plot()
            # if self.labelTool == 'brush':
            #     if len(self.mousePath) > 0:
            #         brush_mask = self._generateLineMask(self.labelImageData.shape, self.mousePath, self.toolSize)
            #         self.labelImageData_preview[(brush_mask==1) &(self.labelImageData>0)] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.plot()
            # elif self.labelTool == 'thresholdlower':
            #     if len(self.mousePath) > 0:
            #         brush_mask = utils.generateAdaptThresMasks(self.imageData, self.mousePath, self.toolSize//2, loweroffset=-255,upperoffset=self.offset_lower)
            #         self.labelImageData_preview[(brush_mask==1) &(self.labelImageData>0)] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.plot()
            # elif self.labelTool == 'thresholdupper':
            #     if len(self.mousePath) > 0:
            #         brush_mask = utils.generateAdaptThresMasks(self.imageData, self.mousePath, self.toolSize//2, loweroffset=self.offset_upper,upperoffset=255)
            #         self.labelImageData_preview[(brush_mask==1) &(self.labelImageData>0)] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.plot()
            # elif self.labelTool == 'thresholdbetween':
            #     if len(self.mousePath) > 0:
            #         brush_mask = utils.generateAdaptThresMasks(self.imageData, self.mousePath, self.toolSize//2, loweroffset=self.loDiff,upperoffset=self.upDiff)
            #         self.labelImageData_preview[(brush_mask==1) &(self.labelImageData>0)] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.plot()
            # else:
            #     pass

    def mouseReleaseEvent(self, event):
        if self.labelImageData is None:
            super().mouseReleaseEvent(event)
            return
        self.is_dragging = False
        self.mousePath = []
        self.mouseButtonPressed = QtCore.Qt.MouseButton.NoButton
        self.labelImageData_prev = self.labelImageData.copy()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.labelImageData[self.labelImageData_preview != 0] = self.currentLabelIdx
            self.plot()
            self.sigLabelImageChanged.emit(
                self.labelImageData, self.labelImageData_preview, self.currentLabelIdx
            )
            self.labelImageData_preview = np.zeros_like(self.labelImageData)

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.labelImageData[self.labelImageData_preview != 0] = 0
            self.plot()
            self.sigLabelImageChanged.emit(
                self.labelImageData, self.labelImageData_preview, 0
            )
            self.labelImageData_preview = np.zeros_like(self.labelImageData)

        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.labelImageData = self.labelImageData_prev.copy()
            self.plot()
            self.sigLabelImageChanged.emit(
                self.labelImageData, self.labelImageData_preview, -1
            )
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        if self.imageData is None:
            super().mousePressEvent(event)
            return
        pos = event.scenePos()
        self.ctrl_pressed = False
        self.mouseButtonPressed = event.button()
        # fillHole = self.parentWindow.ui.spinBox_fill_hole.value()
        # removeRgn = self.parentWindow.ui.spinBox_remove.value()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.draw_center_point:
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    self.center_point = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    self.draw_center_point = False
                    self.sigCenterPointChanged.emit(self.center_point)
                    return
                return

            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                if (
                    self.sceneBoundingRect().contains(pos)
                    and self.imageItem.image is not None
                ):
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.indicatorPos = [
                        np.floor(mouse_point.x()),
                        np.floor(mouse_point.y()),
                    ]
                    self.sigIndicatorChanged.emit(self.indicatorPos)
                    self.sigMousePosChanged.emit(
                        [np.int32(mouse_point.x()), np.int32(mouse_point.y())]
                    )
                    self.ctrl_pressed = True
                    self.plot()
                super().mousePressEvent(event)
                return
            if self.labelTool == "brush":
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    brush_mask = self._generateCircleMask(
                        self.labelImageData.shape,
                        cursor_pos,
                        self.toolSize // 2,
                        thickness=-1,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()

            elif self.labelTool == "thresholdlower":
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    brush_mask = utils.generateAdaptThresMask(
                        self.imageData,
                        cursor_pos,
                        self.toolSize // 2,
                        loweroffset=-255,
                        upperoffset=self.offset_lower,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()
            elif self.labelTool == "thresholdupper":
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    brush_mask = utils.generateAdaptThresMask(
                        self.imageData,
                        cursor_pos,
                        self.toolSize // 2,
                        loweroffset=self.offset_upper,
                        upperoffset=255,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()
            elif self.labelTool == "thresholdbetween":
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    brush_mask = utils.generateAdaptThresMask(
                        self.imageData,
                        cursor_pos,
                        self.toolSize // 2,
                        loweroffset=self.loDiff,
                        upperoffset=self.upDiff,
                    )
                    self.labelImageData_preview[brush_mask == 1] = 1
                    self.labelImageData_preview = self._fill_regions(
                        self.labelImageData,
                        self.labelImageData_preview,
                        self.currentLabelIdx,
                        self.fillHole,
                    )
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()

            else:  # tool is off
                if (
                    self.sceneBoundingRect().contains(pos)
                    and self.imageItem.image is not None
                ):
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.indicatorPos = [
                        np.floor(mouse_point.x()),
                        np.floor(mouse_point.y()),
                    ]
                    self.sigIndicatorChanged.emit(self.indicatorPos)
                    self.sigMousePosChanged.emit(
                        [np.int32(mouse_point.x()), np.int32(mouse_point.y())]
                    )
                super().mousePressEvent(event)

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                super().mousePressEvent(event)
                return
            if self.labelTool != "off":
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array(
                        [mouse_point.x(), mouse_point.y()], dtype=np.int32
                    )
                    brush_mask = self._generateCircleMask(
                        self.labelImageData.shape,
                        cursor_pos,
                        self.toolSize // 2,
                        thickness=-1,
                    )
                    self.labelImageData_preview = brush_mask * (self.labelImageData > 0)
                    self.labelImageData_preview = self._remove_regions(
                        self.labelImageData, self.labelImageData_preview, self.removeRgn
                    )
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()
            # if self.labelTool == 'brush':
            #     if self.sceneBoundingRect().contains(pos):
            #         mouse_point = self.vb.mapSceneToView(pos)
            #         # generate a brush mask, brush size is self.toolSize
            #         cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
            #         brush_mask = self._generateCircleMask(self.labelImageData.shape, cursor_pos, self.toolSize//2,thickness=-1)
            #         self.labelImageData_preview = brush_mask*(self.labelImageData>0)
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.is_dragging = True
            #         self.mousePath = [cursor_pos]
            #         self.handleMouseDrag(event)
            #         self.plot()
            # elif self.labelTool == 'thresholdlower':
            #         mouse_point = self.vb.mapSceneToView(pos)
            #         # generate a brush mask, brush size is self.toolSize
            #         cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
            #         brush_mask = utils.generateAdaptThresMask(self.imageData, cursor_pos, self.toolSize//2, loweroffset=-255,upperoffset=self.offset_lower)
            #         self.labelImageData_preview[brush_mask==1] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.is_dragging = True
            #         self.mousePath = [cursor_pos]
            #         self.handleMouseDrag(event)
            #         self.plot()
            # elif self.labelTool == 'thresholdupper':
            #         mouse_point = self.vb.mapSceneToView(pos)
            #         # generate a brush mask, brush size is self.toolSize
            #         cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
            #         brush_mask = utils.generateAdaptThresMask(self.imageData, cursor_pos, self.toolSize//2, loweroffset=self.offset_upper,upperoffset=255)
            #         self.labelImageData_preview[brush_mask==1] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.is_dragging = True
            #         self.mousePath = [cursor_pos]
            #         self.handleMouseDrag(event)
            #         self.plot()
            # elif self.labelTool == 'thresholdbetween':
            #         mouse_point = self.vb.mapSceneToView(pos)
            #         # generate a brush mask, brush size is self.toolSize
            #         cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
            #         brush_mask = utils.generateAdaptThresMask(self.imageData, cursor_pos, self.toolSize//2, loweroffset=self.loDiff,upperoffset=self.upDiff)
            #         self.labelImageData_preview[brush_mask==1] = 1
            #         self.labelImageData_preview = self._remove_regions(self.labelImageData, self.labelImageData_preview, self.removeRgn)
            #         self.is_dragging = True
            #         self.mousePath = [cursor_pos]
            #         self.handleMouseDrag(event)
            #         self.plot()

            elif self.labelTool == "off":
                super().mousePressEvent(event)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                super().mousePressEvent(event)
                return
            if self.labelTool == "off":
                super().mousePressEvent(event)
            else:
                self.labelImageData = self.labelImageData_prev.copy()
                self.plot()

    def refreshLabelImage(self, labelImageData=None):
        if labelImageData is not None:
            self.labelImageData = labelImageData.copy()
        if self.labelImageData is not None and self.labelColors is not None:
            # use looup table to convert grayimage to color image
            im = np.array(
                self.labelColors[self.labelImageData.astype("uint8")], dtype=np.uint8
            )
            # add transparency
            im = np.stack(
                (
                    im[:, :, 0],
                    im[:, :, 1],
                    im[:, :, 2],
                    255 * self.label_transparency * (self.labelImageData > 0),
                ),
                axis=2,
            )
            if self.transposed:
                im = np.transpose(im, (1, 0, 2))
            self.labelImageItem.setImage(im)

    def plot(self, source=None, *args, **kargs):
        """
        Plot the indicator line.
        """
        # print('plot source:',source)
        self.clear()
        if self.added_items is not None:
            for item in self.added_items:
                self.addItem(item)
        if self.imageItem is not None:
            self.addItem(self.imageItem)

        if self.draw_center_point:
            if self.center_point_indicator_X is not None:
                self.removeItem(self.center_point_indicator_X)
            self.center_point_indicator_X = super().plot(
                [0, self.plotSize[0]],
                [self.cursor_pos[1], self.cursor_pos[1]],
                pen=pg.mkPen("g", width=2),
            )
            if self.center_point_indicator_Y is not None:
                self.removeItem(self.center_point_indicator_Y)
            self.center_point_indicator_Y = super().plot(
                [self.cursor_pos[0], self.cursor_pos[0]],
                [0, self.plotSize[1]],
                pen=pg.mkPen("g", width=2),
            )
            return
        else:
            if self.center_point_indicator_X is not None:
                self.removeItem(self.center_point_indicator_X)
                self.center_point_indicator_X = None
            if self.center_point_indicator_Y is not None:
                self.removeItem(self.center_point_indicator_Y)
                self.center_point_indicator_Y = None

        if self.labelImageData is not None and self.labelColors is not None:
            # use looup table to convert grayimage to color image
            im = np.array(
                self.labelColors[self.labelImageData.astype("uint8")], dtype=np.uint8
            )
            # add transparency
            im = np.stack(
                (
                    im[:, :, 0],
                    im[:, :, 1],
                    im[:, :, 2],
                    255 * self.label_transparency * (self.labelImageData > 0),
                ),
                axis=2,
            )
            if self.transposed:
                im = np.transpose(im, (1, 0, 2))
            self.labelImageItem.setImage(im)
            if self.labelImageItem not in self.items:
                self.addItem(self.labelImageItem)
        if self.labelImageData_preview is not None:
            if self.labelTool != "off" and not self.ctrl_pressed:
                if self.mouseButtonPressed == QtCore.Qt.MouseButton.LeftButton:
                    im_preview = np.stack(
                        (
                            self.labelImageData_preview
                            * self.labelColors[self.currentLabelIdx][0],
                            self.labelImageData_preview
                            * self.labelColors[self.currentLabelIdx][1],
                            self.labelImageData_preview
                            * self.labelColors[self.currentLabelIdx][2],
                            (self.labelImageData_preview > 0)
                            * 200
                            * self.label_transparency,
                        ),
                        axis=2,
                    )
                else:
                    im_preview = np.stack(
                        (
                            self.labelImageData_preview
                            * (255 - self.labelColors[self.currentLabelIdx][0]),
                            self.labelImageData_preview
                            * (255 - self.labelColors[self.currentLabelIdx][1]),
                            self.labelImageData_preview
                            * (255 - self.labelColors[self.currentLabelIdx][2]),
                            (self.labelImageData_preview > 0)
                            * 200
                            * self.label_transparency,
                        ),
                        axis=2,
                    )
                if self.transposed:
                    im_preview = np.transpose(im_preview, (1, 0, 2))
                self.labelImageItem_preview.setImage(im_preview)
                if self.labelImageItem_preview not in self.items: 
                    self.addItem(self.labelImageItem_preview)
                if self.cursorItem not in self.items:
                    self.addItem(self.cursorItem)
            else:
                # remove the label image item and cursor if they are added
                # if self.labelImageItem in self.items:
                #     self.removeItem(self.labelImageItem)
                if self.labelImageItem_preview in self.items:
                    self.removeItem(self.labelImageItem_preview)
                if self.cursorItem in self.items:
                    self.removeItem(self.cursorItem)
                if not self.hided_indicator:
                    if self.direction == 0:
                        if self.transposed:
                            super().plot(
                                [self.indicatorPos[1], self.indicatorPos[1]],
                                [0, self.plotSize[1]],
                                pen=pg.mkPen("y", width=2),
                            )
                        else:
                            super().plot(
                                [self.indicatorPos[1], self.indicatorPos[1]],
                                [0, self.plotSize[0]],
                                pen=pg.mkPen("y", width=2),
                            )
                    elif self.direction == 1:
                        if self.transposed:
                            super().plot(
                                [0, self.plotSize[0]],
                                [self.indicatorPos[0], self.indicatorPos[0]],
                                pen=pg.mkPen("y", width=2),
                            )
                        else:
                            super().plot(
                                [0, self.plotSize[1]],
                                [self.indicatorPos[0], self.indicatorPos[0]],
                                pen=pg.mkPen("y", width=2),
                            )

        if self.show_center_point and self.center_point is not None:
            if self.center_point_plot is not None:
                self.removeItem(self.center_point_plot)
            self.center_point_plot = pg.ScatterPlotItem(
                pos=[self.center_point],
                size=10,
                pen=pg.mkPen(None),
                brush=pg.mkBrush((0, 255, 0, 200)),
            )
            self.center_point_plot.setSymbol("+")
            if self.center_point_plot not in self.items:
                self.addItem(self.center_point_plot)

    def _generateCircleMask(self, mask_size, center, radius, thickness=-1):
        mask = np.zeros(mask_size, dtype=np.uint8)
        mask = cv2.circle(mask, center, radius, 1, thickness)
        return mask

    def _generateLineMask(self, mask_size, points, thickness):
        mask = np.zeros(mask_size, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        mask = cv2.polylines(mask, [pts], False, 1, thickness)
        return mask

    def _fill_regions(self, org_mask, preview_mask, current_label, fillHole):
        tmp_mask = org_mask == current_label
        tmp_mask[preview_mask != 0] = 1
        tmp_mask = utils.fill_holes(tmp_mask, fillHole)
        tmp_mask[org_mask == current_label] = 0
        tmp_mask[preview_mask != 0] = 1
        return tmp_mask

    def _remove_regions(self, org_mask, preview_mask, removeRgn):
        # self.labelImageData_preview[(brush_mask==1) &(self.labelImageData>0)] = 1
        tmp_mask = org_mask > 0
        tmp_mask[preview_mask != 0] = 0
        tmp_mask = utils.binary_area_open(tmp_mask, removeRgn)
        sml_regions = np.logical_and(np.logical_not(tmp_mask), org_mask > 0)
        preview_mask[sml_regions] = 1
        return preview_mask