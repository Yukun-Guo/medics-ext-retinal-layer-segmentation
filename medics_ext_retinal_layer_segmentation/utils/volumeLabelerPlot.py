"""
Various methods of drawing scrolling plots.
"""

import json
from PySide6 import QtCore
from PySide6.QtWidgets import QGraphicsEllipseItem,QGraphicsTextItem,QFileDialog,QMessageBox
from PySide6.QtCore import QThread, Signal

import numpy as np
import pyqtgraph as pg
import cv2
import os
from .utils import utils
import time
from typing import Any, List, Union
import tifffile
import json
# VTK imports with error handling
try:
    import vtkmodules.all as vtk
    VTK_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("VTK modules not available in volumeLabelerPlot: %s", e)
    logger.warning("VTK-related functionality will be disabled.")
    # Create a placeholder vtk module
    class VTKPlaceholder:
        pass
    vtk = VTKPlaceholder()
    VTK_AVAILABLE = False


class StackImageSaveWorker(QThread):
    """
    Worker thread for saving stack images to prevent UI blocking.
    """
    finished = Signal(str)  # Signal emitted when save is complete with success message
    error = Signal(str)     # Signal emitted when an error occurs
    progress = Signal(int)  # Signal for progress updates (0-100)

    def __init__(self, filename, stack_data, save_type, **kwargs):
        super().__init__()
        self.filename = filename
        self.stack_data = stack_data
        self.save_type = save_type  # 'stack', 'labels', 'stack_with_labels'
        self.kwargs = kwargs

    def run(self):
        try:
            if self.save_type == 'stack':
                self._save_stack_image()
            elif self.save_type == 'labels':
                self._save_label_stack()
            elif self.save_type == 'stack_with_labels':
                self._save_stack_with_labels()
            
            self.finished.emit(f"Successfully saved {self.save_type} to {self.filename}")
            
        except Exception as e:
            self.error.emit(f"Error saving {self.save_type}: {str(e)}")

    def _save_stack_image(self):
        """Save regular stack image."""
        self.progress.emit(50)
        self._write_tiff_file(self.filename, self.stack_data)
        self.progress.emit(100)

    def _save_label_stack(self):
        """Save label stack with colormap."""
        self.progress.emit(50)
        colormap = self.kwargs.get('colormap')
        self._write_tiff_file(self.filename, self.stack_data, colormap=colormap)
        self.progress.emit(100)

    def _save_stack_with_labels(self):
        """Save stack with overlay labels."""
        volume_data = self.kwargs.get('volume_data')
        label_volume_data = self.kwargs.get('label_volume_data')
        label_colors = self.kwargs.get('label_colors')
        transparency = self.kwargs.get('transparency', 0.6)
        
        # Create overlay for the entire stack
        stack_with_labels = np.zeros((*volume_data.shape, 4), dtype=np.uint8)
        total_frames = volume_data.shape[0]
        
        for i in range(total_frames):
            # Update progress
            progress = int((i / total_frames) * 90)  # Reserve 10% for file writing
            self.progress.emit(progress)
            
            # Convert grayscale image to RGB
            gray_img = volume_data[i]
            rgb_img = np.stack([gray_img, gray_img, gray_img], axis=2)
            
            # Create label overlay for current frame
            label_colors_frame = np.array(label_colors[label_volume_data[i].astype('uint8')], dtype=np.uint8)
            label_mask = label_volume_data[i] > 0
            
            # Blend the image with labels
            overlay = rgb_img.copy()
            overlay[label_mask] = (1 - transparency) * rgb_img[label_mask] + \
                                 transparency * label_colors_frame[label_mask]
            
            # Create alpha channel (fully opaque for all pixels)
            alpha = np.full(gray_img.shape, 255, dtype=np.uint8)
            
            # Combine RGB + Alpha
            stack_with_labels[i] = np.stack([overlay[:,:,0], overlay[:,:,1], overlay[:,:,2], alpha], axis=2)
        
        # Save the file
        self.progress.emit(95)
        description = {"type": "image_with_labels", "transparency": transparency}
        self._write_tiff_file(self.filename, stack_with_labels, description=description)
        self.progress.emit(100)

    def _write_tiff_file(self, filename, data, colormap=None, description=None):
        """Write TIFF file with optional colormap."""
        if colormap is None:
            tifffile.imwrite(filename, data, metadata={"axes": "ZXY"}, compression="zlib", description=json.dumps(description))
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
            tifffile.imwrite(filename, data, metadata={"axes": "ZXY"}, colormap=colormap, compression="zlib", description=json.dumps(description))


class VolumeLabelerPlot(pg.PlotItem):
    sigIndicatorChanged = QtCore.Signal(object)  ## Emitted when the gCurve has changed
    sigLabelVolumeChanged = QtCore.Signal(object,object)  ## Emitted when the label image has changed
    sigMousePosChanged = QtCore.Signal(object)  ## Emitted when the mouse position has changed
    sigCenterPointChanged = QtCore.Signal(object)  ## Emitted when the center point has changed
    def __init__(
        self, parentWindow=None, parent=None, name=None, labels=None, title=None, viewBox=None, axisItems=None, enableMenu=True,disable_label=False, **kargs
    ):
        """
        Initialize the IndicatorPlot with optional parameters for customization.
        """
        super().__init__(parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs)
        self.vb.setDefaultPadding(0.001)
        self.setAcceptHoverEvents(True)
        self.parentWindow = parentWindow
        self.gCurves = None
        self.roi_position=[0,0]
        self.plotSize = [1,1]
        self.indicatorPos=[0,0]
        self.direction=0
        self.prop_step=1
        self.disable_label=disable_label
        self.hided_indicator=True
        self.added_items=None
        self.transposed = False
        self.draw_center_point = False
        self.show_center_point = True
        self.center_point_plot = None
        self.center_point = None
        self.center_point_indicator_X = None
        self.center_point_indicator_Y = None
        self.imageItemBFrame = pg.ImageItem()
        self.volumeData=None
        self.slab_mask=None
        self.labelImageItemBFrame = pg.ImageItem()
        self.labelVolumeData=None
        self.labelVolumeData=None
        self.labelVolumeData_prev=None
        self.labelImageItemBFrame_preview = pg.ImageItem()
        self.labelVolumeData_preview=None
        self.labelVolumeData_preview=None
        self.upperBoundary = None
        self.lowerBoundary = None
        self.cursorItem = QGraphicsEllipseItem()
        self.cursorImageData=None
        self.labelColors = None
        self.indicator_plot = None
        self.currentLabelIdx = 1
        self.currentFrameIdx = 0
        self.toolSize = 15
        self.labelTool = 'off'
        self.cursor_pos = [0,0]
        self.label_transparency = 0.6
        self.mousePath = []
        self.modifiedFrameIndexs = None
        self.modifiedFrameIndexs_prev = None
        self.is_dragging=False
        self.mouseButtonPressed = QtCore.Qt.MouseButton.NoButton
        self.fillHole=15
        self.removeRgn=15
        self.offset_lower=0
        self.offset_upper=0
        self.loDiff=-20
        self.upDiff=20
        
        # Worker thread for saving stack images
        self.save_worker = None
        
        self.threshold_filter3D = vtk.vtkImageThresholdConnectivity()
        self.vtk_image_data = None
        
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.cursorItem.setPen(pg.mkPen('g', width=2))
        self.cursorItem.setBrush(pg.mkBrush(0, 255, 0, 10))
        self.cursorItem.wheelEvent = self.cursorItem_wheelEvent

        self.setAcceptHoverEvents(True)

        self.menu = self.vb.getMenu(None)
        for i in range(1, len(self.menu.actions())):
            self.menu.removeAction(self.menu.actions()[1])
        # add menu items
        self.menu.addAction('Save image...', self.onSaveImageAs)
        self.menu.addAction('Save stack image...', self.onSaveStackImageAs)
        if not self.disable_label:
            self.menu.addAction('Save label...', self.onSaveLabelAs)
            self.menu.addAction('Save stack label...', self.onSaveLabelStackAs)
            self.menu.addAction('Save image with label...', self.onSaveImageWithLabelAs)
            self.menu.addAction('Save stack image with label...', self.onSaveStackImageWithLabelAs)
            
        self.ctrlMenu.menuAction().setVisible(False)
        if self.parentWindow.theme == 'dark':
            self.menu.setStyleSheet("QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}")
        else:
            self.menu.setStyleSheet("QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}")
        
    def cursorItem_wheelEvent(self, event):
        # check if the key control is pressed
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.wheelEvent(event)
        else:  
            # if scrolling up, increase the tool size by 1, otherwise decrease by 1
            if event.delta() > 0:
                self.toolSize = min(self.toolSize+1,self.parentWindow.ui.spinBox_toolSize.maximum())
            else:
                self.toolSize = max(self.toolSize-1,self.parentWindow.ui.spinBox_toolSize.minimum())
            self.parentWindow.ui.spinBox_toolSize.setValue(self.toolSize)
            self.cursorItem.setRect(self.cursor_pos[0]-self.toolSize/2,self.cursor_pos[1]-self.toolSize/2,self.toolSize,self.toolSize)
            self.refreshLabelImageItemBFrame()
            self.plot()
    
    def set_roi(self, roi_position: list) -> None:
        """Sets the position of the ROI.

        Args:
            roi_position (list): The ROI position.
        """
        self.roi_position = roi_position
    
    def set_curve(self, curve, curve_props:dict=None):
        """
        Set the curves and their properties.
        """
        self.gCurves = curve
        self.curveLen = len(curve[0])
        self.curve_props = curve_props
    
    def setFillHole(self, fillHole):
        self.fillHole = fillHole
    
    def setPropagateStep(self, prop_step):
        self.prop_step = prop_step
    
    def setRemoveRgn(self, removeRgn):
        self.removeRgn = removeRgn
    
    def setOffset_lower(self, offset):
        self.offset_lower = offset
    
    def setOffset_upper(self, offset):
        self.offset_upper = offset
    
    def setLoDiff(self, loDiff):
        self.loDiff = loDiff
    
    def setUpDiff(self, upDiff):
        self.upDiff = upDiff
    
    def set_center_point(self,p):
        self.center_point = p
        self.refreshLabelImageItemBFrame()
        self.plot()
    
    def setCurrentColorIndex(self, idx):
        self.currentLabelIdx = idx
    
    def setLabelTransparency(self, transparency):
        self.label_transparency = transparency
        self.refreshLabelImageItemBFrame()
        self.plot()
    
    def setLabelTool(self, tool:str):
        self.labelTool = tool.lower()
    
    def setLabelColors(self, colors):
        # colors = np.asarray(colors, dtype=np.float32)
        # self.labelColors = np.column_stack((colors, np.ones(len(colors), dtype=np.float32)*255))
        self.labelColors = np.asarray(colors, dtype=np.uint8)
        
    def onLabelColorChanged(self, colors):
        self.labelColors = np.asarray(colors, dtype=np.uint8)
        self.refreshLabelImageItemBFrame()
        self.plot()
    
    def setCursorSize(self, size):
        self.toolSize = size
    
    def setCurrentFrameIndex(self, idx):
        self.currentFrameIdx = idx
        self.imageItemBFrame.setImage(self.volumeData[self.currentFrameIdx])
        self.refreshLabelImageItemBFrame()
        self.plot()
    
    def setSlabInfo(self, slab_mask,upperMin, lowerMax):
        self.upperBoundary = upperMin
        self.lowerBoundary = lowerMax
        pad_before = upperMin
        pad_after = self.volumeData.shape[1]-lowerMax
        self.slab_mask = np.pad(slab_mask,((0,0),(pad_before,pad_after),(0,0)),mode='constant')
        
    
    def setVolumeData(self, data,levels=None):
        self.volumeData = np.ascontiguousarray(data, dtype=np.uint8)
        self.labelVolumeData = np.zeros_like(self.volumeData, dtype=np.uint8)
        self.labelVolumeData_prev = np.zeros_like(self.volumeData, dtype=np.uint8)
        self.labelVolumeData_preview = np.zeros_like(self.volumeData, dtype=np.uint8)
        self.modifiedFrameIndexs = np.zeros(self.volumeData.shape[0],dtype=bool)
        
        self.imageItemBFrame.setImage(self.volumeData[self.currentFrameIdx],levels=levels)
        self.cursorImageData  = np.zeros_like(self.volumeData[0], dtype=np.uint8)
    
    def setLabelVolumeData(self, data,update_range=None):
        if data is not None:
            if update_range is not None:
                self.labelVolumeData[update_range] = data[update_range].astype(np.uint8).copy()
                self.labelVolumeData_prev[update_range] = self.labelVolumeData[update_range].copy()
            else:
                self.labelVolumeData = data.astype(np.uint8)
                self.labelVolumeData_prev = self.labelVolumeData.copy()                
            self.refreshLabelImageItemBFrame()
            
    
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
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Image", '', "Image (*.png)")[0]
        if filename:
            self.imageItemBFrame.save(filename)
    
    def onSaveStackImageAs(self):
        """
        Save the stack image as a file using a worker thread.
        """
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Stack Image", '', "Image (*.tiff)")[0]
        if filename:
            if self.volumeData is not None:
                # Add .tiff extension if not present
                if not filename.endswith('.tiff'):
                    filename += '.tiff'
                
                # Start worker thread
                self._start_save_worker('stack', filename, self.volumeData)

    def onSaveImageWithLabelAs(self):
        """
        Save the image with label as a file.
        """
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Image with Label", '', "Image (*.png)")[0]
        if filename:
            if self.volumeData is not None and self.labelVolumeData is not None:
                # overlay the label on the image
                im = np.array(self.labelColors[self.labelVolumeData[self.currentFrameIdx].astype('uint8')],dtype=np.uint8)
                # add transparency
                im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2],255*self.label_transparency*(self.labelVolumeData[self.currentFrameIdx]>0)),axis=2)
                cv2.imwrite(filename, im)
    
    def onSaveLabelStackAs(self):
        """
        Save the label stack as a file using a worker thread.
        """
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Label Stack", '', "Image (*.tiff)")[0]
        if filename:
            if self.labelVolumeData is not None:
                # Add .tiff extension if not present
                if not filename.endswith('.tiff'):
                    filename += '.tiff'
                
                # Start worker thread
                self._start_save_worker('labels', filename, self.labelVolumeData, colormap=self.labelColors)
    def onSaveStackImageWithLabelAs(self):
        """ 
        Save the stack image with label as a file using a worker thread.
        """
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Stack Image with Label", '', "Image (*.tiff)")[0]
        if filename:
            if self.volumeData is not None and self.labelVolumeData is not None:
                # Add .tiff extension if not present
                if not filename.endswith('.tiff'):
                    filename += '.tiff'
                
                # Start worker thread
                self._start_save_worker('stack_with_labels', filename, None,
                                      volume_data=self.volumeData,
                                      label_volume_data=self.labelVolumeData,
                                      label_colors=self.labelColors,
                                      transparency=self.label_transparency)
            

    def onSaveLabelAs(self):
        """
        Save the label image as a file.
        """
        filename = QFileDialog.getSaveFileName(self.parentWindow, "Save Label", '', "Image (*.png)")[0]
        if filename:
            self.labelImageItemBFrame.save(filename)
    
    def hoverMoveEvent(self, event):
        if self.volumeData is not None:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                self.cursor_pos = (mouse_point.x(),mouse_point.y())             
                self.sigMousePosChanged.emit([np.int32(mouse_point.x()),np.int32(mouse_point.y())]) 
                rt_view = QtCore.QRectF(mouse_point.x()-self.toolSize/2,mouse_point.y()-self.toolSize/2,self.toolSize,self.toolSize)
                self.cursorItem.setRect(rt_view)
                self.plot()
        return super().hoverMoveEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.is_dragging:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                # refresh cursor position
                self.cursor_pos = (mouse_point.x(),mouse_point.y())              
                self.sigMousePosChanged.emit([np.int32(mouse_point.x()),np.int32(mouse_point.y())]) 
                rt_view = QtCore.QRectF(mouse_point.x()-self.toolSize/2,mouse_point.y()-self.toolSize/2,self.toolSize,self.toolSize)
                self.cursorItem.setRect(rt_view) 
                self.mousePath.append(np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32))
                
            self.handleMouseDrag(event)
        super().mouseMoveEvent(event)
    
    def handleMouseDrag(self, event):
        if self.mouseButtonPressed == QtCore.Qt.MouseButton.LeftButton:
            if self.labelTool == 'brush':
                if len(self.mousePath) > 0:
                    brush_mask = self._generateLineMask(self.labelVolumeData[self.currentFrameIdx].shape, self.mousePath, self.toolSize)
                    self.labelVolumeData_preview[self.currentFrameIdx][brush_mask==1] = 1
                    self.labelVolumeData_preview[self.currentFrameIdx] = self._fill_regions(self.labelVolumeData[self.currentFrameIdx], self.labelVolumeData_preview[self.currentFrameIdx],self.currentLabelIdx, self.fillHole)
                    self.modifiedFrameIndexs[self.currentFrameIdx] = True
                    # self.refreshLabelImageItemBFrame()
                    self.plot()
            elif self.labelTool == 'region-grow':
                pass # rewrite 3D region grow
            elif self.labelTool == 'adapt-thresh':
                pass # rewrite 3D adapt-thresh
            else:
                pass
        elif self.mouseButtonPressed == QtCore.Qt.MouseButton.RightButton:
            
            if self.labelTool == 'brush' or self.labelTool == 'thresholdupper' or self.labelTool == 'thresholdlower' or self.labelTool == 'thresholdbetween':
                if len(self.mousePath) > 0:
                    brush_mask = self._generateLineMask(self.labelVolumeData[self.currentFrameIdx].shape, self.mousePath, self.toolSize)
                    self.labelVolumeData_preview[self.currentFrameIdx][(brush_mask==1) &(self.labelVolumeData[self.currentFrameIdx]>0)] = 1
                    self.labelVolumeData_preview[self.currentFrameIdx] = self._remove_regions(self.labelVolumeData[self.currentFrameIdx], self.labelVolumeData_preview[self.currentFrameIdx], self.removeRgn)
                    self.modifiedFrameIndexs[self.currentFrameIdx] = True
                    # self.refreshLabelImageItemBFrame()
                    self.plot()
            else:
                pass
        
    
    def mouseReleaseEvent(self, event):
        if self.labelVolumeData is None:
            super().mouseReleaseEvent(event)
            return
        self.is_dragging = False
        self.mousePath = []
        self.mouseButtonPressed = QtCore.Qt.MouseButton.NoButton
        self.labelVolumeData_prev = self.labelVolumeData.copy()
        self.labelVolumeData_preview[self.slab_mask==0] = 0
        
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            idxs = np.nonzero(self.modifiedFrameIndexs)[0]
            if idxs.size == 0:
                return
            self.labelVolumeData[idxs[0]:idxs[-1]+1][self.labelVolumeData_preview[idxs[0]:idxs[-1]+1]!=0] = self.currentLabelIdx
            self.labelVolumeData_preview = np.zeros_like(self.labelVolumeData)
            self.modifiedFrameIndexs_prev = self.modifiedFrameIndexs.copy()
            self.modifiedFrameIndexs = np.zeros(self.volumeData.shape[0],dtype=bool)
            self.sigLabelVolumeChanged.emit(self.labelVolumeData,range(idxs[0],idxs[-1]+1))
            self.refreshLabelImageItemBFrame()
            self.plot()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            idxs = np.nonzero(self.modifiedFrameIndexs)[0]
            if idxs.size == 0:
                return
            nnzs = np.nonzero(self.labelVolumeData_preview[idxs[0]:idxs[-1]+1])
            if idxs.size == 0 or nnzs[0].size == 0:
                self.labelVolumeData_preview = np.zeros_like(self.labelVolumeData)
                self.modifiedFrameIndexs = np.zeros(self.volumeData.shape[0],dtype=bool)
                return
            self.labelVolumeData[idxs[0]:idxs[-1]+1][self.labelVolumeData_preview[idxs[0]:idxs[-1]+1]!=0] = 0
            self.labelVolumeData_preview = np.zeros_like(self.labelVolumeData)
            self.modifiedFrameIndexs_prev = self.modifiedFrameIndexs.copy()
            self.modifiedFrameIndexs = np.zeros(self.volumeData.shape[0],dtype=bool)
            self.sigLabelVolumeChanged.emit(self.labelVolumeData,range(idxs[0],idxs[-1]+1))
            self.refreshLabelImageItemBFrame()
            self.plot()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            idxs = np.nonzero(self.modifiedFrameIndexs_prev)[0]
            if idxs.size == 0:
                return
            self.labelVolumeData[idxs[0]:idxs[-1]+1] = self.labelVolumeData_prev[idxs[0]:idxs[-1]+1].copy()
            self.labelVolumeData_preview = np.zeros_like(self.labelVolumeData)
            self.sigLabelVolumeChanged.emit(self.labelVolumeData,range(idxs[0],idxs[-1]+1))
            self.refreshLabelImageItemBFrame()
            self.plot()
        super().mouseReleaseEvent(event)
    
    def mousePressEvent(self, event):
        if self.volumeData is None:
            super().mousePressEvent(event)
            return
        pos = event.scenePos()
        self.mouseButtonPressed = event.button()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                if self.sceneBoundingRect().contains(pos) and self.imageItemBFrame.image is not None:
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.indicatorPos = [np.floor(mouse_point.x()),np.floor(mouse_point.y())]
                    self.sigIndicatorChanged.emit(self.indicatorPos)
                    self.sigMousePosChanged.emit([np.int32(mouse_point.x()),np.int32(mouse_point.y())]) 
                super().mousePressEvent(event)
                return
            if self.labelTool == 'brush':
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
                    brush_mask = self._generateCircleMask(self.labelVolumeData[0].shape, cursor_pos, self.toolSize//2,thickness=-1)
                    self.labelVolumeData_preview[self.currentFrameIdx][brush_mask==1] = 1
                    self.labelVolumeData_preview[self.currentFrameIdx] = self._fill_regions(self.labelVolumeData[self.currentFrameIdx], self.labelVolumeData_preview[self.currentFrameIdx],self.currentLabelIdx, self.fillHole)
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()
            elif self.labelTool == 'thresholdlower':
                if self.sceneBoundingRect().contains(pos):
                    # offset = self.parentWindow.ui.spinBox_offset_2.value()
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
                    brush_mask,roi = utils.generateAdaptThresMask3D(self.volumeData, self.currentFrameIdx,cursor_pos, self.toolSize//2,loweroffset=-255,upperoffset=self.offset_lower,propagate_step=self.prop_step)
                    self.labelVolumeData_preview[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]][brush_mask] = 1
                    self.modifiedFrameIndexs[roi[0]:roi[1]] = True
                    self.mousePath = [cursor_pos]
                    # self.handleMouseDrag(event)
                    self.plot()
            elif self.labelTool == 'thresholdupper':
                if self.sceneBoundingRect().contains(pos):
                    # offset = self.parentWindow.ui.spinBox_offset.value()
                    mouse_point = self.vb.mapSceneToView(pos)
                    cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
                    brush_mask,roi = utils.generateAdaptThresMask3D(self.volumeData, self.currentFrameIdx,cursor_pos, self.toolSize//2,loweroffset=self.offset_upper,upperoffset=255,propagate_step=self.prop_step)
                    self.labelVolumeData_preview[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]][brush_mask] = 1
                    self.modifiedFrameIndexs[roi[0]:roi[1]] = True
                    self.mousePath = [cursor_pos]
                    self.plot()
            elif self.labelTool == 'thresholdbetween':
                # pass # rewrite 3D region grow
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
                    brush_mask,roi = utils.generateAdaptThresMask3D(self.volumeData, self.currentFrameIdx,cursor_pos, self.toolSize//2,loweroffset=self.loDiff, upperoffset=self.upDiff,propagate_step=self.prop_step)
                    self.labelVolumeData_preview[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]][brush_mask] = 1
                    self.modifiedFrameIndexs[roi[0]:roi[1]] = True
                    self.mousePath = [cursor_pos]
                    self.plot()
            else: # tool is off
                if self.sceneBoundingRect().contains(pos) and self.imageItemBFrame.image is not None:
                    mouse_point = self.vb.mapSceneToView(pos)
                    self.indicatorPos = [np.floor(mouse_point.x()),np.floor(mouse_point.y())]
                    self.sigIndicatorChanged.emit(self.indicatorPos)
                    self.sigMousePosChanged.emit([np.int32(mouse_point.x()),np.int32(mouse_point.y())]) 
                super().mousePressEvent(event)
                 
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                super().mousePressEvent(event)
                return
            if self.labelTool == 'brush' or self.labelTool == 'thresholdupper' or self.labelTool == 'thresholdlower' or self.labelTool == 'thresholdbetween':
                if self.sceneBoundingRect().contains(pos):
                    mouse_point = self.vb.mapSceneToView(pos)
                    # generate a brush mask, brush size is self.toolSize
                    cursor_pos = np.array([mouse_point.x(),mouse_point.y()],dtype=np.int32)
                    brush_mask = self._generateCircleMask(self.labelVolumeData[0].shape, cursor_pos, self.toolSize//2,thickness=-1)
                    self.labelVolumeData_preview[self.currentFrameIdx] = brush_mask*(self.labelVolumeData[self.currentFrameIdx]>0)
                    self.labelVolumeData_preview[self.currentFrameIdx] = self._remove_regions(self.labelVolumeData[self.currentFrameIdx], self.labelVolumeData_preview[self.currentFrameIdx], self.removeRgn)
                    self.is_dragging = True
                    self.mousePath = [cursor_pos]
                    self.handleMouseDrag(event)
                    self.plot()
            elif self.labelTool == 'off':
                super().mousePressEvent(event)
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                super().mousePressEvent(event)
                return
            if self.labelTool == 'off':
                super().mousePressEvent(event)
            else:
                self.labelVolumeData = self.labelVolumeData_prev.copy()
                self.plot()
                
    def plot_indicator(self, pos: list, direction='vertical'):
        """
        Plot the indicator.
        """
        if pos is None:
            if self.indicator_plot is not None:
                self.removeItem(self.indicator_plot)
            return
        if self.indicator_plot is not None:
            self.removeItem(self.indicator_plot)
            
        if self.cursorItem in self.items:
                self.removeItem(self.cursorItem)
                
        # get plot range
        h,w = self.imageItemBFrame.image.shape
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
    
    def refreshImageItemBFrame(self, img=None, levels=None):
        """
        Refresh the image item.
        """
        if img is not None:
            self.imageItemBFrame.setImage(img, levels=levels)
        else:
            if self.volumeData is not None and self.currentFrameIdx is not None:
                self.imageItemBFrame.setImage(self.volumeData[self.currentFrameIdx])
    
    def refreshLabelImageItemBFrame(self):
        """
        Refresh the label image item.
        """
        if self.labelVolumeData is not None and self.labelColors is not None and self.currentFrameIdx is not None:
            im = np.array(self.labelColors[self.labelVolumeData[self.currentFrameIdx].astype('uint8')],dtype=np.uint8)
            # add transparency
            im = np.stack((im[:,:,0],im[:,:,1],im[:,:,2],255*self.label_transparency*(self.labelVolumeData[self.currentFrameIdx]>0)),axis=2)
            if self.transposed:
                im = np.transpose(im,(1,0,2))
            self.labelImageItemBFrame.setImage(im)
        
    
    def plot(self, *args, **kargs):
        """
        Plot the indicator line.
        """
        self.clear()
        if self.added_items is not None:
            for item in self.added_items:
                self.addItem(item)
        if self.imageItemBFrame is not None:
            self.addItem(self.imageItemBFrame)
        
        if self.labelImageItemBFrame is not None:
            self.addItem(self.labelImageItemBFrame)
            
        
        if self.gCurves is not None and self.curve_props is not None:
            for curve, props in zip(self.gCurves, self.curve_props):
                if props["is_visible"]:
                    super().plot(curve, pen=pg.mkPen(props["color"], style=props["style"], width=props["width"], connect=props["connect"]))
                
        if self.labelVolumeData_preview is not None:
            if self.labelTool != 'off':

                if self.mouseButtonPressed == QtCore.Qt.MouseButton.LeftButton:
                    
                    im_preview = self.labelVolumeData_preview[self.currentFrameIdx] + self.slab_mask[self.currentFrameIdx]*5
                    im_preview[self.labelVolumeData_preview[self.currentFrameIdx]==0] = 0
                    
                    im_preview = np.stack((self.labelVolumeData_preview[self.currentFrameIdx]*self.labelColors[self.currentLabelIdx][0],
                                        self.labelVolumeData_preview[self.currentFrameIdx]*self.labelColors[self.currentLabelIdx][1],
                                        self.labelVolumeData_preview[self.currentFrameIdx]*self.labelColors[self.currentLabelIdx][2],
                                        im_preview*50*self.label_transparency),axis=2)
                else:
                    im_preview = np.stack((self.labelVolumeData_preview[self.currentFrameIdx]*(255-self.labelColors[self.currentLabelIdx][0]),
                                        self.labelVolumeData_preview[self.currentFrameIdx]*(255-self.labelColors[self.currentLabelIdx][1]),
                                        self.labelVolumeData_preview[self.currentFrameIdx]*(255-self.labelColors[self.currentLabelIdx][2]),
                                        self.labelVolumeData_preview[self.currentFrameIdx]*200*self.label_transparency),axis=2)
                if self.transposed:
                    im_preview = np.transpose(im_preview,(1,0,2))
                self.labelImageItemBFrame_preview.setImage(im_preview)
                
                self.addItem(self.labelImageItemBFrame_preview)
                self.addItem(self.cursorItem)
            else:
                # remove the label image item and cursor if they are added
                if self.labelImageItemBFrame_preview in self.items:
                    self.removeItem(self.labelImageItemBFrame_preview)
                if self.cursorItem in self.items:
                    self.removeItem(self.cursorItem)
                if not self.hided_indicator:
                    if self.direction==0:
                        if self.transposed:
                            super().plot([self.indicatorPos[1], self.indicatorPos[1]],[0, self.plotSize[1]] , pen=pg.mkPen("y", width=2))  
                        else:
                            super().plot([self.indicatorPos[1], self.indicatorPos[1]],[0, self.plotSize[0]] , pen=pg.mkPen("y", width=2))  
                    elif self.direction==1:
                        if self.transposed:
                            super().plot([0, self.plotSize[0]],[self.indicatorPos[0], self.indicatorPos[0]], pen=pg.mkPen("y", width=2))
                        else:
                            super().plot([0, self.plotSize[1]],[self.indicatorPos[0], self.indicatorPos[0]], pen=pg.mkPen("y", width=2))

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
            
    def _start_save_worker(self, save_type, filename, stack_data, **kwargs):
        """
        Start the worker thread for saving stack images.
        """
        # Stop any existing worker
        if self.save_worker and self.save_worker.isRunning():
            self.save_worker.terminate()
            self.save_worker.wait()
        
        # Create and start new worker
        self.save_worker = StackImageSaveWorker(filename, stack_data, save_type, **kwargs)
        self.save_worker.finished.connect(self._on_save_finished)
        self.save_worker.error.connect(self._on_save_error)
        self.save_worker.progress.connect(self._on_save_progress)
        
        # Show progress indication
        self.parentWindow.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        
        # Try to show status message safely
        try:
            if hasattr(self.parentWindow, 'ui') and hasattr(self.parentWindow.ui, 'statusbar'):
                self.parentWindow.ui.statusbar.showMessage(f"Saving {save_type}... 0%")
            elif hasattr(self.parentWindow, 'statusBar'):
                self.parentWindow.statusBar().showMessage(f"Saving {save_type}... 0%")
        except:
            pass  # Silently ignore statusbar errors
        
        self.save_worker.start()
    
    def _on_save_progress(self, progress):
        """
        Handle progress updates from the save worker.
        """
        try:
            if hasattr(self.parentWindow, 'ui') and hasattr(self.parentWindow.ui, 'statusbar'):
                self.parentWindow.ui.statusbar.showMessage(f"Saving... {progress}%")
            elif hasattr(self.parentWindow, 'statusBar'):
                self.parentWindow.statusBar().showMessage(f"Saving... {progress}%")
        except:
            pass  # Silently ignore statusbar errors
    
    def _on_save_finished(self, message):
        """
        Handle successful completion of save operation.
        """
        self.parentWindow.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        try:
            if hasattr(self.parentWindow, 'ui') and hasattr(self.parentWindow.ui, 'statusbar'):
                self.parentWindow.ui.statusbar.showMessage(message, 3000)  # Show for 3 seconds
            elif hasattr(self.parentWindow, 'statusBar'):
                self.parentWindow.statusBar().showMessage(message, 3000)
        except:
            pass  # Silently ignore statusbar errors
    
    def _on_save_error(self, error_message):
        """
        Handle errors from the save worker.
        """
        self.parentWindow.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        try:
            if hasattr(self.parentWindow, 'ui') and hasattr(self.parentWindow.ui, 'statusbar'):
                self.parentWindow.ui.statusbar.showMessage("Save failed", 3000)
            elif hasattr(self.parentWindow, 'statusBar'):
                self.parentWindow.statusBar().showMessage("Save failed", 3000)
        except:
            pass  # Silently ignore statusbar errors
        
        # Show error dialog
        msg_box = QMessageBox(self.parentWindow)
        msg_box.setWindowTitle("Save Error")
        msg_box.setText("Failed to save file")
        msg_box.setDetailedText(error_message)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.exec_()
            
    def _generateCircleMask(self, mask_size, center, radius,thickness=-1):
        mask = np.zeros(mask_size, dtype=np.uint8)
        mask = cv2.circle(mask, center, radius, 1, thickness)
        return mask
    
    def _generateLineMask(self, mask_size, points, thickness):
        mask = np.zeros(mask_size, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        mask = cv2.polylines(mask, [pts], False, 1, thickness)
        return mask

    def _fill_regions(self, org_mask, preview_mask,current_label, fillHole):
        tmp_mask = org_mask==current_label
        tmp_mask[preview_mask!=0] = 1
        tmp_mask = utils.fill_holes(tmp_mask, fillHole)
        tmp_mask[org_mask==current_label] = 0
        tmp_mask[preview_mask!=0] = 1
        return tmp_mask
    
    def _remove_regions(self, org_mask, preview_mask, removeRgn):
        # self.labelVolumeData_preview[(brush_mask==1) &(self.labelVolumeData>0)] = 1
        tmp_mask = org_mask>0
        tmp_mask[preview_mask!=0] = 0
        tmp_mask = utils.binary_area_open(tmp_mask, removeRgn)
        sml_regions = np.logical_and(np.logical_not(tmp_mask),org_mask>0)
        preview_mask[sml_regions] = 1
        return preview_mask

    def cleanup(self):
        """
        Cleanup method to stop any running worker threads.
        Call this when the plot is being destroyed.
        """
        if self.save_worker and self.save_worker.isRunning():
            self.save_worker.terminate()
            self.save_worker.wait()
