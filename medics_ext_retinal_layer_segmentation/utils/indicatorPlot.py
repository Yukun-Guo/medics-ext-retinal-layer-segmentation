"""
Various methods of drawing scrolling plots.
"""

from PySide6 import QtCore
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsSceneHoverEvent, QFileDialog


class IndicatorPlot(pg.PlotItem):
    """A class for plotting indicators with interactive functionality."""

    sigIndicatorChanged = QtCore.Signal(object)  # Emitted when the indicator position changes
    sigMousePosChanged = QtCore.Signal(object)  # Emitted when the mouse position changes

    def __init__(
        self,
        parent: QtCore.QObject = None,
        parentWindow: QtCore.QObject = None,
        name: str = None,
        labels: dict = None,
        title: str = None,
        viewBox: pg.ViewBox = None,
        axisItems: dict = None,
        enableMenu: bool = True,
        **kargs,
    ) -> None:
        """Initializes the IndicatorPlot with optional parameters for customization.

        Args:
            parent (QtCore.QObject, optional): Parent object. Defaults to None.
            parentWindow (QtCore.QObject, optional): Parent window. Defaults to None.
            name (str, optional): Name of the plot. Defaults to None.
            labels (dict, optional): Axis labels. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            viewBox (pg.ViewBox, optional): ViewBox for the plot. Defaults to None.
            axisItems (dict, optional): Axis items for the plot. Defaults to None.
            enableMenu (bool, optional): Whether to enable the menu. Defaults to True.
        """
        super().__init__(parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs)
        self.vb.setDefaultPadding(0.001)
        self.parentWindow = parentWindow
        self.plotSize = [1, 1]
        self.indicatorPos = [0, 0]
        self.direction = 0
        self.added_items = None
        self.transposed = False
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

        self.menu = self.vb.getMenu(None)
        # Remove all actions except the first one
        for i in range(1, len(self.menu.actions())):
            self.menu.removeAction(self.menu.actions()[1])
        self.ctrlMenu.menuAction().setVisible(False)
        self.menu.addAction("Save image...", self.onSaveImageAs)

        if self.parentWindow.theme == "dark":
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(37, 37, 38); color: rgb(240, 240, 240);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )
        else:
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(240, 240, 240); color: rgb(37, 37, 38);} QMenu::item:selected {background-color: rgb(0, 120, 212);}"
            )

    def onSaveImageAs(self) -> None:
        """Saves the current plot as an image file."""
        filename,_ = QFileDialog.getSaveFileName(self.parentWindow, "Save Image", "", "Image PNG (*.png)")
        if not filename:
            return
        # Ensure the filename has a .png extension
        if not filename.endswith('.png'):
            filename += '.png'  
        if filename:
            if self.added_items is not None:
                self.added_items[0].save(filename)

    def setTransposed(self, transposed: bool) -> None:
        """Sets whether the plot is transposed.

        Args:
            transposed (bool): Whether the plot is transposed.
        """
        self.transposed = transposed

    def setPlotSize(self, psize: list) -> None:
        """Sets the size of the plot.

        Args:
            psize (list): Size of the plot [width, height].
        """
        self.plotSize = psize

    def setDirection(self, direction: int) -> None:
        """Sets the direction of the indicator.

        Args:
            direction (int): Direction of the indicator (0 for vertical, 1 for horizontal).
        """
        self.direction = direction

    def setIndicatorPos(self, pos: list) -> None:
        """Sets the position of the indicator.

        Args:
            pos (list): Position of the indicator [x, y].
        """
        self.indicatorPos = pos

    def set_added_items(self, added_items: list) -> None:
        """Sets additional items to the plot.

        Args:
            added_items (list): List of additional items.
        """
        if not isinstance(added_items, list):
            added_items = [added_items]
        self.added_items = added_items

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Handles hover move events to update the indicator position.

        Args:
            event (QGraphicsSceneHoverEvent): The hover move event.
        """
        pos = event.scenePos()
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.vb.mapSceneToView(pos)
            self.sigMousePosChanged.emit([np.int32(mouse_point.x()), np.int32(mouse_point.y())])
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                self.indicatorPos = [np.floor(mouse_point.x()), np.floor(mouse_point.y())]
                self.sigIndicatorChanged.emit(self.indicatorPos)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QtCore.QEvent) -> None:
        """Handles mouse press events to update the indicator position.

        Args:
            event (QtCore.QEvent): The mouse press event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                self.indicatorPos = [np.int32(mouse_point.x()), np.int32(mouse_point.y())]
                self.sigIndicatorChanged.emit(self.indicatorPos)
                self.sigMousePosChanged.emit([np.int32(mouse_point.x()), np.int32(mouse_point.y())])
        super().mousePressEvent(event)

    def plot(self, *args, **kargs) -> None:
        """Plots the indicator line."""
        self.clear()
        if self.added_items is not None:
            for item in self.added_items:
                self.addItem(item)

        if self.direction == 0:  # Vertical indicator
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
        elif self.direction == 1:  # Horizontal indicator
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


