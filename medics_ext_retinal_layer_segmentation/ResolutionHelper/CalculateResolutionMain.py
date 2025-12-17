import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QFileInfo, QCoreApplication, QByteArray, Slot, QUrl
from PySide6.QtGui import QImage, QPixmap, QPalette, QPainter, QAction, QIcon, QDoubleValidator
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
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
)
from .CalculateResolutionDlg import Ui_CalculateResolutionDlg

class CalculateResolution(QDialog):
    """Dialog for calculating and updating OCT image resolution."""

    def __init__(self, parentWindow: QWidget = None, parent: QWidget = None) -> None:
        """Initializes the CalculateResolution dialog.

        Args:
            parentWindow (QWidget, optional): The parent window containing scan and resolution info.
            parent (QWidget, optional): The parent widget for the dialog.
        """
        super(CalculateResolution, self).__init__(parent)
        self.parentWindow = parentWindow
        self.ui = Ui_CalculateResolutionDlg()
        self.ui.setupUi(self)

        self.ui.lineEdit.setValidator(QDoubleValidator())
        self.ui.lineEdit.setText(str(self.parentWindow.scan_width_mm))
        self.ui.lineEdit.textChanged.connect(self.onInputChanged)
        self.ui.lineEdit_2.setText(str(self.parentWindow.scan_height_mm))
        self.ui.lineEdit_2.setValidator(QDoubleValidator())
        self.ui.lineEdit_2.textChanged.connect(self.onInputChanged)
        self.ui.lineEdit_6.setValidator(QDoubleValidator())
        self.ui.lineEdit_6.textChanged.connect(self.onInputChanged)
        self.ui.lineEdit.selectAll()
        self.ui.buttonBox.accepted.connect(self.calculateResolution)

        self.ui.pushButton_help.clicked.connect(self.showHelp)

        self.resolution_w = self.parentWindow.resolution_width
        self.ui.lineEdit_8.setText(str(self.resolution_w))
        self.resolution_h = self.parentWindow.resolution_height
        self.ui.lineEdit_5.setText(str(self.resolution_h))
        self.resolution_d = self.parentWindow.resolution_depth
        self.ui.lineEdit_6.setText(str(self.resolution_d))

    @Slot()
    def showHelp(self) -> None:
        """Shows help information about axial resolution."""
        QMessageBox.information(
            self,
            "Help",
            "The axial resolution is defined by the technical design of the OCT device. Please consult the specifications of the OCT device used for the current data acquisition.",
        )

    @Slot()
    def onInputChanged(self) -> None:
        """Handles changes in input fields and updates calculated resolutions."""
        try:
            size_pix_w = float(self.parentWindow.oct_data.shape[2])
            size_pix_h = float(self.parentWindow.oct_data.shape[0])
            size_mm_w = float(self.ui.lineEdit.text())
            size_mm_h = float(self.ui.lineEdit_2.text())
            self.resolution_d = float(self.ui.lineEdit_6.text())
            self.resolution_w = round(size_mm_w / size_pix_w, 4)
            self.resolution_h = round(size_mm_h / size_pix_h, 4)
            self.ui.lineEdit_8.setText(str(self.resolution_h))
            self.ui.lineEdit_5.setText(str(self.resolution_w))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid positive numbers.")
            return

    def calculateResolution(self) -> None:
        """Updates the parent window with the new resolution values and accepts the dialog."""
        self.parentWindow.updateResolution(
            [self.resolution_w, self.resolution_h, self.resolution_d],
            [float(self.ui.lineEdit.text()), float(self.ui.lineEdit_2.text())],
        )
        super().accept()

