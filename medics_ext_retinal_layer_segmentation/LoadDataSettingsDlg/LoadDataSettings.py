import os
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import QDialog, QMessageBox, QFileDialog, QHeaderView, QWidget
from PySide6.QtCore import QStringListModel, QModelIndex
from typing import Optional, Any

from ..utils.fileIO import FileIO
from .LoadDataSettingsDlg import Ui_LoadDataSettingsDlg

pg.setConfigOption("background", "{}{:02x}".format("#151515", 60))


class LoadDataSettings(QDialog):
    """Dialog for loading and selecting data from .med files."""

    def __init__(
        self,
        parentWindow: QWidget=None,
        data_loader_settings: Optional[dict] = None,
    ) -> None:
        """Initializes the LoadDataSettings dialog.

        Args:
            parentWindow (Optional[QWidget]): The parent window for this dialog.
            data_loader_settings (Optional[dict]): Settings for loading data, if any. 
            example: {'path': 'path/to/file.py', 'loaded_functions': {'module1':{'function1': func1, 'function2': func2},'module2':{'function1': func3, 'function2': func4}},
                      'oct_loader': {'module':'module1','function':'function2'},
                      'octa_loader': {'module':'module2','function':'function1'},
                      'seg_loader': {'module':'module2','function':'function2'}}
        """
        self.parentWindow = parentWindow
        self.data_loader_settings: Optional[dict] = data_loader_settings
        self.loaded_functions: Optional[dict] = None
        
        self.setupUI()
        if data_loader_settings is not None:    
            self.loaded_functions = data_loader_settings.get("loaded_functions", None)
            if self.loaded_functions is not None:
                self.ui.lineEdit_path.setText(data_loader_settings.get("path", ""))
                self.ui.comboBox_oct_module.setCurrentText(data_loader_settings["oct_loader"]["module"])
                self.ui.comboBox_oct_func.setCurrentText(data_loader_settings["oct_loader"]["function"])
                self.ui.comboBox_octa_module.setCurrentText(data_loader_settings["octa_loader"]["module"])
                self.ui.comboBox_octa_func.setCurrentText(data_loader_settings["octa_loader"]["function"])
                self.ui.comboBox_seg_module.setCurrentText(data_loader_settings["seg_loader"]["module"])
                self.ui.comboBox_seg_func.setCurrentText(data_loader_settings["seg_loader"]["function"])
        else:
            self.ui.lineEdit_path.setText("")
            self.ui.comboBox_oct_module.clear()
            self.ui.comboBox_octa_module.clear()
            self.ui.comboBox_seg_module.clear()
            # add default to each combobox
            self.ui.comboBox_oct_module.addItem("default")
            self.ui.comboBox_octa_module.addItem("default")
            self.ui.comboBox_seg_module.addItem("default")
            self.ui.comboBox_oct_module.setCurrentIndex(0)
            self.ui.comboBox_octa_module.setCurrentIndex(0)
            self.ui.comboBox_seg_module.setCurrentIndex(0)


    def setupUI(self) -> None:
        """Sets up the user interface and connects signals."""
        QDialog.__init__(self, parent=self.parentWindow)
        self.ui = Ui_LoadDataSettingsDlg()
        self.ui.setupUi(self)
        self.ui.lineEdit_path.textChanged.connect(self.on_text_changed)
        self.ui.pushButton_select_file.clicked.connect(self.on_browse_file)
        self.ui.pushButton_select_folder.clicked.connect(self.on_browse_folder)
        self.ui.comboBox_oct_module.currentIndexChanged.connect(self.on_oct_module_changed)
        self.ui.comboBox_octa_module.currentIndexChanged.connect(self.on_octa_module_changed)
        self.ui.comboBox_seg_module.currentIndexChanged.connect(self.on_seg_module_changed)
        self.ui.comboBox_oct_func.setVisible(False)
        self.ui.comboBox_octa_func.setVisible(False)
        self.ui.comboBox_seg_func.setVisible(False)
        self.ui.label_5.setVisible(False)
        self.ui.label_6.setVisible(False)
        self.ui.label_7.setVisible(False)
        self.setMinimumSize(self.size())
        self.setMaximumSize(self.size())

    
    def on_oct_module_changed(self, index: int) -> None:
        """Handles changes to the OCT module combobox.
        
        Args:
            index (int): The index of the selected module.
        """
        module_name = self.ui.comboBox_oct_module.currentText()
        if module_name == "default":
            self.ui.comboBox_oct_func.clear()
            self.ui.comboBox_oct_func.setVisible(False)
            self.ui.label_5.setVisible(False)
            return
        self.ui.comboBox_oct_func.setVisible(True)
        self.ui.label_5.setVisible(True)
        if self.loaded_functions is None:
            self.loaded_functions = FileIO.load_functions(self.ui.lineEdit_path.text())
        if module_name in self.loaded_functions:
            self.ui.comboBox_oct_func.clear()
            for func_name in self.loaded_functions[module_name].keys():
                self.ui.comboBox_oct_func.addItem(func_name)
            self.ui.comboBox_oct_func.setCurrentIndex(0)
    
    def on_octa_module_changed(self, index: int) -> None:
        """Handles changes to the OCTA module combobox.
        
        Args:
            index (int): The index of the selected module.
        """
        module_name = self.ui.comboBox_octa_module.currentText()
        if module_name == "default":
            self.ui.comboBox_octa_func.clear()
            self.ui.comboBox_octa_func.setVisible(False)
            self.ui.label_6.setVisible(False)
            return
        self.ui.comboBox_octa_func.setVisible(True)
        self.ui.label_6.setVisible(True)
        if self.loaded_functions is None:
            self.loaded_functions = FileIO.load_functions(self.ui.lineEdit_path.text())
        if module_name in self.loaded_functions:
            self.ui.comboBox_octa_func.clear()
            for func_name in self.loaded_functions[module_name].keys():
                self.ui.comboBox_octa_func.addItem(func_name)
            self.ui.comboBox_octa_func.setCurrentIndex(0)
    
    def on_seg_module_changed(self, index: int) -> None:
        """Handles changes to the segmentation module combobox.
        
        Args:
            index (int): The index of the selected module.
        """
        module_name = self.ui.comboBox_seg_module.currentText()
        if module_name == "default":
            self.ui.comboBox_seg_func.clear()
            self.ui.comboBox_seg_func.setVisible(False)
            self.ui.label_7.setVisible(False)
            return
        self.ui.comboBox_seg_func.setVisible(True)
        self.ui.label_7.setVisible(True)
        if self.loaded_functions is None:
            self.loaded_functions = FileIO.load_functions(self.ui.lineEdit_path.text())
        if module_name in self.loaded_functions:
            self.ui.comboBox_seg_func.clear()
            for func_name in self.loaded_functions[module_name].keys():
                self.ui.comboBox_seg_func.addItem(func_name)
            self.ui.comboBox_seg_func.setCurrentIndex(0)
    
    def on_browse_file(self) -> None:
        """Opens a file dialog for selecting a .med file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", "Python (*.py);;All Files (*)"
        )
        if filename:
            self.ui.lineEdit_path.setText(filename)

    def on_browse_folder(self) -> None:
        """Opens a folder dialog for selecting a directory."""
        foldername = QFileDialog.getExistingDirectory(self, "Select Folder")
        if foldername:
            self.ui.lineEdit_path.setText(foldername)
            
    def on_text_changed(self, text: str) -> None:
        """Handles changes to the path text field.
        Args:
            text (str): The current text in the path field.
        """
        if os.path.exists(text) and os.path.isfile(text):
            self.loaded_functions = FileIO.load_functions(text)
            # set all module names from loaded_functions to the combobox_oct_module,combobox_octa_module,combobox_seg_module
            self.ui.comboBox_oct_module.clear()
            self.ui.comboBox_octa_module.clear()
            self.ui.comboBox_seg_module.clear()
            # add default to each combobox
            self.ui.comboBox_oct_module.addItem("default")
            self.ui.comboBox_octa_module.addItem("default")
            self.ui.comboBox_seg_module.addItem("default")
            for module_name in self.loaded_functions.keys():
                self.ui.comboBox_oct_module.addItem(module_name)
                self.ui.comboBox_octa_module.addItem(module_name)
                self.ui.comboBox_seg_module.addItem(module_name)
            # set the current index to 0
            self.ui.comboBox_oct_module.setCurrentIndex(0)
            self.ui.comboBox_octa_module.setCurrentIndex(0)
            self.ui.comboBox_seg_module.setCurrentIndex(0)
    
    def get_loader_settings(self) -> Optional[dict]:
        """Retrieves the selected loader from the dialog.

        Returns:
            Optional[dict]: A dictionary containing the selected loader, or None if no data is selected.
        """
        if not self.ui.lineEdit_path.text():
            QMessageBox.warning(self, "Warning", "Please select a file or folder.")
            return None

        selected_loader = {
            "path": self.ui.lineEdit_path.text(),
            "loaded_functions": self.loaded_functions,
            "oct_loader": {
                "module": self.ui.comboBox_oct_module.currentText(),
                "function": self.ui.comboBox_oct_func.currentText(),
            },
            "octa_loader": {
                "module": self.ui.comboBox_octa_module.currentText(),
                "function": self.ui.comboBox_octa_func.currentText(),
            },
            "seg_loader": {
                "module": self.ui.comboBox_seg_module.currentText(),
                "function": self.ui.comboBox_seg_func.currentText(),
            },
        }
        return selected_loader
            
            

    def accept(self) -> None:
        """Accepts the dialog."""
        super().accept()


# if __name__ == "__main__":
#     from PySide6.QtWidgets import QApplication
#     import sys

#     app = QApplication(sys.argv)
#     dialog = LoadMedData(parentWindow=None)

#     # Show dialog and check result
#     if dialog.exec_() == QDialog.DialogCode.Accepted:
#         print(f"Data entered: {dialog.selected_data}")
#     else:
#         print("Dialog canceled.")