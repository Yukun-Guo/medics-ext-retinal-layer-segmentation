import os
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import QDialog, QMessageBox, QFileDialog, QHeaderView, QWidget
from PySide6.QtCore import QStringListModel, QModelIndex
from typing import Optional, Any

from ..utils.fileIO import FileIO
from .LoadGroupDataDlg import Ui_LoadGroupDataDlg

pg.setConfigOption("background", "{}{:02x}".format("#151515", 60))


class LoadGroupDataClass(QDialog):
    """Dialog for loading and selecting data from .med files."""

    def __init__(
        self,
        parentWindow: QWidget=None,
        filePath: str = "",
    ) -> None:
        """Initializes the LoadGroupData dialog.

        Args:
            parentWindow (Optional[QWidget]): The parent window for this dialog.
        """
        self.parentWindow = parentWindow
        self.dataset: Optional[dict] = None
        self.selected_oct_data: Optional[Any] = None
        self.selected_octa_data: Optional[Any] = None
        self.selected_seg_data: Optional[Any] = None
        self.filePath = filePath
        self.setupUI()

    def setupUI(self) -> None:
        """Sets up the user interface and connects signals."""
        QDialog.__init__(self, parent=self.parentWindow)
        self.ui = Ui_LoadGroupDataDlg()
        self.ui.setupUi(self)
        # OK and Cancel buttons
        self.ui.lineEdit_filename.textChanged.connect(self.on_textChanged)
        
        self.ui.pushButton_browser.clicked.connect(self.on_browse)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.listView_source_oct.clicked.connect(self.on_listView_source_oct_clicked)
        self.ui.listView_source_octa.clicked.connect(self.on_listView_source_octa_clicked)
        self.ui.listView_source_seg.clicked.connect(self.on_listView_source_seg_clicked)
        if self.filePath:
            self.ui.lineEdit_filename.setText(self.filePath)

    def on_browse(self) -> None:
        """Opens a file dialog for selecting a .mat or .med file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", "Mat file (*.mat);;Med file (*.med);;All Files (*)"
        )
        if filename:
            self.ui.lineEdit_filename.setText(filename)

    def on_textChanged(self, text: str) -> None:
        """Handles changes to the filename text field.

        Args:
            text (str): The current text in the filename field.
        """
        # clean up the text input
        text = text.strip().replace("\\", "/").replace('"', '')
        self.ui.lineEdit_filename.setText(text)
        if not text:
            self.dataset = None
            self.ui.listView_source_oct.setModel(QStringListModel())
            self.ui.listView_source_octa.setModel(QStringListModel())
            self.ui.listView_source_seg.setModel(QStringListModel())
            return
        try:
            if os.path.exists(text):
                # check the extension
                _, ext = os.path.splitext(text)
                if ext.lower() not in ['.med', '.mat']:
                    self.dataset = None
                    QMessageBox.critical(self, "Error", "Unsupported file format. Please select a .med or .mat file.")
                    return
                elif ext.lower() == '.mat':
                    data = FileIO.read_mat_file(text)
                elif ext.lower() == '.med':
                    data = FileIO.read_med_file(text)
                else:
                    QMessageBox.critical(self, "Error", "Unsupported file format. Please select a .med or .mat file.")
                    return
                self.dataset = data
                model = QStringListModel()
                keys = []
                self._extract_keys_recursive(data, keys)
                model.setStringList(keys)
                self.ui.listView_source_oct.setModel(model)
                self.ui.listView_source_octa.setModel(model)
                self.ui.listView_source_seg.setModel(model)
            else:
                self.dataset = None
                QMessageBox.critical(self, "Error", "Invalid file")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _extract_keys_recursive(self, data: dict, keys: list, prefix: str = "", depth: int = 0, is_last: bool = True, parent_prefixes: list = None) -> None:
        """Recursively extracts keys from nested dictionaries.
        
        Args:
            data (dict): The dictionary to extract keys from.
            keys (list): The list to append formatted key strings to.
            prefix (str): The prefix for nested keys (for displaying hierarchy).
            depth (int): Current nesting depth for indentation.
            is_last (bool): Whether this is the last item at this level.
            parent_prefixes (list): List of prefix strings for parent levels.
        """
        if parent_prefixes is None:
            parent_prefixes = []
            
        items = list(data.items())
        
        for i, (key, value) in enumerate(items):
            is_last_item = (i == len(items) - 1)
            full_key = f"{prefix}{key}" if prefix else key
            
            # Build the tree-like prefix
            tree_prefix = ""
            for parent_prefix in parent_prefixes:
                tree_prefix += parent_prefix
            
            if depth == 0:
                # Root level - no tree symbols
                tree_symbol = ""
            else:
                # Child level - add tree symbols
                tree_symbol = "└── " if is_last_item else "├── "
            
            # display_key = f"{tree_prefix}{tree_symbol}{key}"
            
            # Use stacked data icon for dicts and data icon for leaf nodes
            stacked_data_icon = "🗄️ " if depth == 0 else ""
            data_icon = "🔢 " if depth > 0 else ""
            if isinstance(value, np.ndarray):
                keys.append(f"{tree_prefix}{tree_symbol}{data_icon}{key} [array {value.dtype}, {value.shape}]")
            elif isinstance(value, dict):
                keys.append(f"{tree_prefix}{tree_symbol}{stacked_data_icon}{key} [dict, {len(value)} items]")
                
                # Prepare prefix for children
                new_parent_prefixes = parent_prefixes.copy()
                if depth > 0:
                    new_parent_prefixes.append("    " if is_last_item else "│   ")
                
                # Recursively process nested dictionary
                self._extract_keys_recursive(value, keys, f"{full_key}.", depth + 1, True, new_parent_prefixes)
            else:
                keys.append(f"{tree_prefix}{tree_symbol}{data_icon}{key} [{type(value).__name__}, {value}]")

    def _get_nested_value(self, data: dict, key_path: str) -> Any:
        """Retrieves a value from nested dictionaries using dot notation.
        
        Args:
            data (dict): The root dictionary.
            key_path (str): The key path using dot notation (e.g., 'parent.child.grandchild').
            
        Returns:
            Any: The value at the specified key path.
        """
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current

    def _build_key_path(self, current_index: int, current_key: str, current_depth: int,string_list: list) -> str:
        """Builds the full key path for a selected item based on its position and indentation.
        
        Args:
            current_index (int): The index of the selected item.
            current_key (str): The key name of the selected item.
            current_depth (int): The indentation depth of the selected item.
            
        Returns:
            str: The full key path using dot notation.
        """
        if current_depth == 0:
            return current_key
            
        # Look backwards to find parent keys
        
        path_parts = []
        
        # Work backwards to find all parent keys
        for i in range(current_index - 1, -1, -1):
            line = string_list[i]
            
            # Calculate depth by counting tree structure characters
            depth = self._calculate_tree_depth(line)
            
            if depth < current_depth:
                key = self._extract_key_from_tree_line(line)
                if key:
                    path_parts.insert(0, key)
                    current_depth = depth
                    if depth == 0:
                        break
        
        path_parts.append(current_key)
        return ".".join(path_parts)
    
    def _calculate_tree_depth(self, line: str) -> int:
        """Calculate the depth of a tree line by counting tree symbols.
        
        Args:
            line (str): The line with tree symbols.
            
        Returns:
            int: The depth level.
        """
        # Count occurrences of tree continuation symbols
        depth = 0
        i = 0
        while i < len(line):
            if line[i:i+4] == "│   " or line[i:i+4] == "    ":
                depth += 1
                i += 4
            elif line[i:i+4] == "├── " or line[i:i+4] == "└── ":
                depth += 1
                break
            else:
                break
        return depth
    
    def _extract_key_from_tree_line(self, line: str) -> str:
        """Extract the actual key name from a tree-formatted line.
        
        Args:
            line (str): The tree-formatted line.
            
        Returns:
            str: The extracted key name.
        """
        # Remove tree symbols and icons
        cleaned = line
        
        # Remove tree structure symbols
        cleaned = cleaned.replace("│   ", "").replace("    ", "")
        cleaned = cleaned.replace("├── ", "").replace("└── ", "")
        
        # Remove icons
        cleaned = cleaned.replace("🗄️ ", "").replace("🔢 ", "")
        
        # Extract the key (first word before space and bracket)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned.split(" ")[0]
        return ""

    def on_listView_source_oct_clicked(self, index: QModelIndex) -> None:
        """Handles selection in the list view.

        Args:
            index (QModelIndex): The index of the selected item.
        """
        selected_text = self.ui.listView_source_oct.model().stringList()[index.row()]
        
        # Extract the key from the tree-formatted line
        key_only = self._extract_key_from_tree_line(selected_text)
        
        # Calculate the depth from tree symbols
        current_depth = self._calculate_tree_depth(selected_text)
        
        # Build the full key path by looking at previous entries
        full_key = self._build_key_path(index.row(), key_only, current_depth, string_list=self.ui.listView_source_oct.model().stringList())
        
        if self.dataset is not None:
            self.selected_oct_data = self._get_nested_value(self.dataset, full_key)
            self.ui.lineEdit_oct_field.setText(full_key)

    def on_listView_source_octa_clicked(self, index: QModelIndex) -> None:
        """Handles selection in the list view.

        Args:
            index (QModelIndex): The index of the selected item.
        """
        selected_text = self.ui.listView_source_octa.model().stringList()[index.row()]
        
        # Extract the key from the tree-formatted line
        key_only = self._extract_key_from_tree_line(selected_text)
        
        # Calculate the depth from tree symbols
        current_depth = self._calculate_tree_depth(selected_text)
        
        # Build the full key path by looking at previous entries
        full_key = self._build_key_path(index.row(), key_only, current_depth, string_list=self.ui.listView_source_octa.model().stringList())
        
        if self.dataset is not None:
            self.selected_octa_data = self._get_nested_value(self.dataset, full_key)
            self.ui.lineEdit_octa_field.setText(full_key)
            
    def on_listView_source_seg_clicked(self, index: QModelIndex) -> None:
        """Handles selection in the list view.

        Args:
            index (QModelIndex): The index of the selected item.
        """
        selected_text = self.ui.listView_source_seg.model().stringList()[index.row()]
        
        # Extract the key from the tree-formatted line
        key_only = self._extract_key_from_tree_line(selected_text)
        
        # Calculate the depth from tree symbols
        current_depth = self._calculate_tree_depth(selected_text)
        
        # Build the full key path by looking at previous entries
        full_key = self._build_key_path(index.row(), key_only, current_depth, string_list=self.ui.listView_source_seg.model().stringList())
        
        if self.dataset is not None:
            self.selected_seg_data = self._get_nested_value(self.dataset, full_key)
            self.ui.lineEdit_seg_field.setText(full_key)
    
    def accept(self) -> None:
        """Accepts the dialog."""
        super().accept()



# if __name__ == "__main__":
#     from PySide6.QtWidgets import QApplication
#     import sys

#     app = QApplication(sys.argv)
#     dialog = LoadGroupDataClass(parentWindow=None)

#     # Show dialog and check result
#     if dialog.exec_() == QDialog.DialogCode.Accepted:
#         print(f"Data entered: {dialog.selected_data}")
#     else:
#         print("Dialog canceled.")