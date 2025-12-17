from PySide6.QtCore import Qt, QAbstractTableModel, QCoreApplication
from PySide6.QtWidgets import QProgressDialog


class TableModel(QAbstractTableModel):
    """A custom table model for displaying data in a QTableView."""

    def __init__(self, data: list, headers: list) -> None:
        """Initializes the TableModel.

        Args:
            data (list): A nested list containing the table data.
            headers (list): A list of column headers.
        """
        super().__init__()
        self._data = data
        self._headers = headers

    def data(self, index, role: int) -> str:
        """Returns the data for a given index and role.

        Args:
            index (QModelIndex): The index of the cell.
            role (int): The role for which data is requested.

        Returns:
            str: The data for the given index and role.
        """
        if not index.isValid():
            return None

        value = self._data[index.row()][index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return value
        elif role == Qt.ItemDataRole.ToolTipRole:
            return value

        return None

    def rowCount(self, index) -> int:
        """Returns the number of rows in the table.

        Args:
            index (QModelIndex): The parent index (not used).

        Returns:
            int: The number of rows.
        """
        return len(self._data)

    def columnCount(self, index) -> int:
        """Returns the number of columns in the table.

        Args:
            index (QModelIndex): The parent index (not used).

        Returns:
            int: The number of columns.
        """
        return len(self._data[0]) if self._data else 0

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> str:
        """Returns the header data for a given section and orientation.

        Args:
            section (int): The section index.
            orientation (Qt.Orientation): The orientation (horizontal or vertical).
            role (int, optional): The role for which header data is requested. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            str: The header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._headers[section]
            elif orientation == Qt.Orientation.Vertical:
                return f"{section + 1}"  # Optional: customize row headers
        return None


def create_progress_dialog(
    title: str, text: str, cancelable: bool = False, cancel_callback=None, parent=None
) -> QProgressDialog:
    """Creates a progress dialog.

    Args:
        title (str): The title of the dialog.
        text (str): The text to display in the dialog.
        cancelable (bool, optional): Whether the dialog can be canceled. Defaults to False.
        cancel_callback (callable, optional): Callback function for cancel action. Defaults to None.
        parent (QWidget, optional): The parent widget. Defaults to None.

    Returns:
        QProgressDialog: The created progress dialog.
    """
    msg = QProgressDialog(text, None, 0, 100, parent)
    msg.setWindowTitle(title)
    msg.setWindowModality(Qt.WindowModality.WindowModal)
    if cancelable:
        msg.canceled.connect(cancel_callback)
    msg.show()
    QCoreApplication.processEvents()
    msg.setValue(0)
    return msg


def update_progress_dialog(msg: QProgressDialog, value: int, text: str = None) -> None:
    """Updates the progress dialog with a new value and optional text.

    Args:
        msg (QProgressDialog): The progress dialog to update.
        value (int): The new progress value (0-100).
        text (str, optional): The new text to display. Defaults to None.
    """
    msg.setValue(value)
    if text is not None:
        msg.setLabelText(text)
    QCoreApplication.processEvents()
    if value == 100:
        msg.close()