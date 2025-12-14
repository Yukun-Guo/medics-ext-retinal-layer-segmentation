"""
Main widget for Example Extension
"""

from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np


class ExampleExtensionWidget(QtWidgets.QWidget):
    """Main widget for the Example Extension.
    
    This widget demonstrates:
    - Creating UI with Qt
    - Interacting with workspace manager
    - Using configuration
    - Logging
    """
    
    def __init__(self, parent=None, extension=None, **kwargs):
        """Initialize the widget.
        
        Args:
            parent: Parent Qt widget
            extension: The ExampleExtension instance
            **kwargs: Additional parameters
        """
        super().__init__(parent)
        self.extension = extension
        self.app_context = extension.app_context if extension else None
        
        self.setup_ui()
        self.connect_signals()
        
        if extension:
            extension.logger.debug("Example Extension widget created")
    
    def setup_ui(self):
        """Create the user interface."""
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("Example Extension")
        title.setAlignment(QtCore.Qt.AlignCenter)
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        
        # Description
        description = QtWidgets.QLabel(
            "This is a template extension demonstrating MedICS plugin system.\n"
            "Use this as a starting point for your own extensions."
        )
        description.setAlignment(QtCore.Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Input section
        input_group = QtWidgets.QGroupBox("Input")
        input_layout = QtWidgets.QFormLayout(input_group)
        
        self.input_text = QtWidgets.QLineEdit()
        self.input_text.setPlaceholderText("Enter some text...")
        input_layout.addRow("Text:", self.input_text)
        
        self.input_number = QtWidgets.QSpinBox()
        self.input_number.setRange(0, 100)
        self.input_number.setValue(42)
        input_layout.addRow("Number:", self.input_number)
        
        layout.addWidget(input_group)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.process_button = QtWidgets.QPushButton("Process Data")
        self.process_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        button_layout.addWidget(self.process_button)
        
        self.load_button = QtWidgets.QPushButton("Load from Workspace")
        self.load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        button_layout.addWidget(self.load_button)
        
        self.save_button = QtWidgets.QPushButton("Save to Workspace")
        self.save_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # Output section
        output_group = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        
        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(150)
        output_layout.addWidget(self.output_text)
        
        layout.addWidget(output_group)
        
        # Status section
        status_layout = QtWidgets.QHBoxLayout()
        
        self.status_label = QtWidgets.QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.clear_button = QtWidgets.QPushButton("Clear Output")
        self.clear_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        status_layout.addWidget(self.clear_button)
        
        layout.addLayout(status_layout)
    
    def connect_signals(self):
        """Connect signals to slots."""
        self.process_button.clicked.connect(self.on_process)
        self.load_button.clicked.connect(self.on_load)
        self.save_button.clicked.connect(self.on_save)
        self.clear_button.clicked.connect(self.on_clear)
    
    def on_process(self):
        """Handle process button click."""
        try:
            # Get input
            text = self.input_text.text()
            number = self.input_number.value()
            
            # Process data
            result = self.process_data(text, number)
            
            # Display result
            self.output_text.append(f"Processed: {result}")
            self.status_label.setText("Processing complete")
            
            if self.extension:
                self.extension.logger.info(f"Processed data: {text}, {number}")
        
        except Exception as e:
            self.show_error(f"Processing failed: {e}")
    
    def on_load(self):
        """Handle load button click."""
        try:
            if not self.app_context:
                raise RuntimeError("No application context")
            
            workspace_manager = self.app_context.get_component("workspace_manager")
            if not workspace_manager:
                raise RuntimeError("Workspace manager not available")
            
            # Load data from workspace
            data = workspace_manager.workspace["example_data"]
            
            if data is None:
                self.output_text.append("No data found in workspace")
                self.status_label.setText("No data to load")
            else:
                self.output_text.append(f"Loaded from workspace: {data}")
                self.status_label.setText("Data loaded")
                
                if self.extension:
                    self.extension.logger.info("Loaded data from workspace")
        
        except Exception as e:
            self.show_error(f"Load failed: {e}")
    
    def on_save(self):
        """Handle save button click."""
        try:
            if not self.app_context:
                raise RuntimeError("No application context")
            
            workspace_manager = self.app_context.get_component("workspace_manager")
            if not workspace_manager:
                raise RuntimeError("Workspace manager not available")
            
            # Prepare data
            data = {
                "text": self.input_text.text(),
                "number": self.input_number.value(),
                "timestamp": QtCore.QDateTime.currentDateTime().toString()
            }
            
            # Save to workspace
            workspace_manager.workspace["example_data"] = data
            workspace_manager.update_workspace_data(workspace_manager.workspace)
            self.output_text.append(f"Saved to workspace: {data}")
            self.status_label.setText("Data saved")
            
            if self.extension:
                self.extension.logger.info("Saved data to workspace")
        
        except Exception as e:
            self.show_error(f"Save failed: {e}")
    
    def on_clear(self):
        """Handle clear button click."""
        self.output_text.clear()
        self.status_label.setText("Output cleared")
    
    def process_data(self, text: str, number: int):
        """Process input data.
        
        Args:
            text: Input text
            number: Input number
        
        Returns:
            str: Processed result
        """
        # Example processing
        processed_text = text.upper()
        processed_number = number * 2
        
        return f"Text: {processed_text}, Number: {processed_number}"
    
    def show_error(self, message: str):
        """Show error message.
        
        Args:
            message: Error message to display
        """
        self.output_text.append(f"ERROR: {message}")
        self.status_label.setText("Error occurred")
        
        if self.extension:
            self.extension.logger.error(message)
        
        QtWidgets.QMessageBox.critical(self, "Error", message)
    
    def cleanup(self):
        """Clean up resources."""
        # Add cleanup code here
        pass
