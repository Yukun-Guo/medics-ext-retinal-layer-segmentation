# MedICS Extension Development Guide

This guide explains how to create independent MedICS extensions that can be distributed as pip packages.

## Table of Contents

- [Overview](#overview)
- [Extension Discovery](#extension-discovery)
- [Quick Start](#quick-start)
- [Extension Package Structure](#extension-package-structure)
- [Extension API](#extension-api)
- [Testing Your Extension](#testing-your-extension)
- [Publishing Your Extension](#publishing-your-extension)
- [Examples](#examples)

## Overview

MedICS extensions are Python packages that integrate with the MedICS application to provide additional functionality. Extensions can add:

- **Image processing tools** - Custom algorithms and filters
- **Data visualization** - Interactive viewers and plots
- **Analysis workflows** - Automated pipelines
- **Import/Export handlers** - Support for new file formats
- **Integration tools** - Connect with external services

### Key Features

✅ **Independent Distribution** - Extensions are separate pip packages  
✅ **Standard Python Packaging** - Uses setuptools entry points  
✅ **Automatic Discovery** - MedICS finds and loads extensions automatically  
✅ **Version Management** - Each extension has its own version and dependencies  

## Extension Discovery

MedICS discovers extensions using the standard Python plugin mechanism via **setuptools entry points**.

Define your extension in `pyproject.toml`:

```toml
[project.entry-points."medics.extensions"]
my_extension = "medics_ext_my_extension:MyExtension"
```

**Benefits:**
- Standard Python packaging
- Easy distribution via PyPI
- Automatic discovery
- Clean separation from MedICS core
- Version management
- Dependency isolation

## Quick Start

### 1. Create Package Structure

```bash
medics-ext-my-extension/
├── pyproject.toml
├── README.md
├── LICENSE
└── medics_ext_my_extension/
    ├── __init__.py
    └── extension.py
```

### 2. Define Package Configuration

**pyproject.toml:**

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "medics-ext-my-extension"
version = "1.0.0"
description = "My custom MedICS extension"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "medics>=2.0.0",
    "medics-extension-sdk>=1.0.0",
    "PySide6>=6.8.0",
    # Add your dependencies here
]

[project.entry-points."medics.extensions"]
my_extension = "medics_ext_my_extension:MyExtension"
```

### 3. Implement Extension Class

**medics_ext_my_extension/__init__.py:**

```python
"""My Extension for MedICS."""

from medics_extension_sdk import BaseExtension
from PySide6 import QtWidgets

class MyExtension(BaseExtension):
    """Custom extension that does something amazing."""
    
    def __init__(self):
        super().__init__(
            extension_name="My Extension",
            author_name="Your Name"
        )
    
    def get_version(self) -> str:
        """Return extension version."""
        return "1.0.0"
    
    def get_description(self) -> str:
        """Return short description."""
        return "A custom extension that does something amazing"
    
    def get_category(self) -> str:
        """Return extension category for UI grouping."""
        return "Custom Tools"
    
    def initialize(self, app_context) -> bool:
        """Initialize extension with application context.
        
        Args:
            app_context: Main application context with access to:
                - workspace_manager: Manage data and workspace
                - config_manager: Access configuration
                - theme_manager: UI theming
                - Other components
        
        Returns:
            True if initialization successful
        """
        self.app_context = app_context
        self.logger.info("My Extension initialized")
        return True
    
    def create_widget(self, parent=None, **kwargs):
        """Create the main extension widget.
        
        Args:
            parent: Parent Qt widget
            **kwargs: Additional parameters
        
        Returns:
            QWidget: Main extension widget
        """
        return MyExtensionWidget(parent, self.app_context)
    
    def cleanup(self) -> None:
        """Clean up resources before unload."""
        self.logger.info("My Extension cleaning up")

# Make extension discoverable
__all__ = ['MyExtension']
```

**medics_ext_my_extension/extension.py:**

```python
"""Extension UI implementation."""

from PySide6 import QtWidgets, QtCore

class MyExtensionWidget(QtWidgets.QWidget):
    """Main widget for My Extension."""
    
    def __init__(self, parent=None, app_context=None):
        super().__init__(parent)
        self.app_context = app_context
        self.setup_ui()
    
    def setup_ui(self):
        """Create UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Add your widgets here
        label = QtWidgets.QLabel("My Extension UI")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        
        button = QtWidgets.QPushButton("Do Something")
        button.clicked.connect(self.on_button_clicked)
        layout.addWidget(button)
    
    def on_button_clicked(self):
        """Handle button click."""
        QtWidgets.QMessageBox.information(
            self,
            "My Extension",
            "Button clicked!"
        )
```

### 4. Install and Test

```bash
# Install in development mode
cd medics-ext-my-extension
pip install -e .

# Run MedICS
python -m medics

# Your extension should appear in the Extensions menu
```

## Extension Package Structure

### Recommended Structure

```
medics-ext-my-extension/          # Package root
├── pyproject.toml                # Package configuration
├── setup.py                      # Optional: for complex builds
├── README.md                     # Package documentation
├── LICENSE                       # License file
├── requirements.txt              # Optional: pin dependencies
├── .gitignore
└── medics_ext_my_extension/      # Extension code
    ├── __init__.py               # Extension class (exports MyExtension)
    ├── extension.py              # Main extension logic
    ├── ui/                       # UI components
    │   ├── __init__.py
    │   ├── main_widget.py
    │   └── dialogs.py
    ├── utils/                    # Utilities
    │   ├── __init__.py
    │   └── helpers.py
    ├── resources/                # Resources (icons, data)
    │   ├── icons/
    │   └── data/
    └── tests/                    # Unit tests
        ├── __init__.py
        └── test_extension.py
```

### Naming Conventions

- **Package name**: `medics-ext-<your-extension>` (PyPI name)
- **Module name**: `medics_ext_<your_extension>` (import name)
- **Entry point name**: `<your_extension>` (short identifier)
- **Extension class**: `<YourExtension>Extension` (class name)

Examples:
- `medics-ext-image-labeler` / `medics_ext_image_labeler` / `ImageLabelerExtension`
- `medics-ext-dicom-viewer` / `medics_ext_dicom_viewer` / `DicomViewerExtension`

## Extension API

### BaseExtension Class

All extensions must inherit from `medics_extension_sdk.BaseExtension`:

```python
from medics_extension_sdk import BaseExtension

class MyExtension(BaseExtension):
    """Extension implementation."""
    
    # Required methods
    def get_version(self) -> str: ...
    def get_description(self) -> str: ...
    def get_category(self) -> str: ...
    def initialize(self, app_context) -> bool: ...
    def create_widget(self, parent=None, **kwargs): ...
    def cleanup(self) -> None: ...
    
    # Inherited from BaseExtension
    # - get_name() -> str
    # - get_author() -> str
    # - logger (logging.Logger)
    # - app_context (ApplicationContext)
```

### Required Methods

#### `get_version() -> str`

Return the extension version string:

```python
def get_version(self) -> str:
    return "1.0.0"
```

#### `get_description() -> str`

Return a short description (shown in extension manager):

```python
def get_description(self) -> str:
    return "Provides advanced image labeling tools"
```

#### `get_category() -> str`

Return category for UI grouping:

```python
def get_category(self) -> str:
    return "Image Analysis"  # or "Data Processing", "Visualization", etc.
```

#### `initialize(app_context) -> bool`

Initialize extension with application context:

```python
def initialize(self, app_context) -> bool:
    self.app_context = app_context
    
    # Access components
    self.workspace_manager = app_context.get_component("workspace_manager")
    self.config_manager = app_context.get_component("config_manager")
    
    # Perform initialization
    self.logger.info("Extension initialized")
    
    return True  # Return False if initialization fails
```

#### `create_widget(parent=None, **kwargs) -> QWidget`

Create and return the main extension widget:

```python
def create_widget(self, parent=None, **kwargs):
    from .ui.main_widget import MyExtensionWidget
    return MyExtensionWidget(parent, self.app_context, **kwargs)
```

#### `cleanup() -> None`

Clean up resources before unload:

```python
def cleanup(self) -> None:
    # Close files, stop threads, release resources
    self.logger.info("Extension cleaning up")
```

### Application Context

The `app_context` provides access to MedICS components:

```python
# Component access
workspace_manager = app_context.get_component("workspace_manager")
config_manager = app_context.get_component("config_manager")
theme_manager = app_context.get_component("theme_manager")
extension_manager = app_context.get_component("extension_manager")

# Application info
root_path = app_context.root_path
main_window = app_context.main_window

# Data access via workspace_manager
data = workspace_manager.get_data("variable_name")
workspace_manager.set_data("variable_name", value)
```

### Configuration

Access extension-specific configuration:

```python
def initialize(self, app_context) -> bool:
    config = app_context.get_component("config_manager")
    
    # Read configuration
    value = config.get_value("MyExtension", "setting", default="default_value")
    
    # Write configuration
    config.set_value("MyExtension", "setting", "new_value")
    config.save()
    
    return True
```

## Testing Your Extension

### Development Installation

Install in editable mode for development:

```bash
cd medics-ext-my-extension
pip install -e .
```

### Manual Testing

1. Install MedICS: `pip install medics`
2. Install your extension: `pip install -e .`
3. Run MedICS: `python -m medics`
4. Check Extensions menu for your extension

### Unit Tests

Create tests in `tests/` directory:

```python
# tests/test_extension.py
import pytest
from medics_ext_my_extension import MyExtension

def test_extension_metadata():
    ext = MyExtension()
    assert ext.get_version() == "1.0.0"
    assert ext.get_category() == "Custom Tools"
    assert len(ext.get_description()) > 0

def test_extension_initialize():
    ext = MyExtension()
    # Mock app_context for testing
    class MockContext:
        def get_component(self, name):
            return None
    
    result = ext.initialize(MockContext())
    assert result is True
```

Run tests:

```bash
pytest tests/
```

### Logging

Use the built-in logger for debugging:

```python
def initialize(self, app_context) -> bool:
    self.logger.debug("Debug message")
    self.logger.info("Info message")
    self.logger.warning("Warning message")
    self.logger.error("Error message")
    return True
```

Logs appear in MedICS console and log files.

## Publishing Your Extension

### 1. Prepare for Release

Update version in `pyproject.toml`:

```toml
[project]
version = "1.0.0"
```

### 2. Build Distribution

```bash
# Install build tools
pip install build twine

# Build distribution packages
python -m build

# This creates:
# - dist/medics-ext-my-extension-1.0.0.tar.gz
# - dist/medics_ext_my_extension-1.0.0-py3-none-any.whl
```

### 3. Test Installation

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/medics_ext_my_extension-1.0.0-py3-none-any.whl

# Test
python -m medics
```

### 4. Publish to PyPI

```bash
# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ medics-ext-my-extension

# If everything works, upload to PyPI
python -m twine upload dist/*
```

### 5. Users Install Your Extension

```bash
pip install medics-ext-my-extension
```

## Examples

### Example 1: Simple Tool Extension

```python
from medics_extension_sdk import BaseExtension
from PySide6 import QtWidgets

class SimpleToolExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Simple Tool",
            author_name="Developer"
        )
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "A simple tool that does one thing"
    
    def get_category(self) -> str:
        return "Tools"
    
    def create_widget(self, parent=None, **kwargs):
        widget = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel("Simple Tool"))
        return widget
```

### Example 2: Data Processing Extension

```python
from medics_extension_sdk import BaseExtension
from PySide6 import QtWidgets
import numpy as np

class DataProcessorExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Data Processor",
            author_name="Developer"
        )
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Process and analyze data"
    
    def get_category(self) -> str:
        return "Data Processing"
    
    def initialize(self, app_context) -> bool:
        self.app_context = app_context
        self.workspace_manager = app_context.get_component("workspace_manager")
        return True
    
    def create_widget(self, parent=None, **kwargs):
        from .ui.processor_widget import ProcessorWidget
        return ProcessorWidget(parent, self)
    
    def process_data(self, data):
        """Process data using workspace manager."""
        result = np.array(data) * 2
        self.workspace_manager.set_data("processed_result", result)
        self.logger.info("Data processed successfully")
        return result
```

### Example 3: Integration Extension

```python
from medics_extension_sdk import BaseExtension
from PySide6 import QtWidgets
import requests

class CloudIntegrationExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Cloud Integration",
            author_name="Developer"
        )
        self.api_endpoint = "https://api.example.com"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Integrate with cloud services"
    
    def get_category(self) -> str:
        return "Integration"
    
    def initialize(self, app_context) -> bool:
        self.app_context = app_context
        
        # Read API key from config
        config = app_context.get_component("config_manager")
        self.api_key = config.get_value("CloudIntegration", "api_key", "")
        
        if not self.api_key:
            self.logger.warning("API key not configured")
        
        return True
    
    def create_widget(self, parent=None, **kwargs):
        from .ui.integration_widget import IntegrationWidget
        return IntegrationWidget(parent, self)
    
    def upload_data(self, data):
        """Upload data to cloud service."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.api_endpoint}/upload",
            json=data,
            headers=headers
        )
        return response.json()
```

---

## Advanced Topics

### Extension Architecture

#### Extension Lifecycle

```
┌─────────────┐
│  Discovery  │  Extension Manager scans entry points
└──────┬──────┘
       │
┌──────▼──────┐
│ Registration│  Extension metadata is collected
└──────┬──────┘
       │
┌──────▼──────┐
│   Loading   │  Extension module is imported
└──────┬──────┘
       │
┌──────▼──────┐
│ Initialize  │  initialize() called with app_context
└──────┬──────┘
       │
┌──────▼──────┐
│   Active    │  Widget created when user opens extension
└──────┬──────┘
       │
┌──────▼──────┐
│   Cleanup   │  cleanup() called on unload/shutdown
└─────────────┘
```

#### Extension Discovery

MedICS uses setuptools entry points for extension discovery:

- Extensions are discovered via the `medics.extensions` entry point group
- Pip-installed packages are automatically found and loaded
- No manual registration or configuration needed

### Resource Management

#### Loading Extension Resources

Extensions can include additional resources (images, data files, configs):

```python
from pathlib import Path
from PySide6 import QtGui

class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Resource Demo",
            author_name="Developer"
        )
        # Get extension installation directory
        self.extension_dir = Path(__file__).parent
    
    def load_icon(self, icon_name: str) -> QtGui.QIcon:
        """Load icon from extension resources."""
        icon_path = self.extension_dir / "resources" / "icons" / icon_name
        if icon_path.exists():
            return QtGui.QIcon(str(icon_path))
        return QtGui.QIcon()
    
    def load_config(self) -> dict:
        """Load extension configuration."""
        config_path = self.extension_dir / "config" / "defaults.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
```

#### Package Data in pyproject.toml

Include non-Python files in your package:

```toml
[project]
name = "medics-ext-my-extension"
# ... other fields ...

[tool.setuptools]
packages = ["medics_ext_my_extension"]

[tool.setuptools.package-data]
medics_ext_my_extension = [
    "resources/**/*",
    "config/*.json",
    "data/*.csv",
    "*.qss"  # Qt stylesheets
]
```

### Asynchronous Operations

#### Background Tasks

For long-running operations, use Qt's threading capabilities:

```python
from PySide6 import QtCore

class DataProcessingWorker(QtCore.QThread):
    """Worker thread for background processing."""
    progress_updated = QtCore.Signal(int)
    processing_complete = QtCore.Signal(object)
    error_occurred = QtCore.Signal(str)
    
    def __init__(self, data, processor):
        super().__init__()
        self.data = data
        self.processor = processor
    
    def run(self):
        """Execute processing in background thread."""
        try:
            result = self.processor.process(
                self.data,
                progress_callback=self.progress_updated.emit
            )
            self.processing_complete.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Async Demo",
            author_name="Developer"
        )
        self.worker = None
    
    def start_processing(self, data):
        """Start background processing."""
        if self.worker and self.worker.isRunning():
            self.logger.warning("Processing already in progress")
            return
        
        self.worker = DataProcessingWorker(data, self.processor)
        self.worker.progress_updated.connect(self.on_progress)
        self.worker.processing_complete.connect(self.on_complete)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def cleanup(self):
        """Ensure worker threads are stopped."""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        super().cleanup()
```

#### Async/Await with asyncio

For I/O-bound operations:

```python
import asyncio
from PySide6 import QtCore, QtAsyncio

class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Async IO Demo",
            author_name="Developer"
        )
    
    async def fetch_data_async(self, url: str) -> dict:
        """Async data fetching."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
    def fetch_data(self, url: str):
        """Wrapper to call async function from sync context."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.fetch_data_async(url))
```

### Configuration Management

#### Persistent Settings

Extensions can store and retrieve persistent configuration:

```python
class MyExtension(BaseExtension):
    def initialize(self, app_context) -> bool:
        self.app_context = app_context
        self.config_manager = app_context.get_component("config_manager")
        
        # Load extension settings
        self.load_settings()
        return True
    
    def load_settings(self):
        """Load extension-specific settings."""
        # Settings are stored in medics config.ini under extension section
        section = f"Extension.{self.get_name()}"
        
        self.setting1 = self.config_manager.get_value(
            section, "setting1", "default_value"
        )
        self.setting2 = int(self.config_manager.get_value(
            section, "setting2", "100"
        ))
    
    def save_settings(self):
        """Save extension settings."""
        section = f"Extension.{self.get_name()}"
        
        self.config_manager.set_value(section, "setting1", self.setting1)
        self.config_manager.set_value(section, "setting2", str(self.setting2))
        self.config_manager.save_config()
    
    def cleanup(self):
        """Save settings before cleanup."""
        self.save_settings()
        super().cleanup()
```

### Extension API System

#### Exposing APIs to Python Console

Extensions can expose methods to the MedICS Python console using the `@api_method` decorator:

```python
from medics_extension_sdk import BaseExtension, api_method

class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="API Demo",
            author_name="Developer"
        )
        self._data = []
    
    @api_method(
        description="Process input data and return results",
        category="Data Processing"
    )
    def process_data(self, input_data: list, threshold: float = 0.5) -> dict:
        """Process data with configurable threshold.
        
        Args:
            input_data: List of values to process
            threshold: Processing threshold (default: 0.5)
        
        Returns:
            dict: Processing results with statistics
        """
        try:
            results = [x for x in input_data if x > threshold]
            return {
                "filtered": results,
                "count": len(results),
                "mean": sum(results) / len(results) if results else 0
            }
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return {"error": str(e)}
    
    @api_method(description="Get extension status")
    def get_status(self) -> dict:
        """Get current extension status."""
        return {
            "name": self.get_name(),
            "version": self.get_version(),
            "data_count": len(self._data),
            "initialized": self.app_context is not None
        }
```

**Usage in Python Console:**

```python
# Access extension API
ext = medics.extensions.get_api("api_demo")

# Call API methods
result = ext.process_data([0.1, 0.6, 0.8, 0.3], threshold=0.4)
status = ext.get_status()

# List available APIs
ext.list_apis()
```

### Testing Strategies

#### Unit Testing

```python
# tests/test_my_extension.py
import pytest
from medics_ext_my_extension import MyExtension

class MockAppContext:
    """Mock application context for testing."""
    def __init__(self):
        self.components = {}
    
    def get_component(self, name: str):
        return self.components.get(name)

@pytest.fixture
def extension():
    """Create extension instance for testing."""
    ext = MyExtension()
    ext.initialize(MockAppContext())
    yield ext
    ext.cleanup()

def test_extension_metadata(extension):
    """Test extension metadata."""
    assert extension.get_name() == "My Extension"
    assert extension.get_version().startswith("1.")
    assert len(extension.get_description()) > 0

def test_data_processing(extension):
    """Test data processing functionality."""
    result = extension.process_data([1, 2, 3])
    assert result is not None
    assert "processed" in result

def test_widget_creation(extension):
    """Test widget creation."""
    widget = extension.create_widget()
    assert widget is not None
    widget.deleteLater()
```

#### Integration Testing

```python
# tests/test_integration.py
import pytest
from PySide6 import QtWidgets
from medics_ext_my_extension import MyExtension

@pytest.fixture
def app(qapp):
    """Create Qt application for testing."""
    return qapp

def test_extension_ui_integration(app):
    """Test extension UI integration."""
    ext = MyExtension()
    ext.initialize(MockAppContext())
    
    widget = ext.create_widget()
    widget.show()
    
    # Test UI interactions
    button = widget.findChild(QtWidgets.QPushButton, "processButton")
    assert button is not None
    
    # Simulate button click
    button.click()
    
    # Verify results
    # ... test specific UI behavior
    
    widget.close()
    ext.cleanup()
```

#### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-qt pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=medics_ext_my_extension tests/

# Run specific test file
pytest tests/test_my_extension.py -v

# Run specific test
pytest tests/test_my_extension.py::test_extension_metadata -v
```

### Performance Optimization

#### Lazy Loading

Load resources only when needed:

```python
class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="Lazy Demo",
            author_name="Developer"
        )
        self._model = None  # Lazy-loaded ML model
    
    @property
    def model(self):
        """Lazy-load ML model."""
        if self._model is None:
            self.logger.info("Loading ML model...")
            import tensorflow as tf
            self._model = tf.keras.models.load_model(
                self.extension_dir / "models" / "model.h5"
            )
        return self._model
    
    def predict(self, data):
        """Make prediction using lazy-loaded model."""
        return self.model.predict(data)
```

#### Caching Results

```python
from functools import lru_cache

class MyExtension(BaseExtension):
    @lru_cache(maxsize=100)
    def expensive_computation(self, param: str) -> dict:
        """Cache expensive computations."""
        # Expensive operation here
        result = self._compute(param)
        return result
    
    def cleanup(self):
        """Clear cache on cleanup."""
        self.expensive_computation.cache_clear()
        super().cleanup()
```

### Security Best Practices

#### Input Validation

```python
class MyExtension(BaseExtension):
    def process_file(self, file_path: str) -> bool:
        """Process file with validation."""
        from pathlib import Path
        
        # Validate file path
        path = Path(file_path)
        
        # Check file exists
        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        # Check file extension
        allowed_extensions = {'.dcm', '.nii', '.png', '.jpg'}
        if path.suffix.lower() not in allowed_extensions:
            self.logger.error(f"Unsupported file type: {path.suffix}")
            return False
        
        # Check file size (e.g., max 100 MB)
        max_size = 100 * 1024 * 1024
        if path.stat().st_size > max_size:
            self.logger.error("File too large")
            return False
        
        # Process file
        return self._safe_process(path)
```

#### Secure Configuration

```python
class MyExtension(BaseExtension):
    def initialize(self, app_context) -> bool:
        self.app_context = app_context
        
        # Load API key from secure config
        config = app_context.get_component("config_manager")
        api_key = config.get_value("MyExtension", "api_key", "")
        
        if not api_key:
            self.logger.warning(
                "API key not configured. "
                "Set it in config.ini under [MyExtension] section."
            )
            return False
        
        # Don't log sensitive data
        self.api_key = api_key
        self.logger.info("Extension initialized with credentials")
        return True
```

---

## Troubleshooting

### Common Issues and Solutions

#### Extension Not Discovered

**Problem:** Extension doesn't appear in MedICS after installation.

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep medics-ext
   ```

2. **Check entry point registration:**
   ```bash
   python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='medics.extensions')))"
   ```

3. **Verify entry point in pyproject.toml:**
   ```toml
   [project.entry-points."medics.extensions"]
   my_extension = "medics_ext_my_extension:MyExtension"
   ```

4. **Reinstall in development mode:**
   ```bash
   pip install -e .
   ```

5. **Check MedICS logs:**
   ```
   Look for "Extension discovery" messages in the console or log file
   ```

#### Import Errors

**Problem:** `ModuleNotFoundError` or `ImportError` when loading extension.

**Solutions:**

1. **Check dependencies in pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       "medics-extension-sdk>=1.0.0",
       "PySide6>=6.0.0",
       "numpy>=1.20.0",
       # ... other dependencies
   ]
   ```

2. **Install missing dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python version compatibility:**
   ```toml
   [project]
   requires-python = ">=3.8"
   ```

4. **Verify module structure:**
   ```
   medics_ext_my_extension/
   ├── __init__.py  ← Must export extension class
   └── ...
   ```

#### Widget Not Displaying

**Problem:** Extension loads but widget doesn't show or appears blank.

**Solutions:**

1. **Check `create_widget()` returns QWidget:**
   ```python
   def create_widget(self, parent=None, **kwargs):
       widget = MyWidget(parent)  # Must be QWidget subclass
       return widget  # Must return the widget!
   ```

2. **Verify widget has layout:**
   ```python
   class MyWidget(QtWidgets.QWidget):
       def __init__(self, parent=None):
           super().__init__(parent)
           layout = QtWidgets.QVBoxLayout(self)  # Set layout
           # Add widgets to layout
   ```

3. **Check for initialization errors:**
   ```python
   def create_widget(self, parent=None, **kwargs):
       try:
           widget = MyWidget(parent, self.app_context)
           return widget
       except Exception as e:
           self.logger.error(f"Widget creation failed: {e}")
           raise
   ```

#### Logger Not Working

**Problem:** Log messages don't appear in console.

**Solutions:**

1. **Use inherited logger:**
   ```python
   class MyExtension(BaseExtension):
       def __init__(self):
           super().__init__(
               extension_name="My Extension",
               author_name="Developer"
           )
           # self.logger is automatically created by BaseExtension
       
       def initialize(self, app_context) -> bool:
           self.logger.info("Initialization started")  # ✓ Correct
           # Don't create new logger
           return True
   ```

2. **Check log level:**
   ```python
   import logging
   self.logger.setLevel(logging.DEBUG)  # Show all messages
   ```

#### Configuration Not Persisting

**Problem:** Extension settings are lost after restart.

**Solutions:**

1. **Save configuration on changes:**
   ```python
   def save_settings(self):
       config = self.app_context.get_component("config_manager")
       section = f"Extension.{self.get_name()}"
       config.set_value(section, "key", "value")
       config.save_config()  # ← Don't forget this!
   ```

2. **Save in cleanup:**
   ```python
   def cleanup(self):
       self.save_settings()
       super().cleanup()
   ```

#### Memory Leaks

**Problem:** Memory usage grows over time.

**Solutions:**

1. **Clean up resources in `cleanup()`:**
   ```python
   def cleanup(self):
       # Close files
       if hasattr(self, 'file_handle'):
           self.file_handle.close()
       
       # Stop threads
       if hasattr(self, 'worker') and self.worker:
           self.worker.quit()
           self.worker.wait()
       
       # Clear caches
       if hasattr(self, 'cache'):
           self.cache.clear()
       
       # Delete large objects
       self._large_data = None
       
       super().cleanup()
   ```

2. **Disconnect signals:**
   ```python
   def cleanup(self):
       # Disconnect all signals
       if self.worker:
           self.worker.progress_updated.disconnect()
           self.worker.finished.disconnect()
       super().cleanup()
   ```

3. **Delete widgets properly:**
   ```python
   def cleanup(self):
       if self.widget:
           self.widget.deleteLater()  # Qt-safe deletion
           self.widget = None
       super().cleanup()
   ```

#### API Methods Not Available in Console

**Problem:** `@api_method` decorated methods don't show up in Python console.

**Solutions:**

1. **Verify decorator import:**
   ```python
   from medics_extension_sdk import api_method  # ✓ Correct
   ```

2. **Check decorator syntax:**
   ```python
   @api_method(description="Method description")
   def my_method(self):  # ← Must be instance method
       pass
   ```

3. **Ensure extension is loaded:**
   ```python
   # In Python console
   medics.extensions.list_loaded()  # Check if extension is loaded
   ```

4. **Widget must be created before API access:**
   ```python
   # Open extension UI first, then:
   ext = medics.extensions.get_api("my_extension")
   ```

### Debugging Tips

#### Enable Debug Logging

```python
class MyExtension(BaseExtension):
    def __init__(self):
        super().__init__(
            extension_name="My Extension",
            author_name="Developer"
        )
        # Enable debug logging
        import logging
        self.logger.setLevel(logging.DEBUG)
        
        # Add file handler for persistent logs
        handler = logging.FileHandler('my_extension.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
```

#### Use Assertions for Development

```python
def process_data(self, data):
    assert data is not None, "Data cannot be None"
    assert len(data) > 0, "Data cannot be empty"
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    
    # Process data
    return result
```

#### Profile Performance

```python
import cProfile
import pstats

def profile_function(self):
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    result = self.expensive_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

---

## Best Practices

### Code Organization

✅ **DO:**
- Separate UI code into dedicated modules (`ui/` directory)
- Use meaningful class and method names
- Keep `__init__.py` focused on the extension class
- Group related functionality into submodules

```
medics_ext_my_extension/
├── __init__.py           # Extension class only
├── core/                 # Core logic
│   ├── processor.py
│   └── analyzer.py
├── ui/                   # UI components
│   ├── main_widget.py
│   └── dialogs.py
└── utils/                # Utilities
    └── helpers.py
```

❌ **DON'T:**
- Put all code in `__init__.py`
- Mix UI and business logic
- Use cryptic abbreviations

### Error Handling

✅ **DO:**
```python
def process_file(self, path: str) -> bool:
    try:
        data = self.load_file(path)
        result = self.process_data(data)
        self.save_result(result)
        return True
    except FileNotFoundError:
        self.logger.error(f"File not found: {path}")
        return False
    except PermissionError:
        self.logger.error(f"Permission denied: {path}")
        return False
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}", exc_info=True)
        return False
```

❌ **DON'T:**
```python
def process_file(self, path: str):
    # Silent failures
    try:
        data = self.load_file(path)
    except:
        pass
    
    # Unhandled exceptions
    result = self.process_data(data)  # May crash
```

### Documentation

✅ **DO:**
- Write clear docstrings for all public methods
- Include type hints
- Document parameters and return values
- Provide usage examples

```python
def process_image(
    self,
    image: np.ndarray,
    threshold: float = 0.5,
    method: str = "otsu"
) -> np.ndarray:
    """Process image with specified method.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        threshold: Binarization threshold, 0.0 to 1.0 (default: 0.5)
        method: Processing method, one of ['otsu', 'adaptive', 'manual']
    
    Returns:
        Processed image as numpy array with same shape as input
    
    Raises:
        ValueError: If threshold is out of range or method is invalid
    
    Example:
        >>> ext = MyExtension()
        >>> result = ext.process_image(img, threshold=0.7, method='otsu')
    """
```

### Version Management

✅ **DO:**
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Document changes in CHANGELOG.md
- Update version in `__init__.py` or `pyproject.toml`

```python
# __init__.py
__version__ = "1.2.3"

# CHANGELOG.md
## [1.2.3] - 2025-12-13
### Fixed
- Fixed memory leak in image processing
### Added
- Added batch processing support
```

### Dependency Management

✅ **DO:**
- Pin major versions, allow minor updates
- List all runtime dependencies
- Separate development dependencies

```toml
[project]
dependencies = [
    "medics-extension-sdk>=1.0.0,<2.0.0",
    "PySide6>=6.0.0,<7.0.0",
    "numpy>=1.20.0,<2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-qt>=4.0.0",
    "black>=22.0.0",
    "mypy>=0.950",
]
```

❌ **DON'T:**
- Use unpinned dependencies (`numpy>=1.0`)
- Mix runtime and dev dependencies
- Forget to specify `medics-extension-sdk`

---

## Additional Resources

- **MedICS Documentation**: https://github.com/medics-dev/MedICS
- **Extension SDK**: https://github.com/medics-dev/MedICS_Extension_Base
- **Python Packaging**: https://packaging.python.org/
- **Entry Points**: https://setuptools.pypa.io/en/latest/userguide/entry_point.html
- **PySide6 Documentation**: https://doc.qt.io/qtforpython/

## Getting Help

- **GitHub Issues**: https://github.com/medics-dev/MedICS/issues
- **Discussions**: https://github.com/medics-dev/MedICS/discussions
- **Example Extensions**: https://github.com/medics-dev?q=medics-ext

## License

MedICS is open source. Your extensions can use any license you choose, but MIT or BSD are recommended for maximum compatibility.
