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
- [Bundled vs External Extensions](#bundled-vs-external-extensions)
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
✅ **Backward Compatible** - Bundled extensions still work  

## Extension Discovery

MedICS discovers extensions using three methods (in priority order):

### 1. Entry Points (Recommended) ⭐

The standard Python plugin mechanism. Define in your `pyproject.toml`:

```toml
[project.entry-points."medics.extensions"]
my_extension = "medics_ext_my_extension:MyExtension"
```

**Benefits:**
- Standard Python packaging
- Easy distribution via PyPI
- Automatic discovery
- Clean separation from MedICS core

### 2. Filesystem (Development/Bundled)

Extensions in `medics/extensions/` directory. Useful for:
- Development and testing
- Bundled extensions (deprecated)
- Local/private extensions

### 3. Priority Rules

When the same extension is found multiple ways:
- **Entry point** versions take highest priority
- **Filesystem** versions are skipped if entry point exists
- First discovered version is used (no duplicates)

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

## Bundled vs External Extensions

### Migration Path

MedICS provides a smooth migration path from bundled to external extensions:

| Phase | Bundled Extensions | External Extensions |
|-------|-------------------|-------------------|
| **Current** | ✅ Fully supported | ✅ Fully supported |
| **Near Future** | ⚠️ Deprecated warnings | ✅ Recommended |
| **Future Major** | ❌ Removed | ✅ Only method |

### Advantages of External Extensions

| Feature | Bundled | External |
|---------|---------|----------|
| **Independent versioning** | ❌ | ✅ |
| **Separate releases** | ❌ | ✅ |
| **Easy distribution** | ❌ | ✅ |
| **Own dependencies** | ❌ | ✅ |
| **PyPI publication** | ❌ | ✅ |
| **User choice** | ❌ | ✅ |
| **Easier testing** | ❌ | ✅ |

### Converting Bundled Extension

To convert a bundled extension to external:

1. **Create new package structure** (see Quick Start)
2. **Copy extension code** to new package
3. **Update imports** to use new package name
4. **Add entry point** in pyproject.toml
5. **Test installation** with `pip install -e .`
6. **Deprecate bundled version** (add warning)
7. **Eventually remove** bundled version

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
