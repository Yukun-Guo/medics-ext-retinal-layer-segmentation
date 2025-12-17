# Extension Template

This directory contains a complete template for creating MedICS extensions as independent pip packages.

## Quick Start

1. **Copy this template** to a new directory:
   ```bash
   cp -r extension_template/ ../medics-ext-myextension/
   cd ../medics-ext-myextension/
   ```

2. **Customize the template**:
   - Update `pyproject.toml` with your extension name, author, and description
   - Rename package directory: `medics_ext_example` → `medics_ext_myextension`
   - Update imports in `__init__.py`
   - Implement your extension logic
   - Update README.md

3. **Install in development mode**:
   ```bash
   pip install -e .
   ```

4. **Test your extension**:
   ```bash
   python -m medics
   ```

## Template Structure

```
extension_template/
├── pyproject.toml              # Package configuration
├── README.md                   # Package documentation
├── LICENSE                     # License file (MIT)
├── .gitignore                 # Git ignore rules
├── medics_ext_example/        # Extension package
│   ├── __init__.py            # Extension class
│   └── ui/                    # UI components
│       ├── __init__.py
│       └── main_widget.py     # Main widget
└── tests/                     # Unit tests
    └── test_extension.py
```

## Customization Checklist

- [ ] Update `name` in pyproject.toml
- [ ] Update `version` in pyproject.toml
- [ ] Update `description` in pyproject.toml
- [ ] Update `authors` in pyproject.toml
- [ ] Update `project.urls` in pyproject.toml
- [ ] Update entry point name in `[project.entry-points."medics.extensions"]`
- [ ] Rename package directory (`medics_ext_example` → `medics_ext_yourname`)
- [ ] Update extension class name in `__init__.py`
- [ ] Update extension metadata (name, author, version, description, category)
- [ ] Implement extension logic
- [ ] Update README.md with your extension details
- [ ] Add tests in `tests/`
- [ ] Update LICENSE if using different license

## Development Workflow

### 1. Initial Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install medics and development dependencies
pip install medics pytest

# Install your extension in development mode
pip install -e .
```

### 2. Development

```bash
# Make changes to your extension code
# Test by running medics
python -m medics

# Your extension should appear in the Extensions menu
```

### 3. Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=medics_ext_example tests/
```

### 4. Building

```bash
# Install build tools
pip install build

# Build distribution
python -m build

# This creates:
# dist/medics-ext-example-1.0.0.tar.gz
# dist/medics_ext_example-1.0.0-py3-none-any.whl
```

### 5. Publishing

```bash
# Install twine
pip install twine

# Upload to Test PyPI (recommended first)
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ medics-ext-example

# Upload to PyPI
python -m twine upload dist/*
```

## Debugging with VS Code (F5)

You can debug your extension in VS Code by opening the generated extension folder as the workspace and pressing F5.

- The included `.vscode/launch.json` has a configuration named `Run MedICS (this extension) - F5` that:
   - Launches MedICS with `python -m medics`.
   - Sets `cwd` to the repository root (workspace parent) so MedICS finds its resources.
   - Adds both the extension folder and its parent to `PYTHONPATH` so MedICS can import the extension without installing it.
   - Provides an alternate configuration `Run MedICS (this extension) - F5 (Use workspace .venv)` which attempts to use `${workspaceFolder}/.venv/bin/python` as the interpreter.

How to use:

1. Open the extension folder (the folder you created with the template) in VS Code.
2. Ensure your workspace has a `.venv` if you want to use the venv configuration, or select the interpreter in the status bar.
3. Press F5 (Run → Start Debugging). The `Run MedICS (this extension) - F5` configuration will start MedICS and load your extension.

Notes:

- If you prefer a specific virtual environment, create a `.env` file in the extension folder and export variables or adjust the interpreter in VS Code's Python selector.
- The venv debug configuration will use `${workspaceFolder}/.venv/bin/python` if present; otherwise select the interpreter manually.

## Entry Point Configuration

The entry point in `pyproject.toml` tells MedICS where to find your extension:

```toml
[project.entry-points."medics.extensions"]
example = "medics_ext_example:ExampleExtension"
#  ^^^      ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
#  |        |                   └─ Extension class name
#  |        └─ Package/module path
#  └─ Short identifier (used in MedICS)
```

## Extension API

Your extension must:

1. **Inherit from BaseExtension**:
   ```python
   from medics_extension_sdk import BaseExtension
   
   class MyExtension(BaseExtension):
       pass
   ```

2. **Implement required methods**:
   - `get_version() -> str`
   - `get_description() -> str`
   - `get_category() -> str`
   - `initialize(app_context) -> bool`
   - `create_widget(parent=None, **kwargs) -> QWidget`
   - `cleanup() -> None`

3. **Export the class**:
   ```python
   __all__ = ['MyExtension']
   ```

## Resources

- **Development Guide**: ../docs/EXTENSION_DEVELOPMENT.md
- **Extension SDK**: https://github.com/medics-dev/MedICS_Extension_Base
- **Python Packaging**: https://packaging.python.org/
- **MedICS Docs**: https://github.com/medics-dev/MedICS

## Support

If you have questions or issues with extension development:

- Open an issue: https://github.com/medics-dev/MedICS/issues
- Start a discussion: https://github.com/medics-dev/MedICS/discussions
