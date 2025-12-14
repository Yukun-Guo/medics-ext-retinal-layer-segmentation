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
