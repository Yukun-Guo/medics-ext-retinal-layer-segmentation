# MedICS Extension: Example

A template extension for MedICS that demonstrates the plugin system.

## Features

- Example feature 1
- Example feature 2
- Example feature 3

## Installation

### From PyPI

```bash
pip install medics-ext-retinal-layer-segmentation
```

### From Source

```bash
git clone https://github.com/your-github-username/medics-ext-retinal-layer-segmentation.git
cd medics-ext-retinal-layer-segmentation
pip install -e .
```

**Note:** This package includes a large model file (~107 MB). The installation may take a few moments to complete as it includes all necessary model files.

### Verifying Installation

To verify that all files (including the large model file) are properly installed:

```bash
python verify_package.py
```

## Usage

1. Install MedICS and this extension
2. Launch MedICS: `python -m medics`
3. Open the extension from the Extensions menu

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/your-github-username/medics-ext-retinal-layer-segmentation.git
cd medics-ext-retinal-layer-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/
```

### Building

```bash
python -m build
```

## Configuration

The extension can be configured in MedICS config.ini:

```ini
[RetinalLayerSegmentationExtension]
setting1 = value1
setting2 = value2
```

## Dependencies

- medics >= 2.0.0
- medics-extension-sdk >= 1.0.0
- PySide6 >= 6.8.0
- numpy >= 1.20.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- Issues: https://github.com/your-github-username/medics-ext-retinal-layer-segmentation/issues
- Discussions: https://github.com/your-github-username/medics-ext-retinal-layer-segmentation/discussions

## Credits

Created by Extension Developer

## Changelog

### 1.0.0 (YYYY-MM-DD)

- Initial release
- Basic functionality implemented
