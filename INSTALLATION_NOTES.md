# Installation Guide for Large Model Files

This document explains how the `layersegmodel.enc` file (107 MB) is included when users install this package via pip from source code.

## Configuration Changes Made

### 1. Updated `pyproject.toml`

The `[tool.setuptools.package-data]` section now includes:
```toml
[tool.setuptools.package-data]
medics_ext_retinal_layer_segmentation = [
    "resources/**/*",
    "model/*.enc",      # ← This ensures model files are included
    "*.ui",
    "*.ini",
    "*.json",
]
```

### 2. Created `MANIFEST.in`

This file explicitly declares which files should be included in source distributions:
```
recursive-include medics_ext_retinal_layer_segmentation/model *.enc
```

## How It Works

When users install the package via pip from source:

```bash
pip install git+https://github.com/your-username/medics-ext-retinal-layer-segmentation.git
```

Or from a local clone:

```bash
git clone https://github.com/your-username/medics-ext-retinal-layer-segmentation.git
cd medics-ext-retinal-layer-segmentation
pip install .
```

The following happens:

1. **setuptools** reads `pyproject.toml` and `MANIFEST.in`
2. The `package-data` configuration tells setuptools to include `model/*.enc` files
3. The `MANIFEST.in` ensures the file is included in source distributions
4. The 107 MB `layersegmodel.enc` file is copied to the installation location

## Testing

To verify the configuration works correctly:

1. **Check files are present locally:**
   ```bash
   python verify_package.py
   ```

2. **Test building the package:**
   ```bash
   pip install build
   python -m build
   ```
   This creates a `.whl` file and `.tar.gz` file in the `dist/` directory.

3. **Verify the model file is in the wheel:**
   ```bash
   unzip -l dist/*.whl | grep layersegmodel.enc
   ```

4. **Test installation in a clean environment:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install .
   python -c "from pathlib import Path; import medics_ext_retinal_layer_segmentation; pkg_path = Path(medics_ext_retinal_layer_segmentation.__file__).parent; model_file = pkg_path / 'model' / 'layersegmodel.enc'; print(f'Model file exists: {model_file.exists()}'); print(f'Size: {model_file.stat().st_size / 1024 / 1024:.2f} MB' if model_file.exists() else 'N/A')"
   ```

## Important Notes

### File Size Considerations

- The model file is **107 MB**, which is acceptable for pip packages
- PyPI has a 100 MB limit per file, so if you plan to upload to PyPI, you may need to:
  - Use Git LFS (Large File Storage) for the model file
  - Host the model separately and download it on first use
  - Split the model into smaller chunks

### Git Considerations

For such a large binary file, consider using **Git LFS**:

```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "medics_ext_retinal_layer_segmentation/model/*.enc"

# Add the .gitattributes file
git add .gitattributes

# Commit as usual
git add medics_ext_retinal_layer_segmentation/model/layersegmodel.enc
git commit -m "Add model file via Git LFS"
```

## Alternative Approaches (if needed)

If the 107 MB file size causes issues, consider:

1. **Lazy Loading**: Download the model on first use
2. **Separate Model Package**: Create a separate package just for models
3. **Cloud Storage**: Host models on cloud storage (S3, Azure Blob, etc.) and download at runtime
4. **Model Compression**: Use compression techniques to reduce file size

## Verification

The `verify_package.py` script confirms that critical files are present before distribution.
