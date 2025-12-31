# Installation Guide for Large Model Files (Chunked Approach)

This document explains how the `layersegmodel.enc` file (107 MB) is split into smaller chunks and automatically reassembled when users install and use this package.

## Overview

The large model file has been split into 3 chunks, each under 50 MB:
- `layersegmodel.part000` - 45 MB
- `layersegmodel.part001` - 45 MB  
- `layersegmodel.part002` - 17 MB
- `layersegmodel.meta.json` - metadata for reassembly

**Total:** ~107 MB when reassembled

## How It Works

### For Package Maintainers

1. **Splitting the Model File** (one-time operation):
   ```bash
   python split_model_file.py
   ```
   This creates chunks in `medics_ext_retinal_layer_segmentation/model/chunks/`

2. **Commit Only the Chunks**:
   - The original `.enc` file is in `.gitignore`
   - Only the chunks and metadata are tracked in git
   - Each chunk is under 50 MB, suitable for GitHub and PyPI

### For End Users

When users install the package:
```bash
pip install git+https://github.com/your-username/medics-ext-retinal-layer-segmentation.git
```

The following happens automatically:

1. **Package Installation**: Chunks are copied to the installation directory
2. **First Import**: When the package is imported:
   ```python
   import medics_ext_retinal_layer_segmentation
   ```
   The `__init__.py` triggers reassembly
3. **Automatic Reassembly**: 
   - Chunks are combined into `layersegmodel.enc`
   - MD5 checksums verify integrity
   - File is cached for subsequent uses
4. **Runtime Usage**: The extension uses `get_model_file_path()` to access the model

## Configuration Files

### 1. `pyproject.toml`
```toml
[tool.setuptools.package-data]
medics_ext_retinal_layer_segmentation = [
    "model/chunks/*",  # Include all chunk files and metadata
    ...
]
```

### 2. `MANIFEST.in`
```
recursive-include medics_ext_retinal_layer_segmentation/model/chunks *
```

### 3. `.gitignore`
```
# Exclude the reassembled model file (auto-generated)
medics_ext_retinal_layer_segmentation/model/layersegmodel.enc
```

## Testing

To verify the reassembly process works correctly:

1. **Run the test script:**
   ```bash
   python test_reassembly.py
   ```
   This simulates the full reassembly process and verifies integrity.

2. **Test building the package:**
   ```bash
   pip install build
   python -m build
   ```
   Creates distributable `.whl` and `.tar.gz` files.

3. **Verify chunks are in the wheel:**
   ```bash
   unzip -l dist/*.whl | grep "model/chunks"
   ```

4. **Test installation in a clean environment:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install .
   python -c "from medics_ext_retinal_layer_segmentation.model_reassembly import get_model_file_path; print(f'Model ready: {get_model_file_path() is not None}')"
   ```

## Technical Details

### File Structure
```
medics_ext_retinal_layer_segmentation/
├── model/
│   ├── chunks/
│   │   ├── layersegmodel.part000    (45 MB)
│   │   ├── layersegmodel.part001    (45 MB)
│   │   ├── layersegmodel.part002    (17 MB)
│   │   └── layersegmodel.meta.json  (metadata)
│   └── layersegmodel.enc            (auto-generated, 107 MB)
└── model_reassembly.py              (reassembly logic)
```

### Reassembly Process

1. **On Package Import**: `__init__.py` calls `ensure_model_ready()`
2. **Check Existing File**: If `layersegmodel.enc` exists and passes MD5 verification, skip reassembly
3. **Load Metadata**: Read `layersegmodel.meta.json` for chunk information
4. **Verify Chunks**: Check MD5 of each chunk before reassembly
5. **Combine Chunks**: Sequentially append chunks to create the full file
6. **Final Verification**: Verify the reassembled file's MD5 matches the original
7. **Cache**: Subsequent calls use the reassembled file without reprocessing

### Thread Safety

The `ModelFileManager` uses:
- Singleton pattern to ensure one instance
- Thread locks for concurrent access protection
- Atomic file operations

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
