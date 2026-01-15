# Model File Split and Reassembly Feature - Implementation Guide

## Overview

This guide documents a robust solution for handling large model files in Python packages by splitting them into smaller chunks for version control and distribution, then automatically reassembling them at runtime. This approach solves the common problem of Git/GitHub file size limits (typically 100MB) while maintaining package integrity and ease of use.

## Problem Statement

**Challenge**: Large ML model files (e.g., 100+ MB) cannot be directly committed to Git repositories due to file size limitations.

**Solution**: Split large files into smaller chunks (< 50 MB each) for version control, then automatically reassemble them when the package is first imported, with full integrity verification using MD5 checksums.

## Architecture

### Components

1. **split_model_file.py** - One-time splitting utility (for maintainers)
2. **model_reassembly.py** - Automatic reassembly module (runtime)
3. **Metadata file** - JSON file containing chunk information and checksums
4. **Package initialization** - Automatic model preparation on first import

### File Structure

```
your_package/
├── split_model_file.py              # Splitting utility (maintainer tool)
├── your_package/
│   ├── __init__.py                  # Automatic reassembly on import
│   ├── model_reassembly.py          # Reassembly logic
│   └── model/
│       ├── chunks/                  # Chunk files (committed to Git)
│       │   ├── model.meta.json      # Metadata with checksums
│       │   ├── model.part000        # First chunk
│       │   ├── model.part001        # Second chunk
│       │   └── model.part00N        # Last chunk
│       └── model.enc                # Reassembled file (not in Git, generated at runtime)
├── pyproject.toml                   # Package configuration
└── MANIFEST.in                      # Include rules for package data
```

## Implementation Steps

### Step 1: Create the Splitting Utility

Create `split_model_file.py` in your project root:

```python
#!/usr/bin/env python3
"""
Utility script to split the large model file into smaller chunks.
This is a one-time operation for package maintainers.
"""
import hashlib
import json
from pathlib import Path


def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def split_file(input_file, chunk_size_mb=45):
    """
    Split a large file into smaller chunks.
    
    Args:
        input_file: Path to the file to split
        chunk_size_mb: Size of each chunk in megabytes (default: 45 MB)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    # Calculate chunk size in bytes
    chunk_size = chunk_size_mb * 1024 * 1024
    
    # Create output directory for chunks
    output_dir = input_path.parent / "chunks"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Splitting {input_path.name}...")
    print(f"File size: {input_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Chunk size: {chunk_size_mb} MB")
    print(f"Output directory: {output_dir}")
    
    # Calculate MD5 of original file
    print("Calculating MD5 checksum of original file...")
    original_md5 = calculate_md5(input_path)
    print(f"Original MD5: {original_md5}")
    
    # Split the file
    chunk_info = []
    chunk_index = 0
    
    with open(input_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            # Create chunk filename
            chunk_filename = f"{input_path.stem}.part{chunk_index:03d}"
            chunk_path = output_dir / chunk_filename
            
            # Write chunk
            with open(chunk_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)
            
            # Calculate chunk MD5
            chunk_md5 = calculate_md5(chunk_path)
            
            chunk_info.append({
                "filename": chunk_filename,
                "size": len(chunk_data),
                "md5": chunk_md5
            })
            
            print(f"  Created: {chunk_filename} ({len(chunk_data) / 1024 / 1024:.2f} MB)")
            chunk_index += 1
    
    # Create metadata file
    metadata = {
        "original_filename": input_path.name,
        "original_size": input_path.stat().st_size,
        "original_md5": original_md5,
        "chunk_size": chunk_size,
        "total_chunks": len(chunk_info),
        "chunks": chunk_info
    }
    
    metadata_path = output_dir / f"{input_path.stem}.meta.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated {len(chunk_info)} chunks")
    print(f"Metadata saved to: {metadata_path}")
    print("\nNext steps:")
    print(f"1. Delete the original file: {input_path}")
    print(f"2. The chunks in {output_dir} will be included in the package")
    print("3. Run the package - it will automatically reassemble the file on first use")
    
    return True


if __name__ == "__main__":
    # Update this path to your model file
    model_file = Path(__file__).parent / "your_package" / "model" / "your_model.enc"
    
    if model_file.exists():
        split_file(model_file, chunk_size_mb=45)  # Use 45MB to stay under 50MB GitHub limit
    else:
        print(f"Error: Model file not found at {model_file}")
        print("Please update the path in this script.")
```

**Key Parameters:**
- `chunk_size_mb=45`: Stay under GitHub's 50MB limit for individual files
- Adjust based on your needs (e.g., 45MB is safe, 48MB allows some overhead)

### Step 2: Create the Reassembly Module

Create `your_package/model_reassembly.py`:

```python
"""
Model file reassembly module.

This module automatically reassembles the model file from chunks when needed.
It verifies file integrity using MD5 checksums and provides thread-safe operations.
"""
import hashlib
import json
import threading
from pathlib import Path
from typing import Optional


class ModelFileManager:
    """Manages model file reassembly from chunks."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model file manager."""
        if not hasattr(self, '_initialized'):
            self.model_dir = Path(__file__).parent / "model"
            self.chunks_dir = self.model_dir / "chunks"
            self.model_file = self.model_dir / "your_model.enc"  # Update filename
            self.metadata_file = self.chunks_dir / "your_model.meta.json"  # Update filename
            self._initialized = True
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def _load_metadata(self) -> Optional[dict]:
        """Load chunk metadata from JSON file."""
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            return None
    
    def _verify_reassembled_file(self, metadata: dict) -> bool:
        """Verify the reassembled file matches the original."""
        if not self.model_file.exists():
            return False
        
        # Check file size
        actual_size = self.model_file.stat().st_size
        expected_size = metadata["original_size"]
        
        if actual_size != expected_size:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {actual_size}")
            return False
        
        # Verify MD5 checksum
        print("Verifying model file integrity...")
        actual_md5 = self._calculate_md5(self.model_file)
        expected_md5 = metadata["original_md5"]
        
        if actual_md5 != expected_md5:
            print(f"Warning: MD5 mismatch. Expected {expected_md5}, got {actual_md5}")
            return False
        
        print("Model file verification successful!")
        return True
    
    def reassemble_model_file(self, force: bool = False) -> bool:
        """
        Reassemble the model file from chunks if needed.
        
        Args:
            force: If True, reassemble even if file already exists
            
        Returns:
            True if file is ready to use, False otherwise
        """
        # Check if file already exists and is valid
        if self.model_file.exists() and not force:
            metadata = self._load_metadata()
            if metadata and self._verify_reassembled_file(metadata):
                return True
            
            # If verification failed, delete and reassemble
            if metadata:
                print("Existing file failed verification, reassembling...")
                self.model_file.unlink()
        
        # Check if chunks directory exists
        if not self.chunks_dir.exists():
            print(f"Error: Chunks directory not found: {self.chunks_dir}")
            return False
        
        # Load metadata
        metadata = self._load_metadata()
        if not metadata:
            print(f"Error: Metadata file not found: {self.metadata_file}")
            return False
        
        print(f"Reassembling {metadata['original_filename']} from {metadata['total_chunks']} chunks...")
        
        # Reassemble file from chunks
        try:
            with open(self.model_file, "wb") as output_file:
                for chunk_info in metadata["chunks"]:
                    chunk_path = self.chunks_dir / chunk_info["filename"]
                    
                    if not chunk_path.exists():
                        print(f"Error: Chunk not found: {chunk_path}")
                        return False
                    
                    # Verify chunk integrity
                    chunk_md5 = self._calculate_md5(chunk_path)
                    if chunk_md5 != chunk_info["md5"]:
                        print(f"Error: Chunk integrity check failed for {chunk_info['filename']}")
                        return False
                    
                    # Append chunk to output file
                    with open(chunk_path, "rb") as chunk_file:
                        output_file.write(chunk_file.read())
                    
                    print(f"  Assembled: {chunk_info['filename']}")
            
            # Verify the reassembled file
            if not self._verify_reassembled_file(metadata):
                print("Error: Reassembled file verification failed")
                self.model_file.unlink()
                return False
            
            print(f"Successfully reassembled {metadata['original_filename']}")
            print(f"File location: {self.model_file}")
            return True
            
        except Exception as e:
            print(f"Error during reassembly: {e}")
            if self.model_file.exists():
                self.model_file.unlink()
            return False
    
    def get_model_file_path(self) -> Optional[Path]:
        """
        Get the path to the model file, reassembling if necessary.
        
        Returns:
            Path to the model file, or None if reassembly fails
        """
        with self._lock:
            if self.reassemble_model_file():
                return self.model_file
            return None
    
    def is_model_ready(self) -> bool:
        """Check if the model file is ready to use."""
        return self.model_file.exists()


# Module-level function for easy access
def get_model_file_path() -> Optional[Path]:
    """
    Get the path to the model file, automatically reassembling from chunks if needed.
    
    Returns:
        Path to the model file, or None if unavailable
        
    Example:
        >>> model_path = get_model_file_path()
        >>> if model_path:
        >>>     # Use the model file
        >>>     with open(model_path, 'rb') as f:
        >>>         model_data = f.read()
    """
    manager = ModelFileManager()
    return manager.get_model_file_path()


def ensure_model_ready() -> bool:
    """
    Ensure the model file is ready to use.
    
    Returns:
        True if model is ready, False otherwise
    """
    manager = ModelFileManager()
    return manager.reassemble_model_file()
```

**Key Features:**
- **Singleton pattern**: Ensures only one instance manages reassembly
- **Thread-safe**: Uses locks to prevent race conditions
- **Integrity verification**: MD5 checksums for both chunks and final file
- **Automatic cleanup**: Removes corrupted files and retries
- **Idempotent**: Safe to call multiple times

### Step 3: Integrate into Package Initialization

Update your `your_package/__init__.py`:

```python
"""
Your Package Name.

Package description.
"""
from .your_main_module import YourMainClass

# Automatically reassemble model file from chunks on package import
# This happens once when the package is first imported
from .model_reassembly import ensure_model_ready
import warnings

try:
    # Attempt to reassemble the model file if needed
    if not ensure_model_ready():
        warnings.warn(
            "Failed to reassemble model file from chunks. "
            "The package may not function properly. "
            "Please check the installation.",
            RuntimeWarning
        )
except Exception as e:
    warnings.warn(
        f"Error during model file reassembly: {e}. "
        "The package may not function properly.",
        RuntimeWarning
    )

__all__ = ['YourMainClass']
```

**Benefits:**
- Automatic: Users don't need to run any setup scripts
- Fail-safe: Provides warnings if reassembly fails
- Non-blocking: Doesn't raise exceptions on import

### Step 4: Configure Package Data Inclusion

#### pyproject.toml

Ensure chunks are included in the package:

```toml
[tool.setuptools.package-data]
your_package = [
    "model/chunks/*",      # Include all chunk files
    "*.ui",                # Include any other necessary files
    "*.ini",
    "*.json",
]
```

#### MANIFEST.in

For source distributions, create or update `MANIFEST.in`:

```plaintext
# Include package metadata and documentation
include README.md
include LICENSE
include pyproject.toml

# Include model chunk files and metadata
recursive-include your_package/model/chunks *

# Exclude the reassembled model file (generated at runtime)
global-exclude *.enc

# Exclude compiled Python files and caches
global-exclude __pycache__
global-exclude *.py[co]
```

### Step 5: Update .gitignore

Ensure the original large file is excluded but chunks are included:

```gitignore
# Exclude the large reassembled model file
your_package/model/*.enc
your_package/model/*.onnx
your_package/model/*.pth
# Add other large model formats as needed

# Keep chunks (explicitly NOT ignored)
# The chunks directory should be committed

# Standard Python excludes
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
```

## Usage Workflow

### For Package Maintainers

1. **Add or update the model file:**
   ```bash
   # Place your large model file in the package
   cp large_model.enc your_package/model/
   ```

2. **Split the model file:**
   ```bash
   python split_model_file.py
   ```

3. **Remove the original file:**
   ```bash
   rm your_package/model/large_model.enc
   ```

4. **Commit chunks to Git:**
   ```bash
   git add your_package/model/chunks/
   git commit -m "Add model chunks"
   git push
   ```

### For Package Users

**No action required!** The model is automatically reassembled on first import:

```python
import your_package  # Model reassembly happens automatically

# Use the package normally
from your_package.model_reassembly import get_model_file_path

model_path = get_model_file_path()
if model_path:
    # Load and use the model
    model = load_model(model_path)
```

## Metadata File Format

The `*.meta.json` file contains all information needed for reassembly:

```json
{
  "original_filename": "model.enc",
  "original_size": 111880262,
  "original_md5": "34d49c1f67314c2996ea1b5ffc160046",
  "chunk_size": 47185920,
  "total_chunks": 3,
  "chunks": [
    {
      "filename": "model.part000",
      "size": 47185920,
      "md5": "7915fb9daff66e487ca35d9d7c0002d1"
    },
    {
      "filename": "model.part001",
      "size": 47185920,
      "md5": "28f32691c5ee720b86e73cc567e4c6e5"
    },
    {
      "filename": "model.part002",
      "size": 17508422,
      "md5": "ada881fe8f9cf422907d935c99522674"
    }
  ]
}
```

## Advanced Usage

### Manual Reassembly

If you need to manually trigger reassembly:

```python
from your_package.model_reassembly import ModelFileManager

manager = ModelFileManager()

# Force reassembly even if file exists
success = manager.reassemble_model_file(force=True)

if success:
    print("Model reassembled successfully")
else:
    print("Reassembly failed")
```

### Check Model Status

```python
from your_package.model_reassembly import ModelFileManager

manager = ModelFileManager()

if manager.is_model_ready():
    print("Model is ready to use")
    print(f"Model path: {manager.model_file}")
else:
    print("Model needs reassembly")
```

### Integration with Model Loading

```python
from your_package.model_reassembly import get_model_file_path
import onnxruntime as ort

def load_inference_model():
    """Load the ONNX model for inference."""
    model_path = get_model_file_path()
    
    if model_path is None:
        raise RuntimeError("Model file could not be loaded")
    
    # Load the model
    session = ort.InferenceSession(str(model_path))
    return session
```

## Error Handling

### Common Issues and Solutions

#### 1. Chunks Not Found
**Error:** `Error: Chunks directory not found`

**Solution:**
- Verify chunks are included in package data (pyproject.toml, MANIFEST.in)
- Check that chunks directory exists after installation

#### 2. Checksum Mismatch
**Error:** `MD5 mismatch` or `Chunk integrity check failed`

**Solution:**
- Chunks may be corrupted during download/installation
- Re-download or reinstall the package
- For maintainers: Re-run the splitting script

#### 3. Permission Errors
**Error:** `Permission denied` when writing reassembled file

**Solution:**
- Ensure write permissions in package directory
- Consider reassembling to a user-writable location (temp directory)

### Robust Error Handling Example

```python
from your_package.model_reassembly import ensure_model_ready
import sys

try:
    if not ensure_model_ready():
        print("ERROR: Model file could not be prepared", file=sys.stderr)
        print("Please reinstall the package or contact support", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# Continue with normal execution
```

## Performance Considerations

### Reassembly Time
- **First import**: 1-5 seconds (depending on model size)
- **Subsequent imports**: Instant (file exists, verification only)
- **Verification**: 1-2 seconds for 100MB+ files

### Disk Space
- **During reassembly**: 2x model size (chunks + reassembled file)
- **After reassembly**: 2x model size (chunks remain for verification)
- **Optimization**: Could delete chunks after successful reassembly (not recommended for verification)

### Memory Usage
- Minimal: Chunks are processed sequentially, not loaded into memory

## Best Practices

### 1. Choose Appropriate Chunk Size
- **45 MB**: Safe for GitHub (50 MB limit with overhead)
- **48 MB**: Maximum for GitHub
- **95 MB**: For GitLab (100 MB limit)

### 2. Version Control
- Always commit chunks and metadata
- Never commit the original large file
- Use `.gitattributes` for LFS if chunks are still too large

### 3. CI/CD Integration
```yaml
# Example GitHub Actions test
- name: Test model reassembly
  run: |
    python -c "from your_package.model_reassembly import ensure_model_ready; assert ensure_model_ready()"
```

### 4. Documentation
- Add a note in README about first-import delay
- Document any environment variables or configuration options
- Provide troubleshooting steps

### 5. Alternative Storage Location
For user-writable locations, modify the manager:

```python
def __init__(self):
    """Initialize with user data directory."""
    if not hasattr(self, '_initialized'):
        # Use user data directory for reassembled file
        import platformdirs
        user_data = platformdirs.user_data_dir("your_package")
        Path(user_data).mkdir(parents=True, exist_ok=True)
        
        self.model_dir = Path(__file__).parent / "model"
        self.chunks_dir = self.model_dir / "chunks"
        self.model_file = Path(user_data) / "your_model.enc"  # User directory
        self.metadata_file = self.chunks_dir / "your_model.meta.json"
        self._initialized = True
```

## Testing

### Unit Tests

```python
import pytest
from pathlib import Path
from your_package.model_reassembly import ModelFileManager

def test_model_reassembly():
    """Test that model can be reassembled."""
    manager = ModelFileManager()
    assert manager.reassemble_model_file()
    assert manager.is_model_ready()
    assert manager.model_file.exists()

def test_model_integrity():
    """Test that reassembled model matches metadata."""
    manager = ModelFileManager()
    manager.reassemble_model_file()
    
    metadata = manager._load_metadata()
    assert metadata is not None
    assert manager._verify_reassembled_file(metadata)

def test_get_model_path():
    """Test the convenience function."""
    from your_package.model_reassembly import get_model_file_path
    
    model_path = get_model_file_path()
    assert model_path is not None
    assert model_path.exists()
```

### Integration Tests

```python
def test_model_loading():
    """Test that model can be loaded and used."""
    from your_package.model_reassembly import get_model_file_path
    import onnxruntime as ort
    
    model_path = get_model_file_path()
    assert model_path is not None
    
    # Attempt to load the model
    session = ort.InferenceSession(str(model_path))
    assert session is not None
```

## Troubleshooting

### Enable Debug Logging

Modify `model_reassembly.py` to add logging:

```python
import logging

logger = logging.getLogger(__name__)

# In reassemble_model_file():
logger.debug(f"Reassembling from {self.chunks_dir}")
logger.info(f"Processing chunk {chunk_info['filename']}")
```

### Manual Verification

```python
# Verify chunks manually
from your_package.model_reassembly import ModelFileManager

manager = ModelFileManager()
metadata = manager._load_metadata()

if metadata:
    print(f"Expected {metadata['total_chunks']} chunks")
    for chunk_info in metadata['chunks']:
        chunk_path = manager.chunks_dir / chunk_info['filename']
        exists = "✓" if chunk_path.exists() else "✗"
        print(f"{exists} {chunk_info['filename']}")
```

## Alternatives and Comparisons

### Git LFS (Large File Storage)
**Pros:**
- Native Git integration
- Transparent to users

**Cons:**
- Requires Git LFS setup
- Additional infrastructure
- Not available on all Git hosts

### External Storage (S3, CDN)
**Pros:**
- Unlimited size
- Fast downloads

**Cons:**
- Requires network access
- Additional complexity
- Authentication needed

### Model Splitting (This Solution)
**Pros:**
- No external dependencies
- Works with standard Git
- Automatic and transparent
- Full integrity verification

**Cons:**
- Slight complexity in codebase
- First-import delay
- 2x disk usage

## Summary

This model split and reassembly feature provides:

✅ **Transparent**: Users don't need to do anything special  
✅ **Robust**: Full integrity verification with MD5 checksums  
✅ **Thread-safe**: Singleton pattern with locking  
✅ **Maintainable**: Simple scripts for splitting and reassembly  
✅ **Git-friendly**: Chunks fit within repository limits  
✅ **Automatic**: Reassembly on first package import  

### Quick Reference

**Maintainer Workflow:**
```bash
python split_model_file.py          # Split the model
rm your_package/model/*.enc          # Remove original
git add your_package/model/chunks/   # Commit chunks
```

**User Experience:**
```python
import your_package  # Everything happens automatically
```

**Manual Usage:**
```python
from your_package.model_reassembly import get_model_file_path
model_path = get_model_file_path()
```

## License

This implementation pattern is free to use and adapt for your projects. Consider crediting the original implementation if you find it useful.

---

**Last Updated:** January 2026  
**Version:** 1.0.0
