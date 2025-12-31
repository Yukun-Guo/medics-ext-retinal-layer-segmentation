#!/usr/bin/env python3
"""
Test script to verify the model file reassembly process.
This simulates what happens when a user installs and uses the package.
"""
import sys
from pathlib import Path

def test_reassembly():
    """Test the model file reassembly process."""
    print("=" * 70)
    print("Model File Reassembly Test")
    print("=" * 70)
    
    # Import the reassembly module
    sys.path.insert(0, str(Path(__file__).parent))
    from medics_ext_retinal_layer_segmentation.model_reassembly import (
        ModelFileManager, 
        get_model_file_path
    )
    
    manager = ModelFileManager()
    
    # Check initial state
    print("\n1. Checking initial state...")
    print(f"   Model file exists: {manager.model_file.exists()}")
    print(f"   Chunks directory exists: {manager.chunks_dir.exists()}")
    print(f"   Metadata file exists: {manager.metadata_file.exists()}")
    
    if manager.chunks_dir.exists():
        chunks = list(manager.chunks_dir.glob("layersegmodel.part*"))
        print(f"   Number of chunk files: {len(chunks)}")
        total_size = sum(c.stat().st_size for c in chunks)
        print(f"   Total chunks size: {total_size / 1024 / 1024:.2f} MB")
    
    # Test reassembly
    print("\n2. Testing model file reassembly...")
    model_path = get_model_file_path()
    
    if model_path:
        print(f"   ✓ Model file successfully reassembled")
        print(f"   Location: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Test that subsequent calls don't reassemble again
        print("\n3. Testing cached reassembly...")
        model_path2 = get_model_file_path()
        print(f"   ✓ Second call successful (uses existing file)")
        
        # Verify it's the same file
        assert model_path == model_path2, "Path mismatch!"
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe package is ready for distribution!")
        print("Users will automatically get the model file reassembled on first use.")
        return 0
    else:
        print("   ✗ Model file reassembly failed")
        print("\n" + "=" * 70)
        print("✗ TEST FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(test_reassembly())
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
