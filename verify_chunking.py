#!/usr/bin/env python3
"""
Quick verification script to check the package is correctly configured
for chunked model file distribution.
"""
from pathlib import Path
import json


def main():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "medics_ext_retinal_layer_segmentation" / "model"
    chunks_dir = model_dir / "chunks"
    
    print("=" * 70)
    print("Package Configuration Verification")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Chunks directory exists
    checks_total += 1
    if chunks_dir.exists():
        print("✓ Chunks directory exists")
        checks_passed += 1
    else:
        print("✗ Chunks directory NOT found")
    
    # Check 2: Metadata file exists
    checks_total += 1
    metadata_file = chunks_dir / "layersegmodel.meta.json"
    if metadata_file.exists():
        print("✓ Metadata file exists")
        checks_passed += 1
        
        # Read and display metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"  - Original file: {metadata['original_filename']}")
        print(f"  - Original size: {metadata['original_size'] / 1024 / 1024:.2f} MB")
        print(f"  - Total chunks: {metadata['total_chunks']}")
    else:
        print("✗ Metadata file NOT found")
    
    # Check 3: All chunk files exist
    checks_total += 1
    chunk_files = list(chunks_dir.glob("layersegmodel.part*"))
    expected_chunks = 3
    if len(chunk_files) == expected_chunks:
        print(f"✓ All {expected_chunks} chunk files present")
        checks_passed += 1
        total_size = sum(f.stat().st_size for f in chunk_files)
        print(f"  - Total chunks size: {total_size / 1024 / 1024:.2f} MB")
        
        # Check chunk sizes
        all_under_50mb = all(f.stat().st_size < 50 * 1024 * 1024 for f in chunk_files)
        if all_under_50mb:
            print("  ✓ All chunks are under 50 MB")
        else:
            print("  ✗ Some chunks exceed 50 MB")
    else:
        print(f"✗ Expected {expected_chunks} chunks, found {len(chunk_files)}")
    
    # Check 4: Original file is NOT in git (should be in .gitignore)
    checks_total += 1
    gitignore = base_dir / ".gitignore"
    if gitignore.exists():
        with open(gitignore) as f:
            gitignore_content = f.read()
        if "layersegmodel.enc" in gitignore_content:
            print("✓ Original .enc file is in .gitignore")
            checks_passed += 1
        else:
            print("✗ Original .enc file NOT in .gitignore")
    else:
        print("✗ .gitignore file not found")
    
    # Check 5: package-data configured correctly
    checks_total += 1
    pyproject = base_dir / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject) as f:
            pyproject_content = f.read()
        if "model/chunks/*" in pyproject_content:
            print("✓ pyproject.toml includes model/chunks/*")
            checks_passed += 1
        else:
            print("✗ pyproject.toml does NOT include model/chunks/*")
    else:
        print("✗ pyproject.toml not found")
    
    # Check 6: Reassembly module exists
    checks_total += 1
    reassembly_module = base_dir / "medics_ext_retinal_layer_segmentation" / "model_reassembly.py"
    if reassembly_module.exists():
        print("✓ Reassembly module exists")
        checks_passed += 1
    else:
        print("✗ Reassembly module NOT found")
    
    # Check 7: Test script exists
    checks_total += 1
    test_script = base_dir / "test_reassembly.py"
    if test_script.exists():
        print("✓ Test script exists")
        checks_passed += 1
    else:
        print("✗ Test script NOT found")
    
    print("=" * 70)
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)
    
    if checks_passed == checks_total:
        print("\n✓ Package is correctly configured for chunked distribution!")
        print("\nNext steps:")
        print("1. Test reassembly: python test_reassembly.py")
        print("2. Commit changes: git add . && git commit -m 'Implement model chunking'")
        print("3. The package is ready for distribution via pip!")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the configuration.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
