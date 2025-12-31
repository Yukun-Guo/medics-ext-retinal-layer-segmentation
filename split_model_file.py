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
    model_file = Path(__file__).parent / "medics_ext_retinal_layer_segmentation" / "model" / "layersegmodel.enc"
    
    if model_file.exists():
        split_file(model_file)
    else:
        print(f"Error: Model file not found at {model_file}")
        print("Please update the path in this script if the file is located elsewhere.")
