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
            self.model_file = self.model_dir / "layersegmodel.enc"
            self.metadata_file = self.chunks_dir / "layersegmodel.meta.json"
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
