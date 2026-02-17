import pytest
import hashlib
from pathlib import Path
from src.prepare_data import get_file_hash

def test_get_file_hash(tmp_path):
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    content = b"Hello World"
    test_file.write_bytes(content)
    
    expected_hash = hashlib.md5(content).hexdigest()
    
    assert get_file_hash(test_file) == expected_hash

def test_get_file_hash_large(tmp_path):
    # Create a larger temporary file (> 4096 bytes)
    test_file = tmp_path / "large_test.txt"
    content = b"A" * 5000
    test_file.write_bytes(content)
    
    expected_hash = hashlib.md5(content).hexdigest()
    
    assert get_file_hash(test_file) == expected_hash
