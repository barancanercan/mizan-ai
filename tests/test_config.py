import pytest
from pathlib import Path
from src.config import discover_parties

def test_discover_parties_with_iyi(mocker):
    # Mock Path objects
    mock_pdf = mocker.MagicMock(spec=Path)
    mock_pdf.stem = "iyi"
    mock_pdf.suffix = ".pdf"
    
    mock_data_dir = mocker.patch("src.config.DATA_DIR")
    mock_data_dir.exists.return_value = True
    mock_data_dir.glob.return_value = [mock_pdf]
    
    parties = discover_parties()
    
    assert "İYİ" in parties
    assert "IYI" not in parties

def test_discover_parties_empty(mocker):
    mock_data_dir = mocker.patch("src.config.DATA_DIR")
    mock_data_dir.exists.return_value = True
    mock_data_dir.glob.return_value = []
    
    parties = discover_parties()
    assert parties == {}

def test_discover_parties_no_dir(mocker):
    mock_data_dir = mocker.patch("src.config.DATA_DIR")
    mock_data_dir.exists.return_value = False
    
    parties = discover_parties()
    assert parties == {}
