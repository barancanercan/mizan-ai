import pytest
from langchain_core.documents import Document
from src.utils import chunk_documents, validate_pdf_exists
from pathlib import Path

def test_chunk_documents():
    pages = [
        Document(page_content="Bu bir test dökümanıdır. " * 100),
        Document(page_content="İkinci sayfa içeriği burada. " * 50)
    ]
    
    chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)
    
    assert len(chunks) > 2
    assert all(isinstance(c, Document) for c in chunks)
    assert all(len(c.page_content) <= 120 for c in chunks) # allowance for overlap and rounding

def test_validate_pdf_exists(mocker):
    mocker.patch("src.config.PARTY_PDFS", {"CHP": Path("data/chp.pdf")})
    
    # PDF exists
    mock_exists = mocker.patch.object(Path, "exists")
    mock_exists.return_value = True
    assert validate_pdf_exists("CHP") is True
    
    # PDF missing
    mock_exists.return_value = False
    assert validate_pdf_exists("CHP") is False
    
    # Party missing
    assert validate_pdf_exists("MAYMUN_PARTISI") is False
