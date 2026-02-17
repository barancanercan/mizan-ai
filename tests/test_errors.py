import pytest
from src.utils import load_pdf, load_vectorstore
from src.query_system import ask_question
from pathlib import Path

def test_load_pdf_missing():
    with pytest.raises(Exception):
        load_pdf(Path("data/non_existent.pdf"))

def test_load_vectorstore_missing():
    with pytest.raises(FileNotFoundError):
        load_vectorstore(Path("non_existent_db"), None)

def test_ask_question_llm_error(mocker):
    mock_vs = mocker.MagicMock()
    mocker.patch("src.query_system.utils.search_similar_docs", return_value=("Context", [1.0], []))
    
    mock_llm = mocker.MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM Down")
    
    with pytest.raises(Exception) as excinfo:
        ask_question("?", mock_vs, mock_llm, "CHP", "ollama")
    assert "LLM Down" in str(excinfo.value)
