import pytest
from src.query_system import ask_question
from langchain_chroma import Chroma
from langchain_core.documents import Document

def test_ask_question_ollama(mocker):
    # Mock vectorstore
    mock_vs = mocker.MagicMock(spec=Chroma)
    mocker.patch("src.query_system.utils.search_similar_docs", return_value=("Test context", [0.9], [Document(page_content="test")]))
    
    # Mock LLM handler
    mock_llm = mocker.MagicMock()
    mock_llm.invoke.return_value = "Test response"
    
    response, docs = ask_question(
        question="Test question",
        vectorstore=mock_vs,
        llm_handler=mock_llm,
        party="CHP",
        llm_type="ollama",
        stream=False
    )
    
    assert response == "Test response"
    assert len(docs) == 1
    mock_llm.invoke.assert_called_once()

def test_ask_question_low_similarity(mocker):
    mock_vs = mocker.MagicMock(spec=Chroma)
    mocker.patch("src.query_system.utils.search_similar_docs", return_value=("", [0.1], [])) # Below threshold
    
    response, docs = ask_question(
        question="Test question",
        vectorstore=mock_vs,
        llm_handler=None,
        party="CHP",
        llm_type="ollama",
        stream=False
    )
    
    assert "yeterli bilgi bulamadım" in response

def test_ask_question_streaming_ollama(mocker):
    mock_vs = mocker.MagicMock(spec=Chroma)
    mocker.patch("src.query_system.utils.search_similar_docs", return_value=("Test context", [0.9], [Document(page_content="test")]))
    
    mock_llm = mocker.MagicMock()
    mock_llm.stream.return_value = iter(["Chunk 1", "Chunk 2"])
    
    stream_gen, docs = ask_question(
        question="Test question",
        vectorstore=mock_vs,
        llm_handler=mock_llm,
        party="CHP",
        llm_type="ollama",
        stream=True
    )
    
    chunks = list(stream_gen)
    assert chunks == ["Chunk 1", "Chunk 2"]
    assert len(docs) == 1
