import pytest
from src.query_system import ask_question
from src.utils import chunk_documents
from langchain_core.documents import Document

def test_full_rag_flow_mock(mocker):
    # 1. Chunking test
    pages = [Document(page_content="CHP parti tüzüğü madde 1: Bağımsızlık.")]
    chunks = chunk_documents(pages, chunk_size=50, chunk_overlap=0)
    assert len(chunks) == 1
    
    # 2. Mocking the search and LLM
    mock_vs = mocker.MagicMock()
    # When search is called, return the chunk we just created
    mocker.patch("src.query_system.utils.search_similar_docs", return_value=(chunks[0].page_content, [1.0], chunks))
    
    mock_llm = mocker.MagicMock()
    mock_llm.invoke.return_value = "CHP Bağımsızlığı savunur."
    
    # 3. Ask question
    response, docs = ask_question(
        question="Madde 1 nedir?",
        vectorstore=mock_vs,
        llm_handler=mock_llm,
        party="CHP",
        llm_type="ollama"
    )
    
    assert "Bağımsızlık" in response or "Bağımsızlığ" in response
    assert response == "CHP Bağımsızlığı savunur."
