from typing import Any


def handle_stream_response(chunk: Any, llm_type: str) -> str:
    """
    Handle stream response from different LLM types (Ollama, HuggingFace, Gemini, or string).
    
    Args:
        chunk: The response chunk from LLM
        llm_type: "ollama", "huggingface", "gemini", or "none"
    
    Returns:
        str: The processed content string
    """
    if isinstance(chunk, str):
        return chunk

    if llm_type == "ollama":
        return str(chunk)

    if llm_type == "gemini":
        try:
            return chunk.content
        except AttributeError:
            return str(chunk)

    if llm_type == "huggingface":
        try:
            return str(chunk.choices[0].delta.content)
        except (AttributeError, IndexError):
            return str(chunk)

    return str(chunk)
