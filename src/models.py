from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class Party(BaseModel):
    code: str = Field(..., description="Party code (e.g., CHP, AKP)")
    name: str = Field(..., description="Full party name")
    short: str = Field(..., description="Short display name")
    website: str = Field(..., description="Official website URL")
    hex_color: str = Field(..., description="Primary color in hex")
    accent_color: str = Field(..., description="Accent color in hex")
    founded: int = Field(..., description="Foundation year")
    ideology: str = Field(..., description="Party ideology")
    description: str = Field(..., description="Brief description")
    logo_path: Optional[Path] = Field(None, description="Path to logo file")


class Source(BaseModel):
    page: int = Field(..., description="PDF page number")
    content: str = Field(..., description="Source text content")
    party: str = Field(..., description="Party code")
    score: Optional[float] = Field(None, description="Similarity score")


class QueryResult(BaseModel):
    answer: str = Field(..., description="LLM generated answer")
    sources: list[Source] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(..., description="Confidence score (0-1)")
    llm_type: str = Field(..., description="LLM used (ollama/huggingface)")


class AppConfig(BaseModel):
    embedding_model: str = "nezahatkorkmaz/turkce-embedding-bge-m3"
    llm_model: str = "qwen2.5:7b"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
