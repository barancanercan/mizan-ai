"""
mizan-ai - LLM Setup Utilities
Gemini (Birincil) + Ollama (Yedek) Configuration
"""

import os
import logging
from typing import Any, Optional, Dict, Tuple

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import config
import utils
from core.parties import normalize_party_name

logger = logging.getLogger(__name__)


def setup_gemini_chain(party: str) -> Tuple[Any, str]:
    """
    Gemini LLM için chain oluşturur (Birincil).

    Args:
        party: Hedef parti kodu.

    Returns:
        Tuple[Any, str]: (handler, llm_type)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠️ GEMINI_API_KEY not set")
        return (None, "none")
    
    normalized_party = normalize_party_name(party)
    
    prompt_template_str = config.SYSTEM_PROMPTS.get(normalized_party)
    if not prompt_template_str:
        prompt_template_str = config.SYSTEM_PROMPTS.get(party)
    if not prompt_template_str:
        prompt_template_str = "Soruyu yanıtla: {question}"

    prompt_template = PromptTemplate.from_template(prompt_template_str)
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
        )
        llm.invoke("test")
        
        chain = prompt_template | llm | StrOutputParser()
        logger.info("✅ Gemini bağlandı (Birincil LLM)")
        return (chain, "gemini")
    except Exception as e:
        logger.warning(f"⚠️ Gemini bağlantısı başarısız: {e}")
        return (None, "none")


def setup_ollama_chain(party: str) -> Tuple[Any, str]:
    """
    Ollama LLM için chain oluşturur (Yedek).

    Args:
        party: Hedef parti kodu.

    Returns:
        Tuple[Any, str]: (handler, llm_type)
    """
    normalized_party = normalize_party_name(party)
    
    prompt_template_str = config.SYSTEM_PROMPTS.get(normalized_party)
    if not prompt_template_str:
        prompt_template_str = config.SYSTEM_PROMPTS.get(party)
    if not prompt_template_str:
        prompt_template_str = "Soruyu yanıtla: {question}"

    prompt_template = PromptTemplate.from_template(prompt_template_str)
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", config.LLM_MODEL)
    
    llm = OllamaLLM(
        model=model, 
        temperature=config.LLM_TEMPERATURE,
        base_url=ollama_base_url,
        num_predict=config.LLM_MAX_TOKENS,
    )
    
    try:
        llm.invoke("test", num_predict=5)
    except Exception as e:
        logger.warning(f"⚠️ Ollama test bağlantısı başarısız: {e}")
    
    chain = prompt_template | llm | StrOutputParser()
    logger.info("✅ Ollama bağlandı (Yedek LLM)")
    return (chain, "ollama")


def create_llm_handler(party: str) -> Tuple[Any, str]:
    """
    LLM handler oluşturur.
    
    Önce Gemini dener, başarısız olursa Ollama'ya geçer.

    Args:
        party: Hedef parti kodu.

    Returns:
        Tuple[Any, str]: (handler, llm_type)
    """
    # Önce Gemini dene
    try:
        handler, llm_type = setup_gemini_chain(party)
        if handler is not None:
            return (handler, llm_type)
    except Exception as e:
        logger.warning(f"⚠️ Gemini başarısız, Ollama deneniyor: {e}")
    
    # Yedek: Ollama
    try:
        handler, llm_type = setup_ollama_chain(party)
        if handler is not None:
            return (handler, llm_type)
    except Exception as e:
        logger.error(f"❌ Ollama da başarısız: {e}")
    
    logger.error("❌ Hiç LLM kullanılamıyor!")
    return (None, "none")


def get_llm_display_name(llm_type: str) -> str:
    """
    LLM tipi için kullanıcı dostu görünen ad döner.

    Args:
        llm_type: LLM tipi

    Returns:
        str: Görünen ad
    """
    display_names = {
        "gemini": "Gemini 1.5 Flash (Birincil)",
        "ollama": "Ollama (Yedek)",
        "none": "LLM Yok",
    }
    return display_names.get(llm_type, "Bilinmiyor")


def check_llm_status() -> Dict[str, Any]:
    """
    LLM durumlarını kontrol eder.
    
    Returns:
        Dict: Durum bilgileri
    """
    status = {
        "gemini": {"available": False, "error": None},
        "ollama": {"available": False, "error": None},
    }
    
    # Gemini kontrolü
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
            llm.invoke("test", max_tokens=5)
            status["gemini"]["available"] = True
    except Exception as e:
        status["gemini"]["error"] = str(e)
    
    # Ollama kontrolü
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", config.LLM_MODEL)
        llm = OllamaLLM(model=model, base_url=ollama_base_url)
        llm.invoke("test", num_predict=5)
        status["ollama"]["available"] = True
    except Exception as e:
        status["ollama"]["error"] = str(e)
    
    return status
