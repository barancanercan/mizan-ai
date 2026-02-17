"""
Turkish Government Intelligence Hub - LLM Setup Utilities
Ollama, HuggingFace ve Gemini LLM yapılandırma fonksiyonları
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


def setup_ollama_chain(party: str) -> Any:
    """
    Ollama LLM için LangChain chain'i oluşturur.

    Args:
        party: Hedef parti kodu.

    Returns:
        Any: Hazırlanmış LangChain chain'i.
    """
    try:
        normalized_party = normalize_party_name(party)

        prompt_template = PromptTemplate.from_template(
            config.SYSTEM_PROMPTS.get(
                normalized_party, config.SYSTEM_PROMPTS.get(party)
            )
        )
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = OllamaLLM(
            model=config.LLM_MODEL, 
            temperature=config.LLM_TEMPERATURE,
            base_url=ollama_base_url
        )
        llm.invoke("test", num_predict=5)
        chain = prompt_template | llm | StrOutputParser()
        utils.logger.info("✅ Ollama bağlantısı başarılı")
        return chain
    except Exception as e:
        utils.logger.error(f"❌ Ollama hatası: {str(e)}")
        raise


def setup_huggingface_config() -> Optional[Dict[str, str]]:
    """
    HuggingFace API bağlantısı için gerekli konfigürasyonu hazırlar.

    Returns:
        Optional[Dict[str, str]]: API token ve model bilgilerini içeren sözlük veya None.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        utils.logger.warning("⚠️ HF_TOKEN not set")
        return None

    utils.logger.info("✅ HuggingFace config hazır")
    return {"token": hf_token, "model": "Qwen/Qwen2.5-7B-Instruct"}


def query_with_huggingface(prompt: str, hf_config: dict, stream: bool = False) -> Any:
    """
    HuggingFace API'ye sorgula (Stream destekli).

    Args:
        prompt: Sorgu metni.
        hf_config: HuggingFace konfigürasyonu.
        stream: Stream modu etkinleştirilsin mi?

    Returns:
        Any: LLM cevabı veya stream iteratörü.
    """
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=hf_config["token"])

        if stream:
            return client.chat_completion(
                model=hf_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )

        response = client.chat_completion(
            model=hf_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )

        if not response or not response.choices:
            return "❌ HuggingFace boş cevap döndürdü"

        return response.choices[0].message.content
    except Exception as e:
        utils.logger.error(f"HuggingFace hatası: {str(e)}")
        raise


def setup_gemini() -> Optional[Any]:
    """
    Gemini API bağlantısı için gerekli konfigürasyonu hazırlar.

    Returns:
        Optional[Any]: Yapılandırılmış Gemini LLM veya None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        utils.logger.warning("⚠️ GEMINI_API_KEY not set")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        llm.invoke("test")
        utils.logger.info("✅ Gemini bağlantısı başarılı")
        return llm
    except Exception as e:
        utils.logger.error(f"❌ Gemini hatası: {str(e)}")
        return None


def query_with_gemini(llm: Any, prompt: str, stream: bool = False) -> Any:
    """
    Gemini API'ye sorgula.

    Args:
        llm: ChatGoogleGenerativeAI instance
        prompt: Sorgu metni.
        stream: Stream modu

    Returns:
        Any: LLM cevabı
    """
    try:
        if stream:
            return llm.stream(prompt)
        response = llm.invoke(prompt)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        utils.logger.error(f"Gemini sorgu hatası: {str(e)}")
        return None
        return None


def create_llm_handler(party: str) -> Tuple[Any, str]:
    """
    LLM handler oluşturur (fallback destekli).

    Önce Ollama dener, başarısız olursa Gemini'ye geçer.
    O da başarısız olursa HuggingFace'e geçer.
    Hepsi başarısız olursa None döner.

    Args:
        party: Hedef parti kodu.

    Returns:
        Tuple[Any, str]: (handler/config, llm_type)
    """
    try:
        handler = setup_ollama_chain(party)
        return (handler, "ollama")
    except Exception as ollama_error:
        utils.logger.warning(f"⚠️ Ollama başarısız, Gemini deneniyor: {ollama_error}")

    try:
        gemini_llm = setup_gemini()
        if gemini_llm:
            return (gemini_llm, "gemini")
    except Exception as gemini_error:
        utils.logger.warning(f"⚠️ Gemini başarısız, HuggingFace deneniyor: {gemini_error}")

    try:
        hf_config = setup_huggingface_config()
        if hf_config:
            return (hf_config, "huggingface")
    except Exception as hf_error:
        utils.logger.error(f"❌ HuggingFace hatası: {hf_error}")

    utils.logger.error("❌ Hiç LLM kullanılamıyor!")
    return (None, "none")


def get_llm_display_name(llm_type: str) -> str:
    """
    LLM tipi için kullanıcı dostu görünen ad döner.

    Args:
        llm_type: LLM tipi ("ollama", "huggingface", "gemini", "none")

    Returns:
        str: Görünen ad
    """
    display_names = {
        "ollama": "Lokal (Ollama)",
        "huggingface": "Bulut (HF)",
        "gemini": "Gemini",
        "none": "LLM Yok",
    }
    return display_names.get(llm_type, "Bilinmiyor")
