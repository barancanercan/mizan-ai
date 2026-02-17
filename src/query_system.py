"""
Turkish Government Intelligence Hub - Query System
Hybrid LLM (Ollama + HuggingFace) with İYİ Normalization
"""

import argparse
from typing import Any, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
import utils
from core.parties import normalize_party_name, normalize_parties_list
from core.llm_setup import (
    setup_ollama_chain,
    setup_huggingface_config,
    query_with_huggingface,
    query_with_gemini,
    create_llm_handler,
)
from core.streaming import handle_stream_response


# ============================================
# QUERY FUNCTION
# ============================================


def ask_question(
    question: str,
    vectorstore: Chroma,
    llm_handler: Any,
    party: str,
    llm_type: str,
    stream: bool = False,
) -> Tuple[Any, List[Document]]:
    """
    Vektör veritabanında arama yapar ve LLM kullanarak soruyu yanıtlar (Stream destekli).
    """

    normalized_party = normalize_party_name(party)

    context, scores, docs = utils.search_similar_docs(
        vectorstore, question, filter_metadata={"party": normalized_party}
    )

    if not scores or scores[0] < config.SIMILARITY_THRESHOLD:
        msg = f"Bu konuda {party} parti tüzüğünde yeterli bilgi bulamadım. Lütfen daha açık bir soru sorun."
        if stream:

            def gen():
                yield msg

            return gen(), docs
        return msg, docs

    system_prompt = config.SYSTEM_PROMPTS.get(
        normalized_party, config.SYSTEM_PROMPTS.get(party)
    )
    full_prompt = system_prompt.format(context=context, question=question)

    try:
        if llm_type == "ollama":
            if stream:
                return llm_handler.stream({"context": context, "question": question}), docs
            return llm_handler.invoke({"context": context, "question": question}), docs
        elif llm_type == "gemini":
            return query_with_gemini(llm_handler, full_prompt, stream=stream), docs
        else:
            return query_with_huggingface(full_prompt, llm_handler, stream=stream), docs

    except Exception as e:
        utils.logger.error(f"Cevap üretme hatası: {str(e)}")
        raise


def stream_response(response_gen: Any, llm_type: str) -> str:
    """Stream response'ı işle ve string olarak döndür."""
    result = ""
    for chunk in response_gen:
        result += handle_stream_response(chunk, llm_type)
    return result


# ============================================
# SINGLE PARTY MODE
# ============================================


def single_party_mode(party: str):
    """Tek parti modu"""
    party = normalize_party_name(party)

    utils.print_header(f"🤖 {party} Soru-Cevap Sistemi")
    utils.print_party_info(party)

    db_path = config.UNIFIED_VECTOR_DB
    if not db_path or not db_path.exists():
        utils.logger.error("❌ Unified vector database bulunamadı!")
        utils.logger.error("💡 Çalıştır: python prepare_data.py")
        return

    embeddings = utils.load_embeddings()
    vectorstore = utils.load_vectorstore(db_path, embeddings)

    llm_handler, llm_type = create_llm_handler(party)

    if llm_type == "none":
        utils.logger.error("❌ Hiç LLM kullanılamıyor!")
        return

    utils.logger.info("✅ Sistem hazır!")
    utils.print_header("💬 Soru-Cevap Başlıyor")
    print("Çıkmak için: q\n")

    while True:
        question = input(f"{config.PARTY_INFO[party]['hex_color']} Sorunuz: ").strip()

        if question.lower() in ["q", "quit", "exit"]:
            print("👋 Hoşça kalın!")
            break
        if not question:
            continue

        try:
            print(f"\n{'='*60}\n🤖 Cevap: ", end="", flush=True)
            response_gen, source_docs = ask_question(
                question, vectorstore, llm_handler, party, llm_type, stream=True
            )

            for chunk in response_gen:
                content = handle_stream_response(chunk, llm_type)
                print(content, end="", flush=True)

            if source_docs:
                print(f"\n\n📚 Kaynaklar: {[doc.metadata.get('page', '?') for doc in source_docs]}")

            print(f"\n{'='*60}\n")
        except Exception as e:
            print(f"⚠️ Hata: {str(e)}\n")


# ============================================
# MULTI PARTY MODE
# ============================================


def multi_party_mode():
    """Çoklu parti modu"""
    utils.print_header("🤖 Çoklu Parti Q&A")

    prepared_parties = normalize_parties_list(utils.get_prepared_parties())

    if not prepared_parties:
        utils.logger.error("❌ Hiç hazır DB yok! Çalıştır: python prepare_data.py")
        return

    utils.logger.info(f"✅ Hazır: {', '.join(prepared_parties)}")

    embeddings = utils.load_embeddings()
    db_path = config.UNIFIED_VECTOR_DB
    if not db_path.exists():
        utils.logger.error("❌ Unified DB yok! Çalıştır: python prepare_data.py")
        return

    vectorstore = utils.load_vectorstore(db_path, embeddings)

    llm_handler, llm_type = create_llm_handler(prepared_parties[0])

    if llm_type == "none":
        utils.logger.error("❌ LLM yok!")
        return

    utils.logger.info("✅ Sistem hazır!")
    utils.print_header("💬 Soru-Cevap Başlıyor")
    print(f"Partiler: {', '.join(prepared_parties)}")
    print("Komutlar: /chp, /akp, /mhp, /dem, /iyi | q=çıkış\n")

    current_party = "CHP" if "CHP" in prepared_parties else prepared_parties[0]

    while True:
        question = input(f"[{current_party}] Sorunuz: ").strip()

        if question.lower() in ["q", "quit", "exit"]:
            print("👋 Hoşça kalın!")
            break

        if question.startswith("/"):
            party_cmd = question[1:].upper()
            party_cmd = normalize_party_name(party_cmd)

            if party_cmd in prepared_parties:
                current_party = party_cmd
                print(f"✅ {current_party}\n")
                utils.print_party_info(current_party)
            else:
                print(f"❌ {party_cmd} yok\n")
            continue

        if not question:
            continue

        try:
            print(f"\n{'='*60}\n🤖 Cevap: ", end="", flush=True)
            response_gen, source_docs = ask_question(
                question,
                vectorstore,
                llm_handler,
                current_party,
                llm_type,
                stream=True,
            )

            for chunk in response_gen:
                content = handle_stream_response(chunk, llm_type)
                print(content, end="", flush=True)

            if source_docs:
                print(f"\n\n📚 Kaynaklar: {[doc.metadata.get('page', '?') for doc in source_docs]}")

            print(f"\n{'='*60}\n")
        except Exception as e:
            print(f"⚠️ Hata: {str(e)}\n")


# ============================================
# MAIN
# ============================================


def main():
    parser = argparse.ArgumentParser(description="Turkish Government Intelligence Hub")
    parser.add_argument("--party", type=str, help="Tek parti modu")
    args = parser.parse_args()

    if args.party:
        single_party_mode(args.party)
    else:
        multi_party_mode()


if __name__ == "__main__":
    main()
