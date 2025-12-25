"""
Turkish Government Intelligence Hub - Query System
Ana soru-cevap programı - Hazır vector DB'leri kullanır

Usage:
    python query_system.py              # Tüm partilerle çalış
    python query_system.py --party CHP  # Sadece CHP ile çalış
"""

import argparse
import sys
from typing import Dict

from langchain_ollama import OllamaLLM  # ✅ Yeni paket (deprecation fix)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma  # ✅ Yeni paket (deprecation fix)

import config
import utils

# ============================================
# LLM SETUP
# ============================================

def setup_llm_chain(party: str):
    """
    LLM ve prompt chain'i hazırla

    Args:
        party: Parti kısa adı

    Returns:
        Chain: LangChain chain
    """
    utils.logger.info(f"LLM chain hazırlanıyor ({party})...")

    try:
        # Prompt template
        prompt_template = PromptTemplate.from_template(
            config.SYSTEM_PROMPTS[party]
        )

        # LLM
        llm = OllamaLLM(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )

        # Test connection
        utils.logger.info("Ollama bağlantısı test ediliyor...")
        llm.invoke("test", num_predict=5)
        utils.logger.info("✅ Ollama bağlantısı başarılı")

        # Chain
        chain = prompt_template | llm | StrOutputParser()

        return chain

    except ConnectionError:
        utils.logger.error("❌ Ollama server'a bağlanılamadı!")
        utils.logger.error("💡 Çözüm: Yeni terminal'de 'ollama serve' çalıştırın")
        raise
    except Exception as e:
        utils.logger.error(f"❌ LLM setup hatası: {str(e)}")
        raise

# ============================================
# QUERY FUNCTION
# ============================================

def ask_question(
    question: str,
    vectorstore: Chroma,
    chain,
    party: str
) -> str:
    """
    Soru sor ve cevap al

    Args:
        question: Kullanıcı sorusu
        vectorstore: Vector database
        chain: LLM chain
        party: Parti adı

    Returns:
        str: LLM cevabı
    """
    # Benzer dökümanları bul
    context, scores = utils.search_similar_docs(vectorstore, question)

    # Skor kontrolü
    if scores[0] < config.SIMILARITY_THRESHOLD:
        utils.logger.warning(f"⚠️ Düşük benzerlik skoru: {scores[0]:.3f}")
        return f"Bu konuda {party} parti tüzüğünde yeterli bilgi bulamadım. Sorunuzu daha açık sorabilir misiniz?"

    # LLM'e gönder
    utils.logger.info("LLM cevap üretiyor...")
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response

# ============================================
# SINGLE PARTY MODE
# ============================================

def single_party_mode(party: str):
    """
    Tek parti modu - sadece bir parti ile çalış

    Args:
        party: Parti kısa adı
    """
    utils.print_header(f"🤖 {party} Soru-Cevap Sistemi")
    utils.print_party_info(party)

    # Vector DB kontrolü
    db_path = config.PARTY_VECTOR_DBS[party]
    if not db_path.exists():
        utils.logger.error(f"❌ {party} için vector database bulunamadı!")
        utils.logger.error("💡 Önce veri hazırlama yapın: python prepare_data.py")
        return

    # Embedding modeli
    utils.logger.info("Embedding modeli yükleniyor...")
    embeddings = utils.load_embeddings()

    # Vector DB yükle
    vectorstore = utils.load_vectorstore(db_path, embeddings)

    # LLM chain hazırla
    chain = setup_llm_chain(party)

    utils.logger.info("✅ Sistem hazır!")

    # Soru-cevap döngüsü
    utils.print_header("💬 Soru-Cevap Başlıyor")
    print("Çıkmak için 'q', 'quit' veya 'exit' yazın\n")

    while True:
        question = input(f"\n{config.PARTY_INFO[party]['color']} Sorunuz: ").strip()

        # Çıkış kontrolü
        if question.lower() in ['q', 'quit', 'exit', 'çıkış']:
            print("\n👋 Görüşmek üzere!")
            break

        if not question:
            print("⚠️ Lütfen bir soru yazın.")
            continue

        # Cevap üret
        try:
            response = ask_question(question, vectorstore, chain, party)

            print("\n" + "="*60)
            print("Cevap:")
            print("="*60)
            print(response)
            print("="*60)

        except Exception as e:
            utils.logger.error(f"❌ Hata: {str(e)}")
            print("⚠️ Bir hata oluştu. Lütfen tekrar deneyin.")

# ============================================
# MULTI PARTY MODE
# ============================================

def multi_party_mode():
    """
    Çoklu parti modu - kullanıcı hangi partiye sormak istediğini seçer
    """
    utils.print_header("🤖 Çok Partili Soru-Cevap Sistemi")

    # Hazır partileri kontrol et
    prepared_parties = utils.get_prepared_parties()

    if not prepared_parties:
        utils.logger.error("❌ Hiç hazır vector database yok!")
        utils.logger.error("💡 Önce veri hazırlama yapın: python prepare_data.py")
        return

    utils.logger.info(f"✅ Hazır partiler: {', '.join(prepared_parties)}")

    # Embedding modeli (tüm partiler için aynı)
    utils.logger.info("Embedding modeli yükleniyor...")
    embeddings = utils.load_embeddings()

    # Tüm partilerin vector DB'lerini yükle
    vectorstores: Dict[str, Chroma] = {}

    for party in prepared_parties:
        db_path = config.PARTY_VECTOR_DBS[party]
        vectorstores[party] = utils.load_vectorstore(db_path, embeddings)

    # Tüm partilerin LLM chain'lerini hazırla
    chains: Dict[str, any] = {}

    for party in prepared_parties:
        chains[party] = setup_llm_chain(party)

    utils.logger.info("✅ Tüm sistemler hazır!")

    # Ana döngü
    utils.print_header("💬 Soru-Cevap Başlıyor")
    print("\nKomutlar:")
    print("  - Parti değiştir: /chp, /akp, /mhp, /iyi")
    print("  - Çıkış: q, quit, exit")
    print("\nVarsayılan parti: CHP\n")

    current_party = "CHP" if "CHP" in prepared_parties else prepared_parties[0]

    while True:
        # Parti göstergesi
        party_color = config.PARTY_INFO[current_party]['color']
        question = input(f"\n{party_color} [{current_party}] Sorunuz: ").strip()

        # Çıkış kontrolü
        if question.lower() in ['q', 'quit', 'exit', 'çıkış']:
            print("\n👋 Görüşmek üzere!")
            break

        # Parti değiştirme (sadece "/" ile başlıyorsa)
        if question.startswith('/'):
            # Sadece ilk kelimeyi (parti adını) al
            parts = question.split(maxsplit=1)
            new_party = parts[0][1:].upper()  # "/" işaretini çıkar

            if new_party in prepared_parties:
                current_party = new_party
                print(f"✅ Parti değiştirildi: {current_party}")
                utils.print_party_info(current_party)

                # Eğer sorunun devamı varsa, onu sor
                if len(parts) > 1:
                    question = parts[1].strip()
                    # Soruyu sor (aşağıdaki kod çalışacak)
                else:
                    continue  # Sadece parti değişikliği, sonraki soruya geç
            else:
                print(f"❌ Parti bulunamadı: {new_party}")
                print(f"Mevcut partiler: {', '.join(prepared_parties)}")
                continue

        if not question:
            print("⚠️ Lütfen bir soru yazın.")
            continue

        # Cevap üret
        try:
            response = ask_question(
                question,
                vectorstores[current_party],
                chains[current_party],
                current_party
            )

            print("\n" + "="*60)
            print(f"Cevap ({current_party}):")
            print("="*60)
            print(response)
            print("="*60)

        except Exception as e:
            utils.logger.error(f"❌ Hata: {str(e)}")
            print("⚠️ Bir hata oluştu. Lütfen tekrar deneyin.")

# ============================================
# CLI INTERFACE
# ============================================

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="Turkish Government Intelligence Hub - Soru-Cevap Sistemi"
    )

    parser.add_argument(
        "--party",
        type=str,
        choices=list(config.PARTY_PDFS.keys()),
        help="Sadece belirtilen parti ile çalış"
    )

    args = parser.parse_args()

    # Tek parti modu
    if args.party:
        single_party_mode(args.party)
    # Çoklu parti modu
    else:
        multi_party_mode()

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()