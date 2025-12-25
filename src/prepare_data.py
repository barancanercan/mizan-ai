"""
Turkish Government Intelligence Hub - Data Preparation
Veri hazırlama script'i - BU DOSYAYI SADECE 1 KERE ÇALIŞTIR!

Usage:
    python prepare_data.py              # Tüm partileri hazırla
    python prepare_data.py --party CHP  # Sadece CHP'yi hazırla
"""

import argparse
import sys
from pathlib import Path

import config
import utils


# ============================================
# MAIN PREPARATION FUNCTION
# ============================================

def prepare_party_data(party: str):
    """
    Bir partinin verisini hazırla

    Args:
        party: Parti kısa adı (CHP, AKP, MHP, İYİ)
    """
    utils.print_header(f"{party} Veri Hazırlama")
    utils.print_party_info(party)

    # 1. PDF kontrolü
    if not utils.validate_pdf_exists(party):
        utils.logger.error(f"❌ {party} için PDF bulunamadı!")
        utils.logger.error(f"PDF'i buraya koyun: {config.PARTY_PDFS[party]}")
        return False

    # 2. PDF'i yükle
    try:
        pdf_path = config.PARTY_PDFS[party]
        pages = utils.load_pdf(pdf_path)
    except Exception as e:
        utils.logger.error(f"❌ PDF yükleme başarısız: {str(e)}")
        return False

    # 3. Chunk'lara böl
    try:
        chunks = utils.chunk_documents(pages)
    except Exception as e:
        utils.logger.error(f"❌ Chunking başarısız: {str(e)}")
        return False

    # 4. Embedding modelini yükle (sadece 1 kere)
    try:
        embeddings = utils.load_embeddings()
    except Exception as e:
        utils.logger.error(f"❌ Embedding model yükleme başarısız: {str(e)}")
        return False

    # 5. Vector database oluştur ve kaydet
    try:
        vector_db_path = config.PARTY_VECTOR_DBS[party]
        vectorstore = utils.create_vectorstore(chunks, embeddings, vector_db_path)

        utils.logger.info(f"✅ {party} veri hazırlama TAMAMLANDI!")
        utils.logger.info(f"📁 Kayıt yeri: {vector_db_path}")
        return True

    except Exception as e:
        utils.logger.error(f"❌ Vector DB oluşturma başarısız: {str(e)}")
        return False


# ============================================
# BATCH PREPARATION
# ============================================

def prepare_all_parties():
    """
    Tüm partilerin verisini hazırla
    """
    utils.print_header("🚀 TÜM PARTİLER VERİ HAZIRLAMA 🚀")

    # Mevcut PDF'leri kontrol et
    available_parties = utils.get_available_parties()

    if not available_parties:
        utils.logger.error("❌ Hiç PDF bulunamadı!")
        utils.logger.error(f"PDF'leri buraya koyun: {config.DATA_DIR}")
        return

    utils.logger.info(f"📋 Mevcut partiler: {', '.join(available_parties)}")

    # Embedding modelini önceden yükle (tüm partiler için aynı model)
    utils.logger.info("\n🔄 Embedding modeli yükleniyor (tüm partiler için kullanılacak)...")
    try:
        embeddings = utils.load_embeddings()
    except Exception as e:
        utils.logger.error(f"❌ Embedding model yüklenemedi: {str(e)}")
        return

    # Her parti için işlem yap
    success_count = 0
    failed_parties = []

    for party in available_parties:
        utils.print_header(f"🔄 {party} İşleniyor...")

        try:
            # PDF yükle
            pdf_path = config.PARTY_PDFS[party]
            pages = utils.load_pdf(pdf_path)

            # Chunk'lara böl
            chunks = utils.chunk_documents(pages)

            # Vector DB oluştur (embedding'i tekrar yüklemeye gerek yok)
            vector_db_path = config.PARTY_VECTOR_DBS[party]
            vectorstore = utils.create_vectorstore(chunks, embeddings, vector_db_path)

            utils.logger.info(f"✅ {party} BAŞARILI!")
            success_count += 1

        except Exception as e:
            utils.logger.error(f"❌ {party} BAŞARISIZ: {str(e)}")
            failed_parties.append(party)

    # Özet
    utils.print_header("📊 VERİ HAZIRLAMA ÖZETİ")
    utils.logger.info(f"✅ Başarılı: {success_count}/{len(available_parties)}")

    if failed_parties:
        utils.logger.warning(f"❌ Başarısız partiler: {', '.join(failed_parties)}")
    else:
        utils.logger.info("🎉 TÜM PARTİLER BAŞARIYLA HAZIRLANDI!")


# ============================================
# STATUS CHECK
# ============================================

def check_status():
    """
    Hazır olan ve eksik olan partileri göster
    """
    utils.print_header("📊 VERİ HAZIRLAMA DURUMU")

    all_parties = list(config.PARTY_PDFS.keys())
    prepared = utils.get_prepared_parties()
    available_pdfs = utils.get_available_parties()

    print("\n📁 PDF Durumu:")
    for party in all_parties:
        pdf_status = "✅" if party in available_pdfs else "❌"
        pdf_path = config.PARTY_PDFS[party]
        print(f"  {pdf_status} {party}: {pdf_path}")

    print("\n💾 Vector Database Durumu:")
    for party in all_parties:
        db_status = "✅" if party in prepared else "❌"
        db_path = config.PARTY_VECTOR_DBS[party]
        print(f"  {db_status} {party}: {db_path}")

    # Özet
    print(f"\n📊 Özet:")
    print(f"  📄 Mevcut PDF'ler: {len(available_pdfs)}/{len(all_parties)}")
    print(f"  💾 Hazır Vector DB'ler: {len(prepared)}/{len(all_parties)}")

    if len(prepared) == 0:
        print("\n⚠️ Henüz hiç veri hazırlanmamış!")
        print("💡 Çalıştır: python prepare_data.py")
    elif len(prepared) < len(available_pdfs):
        missing = set(available_pdfs) - set(prepared)
        print(f"\n⚠️ Eksik partiler: {', '.join(missing)}")
        print(f"💡 Çalıştır: python prepare_data.py --party {missing.pop()}")
    else:
        print("\n✅ Tüm partiler hazır!")


# ============================================
# CLI INTERFACE
# ============================================

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description="Turkish Government Intelligence Hub - Veri Hazırlama"
    )

    parser.add_argument(
        "--party",
        type=str,
        choices=list(config.PARTY_PDFS.keys()),
        help="Sadece belirtilen partiyi hazırla"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Veri hazırlama durumunu göster"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Mevcut vector DB'yi sil ve yeniden oluştur"
    )

    args = parser.parse_args()

    # Status kontrolü
    if args.status:
        check_status()
        return

    # Tek parti hazırlama
    if args.party:
        # Force flag kontrolü
        if args.force:
            db_path = config.PARTY_VECTOR_DBS[args.party]
            if db_path.exists():
                utils.logger.warning(f"⚠️ Mevcut DB siliniyor: {db_path}")
                import shutil
                shutil.rmtree(db_path)

        success = prepare_party_data(args.party)
        sys.exit(0 if success else 1)

    # Tüm partileri hazırla
    else:
        prepare_all_parties()


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()