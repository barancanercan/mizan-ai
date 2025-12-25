"""
Turkish Government Intelligence Hub - Utility Functions
Yardımcı fonksiyonlar
"""

import logging
from pathlib import Path
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # ✅ Yeni paket (deprecation fix)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# PDF PROCESSING FUNCTIONS
# ============================================

def load_pdf(pdf_path: Path) -> List[Document]:
    """
    PDF dosyasını yükle

    Args:
        pdf_path: PDF dosya yolu

    Returns:
        List[Document]: Yüklenmiş sayfalar
    """
    try:
        logger.info(f"PDF yükleniyor: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        logger.info(f"✅ {len(pages)} sayfa yüklendi")
        return pages
    except FileNotFoundError:
        logger.error(f"❌ PDF bulunamadı: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"❌ PDF yükleme hatası: {str(e)}")
        raise


def chunk_documents(
    pages: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> List[Document]:
    """
    Dökümanları chunk'lara böl

    Args:
        pages: PDF sayfaları
        chunk_size: Chunk boyutu
        chunk_overlap: Chunk overlap

    Returns:
        List[Document]: Chunk'lanmış dökümanlar
    """
    logger.info(f"Metin chunk'lara bölünüyor (size={chunk_size}, overlap={chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages)
    logger.info(f"✅ {len(chunks)} chunk oluşturuldu")

    return chunks

# ============================================
# EMBEDDING FUNCTIONS
# ============================================

def load_embeddings(model_name: str = config.EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Türkçe embedding modelini yükle

    Args:
        model_name: HuggingFace model adı

    Returns:
        HuggingFaceEmbeddings: Yüklenmiş embedding modeli
    """
    logger.info(f"Embedding modeli yükleniyor: {model_name}")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("✅ Embedding modeli hazır")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Embedding yükleme hatası: {str(e)}")
        raise

# ============================================
# VECTOR DATABASE FUNCTIONS
# ============================================

def create_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path
) -> Chroma:
    """
    Vector database oluştur ve kaydet

    Args:
        chunks: Chunk'lanmış dökümanlar
        embeddings: Embedding modeli
        persist_dir: Kaydedilecek dizin

    Returns:
        Chroma: Vector database
    """
    logger.info(f"Vector database oluşturuluyor: {persist_dir}")

    try:
        # Dizini oluştur
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Vector DB oluştur
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )

        logger.info(f"✅ Vector database kaydedildi: {persist_dir}")
        return vectorstore

    except Exception as e:
        logger.error(f"❌ Vector DB oluşturma hatası: {str(e)}")
        raise


def load_vectorstore(
    persist_dir: Path,
    embeddings: HuggingFaceEmbeddings
) -> Chroma:
    """
    Hazır vector database'i yükle

    Args:
        persist_dir: Vector DB dizini
        embeddings: Embedding modeli

    Returns:
        Chroma: Yüklenmiş vector database
    """
    logger.info(f"Vector database yükleniyor: {persist_dir}")

    try:
        if not persist_dir.exists():
            raise FileNotFoundError(f"Vector DB bulunamadı: {persist_dir}")

        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings
        )

        logger.info(f"✅ Vector database yüklendi")
        return vectorstore

    except Exception as e:
        logger.error(f"❌ Vector DB yükleme hatası: {str(e)}")
        raise

# ============================================
# SEARCH FUNCTIONS
# ============================================

def search_similar_docs(
    vectorstore: Chroma,
    question: str,
    top_k: int = config.TOP_K
) -> Tuple[str, List[float]]:
    """
    Benzer dökümanları bul

    Args:
        vectorstore: Vector database
        question: Kullanıcı sorusu
        top_k: Kaç chunk getireceğiz

    Returns:
        Tuple[str, List[float]]: (context, similarity_scores)
    """
    logger.info(f"Arama yapılıyor: '{question}'")

    try:
        relevant_docs = vectorstore.similarity_search_with_score(question, k=top_k)

        relevant_chunks = [doc.page_content for doc, score in relevant_docs]
        context = "\n\n".join(relevant_chunks)
        scores = [score for doc, score in relevant_docs]

        logger.info(f"✅ {len(relevant_docs)} chunk bulundu, skorlar: {scores}")

        return context, scores

    except Exception as e:
        logger.error(f"❌ Arama hatası: {str(e)}")
        raise

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_pdf_exists(party: str) -> bool:
    """
    Parti PDF'inin var olup olmadığını kontrol et

    Args:
        party: Parti kısa adı (CHP, AKP, etc.)

    Returns:
        bool: PDF var mı?
    """
    pdf_path = config.PARTY_PDFS.get(party)

    if pdf_path is None:
        logger.error(f"❌ Parti bulunamadı: {party}")
        return False

    if not pdf_path.exists():
        logger.warning(f"⚠️ PDF bulunamadı: {pdf_path}")
        return False

    return True


def get_available_parties() -> List[str]:
    """
    Mevcut PDF'leri olan partileri listele

    Returns:
        List[str]: Mevcut partiler
    """
    available = []

    for party, pdf_path in config.PARTY_PDFS.items():
        if pdf_path.exists():
            available.append(party)

    return available


def get_prepared_parties() -> List[str]:
    """
    Vector DB'si hazır olan partileri listele

    Returns:
        List[str]: Hazır partiler
    """
    prepared = []

    for party, db_path in config.PARTY_VECTOR_DBS.items():
        if db_path.exists():
            prepared.append(party)

    return prepared

# ============================================
# DISPLAY FUNCTIONS
# ============================================

def print_header(text: str, width: int = 60):
    """Başlık yazdır"""
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width)


def print_party_info(party: str):
    """Parti bilgilerini yazdır"""
    info = config.PARTY_INFO.get(party)
    if info:
        print(f"\n{info['color']} {info['name']} ({info['short']})")
        print(f"🌐 Website: {info['website']}")