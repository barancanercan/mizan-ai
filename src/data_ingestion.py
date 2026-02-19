"""
FAZ 1: Data Ingestion Agent
Gelişmiş metadata extraction ve veri yükleme pipeline'ı.

Bu modül:
- PDF parsing ve metadata extraction
- Kaynak doğrulama (whitelist)
- Versioning desteği
- Chunking stratejisi ile vektör DB'ye yükleme
"""

import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
from dataclasses import dataclass, field, asdict

from langchain_core.documents import Document

from config import SourceWhitelist, SourceType
from data_cleaning import DataCleaningAgent, DocumentMetadata, CleaningConfig

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Veri yükleme sonucu."""
    success: bool
    source_key: str
    chunks_count: int = 0
    articles_count: int = 0
    metadata: Optional[DocumentMetadata] = None
    error: str = ""
    version: str = ""


class MetadataExtractor:
    """
    Gelişmiş metadata extraction agent'ı.
    Belge içeriğinden metadata çıkarır.
    """
    
    # Belge türü kalıpları
    DOCUMENT_TYPE_PATTERNS = {
        'tuzuk': [r'tüzük', r'tüzüğü', r'ana tüzük', r'tüzükleri'],
        'program': [r'program', r'seçim programı', r'parti programı'],
        'tutanak': [r'tutanak', r'oturum tutanak', r'tbmm tutanak'],
        'yasa': [r'kanun', r'yasa', r'kanunu', r'tasarı', r'teklif'],
    }
    
    # Başlık kalıpları
    TITLE_PATTERNS = [
        r'^(.+?)(?:Tüzük|Program|Kanun|Yasa|Tutanak)',
        r'^([A-ZÇĞİÖŞÜ][A-Za-zçğıöşü\s]{5,50})$',
    ]
    
    @classmethod
    def extract_from_content(cls, text: str, source_key: str = "") -> Dict[str, Any]:
        """
        İçerikten metadata çıkarır.
        
        Args:
            text: Belge içeriği
            source_key: Kaynak anahtarı (parti kodu vb.)
            
        Returns:
            Dict: Çıkarılan metadata
        """
        metadata = {
            'document_type': 'unknown',
            'title': '',
            'language': 'tr',
            'country': 'TR',
            'extracted_at': datetime.now().isoformat(),
        }
        
        # Belge türünü tespit et
        text_lower = text.lower()
        for doc_type, patterns in cls.DOCUMENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    metadata['document_type'] = doc_type
                    break
        
        # Başlığı çıkar
        first_lines = text.split('\n')[:10]
        for line in first_lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Sayı veya madde içermeyen satırı başlık olarak kabul et
                if not re.match(r'^(MADDE|\d+|\([a-zA-Z]\)|\.)', line):
                    metadata['title'] = line
                    break
        
        # Kaynak bilgisi ekle
        if source_key and source_key in SourceWhitelist.SOURCES:
            source_info = SourceWhitelist.SOURCES[source_key]
            metadata['source_name'] = source_info.get('name', '')
            metadata['source_url'] = source_info.get('url', '')
            metadata['source_type'] = source_info.get('type', '')
        
        return metadata
    
    @classmethod
    def extract_from_filename(cls, filepath: Path, source_key: str = "") -> Dict[str, Any]:
        """Dosya adından metadata çıkarır."""
        metadata = {
            'filename': filepath.name,
            'file_size': filepath.stat().st_size if filepath.exists() else 0,
            'file_extension': filepath.suffix.lower(),
        }
        
        # Tarih kalıpları
        date_patterns = [
            r'(\d{4})[_\-](\d{2})[_\-](\d{2})',  # 2024_01_15
            r'(\d{4})',  # Sadece yıl
        ]
        
        filename_str = filepath.stem
        for pattern in date_patterns:
            match = re.search(pattern, filename_str)
            if match:
                metadata['year'] = match.group(1)
                break
        
        return metadata


class VersioningManager:
    """
    Belge versioning manager'ı.
    Her belge için hash tabanlı versiyonlama yapar.
    """
    
    @staticmethod
    def compute_version(content: str, source_key: str) -> str:
        """İçerikten versiyon hash'i oluşturur."""
        version_string = f"{source_key}:{content[:1000]}:{datetime.now().date()}"
        return hashlib.sha256(version_string.encode()).hexdigest()[:12]
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Sadece içerik hash'i."""
        return hashlib.md5(content.encode()).hexdigest()
    
    @classmethod
    def check_needs_update(
        cls, 
        source_key: str, 
        current_content_hash: str, 
        stored_hashes: Dict[str, str]
    ) -> bool:
        """Güncelleme gerekip gerekmediğini kontrol eder."""
        stored_hash = stored_hashes.get(source_key, "")
        return stored_hash != current_content_hash


class DataIngestionAgent:
    """
    Ana Data Ingestion Agent.
    PDF'leri işler, temizler ve vector DB'ye yükler.
    """
    
    def __init__(
        self,
        cleaning_config: Optional[CleaningConfig] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.cleaning_agent = DataCleaningAgent(cleaning_config or CleaningConfig())
        self.metadata_extractor = MetadataExtractor()
        self.versioning_manager = VersioningManager()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(
        self,
        pdf_path: Path,
        source_key: str,
        embeddings: Any = None,
    ) -> IngestionResult:
        """
        PDF'i işler ve vector DB'ye yükler.
        
        Args:
            pdf_path: PDF dosya yolu
            source_key: Kaynak anahtarı (parti kodu)
            embeddings: Embedding modeli
            
        Returns:
            IngestionResult: İşleme sonucu
        """
        logger.info(f"PDF işleniyor: {pdf_path.name}")
        
        # 1. PDF'i yükle
        try:
            from utils import load_pdf
            pages = load_pdf(pdf_path)
            text = '\n'.join([page.page_content for page in pages])
        except Exception as e:
            logger.error(f"PDF yükleme hatası: {e}")
            return IngestionResult(
                success=False,
                source_key=source_key,
                error=str(e),
            )
        
        # 2. Kaynak doğrulama
        if not SourceWhitelist.is_trusted(source_key):
            logger.warning(f"Kaynak güvenilir değil: {source_key}")
        
        # 3. Temizleme
        cleaned_text, metadata = self.cleaning_agent.clean_document(text, source_key)
        
        # 4. Metadata extraction
        content_metadata = self.metadata_extractor.extract_from_content(cleaned_text, source_key)
        filename_metadata = self.metadata_extractor.extract_from_filename(pdf_path, source_key)
        
        # Metadata'yı birleştir
        metadata.raw_attributes = {**content_metadata, **filename_metadata}
        metadata.document_type = content_metadata.get('document_type', 'unknown')
        metadata.title = content_metadata.get('title', pdf_path.stem)
        metadata.page_count = len(pages)
        
        # 5. Versiyon oluştur
        version = self.versioning_manager.compute_version(cleaned_text, source_key)
        content_hash = self.versioning_manager.compute_content_hash(cleaned_text)
        metadata.version = version
        
        # 6. Chunk'lara böl
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(cleaned_text)
        
        # 7. Document objelerine dönüştür
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                'source_key': source_key,
                'source_type': metadata.source_type,
                'document_type': metadata.document_type,
                'title': metadata.title,
                'page_count': metadata.page_count,
                'version': version,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'language': metadata.language,
                'country': metadata.country,
                'is_verified': metadata.is_verified,
                'extracted_date': metadata.extracted_date,
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        logger.info(f"✅ {len(documents)} chunk oluşturuldu, madde sayısı: {metadata.article_count}")
        
        # 8. Vector DB'ye ekle
        if embeddings:
            try:
                from utils import create_vectorstore, load_vectorstore, add_to_vectorstore
                from config import UNIFIED_VECTOR_DB, COLLECTION_NAME
                
                if UNIFIED_VECTOR_DB.exists():
                    vectorstore = load_vectorstore(UNIFIED_VECTOR_DB, embeddings)
                    # Eski verileri temizle
                    try:
                        vectorstore.delete(where={"source_key": source_key})
                    except:
                        pass
                    add_to_vectorstore(documents, vectorstore)
                else:
                    create_vectorstore(documents, embeddings, UNIFIED_VECTOR_DB)
                
                logger.info(f"✅ Vector DB'ye eklendi: {source_key}")
                
                return IngestionResult(
                    success=True,
                    source_key=source_key,
                    chunks_count=len(documents),
                    articles_count=metadata.article_count,
                    metadata=metadata,
                    version=version,
                )
                
            except Exception as e:
                logger.error(f"Vector DB hatası: {e}")
                return IngestionResult(
                    success=False,
                    source_key=source_key,
                    error=str(e),
                    metadata=metadata,
                )
        
        # Embedding yoksa sadece işle
        return IngestionResult(
            success=True,
            source_key=source_key,
            chunks_count=len(documents),
            articles_count=metadata.article_count,
            metadata=metadata,
            version=version,
        )
    
    def process_batch(
        self,
        pdf_paths: Dict[str, Path],
        embeddings: Any = None,
    ) -> List[IngestionResult]:
        """
        Birden fazla PDF'i işler.
        
        Args:
            pdf_paths: {source_key: pdf_path} sözlüğü
            embeddings: Embedding modeli
            
        Returns:
            List[IngestionResult]: İşleme sonuçları
        """
        results = []
        for source_key, pdf_path in pdf_paths.items():
            result = self.process_pdf(pdf_path, source_key, embeddings)
            results.append(result)
        
        return results
    
    def validate_sources(self, source_keys: List[str]) -> Dict[str, bool]:
        """
        Kaynakların güvenilirliğini doğrular.
        
        Returns:
            Dict: {source_key: is_valid}
        """
        validation = {}
        for key in source_keys:
            validation[key] = SourceWhitelist.is_trusted(key)
        
        return validation


def create_ingestion_agent(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> DataIngestionAgent:
    """Ingestion agent factory."""
    return DataIngestionAgent(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
