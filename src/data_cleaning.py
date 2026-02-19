"""
FAZ 1: Data Cleaning Agent
OCR temizleme, gürültü azaltma ve madde bazlı ayrıştırma.

Bu modül:
- PDF'den gelen OCR hatalarını temizler
- Gereksiz karakter/gürültüyü azaltır
- Metni yapısal olarak ayrıştırır (madde, madde altı, vb.)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Temizlik konfigürasyonu."""
    remove_page_numbers: bool = True
    remove_headers_footers: bool = True
    normalize_whitespace: bool = True
    fix_ocr_errors: bool = True
    extract_articles: bool = True
    min_article_length: int = 50


@dataclass
class DocumentMetadata:
    """Gelişmiş belge metadata'sı."""
    source_key: str = ""
    source_type: str = ""
    document_type: str = ""
    title: str = ""
    page_count: int = 0
    extracted_date: str = ""
    version: str = ""
    language: str = "tr"
    country: str = "TR"
    is_verified: bool = False
    article_count: int = 0
    raw_attributes: Dict[str, Any] = field(default_factory=dict)


class OCRCleaner:
    """
    OCR hatalarını temizleyen agent.
    Türkçe karakter setine özel iyileştirmeler içerir.
    """
    
    # Sık görülen OCR hataları - Türkçe
    OCR_ERRORS = {
        # Yanlış karakterler
        r'\b0\b': 'O',  # Sıfır -> O (kelime içinde)
        r'\b1\b': 'I',  # Bir -> I
        r'\b5\b': 'S',  # Beş -> S
        r'\b8\b': 'B',  # Sekiz -> B
        # Tire ve bağlaç hataları
        r'—': '-',
        r'–': '-',
        r'­': '',  # Soft hyphen
        # Tırnak hataları
        r'"': '"',
        r'"': '"',
        r''': "'",
        r''': "'",
        # Parantez hataları
        r'\[\]': '',
        r'\(\)': '',
        # Özel karakterler
        r'…': '...',
        r'•': '-',
        r'·': '-',
        r'○': '',
        # Çoklu boşluklar
        r'\s+': ' ',
    }
    
    # Türkçe'ye özel düzeltmeler
    TURKISH_FIXES = {
        # Büyük harf düzeltmeleri
        r'\bİ\b': 'İ',
        r'\bI\b': 'ı',  # Turkish lowercase I
        r'\bI\b': 'I',  # Keep uppercase I
        # Noktalama
        r'\s\.\.\.': '...',
        r'\.\.\.\s': '...',
    }
    
    @classmethod
    def clean_text(cls, text: str, config: Optional[CleaningConfig] = None) -> str:
        """
        OCR hatalarını temizler.
        
        Args:
            text: Temizlenecek metin
            config: Temizlik konfigürasyonu
            
        Returns:
            str: Temizlenmiş metin
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Satır sonlarını normalize et
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # OCR hatalarını düzelt
        if config and config.fix_ocr_errors:
            for pattern, replacement in cls.OCR_ERRORS.items():
                cleaned = re.sub(pattern, replacement, cleaned)
            
            for pattern, replacement in cls.TURKISH_FIXES.items():
                cleaned = re.sub(pattern, replacement, cleaned)
        
        # Çoklu boşlukları temizle
        if config and config.normalize_whitespace:
            cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Tab ve çoklu boşluk
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # 3+ satır sonu -> 2
        
        return cleaned.strip()
    
    @classmethod
    def remove_page_numbers(cls, text: str) -> str:
        """Sayfa numaralarını kaldırır."""
        # Tek başına sayfa numaraları (rakamlar)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Sayfa X / Y formatı
        text = re.sub(r'Sayfa\s*\d+\s*/\s*\d+', '', text, flags=re.IGNORECASE)
        return text
    
    @classmethod
    def remove_headers_footers(cls, text: str, header_patterns: Optional[List[str]] = None) -> str:
        """Header ve footer'ları kaldırır."""
        if header_patterns is None:
            header_patterns = [
                r'^(Tüzük|Program|Madde)\s*\d+.*$',
                r'^www\..+$',
                r'^https?://.*$',
            ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_header_footer = False
            for pattern in header_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class NoiseReducer:
    """
    Gürültü azaltma agent'ı.
    Anlamsız karakter dizilerini ve kalıpları kaldırır.
    """
    
    NOISE_PATTERNS = [
        r'[{}\[\]<>]',  # Kötü formatlanmış parantez
        r'[^\S\n]+',    # Görünmez karakterler
        r'[\x00-\x1F\x7F-\x9F]',  # Kontrol karakterleri
    ]
    
    @classmethod
    def reduce_noise(cls, text: str) -> str:
        """Gürültüyü azaltır."""
        cleaned = text
        
        for pattern in cls.NOISE_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned


class ArticleParser:
    """
    Madde bazlı ayrıştırma agent'ı.
    Türkçe yasal/metinsel belgelerdeki madde yapısını parser eder.
    """
    
    # Madde kalıpları
    ARTICLE_PATTERNS = [
        r'^(MADDE\s*(\d+|[IVXLCDM]+))[\.\:\-]?\s*(.*)$',  # MADDE 1, MADDE I
        r'^(\d+)[\.\:\-]\s*(.*)$',  # 1. Madde, 1: Madde
        r'^([A-Z])[\.\:\-]\s*(.*)$',  # A. Madde
        r'^(m\.)?\s*(\d+)[\.\:\-]?\s*(.*)$',  # m.1, m.1.
    ]
    
    SECTION_PATTERNS = [
        r'^(BÖLÜM|CHAPTER|SECTION|KISIM)\s*(\d+|[IVXLCDM]+)[\.\:\-]?\s*(.*)$',
        r'^(İÇİNDEKİLER|INDEX|Contents)',  # İçindekiler sayfası
    ]
    
    @classmethod
    def parse_structure(cls, text: str) -> List[Dict[str, Any]]:
        """
        Metni yapısal olarak ayrıştırır.
        
        Returns:
            List[Dict]: Her madde için {type, number, title, content}
        """
        lines = text.split('\n')
        structure = []
        
        current_article = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Bölüm kontrolü
            section_match = cls._match_section(line)
            if section_match:
                if current_article:
                    current_article['content'] = '\n'.join(current_content)
                    structure.append(current_article)
                    current_article = None
                    current_content = []
                
                structure.append({
                    'type': 'section',
                    'number': section_match.group(2) if section_match.lastindex and section_match.lastindex >= 2 else '',
                    'title': section_match.group(3) if section_match.lastindex and section_match.lastindex >= 3 else line,
                    'content': ''
                })
                continue
            
            # Madde kontrolü
            article_match = cls._match_article(line)
            if article_match:
                if current_article:
                    current_article['content'] = '\n'.join(current_content)
                    structure.append(current_article)
                
                current_article = {
                    'type': 'article',
                    'number': article_match.group(2) if article_match.lastindex and article_match.lastindex >= 2 else article_match.group(1),
                    'title': article_match.group(3) if article_match.lastindex and article_match.lastindex >= 3 else '',
                    'content': ''
                }
                current_content = []
            elif current_article:
                current_content.append(line)
        
        # Son maddeyi ekle
        if current_article:
            current_article['content'] = '\n'.join(current_content)
            structure.append(current_article)
        
        return structure
    
    @classmethod
    def _match_article(cls, line: str) -> Optional[re.Match]:
        """Satırın madde olup olmadığını kontrol eder."""
        for pattern in cls.ARTICLE_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match
        return None
    
    @classmethod
    def _match_section(cls, line: str) -> Optional[re.Match]:
        """Satırın bölüm olup olmadığını kontrol eder."""
        for pattern in cls.SECTION_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match
        return None
    
    @classmethod
    def extract_articles(cls, text: str) -> Tuple[List[str], int]:
        """
        Metinden madde içeriklerini çıkarır.
        
        Returns:
            Tuple[List[str], int]: (madde listesi, toplam madde sayısı)
        """
        structure = cls.parse_structure(text)
        articles = [item['content'] for item in structure if item['type'] == 'article']
        
        # Kısa maddeleri filtrele
        articles = [a for a in articles if len(a) >= 50]
        
        return articles, len(articles)


class DataCleaningAgent:
    """
    Ana Data Cleaning Agent.
    Diğer temizlik sınıflarını birleştirir.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.ocr_cleaner = OCRCleaner()
        self.noise_reducer = NoiseReducer()
        self.article_parser = ArticleParser()
    
    def clean_document(self, text: str, source_key: str = "") -> Tuple[str, DocumentMetadata]:
        """
        Belgeyi temizler ve metadata çıkarır.
        
        Returns:
            Tuple[str, DocumentMetadata]: (temizlenmiş metin, metadata)
        """
        logger.info(f"Belge temizleniyor: {source_key}")
        
        # Metadata oluştur
        metadata = DocumentMetadata(
            source_key=source_key,
            extracted_date=datetime.now().isoformat(),
        )
        
        # Whitelist kontrolü
        from config import SourceWhitelist
        if source_key and SourceWhitelist.is_trusted(source_key):
            metadata.is_verified = True
            metadata.source_type = SourceWhitelist.get_source_type(source_key)
        
        # 1. OCR temizleme
        cleaned = self.ocr_cleaner.clean_text(text, self.config)
        
        # 2. Sayfa numaralarını kaldır
        if self.config.remove_page_numbers:
            cleaned = self.ocr_cleaner.remove_page_numbers(cleaned)
        
        # 3. Header/Footer kaldır
        if self.config.remove_headers_footers:
            cleaned = self.ocr_cleaner.remove_headers_footers(cleaned)
        
        # 4. Gürültü azaltma
        cleaned = self.noise_reducer.reduce_noise(cleaned)
        
        # 5. Madde sayısını hesapla
        if self.config.extract_articles:
            _, article_count = self.article_parser.extract_articles(cleaned)
            metadata.article_count = article_count
        
        logger.info(f"Temizleme tamamlandı: {source_key}, Madde sayısı: {metadata.article_count}")
        
        return cleaned, metadata
    
    def clean_batch(self, texts: List[Tuple[str, str]]) -> List[Tuple[str, DocumentMetadata]]:
        """
        Birden fazla belgeyi temizler.
        
        Args:
            texts: [(text, source_key), ...]
            
        Returns:
            List[Tuple[str, DocumentMetadata]]: [(cleaned_text, metadata), ...]
        """
        results = []
        for text, source_key in texts:
            cleaned, metadata = self.clean_document(text, source_key)
            results.append((cleaned, metadata))
        
        return results


def create_cleaning_agent(config: Optional[CleaningConfig] = None) -> DataCleaningAgent:
    """Cleaning agent factory."""
    return DataCleaningAgent(config)
