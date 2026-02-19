"""
Turkish Government Intelligence Hub - Configuration
Local Image Serving + İYİ Normalization
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

# ============================================
# TELEMETRY - Devre Dışı
# ============================================

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ============================================
# PATHS
# ============================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
SRC_DIR = PROJECT_ROOT / "src"
PICTURE_DIR = PROJECT_ROOT / "picture"  # ← LOCAL IMAGES

# ============================================
# DYNAMIC PARTY DISCOVERY WITH İYİ NORMALIZATION
# ============================================


def discover_parties() -> Dict[str, Path]:
    """
    Dinamik olarak data/ klasöründeki tüm PDF'leri keşfederek bir sözlük döndürür.
    "IYI" isimlendirmesini "İYİ" olarak normalize eder.

    Returns:
        Dict[str, Path]: Parti kodu ve ilgili PDF dosyasının yolu.
    """
    discovered: Dict[str, Path] = {}
    if not DATA_DIR.exists():
        return discovered

    for pdf_file in sorted(DATA_DIR.glob("*.pdf")):
        party_code: str = pdf_file.stem.upper()

        if party_code == "IYI":
            party_code = "İYİ"

        discovered[party_code] = pdf_file

    return discovered


PARTY_PDFS = discover_parties()
PARTY_VECTOR_DBS = {
    party: VECTOR_DB_DIR / f"{party.replace('İ', 'i').lower()}_db"  # ← ORDER DEĞİŞTİ!
    for party in PARTY_PDFS.keys()
}

# ============================================
# PARTY METADATA
# ============================================

PARTY_METADATA: Dict[str, Any] = {
    "CHP": {
        "name": "Cumhuriyet Halk Partisi",
        "short": "CHP",
        "website": "https://chp.org.tr",
        "hex_color": "#FF0000",
        "accent_color": "#CC0000",
        "founded": 1923,
        "ideology": "Sosyal Demokrasi, Laïklik",
        "description": "Türkiye'nin en eski siyasi partisi, merkez-sol",
    },
    "AKP": {
        "name": "Adalet ve Kalkınma Partisi",
        "short": "AKP",
        "website": "https://www.akparti.org.tr",
        "hex_color": "#E55100",
        "accent_color": "#D84315",
        "founded": 2001,
        "ideology": "Muhafazakâr, İslami Demokrasi",
        "description": "Mevcut iktidar partisi, merkez-sağ muhafazakâr",
    },
    "MHP": {
        "name": "Milliyetçi Hareket Partisi",
        "short": "MHP",
        "website": "https://mhp.org.tr",
        "hex_color": "#0066FF",
        "accent_color": "#0052CC",
        "founded": 1969,
        "ideology": "Milliyetçilik, Muhafazakârlık",
        "description": "Milliyetçi muhafazakâr parti, hükümet ortağı",
    },
    "İYİ": {  # ← Turkish İ character
        "name": "İYİ Parti",
        "short": "İYİ",
        "website": "https://iyiparti.org.tr",
        "hex_color": "#FFD700",
        "accent_color": "#FFC700",
        "founded": 2017,
        "ideology": "Liberalizm, Milliyetçilik",
        "description": "Merkez-sağ liberal parti",
    },
    "DEM": {
        "name": "Halkların Eşitlik ve Demokrasi Partisi",
        "short": "DEM",
        "website": "https://www.dem.org.tr",
        "hex_color": "#9933FF",
        "accent_color": "#7722CC",
        "founded": 2022,
        "ideology": "Sosyalizm, Halkçılık",
        "description": "Sol-liberal demokratik parti",
    },
    "SP": {
        "name": "Saadet Partisi",
        "short": "SP",
        "website": "https://saadet.org.tr",
        "hex_color": "#00AA00",
        "accent_color": "#008800",
        "founded": 1997,
        "ideology": "İslami Demokrasi",
        "description": "İslami değerlere yakın parti",
    },
    "ZP": {
        "name": "Zafer Partisi",
        "short": "ZP",
        "website": "https://www.zaferpartisi.org.tr",
        "hex_color": "#000000",
        "accent_color": "#333333",
        "founded": 2021,
        "ideology": "Milliyetçilik",
        "description": "Milliyetçi parti",
    },
    "BBP": {
        "name": "Büyük Birlik Partisi",
        "short": "BBP",
        "website": "https://www.bbp.org.tr",
        "hex_color": "#CC0000",
        "accent_color": "#990000",
        "founded": 1993,
        "ideology": "Milliyetçilik, Muhafazakârlık",
        "description": "Milliyetçi muhafazakâr parti",
    },
}


def create_party_info(parties: Dict[str, Path]) -> Dict[str, Any]:
    """
    Parti bilgilerini temel alan PARTY_INFO sözlüğünü otomatik olarak oluşturur.

    Args:
        parties: Keşfedilen partilerin sözlüğü.

    Returns:
        Dict[str, Any]: Parti özelliklerini (isim, renk, vb.) içeren sözlük.
    """
    party_info: Dict[str, Any] = {}
    for party in parties.keys():
        if party in PARTY_METADATA:
            party_info[party] = PARTY_METADATA[party]
        else:
            party_info[party] = {
                "name": party.capitalize(),
                "short": party,
                "website": "https://example.com",
                "hex_color": "#0066FF",
                "accent_color": "#0052CC",
                "founded": 2020,
                "ideology": "Bilinmiyor",
                "description": "Açıklama bulunmamaktadır",
            }
    return party_info


PARTY_INFO = create_party_info(PARTY_PDFS)

# ============================================
# MODEL CONFIGS
# ============================================

EMBEDDING_MODEL = "nezahatkorkmaz/turkce-embedding-bge-m3"
LLM_MODEL = "qwen2.5:7b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# ============================================
# SOURCE WHITELIST - Güvenilir Kaynak Yönetimi (FAZ 1)
# ============================================

class SourceType:
    """Kaynak türleri."""
    PARTY_STATUTE = "party_statute"      # Parti tüzükleri
    PARTY_PROGRAM = "party_program"      # Resmi programlar
    TBMM_DOCUMENT = "tbmm_document"     # TBMM belgeleri
    YSK_PUBLICATION = "ysk_publication"  # YSK yayınları


class SourceWhitelist:
    """
    Whitelist edilmiş güvenilir kaynaklar.
    Her kaynak: {type, name, url, verified_date, trusted}
    """
    
    SOURCES = {
        # Parti tüzükleri (PDF)
        "CHP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Cumhuriyet Halk Partisi Tüzüğü",
            "url": "https://chp.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "AKP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Adalet ve Kalkınma Partisi Tüzüğü",
            "url": "https://www.akparti.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "MHP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Milliyetçi Hareket Partisi Tüzüğü",
            "url": "https://mhp.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "İYİ": {
            "type": SourceType.PARTY_STATUTE,
            "name": "İYİ Parti Tüzüğü",
            "url": "https://iyiparti.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "DEM": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Halkların Eşitlik ve Demokrasi Partisi Tüzüğü",
            "url": "https://www.dem.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "SP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Saadet Partisi Tüzüğü",
            "url": "https://saadet.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "ZP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Zafer Partisi Tüzüğü",
            "url": "https://www.zaferpartisi.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
        "BBP": {
            "type": SourceType.PARTY_STATUTE,
            "name": "Büyük Birlik Partisi Tüzüğü",
            "url": "https://www.bbp.org.tr",
            "verified_date": "2024-01-01",
            "trusted": True,
        },
    }
    
    @classmethod
    def is_trusted(cls, source_key: str) -> bool:
        """Kaynağın güvenilir olup olmadığını kontrol eder."""
        source = cls.SOURCES.get(source_key)
        return source.get("trusted", False) if source else False
    
    @classmethod
    def get_source_type(cls, source_key: str) -> str:
        """Kaynağın türünü döndürür."""
        source = cls.SOURCES.get(source_key)
        return source.get("type", "unknown") if source else "unknown"
    
    @classmethod
    def get_all_trusted(cls) -> dict:
        """Tüm güvenilir kaynakları döndürür."""
        return {k: v for k, v in cls.SOURCES.items() if v.get("trusted")}


# ============================================
# RAG CONFIGS
# ============================================

CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
TOP_K: int = 3
SIMILARITY_THRESHOLD: float = 0.3  # ChromaDB similarity (yüksek = iyi)

# Unified Database Configuration
UNIFIED_VECTOR_DB: Path = VECTOR_DB_DIR / "unified_parties_db"
COLLECTION_NAME: str = "turkish_parties"

# ============================================
# ROUTER & SEARCH CONFIGS (FREE)
# ============================================

ROUTER_THRESHOLD: float = 0.15  # Düşük - tek kelime eşleşmesi yeterli
WEB_SEARCH_MAX_RESULTS: int = 5
WEB_SEARCH_TIMEOUT: int = 10

# ============================================
# SYSTEM PROMPTS
# ============================================


def create_professional_prompts(parties: Dict[str, Path]) -> Dict[str, str]:
    """
    Her siyasi parti için özelleştirilmiş profesyonel sistem promptları oluşturur.

    Args:
        parties: Keşfedilen partilerin sözlüğü.

    Returns:
        Dict[str, str]: Parti bazlı sistem promptları.
    """
    base_prompts: Dict[str, str] = {
        "CHP": "Sen CHP (Cumhuriyet Halk Partisi) hakkında uzman bir bilgi asistanısın.",
        "AKP": "Sen AKP (Adalet ve Kalkınma Partisi) hakkında uzman bir bilgi asistanısın.",
        "MHP": "Sen MHP (Milliyetçi Hareket Partisi) hakkında uzman bir bilgi asistanısın.",
        "İYİ": "Sen İYİ Parti hakkında uzman bir bilgi asistanısın.",
        "DEM": "Sen DEM (Halkların Eşitlik ve Demokrasi Partisi) hakkında uzman bir bilgi asistanısın.",
        "SP": "Sen Saadet Partisi hakkında uzman bir bilgi asistanısın.",
        "ZP": "Sen Zafer Partisi hakkında uzman bir bilgi asistanısın.",
        "BBP": "Sen Büyük Birlik Partisi hakkında uzman bir bilgi asistanısın.",
    }

    prompts: Dict[str, str] = {}
    for party, intro in base_prompts.items():
        prompt: str = f"""Sen {party} Partisi hakkında uzman bir asistansın. Aşağıdaki bilgilere dayanarak soruyu Türkçe olarak yanıtla.

Bilgi:
{{context}}

Soru: {{question}}

Kısa ve öz cevap ver:"""
        prompts[party] = prompt

    return prompts


SYSTEM_PROMPTS = create_professional_prompts(PARTY_PDFS)

# ============================================
# LOGGING
# ============================================

LOG_FILE: Path = PROJECT_ROOT / "app.log"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: str = "INFO"

# ============================================
# UI & STREAMLIT CONFIGS
# ============================================

APP_TITLE: str = "mizan-ai | Siyasi Belge Analiz Platformu"
APP_SUBTITLE: str = "T-RAG: Tool-Augmented RAG for Political Documents"
APP_ICON: str = "⚖️"
APP_LAYOUT: str = "wide"
SIDEBAR_STATE: str = "expanded"

PARTY_LOGOS: Dict[str, str] = {
    "CHP": "chp.png",
    "AKP": "akp.png",
    "MHP": "mhp.png",
    "İYİ": "iyi.png",
    "DEM": "dem.png",
    "SP": "sp.png",
    "ZP": "zp.png",
    "BBP": "bbp.png",
}
