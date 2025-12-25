"""
Turkish Government Intelligence Hub - Configuration
Merkezi konfigürasyon dosyası
"""

from pathlib import Path

# ============================================
# PATHS - Dosya Yolları
# ============================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
SRC_DIR = PROJECT_ROOT / "src"

# Parti PDF'leri
PARTY_PDFS = {
    "CHP": DATA_DIR / "chp.pdf",
    "AKP": DATA_DIR / "akp.pdf",
    "MHP": DATA_DIR / "mhp.pdf",
    "İYİ": DATA_DIR / "iyi.pdf"
}

# Vector Database paths
PARTY_VECTOR_DBS = {
    "CHP": VECTOR_DB_DIR / "chp_db",
    "AKP": VECTOR_DB_DIR / "akp_db",
    "MHP": VECTOR_DB_DIR / "mhp_db",
    "İYİ": VECTOR_DB_DIR / "iyi_db"
}

# ============================================
# MODEL CONFIGS - Model Ayarları
# ============================================

# Embedding Model
EMBEDDING_MODEL = "nezahatkorkmaz/turkce-embedding-bge-m3"

# LLM Model
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 512

# ============================================
# RAG CONFIGS - RAG Ayarları
# ============================================

# Text Splitting
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 3  # Kaç chunk getireceğiz
SIMILARITY_THRESHOLD = 0.6  # Minimum benzerlik skoru

# ============================================
# PROMPT TEMPLATES - Prompt Şablonları
# ============================================

SYSTEM_PROMPTS = {
    "CHP": """Sen CHP (Cumhuriyet Halk Partisi) hakkında bilgi veren bir asistansın.

Aşağıdaki CHP Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. Daha fazla bilgi için https://chp.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
""",

    "AKP": """Sen AKP (Adalet ve Kalkınma Partisi) hakkında bilgi veren bir asistansın.

Aşağıdaki AKP Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. Daha fazla bilgi için https://akparti.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
""",

    "MHP": """Sen MHP (Milliyetçi Hareket Partisi) hakkında bilgi veren bir asistansın.

Aşağıdaki MHP Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. Daha fazla bilgi için https://mhp.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
""",

    "İYİ": """Sen İYİ Parti hakkında bilgi veren bir asistansın.

Aşağıdaki İYİ Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. Daha fazla bilgi için https://iyiparti.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
"""
}

# ============================================
# PARTY INFO - Parti Bilgileri
# ============================================

PARTY_INFO = {
    "CHP": {
        "name": "Cumhuriyet Halk Partisi",
        "short": "CHP",
        "website": "https://chp.org.tr",
        "color": "🔴"
    },
    "AKP": {
        "name": "Adalet ve Kalkınma Partisi",
        "short": "AKP",
        "website": "https://akparti.org.tr",
        "color": "🟠"
    },
    "MHP": {
        "name": "Milliyetçi Hareket Partisi",
        "short": "MHP",
        "website": "https://mhp.org.tr",
        "color": "🔵"
    },
    "İYİ": {
        "name": "İYİ Parti",
        "short": "İYİ",
        "website": "https://iyiparti.org.tr",
        "color": "🟡"
    }
}

# ============================================
# LOGGING CONFIG - Log Ayarları
# ============================================

LOG_FILE = PROJECT_ROOT / "app.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"