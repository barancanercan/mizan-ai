# Turkish Government Intelligence Hub - Proje Haritası

## 📁 Proje Yapısı

```
Turkish-Government-Intelligence-Hub/
├── src/                        # Ana kaynak kod
│   ├── app.py                  # Streamlit UI (466 satır)
│   ├── query_system.py         # RAG sorgu sistemi (365 satır)
│   ├── prepare_data.py         # Veri hazırlama (280 satır)
│   ├── utils.py                # Yardımcı fonksiyonlar (438 satır)
│   └── config.py               # Konfigürasyon (280 satır)
├── tests/                      # Test dosyaları
├── scripts/                    # Yardımcı scriptler
├── data/                       # PDF dosyaları (8 parti)
├── picture/                    # Parti logoları (8 PNG)
├── vector_db/                  # ChromaDB vektör veritabanları
└── docs/                      # Dokümanlar
```

---

## 🔴 Tespit Edilen Hatalar (Bugs)

### 1. Hardcoded Path Sorunları
- **Konum:** `scripts/dignostic.py:9`, `scripts/fix_iyi.py:9`
- **Sorun:** Linux path (`/home/baran/Desktop/...`) Windows ile uyumsuz
- **Çözüm:** `pathlib.Path(__file__).parent.parent` kullanarak dinamik yol

### 2. İYİ/İYİ Normalization Tutarsızlığı
- **Konum:** `src/app.py:119`, `src/query_system.py:36,136,165,249,302`
- **Sorun:** "IYI" → "İYİ" dönüşümü birçok yerde tekrarlanıyor, tutarsız kullanım
- **Çözüm:** Tek bir yardımcı fonksiyonda topla (`utils.normalize_party_name()`)

### 3. Stream/Cevap Üretme Kod Tekrarı
- **Konum:** `src/app.py:329-341` ve `src/query_system.py:217-227, 327-337`
- **Sorun:** Aynı stream işleme mantığı 3 kez tekrarlanıyor
- **Çözüm:** `utils.py`'de `format_stream_response()` fonksiyonu oluştur

### 4. Vektör DB Yükleme Kod Tekrarı
- **Konum:** `src/utils.py:187-222`, `src/query_system.py:171-179`, `src/app.py:215-219`
- **Sorun:** `load_vectorstore()` çağrısı birçok yerde tekrarlanıyor
- **Çözüm:** Singleton pattern veya cached fonksiyon kullan

### 5. Embeddings Yükleme Tekrarı
- **Konum:** `src/utils.py:101-119`, `src/app.py:218`, `src/prepare_data.py:103`
- **Sorun:** Embeddings birçok kez yükleniyor
- **Çözüm:** `@st.cache_resource` veya global singleton

### 6. LLM Handler Setup Tekrarı
- **Konum:** `src/app.py:223-233` ve `src/query_system.py:182-196, 267-281`
- **Sorun:** Ollama → HuggingFace fallback mantığı 3 kez yazılmış
- **Çözüm:** `utils.py`'de `setup_llm_handler()` fonksiyonu oluştur

### 7. Error Handling Eksikliği
- **Konum:** `src/query_system.py:93-99`
- **Sorun:** HuggingFace hata durumunda `raise` ediyor, kullanıcı dostu mesaj yok
- **Çözüm:** Graceful fallback veya kullanıcıya bilgi

### 8. Type Hint Eksiklikleri
- **Konum:** `src/app.py`, `src/query_system.py`
- **Sorun:** Birçok fonksiyonda `Any` tipi kullanılmış
- **Çözüm:** Proper type hinting ekle

### 9. Unused Import
- **Konum:** `src/query_system.py:6`
- **Sorun:** `argparse` import edilmiş ama sadece `main()` için kullanılıyor
- **Çözüm:** Import mantıklı, ancak fonksiyon bölme düşünülebilir

### 10. Magic Number/String
- **Konum:** `src/app.py:354`, `src/config.py:196`
- **Sorun:** `500` (karakter sınırı), `0.5` (threshold) sabit olarak yazılmış
- **Çözüm:** config.py'de değişken olarak tanımla

---

## 🟡 Spagetti Kod & Kod Tekrarı (Code Smells)

### 1. İYİ Normalization Tekrarı (CRITICAL)
**Tekrar Sayısı:** 6+ kez

```python
# Bu kod parçası birçok dosyada tekrar ediyor:
if party.upper() in ["IYI", "İYİ"]:
    party = "İYİ"
```

**Öneri:** `utils/parties.py` dosyası oluştur:
```python
def normalize_party_name(party: str) -> str:
    """Tüm parti isimlerini normalize et"""
    if party.upper() in ("IYI", "İYİ"):
        return "İYİ"
    return party
```

### 2. Stream Response Handler Tekrarı
**Tekrar Sayısı:** 3 kez (app.py, query_system.py)

**Öneri:** `utils/streaming.py`:
```python
def handle_stream_response(chunk, llm_type: str) -> str:
    """Tüm LLM tipleri için stream response'ı işle"""
    if isinstance(chunk, str):
        return chunk
    if llm_type == "ollama":
        return str(chunk)
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError):
        return str(chunk)
```

### 3. LLM Setup Fallback Mantığı Tekrarı
**Tekrar Sayısı:** 3 kez

**Öneri:** `utils/llm_setup.py`:
```python
def create_llm_handler(party: str) -> tuple[Any, str]:
    """Ollama → HuggingFace fallback mantığı tek yerde"""
    try:
        return setup_ollama_chain(party), "ollama"
    except Exception:
        hf_config = setup_huggingface_config()
        if hf_config:
            return hf_config, "huggingface"
        return None, "none"
```

### 4. VectorStore Lazy Loading Tekrarı
**Tekrar Sayısı:** 3 kez

**Öneri:** `utils/cache.py`:
```python
@st.cache_resource
def get_vectorstore():
    """Singleton vectorstore"""
    embeddings = utils.load_embeddings()
    return utils.load_vectorstore(config.UNIFIED_VECTOR_DB, embeddings)
```

### 5. Party Info Display Tekrarı
**Konum:** `utils.py:343-348`, `app.py:181-198`

**Öneri:** Tek bir `display_party_card()` fonksiyonu kullan

### 6. Hash Kontrolü Kod Tekrarı
**Konum:** `prepare_data.py:68-74`

**Öneri:** İyi yapılmış, mevcut durumu koru ama test et

### 7. Logging Setup Tekrarı
**Konum:** `utils.py:27-35`, `config.py:16`

**Öneri:** Tek bir `setup_logging()` fonksiyonunda topla

---

## 🟢 Profesyonel İyileştirme Önerileri

### 1. Modüler Yapı Oluştur

```
src/
├── core/                    # Çekirdek fonksiyonlar
│   ├── __init__.py
│   ├── parties.py          # Parti normalize, metadata
│   ├── vectorstore.py      # Vector DB yönetimi
│   └── llm.py              # LLM setup & handler
├── ui/                     # UI bileşenleri
│   ├── __init__.py
│   ├── components.py       # Streamlit bileşenleri
│   └── styles.py           # CSS & theming
├── utils/                  # Yardımcılar
│   ├── __init__.py
│   ├── logging.py          # Logging setup
│   └── stream.py           # Stream handlers
└── app.py                  # Ana uygulama
```

### 2. Abstract Base Class Kullanımı

```python
from abc import ABC, abstractmethod

class BaseLLMHandler(ABC):
    @abstractmethod
    def ask(self, question: str, context: str) -> str:
        pass
    
    @abstractmethod
    def stream(self, question: str, context: str) -> Generator:
        pass

class OllamaHandler(BaseLLMHandler):
    ...

class HuggingFaceHandler(BaseLLMHandler):
    ...
```

### 3. Configuration Management

```python
# config/ settings kullan
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    embedding_model: str = "nezahatkorkmaz/turkce-embedding-bge-m3"
    llm_model: str = "qwen2.5:7b"
    chunk_size: int = 512
    # ...
```

### 4. Dependency Injection

```python
from functools import lru_cache

@lru_cache()
def get_embeddings():
    return utils.load_embeddings()

def get_vectorstore(embeddings=get_embeddings()):
    ...
```

### 5. Data Class / Pydantic Models

```python
from pydantic import BaseModel
from typing import Optional

class Party(BaseModel):
    code: str
    name: str
    short: str
    website: str
    hex_color: str
    founded: int
    logo_path: Optional[Path] = None

class QueryResult(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
```

### 6. Repository Pattern

```python
class VectorDBRepository:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def search(self, query: str, party: str) -> list[Document]:
        ...
    
    def add_documents(self, chunks: list[Document]):
        ...
```

### 7. Service Layer

```python
class QueryService:
    def __init__(self, vector_repo, llm_handler):
        self.vector_repo = vector_repo
        self.llm_handler = llm_handler
    
    def process_query(self, question: str, party: str) -> QueryResult:
        ...
```

### 8. Exception Handling

```python
class VectorDBError(Exception):
    ...

class LLMError(Exception):
    ...

class PartyNotFoundError(VectorDBError):
    ...
```

### 9. Environment Variables

```python
# .env dosyası kullan
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

### 10. Constants Enum

```python
from enum import Enum

class PartyCode(str, Enum):
    CHP = "CHP"
    AKP = "AKP"
    MHP = "MHP"
    IYI = "İYI"
    # ...

class LLMType(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    NONE = "none"
```

---

## ✅ Yapılacaklar Listesi (TAMAMLANDI)

| Öncelik | Görev | Dosya | Durum |
|---------|-------|-------|-------|
| 🔴 High | İYİ normalization fonksiyonu oluştur | utils/parties.py | ✅ |
| 🔴 High | Stream response handler birleştir | utils/streaming.py | ✅ |
| 🔴 High | LLM setup fonksiyonu birleştir | utils/llm_setup.py | ✅ |
| 🟡 Medium | Lazy vectorstore singleton | utils/cache.py | ✅ |
| 🟡 Medium | Type hints ekle | Tüm dosyalar | ✅ |
| 🟡 Medium | Pydantic models ekle | src/models.py | ✅ |
| 🟢 Low | Config class yapısı | config/settings.py | ❌ |
| 🟢 Low | Exception sınıfları | src/exceptions.py | ✅ |
| 🟢 Low | .env desteği ekle | - | ❌ |

---

## 📊 Metrikler (Güncellendi)

- **Toplam Satır:** ~1,850 (src/)
- **Eski Tekrar Eden Kod:** ~300 satır
- **Yeni Dosyalar:** 5 adet
  - `src/utils/parties.py` - Parti normalizasyonu
  - `src/utils/streaming.py` - Stream handler
  - `src/utils/llm_setup.py` - LLM fallback
  - `src/utils/cache.py` - Vectorstore cache
  - `src/exceptions.py` - Exception sınıfları
  - `src/models.py` - Pydantic models
- **Modülerlik Skoru:** 7/10 (İyileştirildi)

---

*Bu rapor otomatik olarak oluşturulmuştur.*
