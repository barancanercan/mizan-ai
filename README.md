# 🇹🇷 Turkish Government Intelligence Hub

**Türkiye'deki siyasi partilerin tüzüklerini analiz eden, tamamen lokal çalışan RAG tabanlı soru-cevap sistemi**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--7B-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Teknoloji Stack](#-teknoloji-stack)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Performans](#-performans)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🎯 Özellikler

- ✅ **Tamamen Lokal:** İnternet bağlantısı gerektirmez, veriler bilgisayarınızda kalır
- ✅ **Ücretsiz:** API key veya ödeme gerektirmez
- ✅ **Türkçe Optimizasyonlu:** Türkçe embedding ve LLM modelleri kullanır
- ✅ **GPU Hızlandırmalı:** NVIDIA GPU desteği ile hızlı yanıt süreleri
- ✅ **RAG (Retrieval-Augmented Generation):** Doğru ve kaynak tabanlı cevaplar
- ✅ **Kolay Genişletilebilir:** Yeni parti tüzükleri kolayca eklenebilir

---

## 🛠️ Teknoloji Stack

### Core Framework
- **LangChain** - LLM orchestration ve RAG pipeline
- **Qwen2.5-7B-Instruct** - Lokal LLM (Alibaba)
- **Ollama** - Lokal LLM inference server

### Embeddings & Vector DB
- **HuggingFace Transformers** - Türkçe text embeddings
- **ChromaDB** - Vector database

### Document Processing
- **PyPDF** - PDF döküman parsing
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

---

## 📦 Kurulum

### Gereksinimler

- **Python 3.10+**
- **NVIDIA GPU** (önerilen, CPU'da da çalışır)
- **6GB+ VRAM** (RTX 3050 veya üzeri)
- **10GB+ Disk Alanı**

### Adım 1: Ollama Kurulumu

1. Ollama'yı indirin ve kurun:
   ```
   https://ollama.com/download/windows
   ```

2. Qwen2.5 modelini indirin:
   ```bash
   ollama pull qwen2.5:7b-instruct-q4_K_M
   ```

3. Model durumunu kontrol edin:
   ```bash
   ollama list
   ```

### Adım 2: Python Ortamını Hazırlayın

1. Repository'yi klonlayın:
   ```bash
   git clone https://github.com/barancanercan/turkish-government-intelligence-hub.git
   cd turkish-government-intelligence-hub
   ```

2. Virtual environment oluşturun:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Kullanım

### Temel Kullanım

```bash
python rag_qwen_local.py
```

Program çalıştığında soru sorabilirsiniz:

```
============================================================
CHP Parti Tüzüğü - Soru-Cevap Sistemi (LOKAL QWEN)
============================================================

Sorunuz: CHP genel başkanı nasıl seçilir?
```

### Python Kodu ile Kullanım

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

# 1. PDF'i yükle
loader = PyPDFLoader("data/chp.pdf")
pages = loader.load()

# 2. Embedding modeli
embeddings = HuggingFaceEmbeddings(
    model_name="nezahatkorkmaz/turkce-embedding-bge-m3"
)

# 3. Vector database
vectorstore = Chroma.from_documents(
    documents=pages,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. Lokal LLM
llm = Ollama(model="qwen2.5:7b-instruct-q4_K_M")

# 5. Soru sor
question = "CHP'nin kuruluş tarihi nedir?"
docs = vectorstore.similarity_search(question, k=3)
context = "\n".join([doc.page_content for doc in docs])

response = llm.invoke(f"Context: {context}\n\nSoru: {question}")
print(response)
```

---

## ⚡ Performans

### Test Sistemi
- **GPU:** NVIDIA RTX 3060 6GB
- **CPU:** 4 cores
- **RAM:** 9GB
- **OS:** Windows 10

### Benchmark Sonuçları

| Metrik | Değer |
|--------|-------|
| İlk Token Süresi | 0.5-1s |
| Token/Saniye | 25-30 |
| Ortalama Cevap Süresi | 3-8s |
| VRAM Kullanımı | ~4GB |
| Embedding Süresi | ~2s (328 chunks) |
| Vector Search | <0.5s |

### Örnek Çalıştırma

```
PDF yükleniyor...     
140 sayfa yüklendi
Metin chunk'lara bölünüyor...                               
328 chunk oluşturuldu                                      
Türkçe Embedding Modeli yükleniyor...            
Embedding modeli hazır                                
Vector database oluşturuluyor...   
Vector database hazır

Sorunuz: CHP genel başkanı nasıl seçilir? 
Benzerlik hesaplanıyor...
En benzer 3 bölüm bulundu
Benzerlik skorları: [0.77, 0.77, 0.77]
Lokal Qwen modeline gönderiliyor...

Cevap:
CHP genel başkanı, kurultayda gizli oyla ve üye tam sayısının 
salt çoğunluğuyla seçilir. İlk iki oylamada sonuç alınamazsa, 
üçüncü oylamanın en çok oy alan adayı seçilir.
```

---

## 📁 Proje Yapısı

```
turkish-government-intelligence-hub/
├── data/
│   ├── chp.pdf              # CHP Parti Tüzüğü
│   ├── akp.pdf              # (Eklenecek)
│   └── mhp.pdf              # (Eklenecek)
├── chroma_db/               # Vector database (otomatik oluşur)
├── rag_qwen_local.py        # Ana uygulama
├── requirements.txt         # Python bağımlılıkları
├── KURULUM.md              # Detaylı kurulum rehberi
└── README.md               # Bu dosya
```

---

## 🔧 Yapılandırma

### Chunk Boyutlarını Ayarlama

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # Chunk boyutu (256-1024 arası önerilir)
    chunk_overlap=50,      # Chunk overlap (10-20% chunk_size)
    length_function=len
)
```

### LLM Parametrelerini Ayarlama

```python
llm = Ollama(
    model="qwen2.5:7b-instruct-q4_K_M",
    temperature=0,         # Yaratıcılık (0.0-1.0)
    num_predict=512,       # Maksimum token
    num_ctx=4096          # Context window
)
```

### Farklı Model Kullanma

```bash
# Daha küçük model (daha hızlı)
ollama pull qwen2.5:3b-instruct-q4_K_M

# Daha büyük model (daha kaliteli)
ollama pull qwen2.5:14b-instruct-q4_K_M
```

Kod'da:
```python
llm = Ollama(model="qwen2.5:3b-instruct-q4_K_M")  # Küçük model
```

---

## 🐛 Sorun Giderme

### Model Bulunamıyor Hatası

```bash
# Modeli kontrol et
ollama list

# Modeli tekrar indir
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### GPU Kullanılmıyor

```bash
# GPU durumunu kontrol et
nvidia-smi

# Ollama'yı restart et (Windows Services)
```

### VRAM Yetersiz

```python
# Daha küçük model kullan
llm = Ollama(model="qwen2.5:3b-instruct-q4_K_M")

# Veya chunk sayısını azalt
top_k = 2  # 3 yerine 2 chunk kullan
```

---

## 🚧 Gelecek Özellikler

- [ ] Multi-party comparison (Partileri karşılaştırma)
- [ ] Streamlit web UI
- [ ] Conversation history (Sohbet geçmişi)
- [ ] Export to PDF/DOCX
- [ ] Voice interface (Sesli soru-cevap)
- [ ] Fine-tuned Turkish political model

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Pull request'lerinizi gönderin veya issue açın.

### Katkı Adımları

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit'leyin (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👤 Geliştirici

**Baran Can Ercan**

- 🌐 LinkedIn: [@barancanercan](https://www.linkedin.com/in/barancanercan)
- 📝 Medium: [@barancanercan](https://barancanercan.medium.com)
- 📧 Email: barancanercan@gmail.com
- 🐙 GitHub: [@barancanercan](https://github.com/barancanercan)

---

## 🙏 Teşekkürler

- [Ollama](https://ollama.com/) - Lokal LLM inference
- [LangChain](https://www.langchain.com/) - RAG framework
- [Alibaba Qwen Team](https://qwenlm.github.io/) - Qwen2.5 model
- [HuggingFace](https://huggingface.co/) - Turkish embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

## 📊 İstatistikler

![GitHub stars](https://img.shields.io/github/stars/barancanercan/turkish-government-intelligence-hub?style=social)
![GitHub forks](https://img.shields.io/github/forks/barancanercan/turkish-government-intelligence-hub?style=social)
![GitHub issues](https://img.shields.io/github/issues/barancanercan/turkish-government-intelligence-hub)

---

<div align="center">

**"Verilerle Aydınlanan Siyaset"** 🏛️

Made with ❤️ by [Baran Can Ercan](https://github.com/barancanercan)

</div>