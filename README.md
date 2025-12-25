# 🇹🇷 Turkish Government Intelligence Hub

**Production-ready multi-party RAG system for analyzing Turkish political party documents with 68x faster initialization**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: clean](https://img.shields.io/badge/Code%20style-clean-black)](https://github.com/barancanercan/Turkish-Government-Intelligence-Hub)

---

## 🎯 Overview

A sophisticated Question-Answering system that enables users to interact with Turkish political party constitutions through natural language. Built with production-ready patterns, modular architecture, and optimized for performance with persistent vector databases.

### Key Highlights

- 🚀 **68x Faster Initialization**: 960s → 14s for 4 parties through persistent vector databases
- 🔄 **Multi-Party Support**: CHP, AKP, MHP, İYİ Parti with instant switching (<0.1s)
- 📊 **1,505 Document Chunks**: Processed from 490 pages of political constitutions
- ⚡ **Sub-second Retrieval**: <0.5s similarity search across all documents
- 🏗️ **Production-Ready**: Modular architecture with separation of concerns
- 💾 **Fully Local**: No external API calls, complete privacy
- 🇹🇷 **Turkish-Optimized**: Custom Turkish embeddings and prompts

---

## ✨ Features

### Core Capabilities

- **Intelligent Q&A**: Ask questions about any Turkish political party's constitution
- **Party Switching**: Seamlessly switch between parties mid-conversation
- **Context-Aware**: RAG system retrieves relevant context before answering
- **CLI Interface**: Professional command-line interface with argparse
- **Persistent Storage**: Vector databases cached for instant reuse
- **Error Handling**: Comprehensive logging and error management

### Technical Features

- **Modular Design**: Separate data preparation from inference
- **Configuration Management**: Centralized config for easy customization
- **Deprecation-Free**: Latest LangChain packages (langchain-chroma, langchain-ollama)
- **Type Hints**: Partial type annotations for better code quality
- **Clean Code**: Following Python best practices

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION (One-time, ~8-12 min)             │
│     prepare_data.py                                      │
│     ├── Load PDFs (4 parties)                           │
│     ├── Split into chunks (512 chars, 50 overlap)       │
│     ├── Generate embeddings (Turkish BGE-M3)            │
│     └── Store in vector DBs (ChromaDB)                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. QUERY SYSTEM (Fast, ~5-14s initial load)            │
│     query_system.py                                      │
│     ├── Load vector DBs (5s)                            │
│     ├── Initialize LLM (Qwen2.5-7B)                     │
│     ├── Process queries (3-8s each)                     │
│     └── Multi-party switching (instant)                 │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Turkish-Government-Intelligence-Hub/
├── data/                          # Party constitution PDFs
│   ├── chp.pdf
│   ├── akp.pdf
│   ├── mhp.pdf
│   └── iyi.pdf
│
├── vector_db/                     # Persistent vector databases (gitignored)
│   ├── chp_db/
│   ├── akp_db/
│   ├── mhp_db/
│   └── iyi_db/
│
├── src/
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Reusable utility functions
│   ├── prepare_data.py            # Data preparation script
│   └── query_system.py            # Main Q&A interface
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** (recommended, 6GB+ VRAM) or CPU
- **Ollama** (for local LLM)
- **10GB+ Disk Space**

### Installation

**Step 1: Install Ollama**

```bash
# Download from https://ollama.com/download/windows
# Then pull the Qwen model:
ollama pull qwen2.5:7b-instruct-q4_K_M
```

**Step 2: Clone Repository**

```bash
git clone https://github.com/barancanercan/Turkish-Government-Intelligence-Hub.git
cd Turkish-Government-Intelligence-Hub
```

**Step 3: Setup Python Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 4: Prepare Data (One-time, ~8-12 minutes)**

```bash
cd src

# Check status
python prepare_data.py --status

# Prepare all parties
python prepare_data.py

# Or prepare single party
python prepare_data.py --party CHP
```

**Step 5: Start Ollama Server**

```bash
# In a separate terminal
ollama serve
```

**Step 6: Run Query System**

```bash
# Single party mode
python query_system.py --party CHP

# Multi-party mode
python query_system.py
```

---

## 💡 Usage Examples

### Single Party Mode

```bash
python query_system.py --party CHP
```

**Example Session:**
```
🔴 Sorunuz: CHP genel başkanı nasıl seçilir?

Cevap:
CHP genel başkanı, kurultayda gizli oyla ve üye tam sayısının 
salt çoğunluğuyla seçilir. İlk iki oylamada sonuç alınamazsa, 
üçüncü oylamanın en çok oy alan adayı seçilir.
```

### Multi-Party Mode

```bash
python query_system.py
```

**Example Session:**
```
🔴 [CHP] Sorunuz: CHP ne zaman kuruldu?
Cevap: CHP, 9 Eylül 1923'te kurulmuştur.

🔴 [CHP] Sorunuz: /akp
✅ Parti değiştirildi: AKP

🟠 [AKP] Sorunuz: AKP'nin sembolü nedir?
Cevap: AKP'nin sembolü ampuldür.
```

### Commands

- `/chp`, `/akp`, `/mhp`, `/iyi` - Switch party
- `q`, `quit`, `exit` - Exit program

---

## 📊 Performance Metrics

### Initialization Time

| Mode | Old System | New System | Improvement |
|------|------------|------------|-------------|
| Single Party (CHP) | 240s | 10s | **24x faster** |
| All 4 Parties | 960s | 14s | **68x faster** |

### Query Performance

| Metric | Performance |
|--------|-------------|
| First Token | 0.5-1s |
| Token/Second | 25-30 |
| Response Time | 3-8s |
| Retrieval Time | <0.5s |
| Party Switch | <0.1s |

### Resource Usage

| Resource | Usage |
|----------|-------|
| VRAM | ~4GB (RTX 3060) |
| Disk Space | ~555 MB (vector DBs) |
| RAM | ~2GB |
| CPU Cores | 4 |

### Data Statistics

| Party | Pages | Chunks | Vector DB Size |
|-------|-------|--------|----------------|
| CHP | 140 | 328 | ~120 MB |
| AKP | 144 | 466 | ~170 MB |
| MHP | 152 | 377 | ~140 MB |
| İYİ | 54 | 334 | ~125 MB |
| **Total** | **490** | **1,505** | **~555 MB** |

---

## 🛠️ Tech Stack

### Core Framework
- **LangChain 0.3+** - LLM orchestration and RAG pipeline
- **Qwen2.5-7B-Instruct** - Local LLM (Alibaba Cloud)
- **Ollama** - Local LLM inference server

### Embeddings & Vector DB
- **Turkish BGE-M3** - Turkish-optimized embeddings
- **ChromaDB 0.5+** - Vector database with persistence
- **Sentence-Transformers** - Embedding model framework

### Document Processing
- **PyPDF 5.1+** - PDF parsing
- **LangChain Text Splitters** - Intelligent chunking

### ML/DL Frameworks
- **PyTorch 2.5+** - Deep learning backend
- **Transformers 4.46+** - HuggingFace transformers

---

## 🎯 Advanced Usage

### Data Preparation Options

```bash
# Check which parties are ready
python prepare_data.py --status

# Prepare specific party
python prepare_data.py --party MHP

# Force rebuild (overwrites existing)
python prepare_data.py --party CHP --force

# Prepare all parties
python prepare_data.py
```

### Query System Options

```bash
# Single party mode (faster initial load)
python query_system.py --party CHP

# Multi-party mode (load all parties)
python query_system.py

# Help
python query_system.py --help
```

### Configuration

Edit `src/config.py` to customize:

```python
# Model settings
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
EMBEDDING_MODEL = "nezahatkorkmaz/turkce-embedding-bge-m3"

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3

# Paths
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
```

---

## 🔬 How It Works

### RAG (Retrieval-Augmented Generation) Pipeline

1. **User Query** → "CHP genel başkanı nasıl seçilir?"
2. **Embedding** → Convert query to vector (768 dimensions)
3. **Similarity Search** → Find top-3 most similar chunks
4. **Context Building** → Combine retrieved chunks
5. **LLM Generation** → Generate answer using context
6. **Response** → Return formatted answer to user

### Why Local LLM?

- ✅ **Privacy**: No data sent to external APIs
- ✅ **Cost**: Zero API costs
- ✅ **Offline**: Works without internet
- ✅ **Control**: Full control over model behavior
- ✅ **Speed**: No network latency

---

## 📈 Roadmap

### Phase 1: Core System ✅
- [x] Multi-party RAG system
- [x] Persistent vector databases
- [x] CLI interface
- [x] Production-ready architecture

### Phase 2: UI & Features (In Progress)
- [ ] Streamlit web interface
- [ ] Conversation history
- [ ] Export to PDF/DOCX
- [ ] Comparative analysis (compare parties)

### Phase 3: Advanced Features
- [ ] Fine-tuned Turkish political LLM
- [ ] Multi-turn conversations with memory
- [ ] Source citation with page numbers
- [ ]党 similarity analysis

### Phase 4: Deployment
- [ ] Docker containerization
- [ ] REST API (FastAPI)
- [ ] CI/CD pipeline
- [ ] Cloud deployment option

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repo
git clone https://github.com/barancanercan/Turkish-Government-Intelligence-Hub.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes and test
python src/query_system.py --party CHP

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Keep functions small and focused
- Write clean, readable code

---

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{ercan2024turkish_gov_intel,
  author = {Ercan, Baran Can},
  title = {Turkish Government Intelligence Hub: Multi-Party RAG System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/barancanercan/Turkish-Government-Intelligence-Hub}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Baran Can Ercan**

- 🌐 LinkedIn: [@barancanercan](https://www.linkedin.com/in/barancanercan)
- 📝 Medium: [@barancanercan](https://barancanercan.medium.com)
- 📧 Email: barancanercan@gmail.com
- 🐙 GitHub: [@barancanercan](https://github.com/barancanercan)

**Position:** Senior Data Scientist @ Ankara Metropolitan Municipality  
**Focus:** NLP, LLM Engineering, RAG Systems, Turkish Language Processing

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM inference
- [LangChain](https://www.langchain.com/) - RAG framework
- [Alibaba Qwen Team](https://qwenlm.github.io/) - Qwen2.5 model
- [HuggingFace](https://huggingface.co/) - Turkish embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

## 🐛 Known Issues

- Emoji rendering issues in Windows PowerShell (cosmetic only)
- Some deprecation warnings (will be fixed in next release)

---

## ❓ FAQ

**Q: Do I need an API key?**  
A: No! The system is completely local. No API keys needed.

**Q: Can I add more parties?**  
A: Yes! Just add the PDF to `data/` and run `python prepare_data.py --party PARTYNAME`

**Q: How much VRAM do I need?**  
A: Recommended 6GB+, but it works on CPU too (slower).

**Q: Can I use a different LLM?**  
A: Yes! Edit `config.py` and change `LLM_MODEL` to any Ollama model.

**Q: Is this production-ready?**  
A: Yes for local use. For web deployment, add authentication and rate limiting.

---

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/barancanercan/Turkish-Government-Intelligence-Hub/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/barancanercan/Turkish-Government-Intelligence-Hub/discussions)
- 📧 **Email**: barancanercan@gmail.com

---

<div align="center">

**"Harnessing AI for Political Transparency"** 🇹🇷

⭐ Star this repo if you find it useful!

Made with ❤️ by [Baran Can Ercan](https://github.com/barancanercan)

</div>