# RAG_Weaviate 🔬

A **Multimodal RAG** (Retrieval-Augmented Generation) system for scientific papers and PDF documents. Combines visual page understanding with semantic text search to answer questions with full context — text and images together.

---

## How It Works

```
PDF Upload
    ↓
PyMuPDF
    ├── Page Images → colSmol-500M (CPU) → Weaviate [Pages]
    └── Raw Text → Semantic Chunking → all-MiniLM-L6-v2 → Weaviate [Chunks]

User Question
    ↓
Dual Embedding (colSmol + MiniLM)
    ↓
Weaviate Hybrid Search (BM25 + Vector)
    ↓
Gemini 2.0 Flash → (fallback) OpenRouter → (fallback) Groq
    ↓
Answer + Page References
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| Image Embedding | `vidore/colSmol-500M` (local, CPU) |
| Text Embedding | `all-MiniLM-L6-v2` (local, CPU) |
| Chunking | Semantic (sentence similarity) |
| Vector DB | Weaviate Cloud (MultiVectors for ColPali) |
| Generation | Gemini 2.0 Flash → OpenRouter → Groq |
| UI | Streamlit |

---

## Requirements

- Python 3.11
- No GPU required — runs fully on CPU
- ~5 GB RAM minimum

---

## Setup

**1. Clone and create environment**
```bash
git clone https://github.com/Marwan-ALMasrat/RAG_Weaviate.git
cd RAG_Weaviate

py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file in the project root and add your keys:

```env
GEMINI_API_KEY=...       # aistudio.google.com (free)
GROQ_API_KEY=...         # console.groq.com (free)
WEAVIATE_URL=...         # console.weaviate.cloud (free sandbox)
WEAVIATE_API_KEY=...
OPENROUTER_API_KEY=...   # openrouter.ai (free tier)
```

**3. Run**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. Upload a PDF using the sidebar
2. Wait for ingestion (page embedding takes ~5-8 sec/page on CPU)
3. Ask questions in the chat
4. See retrieved page images alongside the answer

---

## Project Structure

```
RAG_Weaviate/
├── app.py                     # Streamlit UI
├── config.py                  # All settings in one place
├── requirements.txt
│
├── ingestion/
│   ├── pdf_processor.py       # PyMuPDF → page images + text
│   ├── chunker.py             # Semantic chunking
│   └── embedder.py            # colSmol-500M + all-MiniLM
│
├── db/
│   ├── weaviate_client.py     # Connection + collection setup
│   └── ingestion_pipeline.py  # Full ingestion orchestration
│
├── retrieval/
│   └── searcher.py            # Hybrid search (BM25 + vector)
│
├── generation/
│   ├── gemini_client.py       # Primary LLM
│   ├── openrouter_client.py   # Fallback 1
│   ├── groq_client.py         # Fallback 2
│   └── generator.py           # Automatic fallback chain
│
└── memory/
    └── chat_history.py        # Multi-turn conversation context
```

---

## Free Tier Limits

| Service | Free Limit |
|---------|-----------|
| Gemini 2.0 Flash | 15 req/min · 1,500/day |
| OpenRouter (Gemma 4 31B) | Free tier available |
| Groq (Llama 4 Scout) | 30 req/min · 7,000/day |
| Weaviate Cloud | 14-day sandbox |

---

## Performance (CPU · Intel Core i3 · 20GB RAM)

| Operation | Time |
|-----------|------|
| colSmol-500M load | ~45 sec (first time) |
| Page embedding | ~4-8 sec/page |
| Semantic chunking | ~0.5 sec/page |
| Weaviate search | <1 sec |
| Generation | ~2-3 sec |

---

## Notes

- `.env` is excluded from git — never commit your API keys
- Models are cached in `./hf_model_cache` after first download
- Uploaded PDFs and extracted images are excluded from git