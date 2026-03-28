# 📊 Financial RAG Assistant

> A fully local Retrieval-Augmented Generation (RAG) system for querying SEC 10-K filings — no OpenAI API key required.

Built with **MiniLM embeddings**, **FAISS vector search**, **Ollama LLM**, and an agentic **tool-calling** loop that lets the model decide when to search, calculate, or compare.
> A lightweight, fully local RAG system designed for learning and experimentation.  
> Released under the MIT License for maximum flexibility and reuse.
---

## ✨ Features

- 🔍 **Semantic search** over real SEC 10-K filings (AAPL, MSFT, NVDA)
- 🤖 **Fully local** — MiniLM for embeddings, Ollama (llama3.2) for generation
- 🔧 **Tool calling** — LLM autonomously selects from 3 tools:
  - `search_filings` — vector search over indexed filings
  - `calculate` — growth rates, margins, ratios
  - `compare_companies` — side-by-side metric comparison
- 🧹 **HTML-aware ingestion** — decodes SEC HTML entities, strips tags, extracts year from document content
- ⚡ **FAISS IndexFlatIP** — exact cosine similarity search over 68k+ vectors
- 🖥️ **Terminal-first** — interactive Q&A in the terminal, no UI dependency

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│         Ollama llama3.2         │  ← decides which tool to call
│     (Tool-Calling Agent)        │
└────────────┬────────────────────┘
             │ tool_calls
     ┌───────┼───────────┐
     ▼       ▼           ▼
search_  calculate  compare_
filings             companies
     │
     ▼
┌─────────────────────────────────┐
│   MiniLM Embedder (local)       │  ← encodes query
└────────────┬────────────────────┘
             │ query vector
             ▼
┌─────────────────────────────────┐
│   FAISS Index (68k vectors)     │  ← top-k retrieval
└────────────┬────────────────────┘
             │ chunks + metadata
             ▼
┌─────────────────────────────────┐
│   Ollama llama3.2               │  ← generates final answer
│   (Answer Generation)           │     with [Source N] citations
└─────────────────────────────────┘
```

---

## 📁 File Structure

```
financial-rag/
│
├── build_index.py          # One-shot script: download → chunk → embed → save FAISS index
├── test_rag.py             # Terminal interface for interactive Q&A and auto testing
│
├── src/
│   ├── app.py              # (Optional) Gradio web UI
│   ├── chain.py            # Core RAG chain: tool-calling loop + tool implementations
│   ├── chunker.py          # Text chunking strategies (fixed, recursive, token, semantic)
│   ├── embedder.py         # Embedding interface (MiniLM local / OpenAI)
│   ├── evaluator.py        # RAGAS + classical IR metrics (Precision@k, MRR, NDCG)
│   ├── ingest.py           # SEC EDGAR downloader + HTML cleaner + section parser
│   ├── retriever.py        # High-level retrieval: dense search + MMR deduplication
│   └── vectorstore.py      # FAISSStore and PineconeStore implementations
│
├── data/
│   ├── sec_filings/        # Raw downloaded SEC filings (auto-created)
│   └── indices/
│       ├── financial_index.faiss       # FAISS binary index
│       └── financial_index_meta.json   # Metadata sidecar (company, year, section, text)
│
├── tests/
│   └── eval_dataset.json   # Sample evaluation questions + ground truth
│
├── .env                    # Environment variables (OPENAI_API_KEY optional)
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup

### 1. Clone and create environment

```bash
git clone https://github.com/yourname/financial-rag.git
cd financial-rag
conda create -n financial-rag python=3.11 -y
conda activate financial-rag
```

### 2. Install dependencies

```bash
# FAISS — install via conda for best ARM/macOS compatibility
conda install -c conda-forge faiss-cpu=1.8.0 -y

# Python packages
pip install -r requirements.txt
```

### 3. Install and start Ollama

```bash
# Download from https://ollama.com, then:
ollama pull llama3.2
# Ollama starts automatically as a background service
```

### 4. Environment variables (optional)

```bash
cp .env.example .env
# Only needed if using OpenAI embeddings instead of MiniLM
# OPENAI_API_KEY=sk-...
```

---

## 🚀 Usage

### Step 1 — Build the index

Download and index SEC 10-K filings for one or more companies:

```bash
# Full run — downloads real filings from SEC EDGAR
TRANSFORMERS_OFFLINE=1 python build_index.py \
  --tickers AAPL MSFT NVDA \
  --strategy recursive \
  --chunk-size 512 \
  --overlap 64 \
  --embed-model minilm

# Quick demo — uses built-in synthetic data, no download needed
python build_index.py --demo --tickers AAPL MSFT NVDA
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | `AAPL MSFT NVDA` | Companies to index |
| `--strategy` | `recursive` | Chunking: `fixed` / `recursive` / `token` / `semantic` |
| `--chunk-size` | `512` | Target chunk size in characters |
| `--overlap` | `64` | Overlap between chunks |
| `--embed-model` | `minilm` | `minilm` (local) or `openai-small` (API) |

### Step 2 — Run the terminal assistant

```bash
# Interactive mode — type questions, get answers
TRANSFORMERS_OFFLINE=1 python test_rag.py

# Auto mode — runs preset benchmark questions
TRANSFORMERS_OFFLINE=1 python test_rag.py --auto --k 3
```

**Example session:**

```
You: What was Apple's total revenue in fiscal year 2023?

  🔧 search_filings({"query": "Apple total revenue FY2023", "company": "AAPL", "k": 3})

💬 Answer:
According to [Source 1] (AAPL 2023 — Financial Statements), Apple's total
net sales for fiscal year 2023 were $383.3 billion, compared to $394.3
billion in fiscal year 2022, a decrease of approximately 3 percent.

⏱  12.4s total
```

```
You: Calculate Apple's revenue growth from 2022 to 2023.

  🔧 calculate({"operation": "growth_rate", "value_a": 383.3, "value_b": 394.3})

💬 Answer:
Apple's revenue declined by 2.79% from FY2022 to FY2023
(from $394.3B to $383.3B).

⏱  2.1s total
```

```
You: Compare Apple and Microsoft gross margins.

  🔧 compare_companies({"company_a": "AAPL", "company_b": "MSFT", "metric": "gross margin"})

💬 Answer:
• Apple gross margin: 44.1% (FY2023), up from 43.3% in FY2022
• Microsoft gross margin: 69% (FY2023), up from 68% in FY2022
Microsoft's higher margin reflects its cloud-heavy revenue mix.

⏱  14.7s total
```

---

## 🔧 Tool Calling

The LLM autonomously selects tools based on the question type:

| Tool | Triggered by | What it does |
|------|-------------|-------------|
| `search_filings` | Factual questions about filings | Vector search over FAISS index |
| `calculate` | "calculate", "growth", "margin", "ratio" | Python math on provided numbers |
| `compare_companies` | "compare X vs Y" | Parallel search for both tickers |

The agent loop runs up to 6 rounds — tool results are fed back to the LLM as context until it produces a final text answer.

---

## 📐 Key Modules

### `chain.py` — Agentic RAG loop
- Sends question + tool definitions to Ollama `/api/chat`
- Detects `tool_calls` in response and dispatches to local Python functions
- Handles llama3.2 quirk where it returns raw JSON instead of structured tool calls
- Loops until LLM produces a plain-text final answer

### `vectorstore.py` — FAISSStore
- Wraps `faiss.IndexFlatIP` for exact cosine similarity (via L2-normalized inner product)
- Saves/loads as `.faiss` binary + `_meta.json` sidecar
- Supports optional metadata `filter_fn` for ticker/year filtering

### `chunker.py` — Chunking strategies
- `fixed` — character-count windows
- `recursive` — splits on `\n\n`, `\n`, `. ` in order
- `token` — splits by token count (requires tiktoken)
- `semantic` — groups sentences by embedding similarity

### `ingest.py` — SEC filing parser
- Downloads filings via `sec-edgar-downloader`
- Strips HTML tags, decodes entities (`&#8217;` → `'`)
- Extracts fiscal year from document text (not folder name)
- Splits into sections by Item number (Item 1, 1A, 7, 8, etc.)

### `evaluator.py` — Evaluation metrics
- Classical IR: `Precision@k`, `Recall@k`, `MAP`, `MRR`, `NDCG`
- RAGAS suite: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`

---

## 🧪 Evaluation

```bash
# Run classical retrieval metrics
python src/evaluator.py

# Run RAGAS evaluation (requires eval_dataset.json)
python -c "
from src.chain import FinancialRAGChain
from src.evaluator import run_ragas
rag = FinancialRAGChain('data/indices/financial_index')
run_ragas(rag)
"
```

---

## 📦 Requirements

```
sentence-transformers
faiss-cpu          # install via conda, not pip
langchain-core
langchain-ollama
langchain-community
python-dotenv
gradio             # optional, for web UI
sec-edgar-downloader
pypdf
requests
numpy
```

---

## 🗺️ Roadmap

- [ ] Fix year metadata extraction accuracy
- [ ] Add BM25 hybrid search (dense + sparse)
- [ ] Support more tickers beyond AAPL/MSFT/NVDA
- [ ] Add streaming output to terminal
- [ ] Evaluation dashboard with RAGAS scores

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Answers are grounded in indexed SEC filings but may be incomplete or inaccurate. Do not use for investment decisions.

---

## 📄 License

MIT