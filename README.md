# 📚 Academic Literature RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system for research/thesis
"related work" folders. Drop all the reference PDFs into a folder, ask questions in a chat UI, and
get answers with **clickable citations** that open the exact cited page in a live PDF
preview panel — no more manually re-reading or re-uploading PDFs to a chatbot.

Built entirely on **free** infrastructure:

| Component        | Choice                                   | Why |
|-------------------|-------------------------------------------|-----|
| LLM               | Google **Gemini** (`gemini-3.5-flash`)    | Generous free tier |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2`  | Local, free, no API key |
| Vector DB         | **ChromaDB**                              | Free, local, persists to disk |
| Keyword search    | `rank_bm25`                               | Free, local |
| Re-ranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`    | Free, local (replaces paid Cohere Rerank) |
| UI                | **Streamlit**                             | Free, deployable to Streamlit Community Cloud |
| PDF parsing/render| **PyMuPDF**                               | Free, fast |

I built this **project-based learning** for the concepts in
LangChain's [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) notebook series (indexing, hybrid retrieval,
query routing, decomposition, re-ranking). See [Concepts & where they live](#concepts--where-they-live-in-this-repo)
below to map each notebook idea to the actual production code.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)

---

## How it works

```
                 ┌──────────────────────────────────────────┐
   PDFs  ──────▶ │ 1. Ingestion (src/ingestion.py)           │
 (data/pdfs/)    │    - extract text per page (PyMuPDF)      │
                 │    - flag & SKIP scanned/no-text PDFs     │
                 │    - detect section headings              │
                 │    - chunk (RecursiveCharacterTextSplitter)│
                 │    - prepend CONTEXTUAL HEADER to chunk   │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 2. Indexing (src/indexing.py)             │
                 │    - Chroma vector index (MiniLM embeds)  │
                 │    - BM25 keyword index                   │
                 └───────────────────┬────────────────────────┘
                                     ▼
 Question ─────▶ ┌──────────────────────────────────────────┐
                 │ 3. Router (src/router.py)                  │
                 │    Gemini classifies: FACT / SUMMARY /     │
                 │    COMPARISON / GENERAL, extracts paper    │
                 │    hint, decomposes comparisons             │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 4. Retrieval (src/retrieval.py)            │
                 │    FACT       → hybrid search, top 20      │
                 │                 candidates → rerank → top 8│
                 │    SUMMARY    → wide per-paper pool → rerank│
                 │    COMPARISON → retrieve per sub-question,  │
                 │                 merge, rerank                │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 5. Generation (src/generation.py)          │
                 │    Gemini answers ONLY from retrieved       │
                 │    chunks, citing [Paper Title, p. X]       │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 6. Streamlit UI (app.py)                   │
                 │    Left: chat + clickable citation buttons │
                 │    Right: rendered PDF page with the cited │
                 │           snippet highlighted               │
                 └──────────────────────────────────────────┘
```

## Project structure

```
academic-rag/
├── app.py                  # Streamlit UI
├── config.py               
├── requirements.txt
├── .env.example
├── LICENSE                 
├── src/
│   ├── ingestion.py         # PDF text extraction, scanned-PDF filter, contextual headers, chunking
│   ├── indexing.py          # Chroma + BM25 index build/load
│   ├── retrieval.py         # Hybrid search (ensemble) + cross-encoder re-ranking
│   ├── router.py            # Query classification + decomposition (Gemini structured output)
│   ├── generation.py        # Prompting + citation parsing
│   └── pipeline.py          # Orchestrates router → retrieval → generation
├── scripts/
│   ├── build_index.py       # (Re)build the indexes from data/pdfs/
│   └── evaluate.py          # Quantifies retrieval quality improvement
├── eval/
│   └── sample_qa.json       # Template eval set (replace with your own Q/A pairs)
└── data/
    ├── pdfs/                 # PDFs storage
    └── chroma_db/            # generated vector index
```

## Installation

**Requirements:** Python 3.10–3.12.

```bash
git clone https://github.com/Misiker101/academic-rag-assistant.git

cd academic-rag-assistant

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

The first run will download the embedding model (~90 MB) and the re-ranker model
(~80 MB) from HuggingFace automatically (this needs internet access once, after
which they're cached locally).

## Configuration

1. Get a **free** Gemini API key: https://aistudio.google.com/app/apikey
2. Copy the env template and fill it in:

   ```bash
   cp .env.example .env
   ```

   ```env
   GOOGLE_API_KEY=your_actual_key_here
   ```

3. (Optional) Tune retrieval/chunking behavior in `.env` or `config.py` — chunk size,
   overlap, `TOP_K_CANDIDATES` (default 20, per your "high precision" requirement),
   `TOP_K_FINAL` (chunks actually sent to the LLM), hybrid search weighting, etc.

## Usage

1. **Add your PDFs.** Copy all your related-work / bibliography PDFs into `data/pdfs/`.

2. **Build the index** (run again any time you add/remove PDFs):

   ```bash
   python scripts/build_index.py
   ```

   This prints how many PDFs were indexed and how many were **skipped** (e.g. scanned
   image-only PDFs with no extractable text). Skipped files are logged to
   `data/skipped_pdfs.json` — using OCR for the future (e.g., `pytesseract`)

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (defaults to `http://localhost:8501`).

4. **Ask questions**, e.g.:
   - *"Summarize the main contribution of the XYZ paper."* → routed to **SUMMARY**
   - *"What accuracy did the ViT paper report on the trivia QA benchmark?"* → **FACT**
   - *"How do the re-ranking approaches in these papers differ from the query
     decomposition approaches?"* → **COMPARISON** (decomposed into sub-questions,
     retrieved per aspect, then synthesized)

   Click any `[Paper Title, p. X]` citation button under an answer to load that exact
   page in the right-hand preview panel, with the cited passage highlighted.