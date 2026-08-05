"""
Builds and persists two indexes over the same chunk set so retrieval can be
hybrid:

* A dense vector index (Chroma + all-MiniLM-L6-v2 embeddings) for semantic
  similarity search.
* A sparse BM25 index (rank_bm25) for exact keyword / terminology matches,
  which academic text (method names, dataset names, acronyms) really
  benefits from.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """instantiate the local embedding model"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vectorstore(chunks: List[Document]) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25_index(chunks: List[Document]) -> None:
    corpus_tokens = [_tokenize(d.page_content) for d in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)
    logger.info("BM25 index built with %d documents -> %s", len(chunks), config.BM25_INDEX_PATH)


def load_bm25_index() -> dict:
    with open(config.BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def save_docstore(chunks: List[Document]) -> None:
    with open(config.DOCSTORE_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_docstore() -> List[Document]:
    with open(config.DOCSTORE_PATH, "rb") as f:
        return pickle.load(f)


def build_all_indexes(chunks: List[Document]) -> None:
    """Convenience wrapper used by scripts/build_index.py."""
    if not chunks:
        raise ValueError(
            "No chunks to index. Add PDFs to data/pdfs/ and check data/skipped_pdfs.json."
        )
    logger.info("Building Chroma vector index (%d chunks)...", len(chunks))
    build_vectorstore(chunks)
    logger.info("Building BM25 keyword index...")
    build_bm25_index(chunks)
    save_docstore(chunks)
    logger.info("All indexes built successfully.")
