"""
Ingestion pipeline
===================
1. Walk the PDF folder.
2. For each PDF, extract text per page with PyMuPDF (fitz).
3. Skip PDFs that are mostly scanned images with no extractable text
   (logged to data/skipped_pdfs.json so the user knows what was excluded).
4. Detect the largest-font line on each page as a pseudo "section heading"
   (cheap, model-free heuristic) to build a contextual header per chunk,
   inspired by Anthropic's "contextual retrieval" idea: every chunk is
   prefixed with a short piece of context (paper title + nearest heading)
   before it is embedded, which measurably improves retrieval precision
   because the chunk is no longer a decontextualized fragment.
5. Split page text into overlapping chunks with LangChain's
   RecursiveCharacterTextSplitter.
6. Return a list of langchain Document objects, each carrying rich
   metadata (source path, paper title, page number, section heading)
   that the UI later uses to open the right PDF page.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed
    text: str
    heading: str


def _guess_paper_title(doc: "fitz.Document", filename: str) -> str:
    """Best-effort title guess: PDF metadata title, else largest text on page 1,
    else the filename."""
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if meta_title and len(meta_title) > 3:
        return meta_title

    try:
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        best_text, best_size = "", 0.0
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > best_size and len(span["text"].strip()) > 4:
                        best_size = span["size"]
                        best_text = span["text"].strip()
        if best_text:
            return best_text
    except Exception:
        pass

    return Path(filename).stem


def _extract_headings_per_page(page: "fitz.Page") -> str:
    """Return the largest-font text line on the page as a pseudo heading."""
    try:
        blocks = page.get_text("dict")["blocks"]
        best_text, best_size = "", 0.0
        for b in blocks:
            for line in b.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", [])).strip()
                if not line_text:
                    continue
                size = max((s["size"] for s in line.get("spans", [])), default=0)
                if size > best_size and len(line_text) < 120:
                    best_size = size
                    best_text = line_text
        return best_text
    except Exception:
        return ""


def _extract_pages(pdf_path: Path) -> List[PageExtraction]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            heading = _extract_headings_per_page(page)
            pages.append(PageExtraction(page_number=i + 1, text=text, heading=heading))
    return pages


def _is_scanned(pages: List[PageExtraction]) -> bool:
    if not pages:
        return True
    low_text_pages = sum(1 for p in pages if len(p.text.strip()) < config.MIN_CHARS_PER_PAGE)
    return (low_text_pages / len(pages)) >= config.SCANNED_PAGE_RATIO


def load_and_chunk_pdfs(pdf_dir: Path = config.PDF_DIR) -> List[Document]:
    """Main entry point used by scripts/build_index.py.

    Returns a list of chunked LangChain Documents ready for embedding, and
    writes a JSON report of any PDFs that were skipped (scanned / unreadable)
    to config.SKIPPED_LOG_PATH.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    skipped: List[dict] = []
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDFs found in %s", pdf_dir)

    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as doc:
                title = _guess_paper_title(doc, pdf_path.name)
            pages = _extract_pages(pdf_path)
        except Exception as e:  # noqa: BLE001
            skipped.append({"file": pdf_path.name, "reason": f"failed to open/parse: {e}"})
            continue

        if _is_scanned(pages):
            skipped.append(
                {
                    "file": pdf_path.name,
                    "reason": "appears to be a scanned/image-only PDF with little "
                    "or no extractable text (consider OCR-ing it first)",
                }
            )
            continue

        current_heading = title
        for page in pages:
            if page.heading:
                current_heading = page.heading
            if not page.text.strip():
                continue

            # Contextual header prepended to every chunk before embedding.
            # This is the "Add Contextual Headers" technique the user asked for:
            # a short prefix that re-situates an otherwise decontextualized
            # chunk within its paper and section, boosting retrieval precision.
            context_header = f"[Paper: {title} | Section: {current_heading} | Page: {page.page_number}]\n"

            splits = splitter.split_text(page.text)
            for j, chunk_text in enumerate(splits):
                all_chunks.append(
                    Document(
                        page_content=context_header + chunk_text,
                        metadata={
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "paper_title": title,
                            "page": page.page_number,
                            "section": current_heading,
                            "chunk_index": j,
                        },
                    )
                )

    config.SKIPPED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SKIPPED_LOG_PATH, "w") as f:
        json.dump(skipped, f, indent=2)

    logger.info(
        "Ingested %d PDFs into %d chunks. Skipped %d PDFs (see %s).",
        len(pdf_paths) - len(skipped),
        len(all_chunks),
        len(skipped),
        config.SKIPPED_LOG_PATH,
    )
    return all_chunks
