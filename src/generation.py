"""
Builds the final prompt from retrieved chunks (each carrying paper title +
page number metadata), calls Gemini, and returns both the answer text and
a structured citation map so the Streamlit UI can render
`[Paper Title, p. X]` as a clickable button that opens the right PDF page
in the right-hand preview panel.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
        )
    return _llm


SYSTEM_PROMPT = """You are a research assistant helping a student navigate a folder of \
academic papers (a thesis/research "related work" library). Answer strictly using the \
provided context chunks. Rules:

1. Every factual claim must end with a citation tag in the exact form [Paper Title, p. X], \
   using the paper title and page number given in the chunk's context header.
2. If different chunks disagree or come from different papers, cite each separately.
3. If the answer requires combining multiple papers, synthesize clearly and cite each source.
4. If the context does not contain the answer, say so plainly. Do not invent facts, numbers, \
   citations, or paper titles that are not present in the context.
5. Be precise about experimental results, numbers, and methodology when asked - quote figures \
   exactly as they appear in the context.
6. Keep the answer well-structured (short paragraphs / bullet points where helpful).
"""

USER_TEMPLATE = """Question: {question}

Context chunks (each preceded by its [Paper / Section / Page] header):
---
{context}
---

Answer the question using only the context above, citing as instructed."""

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?),\s*p\.\s*(\d+)\]")


def _extract_text(content) -> str:
    # Normalize an AIMessage.content value to plain text.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Skip non-text blocks (e.g. "thinking"/"thought" parts)
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


@dataclass
class RagResult:
    answer: str
    citations: List[Dict] = field(default_factory=list)  # [{id, paper_title, page, source, snippet}]
    source_chunks: List[Document] = field(default_factory=list)


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _build_citation_map(docs: List[Document]) -> Dict[str, Dict]:
    """Map 'Paper Title|page' -> metadata, used to resolve citation tags in
    the answer to an actual file path + page for the PDF preview panel."""
    mapping = {}
    for d in docs:
        key = f"{d.metadata.get('paper_title')}|{d.metadata.get('page')}"
        if key not in mapping:
            mapping[key] = {
                "paper_title": d.metadata.get("paper_title"),
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source"),
                "filename": d.metadata.get("filename"),
                "snippet": d.page_content[:400],
            }
    return mapping


def generate_answer(question: str, docs: List[Document]) -> RagResult:
    if not docs:
        return RagResult(
            answer="I couldn't find relevant passages in the indexed papers to answer this. "
            "Try rephrasing, naming the paper explicitly, or check that the relevant PDF "
            "was not skipped as a scanned document (see data/skipped_pdfs.json).",
            citations=[],
            source_chunks=[],
        )

    llm = get_llm()
    context = _format_context(docs)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE.format(question=question, context=context)),
    ]
    response = llm.invoke(messages)
    answer_text = _extract_text(response.content)

    citation_map = _build_citation_map(docs)
    found = []
    for match in CITATION_PATTERN.finditer(answer_text):
        title, page = match.group(1).strip(), match.group(2)
        key = f"{title}|{page}"
        meta = citation_map.get(key)
        if meta:
            found.append(meta)

    # De-duplicate while preserving order
    dedup = []
    seen_keys = set()
    for c in found:
        k = (c["paper_title"], c["page"])
        if k not in seen_keys:
            seen_keys.add(k)
            dedup.append(c)

    return RagResult(answer=answer_text, citations=dedup, source_chunks=docs)
