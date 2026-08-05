"""
The single entry point the Streamlit app (and the evaluation script) calls.
Ties together: query routing -> intent-specific retrieval strategy ->
cross-encoder re-ranking -> Gemini generation with citations.
"""
from __future__ import annotations

import logging

from src import generation, retrieval, router
from src.generation import RagResult

logger = logging.getLogger(__name__)


def answer_question(question: str) -> RagResult:
    decision = router.route(question)
    logger.info("Routed '%s' -> %s (paper_hint=%s)", question, decision.query_type, decision.paper_hint)

    if decision.query_type == "SUMMARY" and decision.paper_hint:
        docs = retrieval.retrieve_for_summary(decision.paper_hint, question)
    elif decision.query_type == "COMPARISON" and decision.sub_questions:
        docs = retrieval.retrieve_multi(decision.sub_questions)
    else:
        # FACT, GENERAL, or COMPARISON/SUMMARY without enough info to
        # specialize -> standard high-precision hybrid retrieval.
        docs = retrieval.retrieve(question, paper_hint=decision.paper_hint)

    result = generation.generate_answer(question, docs)
    result.query_type = decision.query_type  # type: ignore[attr-defined]
    return result
