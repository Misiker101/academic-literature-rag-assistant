"""
Classifies each incoming question into one of four intents and, for
cross-paper comparisons, decomposes it into focused sub-questions.

Query types:
  * FACT        -> a specific, narrow fact/number/definition/experiment
                   result, usually scoped to one paper.
  * SUMMARY      -> "summarize X", "what does paper Y argue", broad
                   whole-paper or whole-concept synthesis.
  * COMPARISON   -> requires combining/contrasting information from
                   multiple papers.
  * GENERAL      -> anything else / small talk / doesn't need retrieval.
"""
from __future__ import annotations

import json
import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    query_type: Literal["FACT", "SUMMARY", "COMPARISON", "GENERAL"] = Field(
        description="The category of the user's question."
    )
    paper_hint: Optional[str] = Field(
        default=None,
        description="If the user named or clearly implied a specific paper/title/filename, extract it. Else null.",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Only for COMPARISON queries: 2-4 focused, self-contained sub-questions "
            "that together cover what needs to be retrieved from each relevant paper."
        ),
    )


_llm = None


def get_router_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
        )
    return _llm


ROUTER_SYSTEM_PROMPT = """You are a query router for an academic literature RAG system \
over a folder of research papers. Classify the user's question and, if it is a \
COMPARISON question (needs combining info from 2+ papers), break it into 2-4 \
focused sub-questions, one per aspect/paper needed. Respond ONLY with the \
structured output requested, no extra commentary."""


def route(question: str) -> RouteDecision:
    llm = get_router_llm().with_structured_output(RouteDecision)
    try:
        decision = llm.invoke(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", question),
            ]
        )
        return decision
    except Exception as e:  # noqa: BLE001
        logger.warning("Router failed (%s); falling back to FACT retrieval.", e)
        return RouteDecision(query_type="FACT", paper_hint=None, sub_questions=[])
