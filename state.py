"""
state.py — Typed LangGraph state shared across all SARAS agents.

The ResearchState object is passed through every node in the graph.
Each agent reads from and writes to this state. LangGraph merges
updates automatically between node calls.
"""

from __future__ import annotations
from typing import Annotated, Optional
from pydantic import BaseModel, Field
import operator


class SourceRecord(BaseModel):
    """A single retrieved source with metadata."""
    url: str
    title: str
    snippet: str
    credibility_score: float = Field(
        default=0.5,
        description="0.0–1.0 trust score based on domain authority"
    )
    origin: str = Field(
        default="web",
        description="'web' or 'document' — where this source came from"
    )


class ResearchState(BaseModel):
    """
    Central state object for the SARAS graph.

    Fields with Annotated[list, operator.add] accumulate across nodes
    (LangGraph merges them by appending). All other fields are
    overwritten by whichever node last sets them.
    """

    # ── Input ─────────────────────────────────────────────────────────
    query: str = ""
    has_uploaded_docs: bool = False

    # ── Planner output ─────────────────────────────────────────────────
    research_plan: list[str] = Field(
        default_factory=list,
        description="Ordered list of sub-tasks decomposed from the query"
    )

    # ── Web Scripter output ────────────────────────────────────────────
    raw_sources: Annotated[list[SourceRecord], operator.add] = Field(
        default_factory=list,
        description="All retrieved sources (web + document). Accumulates across retries."
    )
    raw_text_chunks: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Raw text passages gathered per sub-task"
    )

    # ── Analyst output ─────────────────────────────────────────────────
    analyst_verdict: str = Field(
        default="pending",
        description="'approved' | 'retry' | 'pending'"
    )
    contradiction_notes: str = ""
    retry_instruction: str = Field(
        default="",
        description="Targeted instruction sent back to Web Scripter on retry"
    )
    retry_count: int = 0

    # ── Synthesizer output ─────────────────────────────────────────────
    report_markdown: str = ""

    # ── Source Aggregator output ───────────────────────────────────────
    final_sources: list[SourceRecord] = Field(
        default_factory=list,
        description="Deduplicated, ranked, credibility-scored final source list"
    )
    bibliography: str = Field(
        default="",
        description="Formatted bibliography string appended to the report"
    )

    # ── Pipeline control ───────────────────────────────────────────────
    error: Optional[str] = None
    current_step: str = "idle"

    class Config:
        arbitrary_types_allowed = True
