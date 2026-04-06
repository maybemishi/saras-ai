"""
agents/source_aggregator.py — Source Aggregator Agent

Responsibility:
  Post-process all accumulated raw_sources into a clean, ranked,
  deduplicated final source list and a formatted bibliography string.

  This is a deterministic (non-LLM) node — it just does data processing.
  No LLM call needed here; the ranking logic is rule-based.
"""

from __future__ import annotations
from state import ResearchState, SourceRecord
from utils.credibility import score_url, label_for_score


def source_aggregator_node(state: ResearchState) -> dict:
    """
    LangGraph node: Source Aggregator.
    Reads:  state.raw_sources
    Writes: state.final_sources, state.bibliography, state.current_step
    """
    # --- Deduplicate by URL ---
    seen_urls: set[str] = set()
    unique_sources: list[SourceRecord] = []
    for source in state.raw_sources:
        if source.url not in seen_urls:
            seen_urls.add(source.url)
            # Re-score to ensure credibility is set
            score = score_url(source.url)
            unique_sources.append(
                SourceRecord(
                    url=source.url,
                    title=source.title,
                    snippet=source.snippet,
                    credibility_score=score,
                    origin=source.origin,
                )
            )

    # --- Sort: documents first (researcher's own material), then by credibility ---
    unique_sources.sort(
        key=lambda s: (0 if s.origin == "document" else 1, -s.credibility_score)
    )

    # --- Build bibliography string ---
    bib_lines = ["## Bibliography\n"]
    for i, source in enumerate(unique_sources, 1):
        label = label_for_score(source.credibility_score)
        trust_badge = f"[{label} credibility]"
        if source.origin == "document":
            entry = f"{i}. **{source.title}** {trust_badge} *(Uploaded document)*"
        else:
            entry = f"{i}. **{source.title}** {trust_badge}  \n   {source.url}"
        bib_lines.append(entry)

    bibliography = "\n".join(bib_lines)

    return {
        "final_sources": unique_sources,
        "bibliography": bibliography,
        "current_step": "aggregation_complete",
    }
