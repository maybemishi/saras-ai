"""
agents/web_scripter.py — Web Scripter Agent

Responsibility:
  Execute each sub-task in the research plan.
  For each task:
    1. Search the web via Tavily
    2. Retrieve relevant local chunks from ChromaDB (if docs were uploaded)
    3. Accumulate results into state.raw_sources and state.raw_text_chunks

On retry: uses state.retry_instruction instead of the original plan.
"""

from __future__ import annotations
from tavily import TavilyClient

from config import settings
from state import ResearchState, SourceRecord
from utils.pdf_parser import retrieve_relevant_chunks
from utils.credibility import score_url


def _get_tavily() -> TavilyClient:
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def _search_web(client: TavilyClient, task: str) -> list[SourceRecord]:
    """Run a single Tavily search and return structured SourceRecords."""
    try:
        response = client.search(
            query=task,
            max_results=settings.MAX_SEARCH_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception:
        return []

    sources = []
    for result in response.get("results", []):
        url = result.get("url", "")
        sources.append(
            SourceRecord(
                url=url,
                title=result.get("title", url),
                snippet=result.get("content", "")[:600],
                credibility_score=score_url(url),
                origin="web",
            )
        )
    return sources


def _retrieve_docs(task: str) -> tuple[list[SourceRecord], list[str]]:
    """Retrieve relevant chunks from ChromaDB and wrap as SourceRecords."""
    chunks = retrieve_relevant_chunks(task, n_results=5)
    sources, texts = [], []
    for chunk in chunks:
        sources.append(
            SourceRecord(
                url=chunk["source"],
                title=f"Uploaded: {chunk['source']}",
                snippet=chunk["text"][:400],
                credibility_score=0.85,
                origin="document",
            )
        )
        texts.append(chunk["text"])
    return sources, texts


def web_scripter_node(state: ResearchState) -> dict:
    """
    LangGraph node: Web Scripter.
    Reads:  state.research_plan, state.retry_instruction, state.has_uploaded_docs
    Writes: state.raw_sources (accumulated), state.raw_text_chunks (accumulated),
            state.current_step
    """
    client = _get_tavily()
    new_sources: list[SourceRecord] = []
    new_chunks: list[str] = []

    # On retry, use the analyst's targeted instruction
    if state.retry_instruction and state.retry_count > 0:
        tasks = [state.retry_instruction]
    else:
        tasks = state.research_plan

    for task in tasks:
        # --- Web search ---
        web_sources = _search_web(client, task)
        new_sources.extend(web_sources)
        new_chunks.extend([s.snippet for s in web_sources if s.snippet])

        # --- Document retrieval (only if docs were ingested) ---
        if state.has_uploaded_docs:
            doc_sources, doc_chunks = _retrieve_docs(task)
            new_sources.extend(doc_sources)
            new_chunks.extend(doc_chunks)

    return {
        "raw_sources": new_sources,       # operator.add will append
        "raw_text_chunks": new_chunks,    # operator.add will append
        "current_step": "scripter_complete",
        "error": None,
    }
