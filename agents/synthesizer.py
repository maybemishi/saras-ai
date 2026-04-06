"""
agents/synthesizer.py — Synthesizer Agent

Responsibility:
  Take all verified, analyst-approved text chunks and sources and
  produce a structured academic report in Markdown with:
    - A clear introduction
    - Themed sections based on the research plan
    - Inline citations in [Source N] format
    - A conclusion
"""

from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from state import ResearchState

SYSTEM_PROMPT = """You are the Synthesizer agent in SARAS, an autonomous research system.

Your job is to produce a high-quality, structured academic research report in Markdown.

You will receive:
- The original research query
- Verified research material (text chunks from web and documents)
- The research plan (sub-tasks that were investigated)
- Analyst notes on data quality

Your report MUST:
1. Start with a ## Introduction section
2. Have one ## section per sub-task from the research plan
3. Use inline citations like [Source 1], [Source 2] wherever you reference material
4. End with a ## Conclusion section
5. End with a ## References section listing each [Source N] with its title and URL

Format every section with proper Markdown headings.
Write in a formal, academic tone.
Do NOT hallucinate facts — only use what is in the provided material.
If the material on a sub-task is thin, say so explicitly in that section.
"""


def _build_llm() -> ChatGoogleGenerativeAI:
    # Moderate thinking budget — synthesis needs creativity but not deep reasoning
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3,
        thinking_budget=1024,
    )


def _prepare_material(state: ResearchState) -> str:
    """Build numbered source material for the synthesizer."""
    lines = []
    seen_urls = set()
    source_index = 1
    for source in state.raw_sources:
        if source.url in seen_urls:
            continue
        seen_urls.add(source.url)
        lines.append(
            f"[Source {source_index}] ({source.origin.upper()}) "
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Content: {source.snippet}"
        )
        source_index += 1
    return "\n\n".join(lines)


def synthesizer_node(state: ResearchState) -> dict:
    """
    LangGraph node: Synthesizer.
    Reads:  state.query, state.raw_sources, state.raw_text_chunks,
            state.research_plan, state.contradiction_notes
    Writes: state.report_markdown, state.current_step
    """
    llm = _build_llm()
    material = _prepare_material(state)
    plan_str = "\n".join(f"- {t}" for t in state.research_plan)

    user_content = (
        f"Research query: {state.query}\n\n"
        f"Research plan (sub-tasks investigated):\n{plan_str}\n\n"
        f"Analyst notes: {state.contradiction_notes or 'None'}\n\n"
        f"Verified source material:\n{material}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        report = response.content.strip()
    except Exception as e:
        report = (
            f"# Research Report\n\n"
            f"**Error during synthesis:** {str(e)}\n\n"
            f"Query: {state.query}"
        )

    return {
        "report_markdown": report,
        "current_step": "synthesis_complete",
    }
