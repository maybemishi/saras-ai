"""
agents/analyst.py — Analyst Agent

Responsibility:
  The critical gatekeeper.
  Given all gathered text chunks and sources:
    1. Cross-reference claims across multiple sources
    2. Detect contradictions or gaps
    3. Decide: 'approved' (proceed to synthesis) OR 'retry' (send back to scripter)

Thinking mode is ON for this agent — it requires the deepest reasoning.
"""

from __future__ import annotations
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from state import ResearchState

SYSTEM_PROMPT = """You are the Analyst agent in SARAS, an autonomous research system.

Your job is to critically evaluate the gathered research material and decide
whether it is sufficient and consistent for a reliable academic report.

You will receive:
- The original research query
- A list of text snippets from web sources and uploaded documents

Your tasks:
1. Identify any CONTRADICTIONS between sources
2. Identify critical GAPS (important aspects of the query not covered)
3. Make a verdict

Respond with ONLY a JSON object in this exact format:
{
  "verdict": "approved" | "retry",
  "contradiction_notes": "Summary of contradictions found, or 'None detected'",
  "retry_instruction": "If retry: a single specific targeted search query to fill the gap. If approved: empty string."
}

IMPORTANT RULES:
- If material is reasonably sufficient (even if imperfect), choose 'approved'.
- Only choose 'retry' if there is a CRITICAL gap or a major unresolved contradiction.
- retry_instruction must be a single, short, specific search query.
- No preamble. No markdown. Only the JSON object.
"""


def _build_llm() -> ChatGoogleGenerativeAI:
    # Thinking ON for the Analyst — this is the reasoning-heavy node
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        thinking_budget=2048,
    )


def _prepare_context(state: ResearchState) -> str:
    """Build a condensed context string from gathered chunks."""
    chunks = state.raw_text_chunks
    # Take up to 20 chunks to stay within token budget
    sample = chunks[:20]
    lines = []
    for i, chunk in enumerate(sample, 1):
        lines.append(f"[Source {i}]: {chunk[:400]}")
    return "\n\n".join(lines)


def analyst_node(state: ResearchState) -> dict:
    """
    LangGraph node: Analyst.
    Reads:  state.query, state.raw_text_chunks, state.retry_count
    Writes: state.analyst_verdict, state.contradiction_notes,
            state.retry_instruction, state.retry_count, state.current_step
    """
    # Safety: if max retries exceeded, force approval to prevent infinite loop
    if state.retry_count >= settings.MAX_RETRY_LOOPS:
        return {
            "analyst_verdict": "approved",
            "contradiction_notes": f"Max retries ({settings.MAX_RETRY_LOOPS}) reached. Proceeding with available data.",
            "retry_instruction": "",
            "current_step": "analyst_approved",
        }

    if not state.raw_text_chunks:
        return {
            "analyst_verdict": "retry",
            "contradiction_notes": "No data gathered yet.",
            "retry_instruction": f"Comprehensive overview of: {state.query}",
            "retry_count": state.retry_count + 1,
            "current_step": "analyst_retry",
        }

    llm = _build_llm()
    context = _prepare_context(state)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original query: {state.query}\n\n"
                f"Gathered research material:\n{context}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        result = json.loads(raw)

        verdict = result.get("verdict", "approved")
        contradiction_notes = result.get("contradiction_notes", "")
        retry_instruction = result.get("retry_instruction", "")

        if verdict not in ("approved", "retry"):
            verdict = "approved"

    except Exception:
        # Parsing failure → approve and move on
        verdict = "approved"
        contradiction_notes = "Analyst response parsing failed. Proceeding."
        retry_instruction = ""

    new_retry_count = state.retry_count + (1 if verdict == "retry" else 0)
    next_step = "analyst_approved" if verdict == "approved" else "analyst_retry"

    return {
        "analyst_verdict": verdict,
        "contradiction_notes": contradiction_notes,
        "retry_instruction": retry_instruction,
        "retry_count": new_retry_count,
        "current_step": next_step,
    }
