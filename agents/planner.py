"""
agents/planner.py — Planner Agent

Responsibility:
  Receive the raw user query and decompose it into 3–5 concrete,
  actionable research sub-tasks. Each sub-task becomes a targeted
  search instruction for the Web Scripter.

LangGraph node signature: (state: ResearchState) -> dict
"""

from __future__ import annotations
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from state import ResearchState

SYSTEM_PROMPT = """You are the Planner agent in SARAS, an autonomous research system.

Your ONLY job is to decompose a research query into 3–5 focused sub-tasks.
Each sub-task must be:
- A specific, self-contained search instruction
- Distinct from the others (no overlap)
- Phrased as an action: "Find...", "Retrieve...", "Search for..."

Respond with ONLY a JSON array of strings. No preamble, no explanation.
Example output:
["Find recent statistics on X", "Retrieve academic definitions of Y", "Search for case studies about Z"]
"""


def _build_llm(thinking: bool = False) -> ChatGoogleGenerativeAI:
    kwargs = dict(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
    )
    # Thinking mode OFF for Planner — speed matters more than deep reasoning here
    if thinking:
        kwargs["thinking_budget"] = 1024
    return ChatGoogleGenerativeAI(**kwargs)


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node: Planner.
    Reads: state.query
    Writes: state.research_plan, state.current_step
    """
    llm = _build_llm(thinking=False)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {state.query}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if the model wraps output
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        plan: list[str] = json.loads(raw)
        if not isinstance(plan, list) or not plan:
            raise ValueError("Plan must be a non-empty list")
        plan = [str(t).strip() for t in plan if str(t).strip()]

    except Exception as e:
        # Fallback: treat the whole query as a single task
        plan = [f"Research comprehensively: {state.query}"]

    return {
        "research_plan": plan,
        "current_step": "planner_complete",
        "error": None,
    }
