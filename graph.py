"""
graph.py — SARAS LangGraph State Machine

Nodes (in order):
  planner → web_scripter → analyst → [retry → web_scripter] | synthesizer → source_aggregator → END

The feedback loop:
  analyst node returns verdict 'retry' → edge goes back to web_scripter
  analyst node returns verdict 'approved' → edge goes to synthesizer

All nodes read from and write to ResearchState.
The graph is compiled once at module load and reused across sessions.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from state import ResearchState
from agents import (
    planner_node,
    web_scripter_node,
    analyst_node,
    synthesizer_node,
    source_aggregator_node,
)


# ── Conditional routing function ────────────────────────────────────────────

def route_analyst(state: ResearchState) -> str:
    """
    After the Analyst runs, decide the next node.
    Returns the name of the next node as a string.
    """
    if state.analyst_verdict == "retry":
        return "web_scripter"
    return "synthesizer"


# ── Build graph ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("web_scripter", web_scripter_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("source_aggregator", source_aggregator_node)

    # Linear edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "web_scripter")
    graph.add_edge("web_scripter", "analyst")

    # Conditional feedback loop from analyst
    graph.add_conditional_edges(
        "analyst",
        route_analyst,
        {
            "web_scripter": "web_scripter",   # retry path
            "synthesizer": "synthesizer",     # approved path
        },
    )

    graph.add_edge("synthesizer", "source_aggregator")
    graph.add_edge("source_aggregator", END)

    return graph.compile()


# Compile once — import this in app.py
saras_graph = build_graph()
