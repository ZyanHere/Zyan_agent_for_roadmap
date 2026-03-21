"""
Phase 3: Run the LLM-driven LangGraph agent.
Planner now uses Claude to dynamically generate tasks from any goal.
"""

from state import AgentState
from graph import build_graph
from typing import cast


def run_agent(goal: str) -> AgentState:
    """Run the full agent loop for a given goal."""

    initial_state: AgentState = {
        "goal": goal,
        "plan": None,
        "tasks": [],
        "current_task_id": None,
        "progress_pct": 0.0,
    }

    app = build_graph()
    print("=" * 50)
    print(f"GOAL: {goal}")
    print("=" * 50)

    final_state = cast(AgentState, app.invoke(initial_state))

    ## summary
    print("\n" + "=" * 50)
    print("DONE — Final State")
    print("=" * 50)
    print(f"Goal: {final_state['goal']}")
    print(f"Progress: {final_state['progress_pct']}%")
    if final_state["plan"]:
        print(f"Phases: {[p['name'] for p in final_state['plan']['phases']]}")
    print(f"Tasks:")
    for t in final_state["tasks"]:
        print(f"  [{t['status']:>11}] {t['title']} (prio={t['priority']}, phase={t['phase']})")

    return final_state


if __name__ == "__main__":
    run_agent("Learn LangGraph in 2 weeks")
