"""
Phase 2: Run the LangGraph agent.
Invoke the compiled graph with a goal and watch it autonomously
walk through planner → scheduler → executor → evaluator → loop → done.
"""

from state import AgentState
from graph import build_graph
from typing import cast


def run_agent(goal: str) -> AgentState:
    """Run the full agent loop for a given goal."""

    ## initial state
    initial_state: AgentState = {
        "goal": goal,
        "plan": None,
        "tasks": [],
        "current_task_id": None,
        "progress_pct": 0.0,
    }

    ## build and invoke graph
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
    print(f"Tasks:")
    for t in final_state["tasks"]:
        print(f"  [{t['status']:>11}] {t['title']} (prio={t['priority']}, phase={t['phase']})")

    return final_state


if __name__ == "__main__":
    ## Test 1: default goal (uses A/B/C/D from registry)
    run_agent("Complete the project")

    print("\n\n")

    ## Test 2: "learn python" goal (uses learn python blueprints)
    run_agent("Learn Python programming")
