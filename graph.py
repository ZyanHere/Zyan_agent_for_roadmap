"""
Phase 2: LangGraph State Machine
Nodes: planner → scheduler → executor → evaluator
Edges: conditional routing based on state
NO LLM — pure deterministic logic
"""

from langgraph.graph import StateGraph, END
from state import AgentState, Task, Plan
from task_manager import task_manager


# ─────────────────────────────────────────────
# GOAL REGISTRY: rule-based task definitions
# ─────────────────────────────────────────────
# Maps goal keywords to a list of task blueprints.
# The planner node looks up the goal here.
# Each blueprint defines title, phase, priority, and
# dependency indices (resolved to real IDs at creation time).

GOAL_TASK_REGISTRY: dict[str, list[dict]] = {
    "learn python": [
        {"title": "Setup environment",    "phase": "setup",    "priority": 1, "dep_indices": []},
        {"title": "Learn basics",         "phase": "learn",    "priority": 2, "dep_indices": [0]},
        {"title": "Build a project",      "phase": "apply",    "priority": 1, "dep_indices": [0, 1]},
        {"title": "Review and reflect",   "phase": "review",   "priority": 3, "dep_indices": [2]},
    ],
    "default": [
        {"title": "Task A", "phase": "phase1", "priority": 1, "dep_indices": []},
        {"title": "Task B", "phase": "phase1", "priority": 2, "dep_indices": []},
        {"title": "Task C", "phase": "phase2", "priority": 1, "dep_indices": [0, 1]},
        {"title": "Task D", "phase": "phase2", "priority": 3, "dep_indices": [2]},
    ],
}


# ─────────────────────────────────────────────
# NODE: planner
# ─────────────────────────────────────────────
# Rule-based. Looks up goal in registry, creates
# tasks with proper dependency wiring.

def planner(state: AgentState) -> dict:
    """Create tasks from goal using rule-based registry. No LLM."""

    ## skip if tasks already exist (replan guard)
    if state["tasks"]:
        return {}

    ## lookup goal in registry, fallback to default
    goal_lower = state["goal"].lower()
    blueprints = None
    for key, bps in GOAL_TASK_REGISTRY.items(): ## keyword match
        if key in goal_lower:
            blueprints = bps
            break
    if blueprints is None:
        blueprints = GOAL_TASK_REGISTRY["default"]

    ## create tasks, resolving dep_indices → real task IDs
    current_state = state
    created_ids: list[str] = [] ## to track created task IDs for dependency wiring

    for bp in blueprints:
        dep_ids = [created_ids[i] for i in bp["dep_indices"]] ## converts [0,1] → real task IDs
        ## create task via task_manager
        current_state, task = task_manager(current_state, "create", {
            "title": bp["title"],
            "phase": bp["phase"],
            "priority": bp["priority"],
            "dependencies": dep_ids,
        })
        created_ids.append(task["id"])

    ## build plan metadata
    phases_seen: list[str] = []
    phase_task_map: dict[str, list[str]] = {}
    for t in current_state["tasks"]:
        if t["phase"] not in phase_task_map:
            phases_seen.append(t["phase"])
            phase_task_map[t["phase"]] = []
        phase_task_map[t["phase"]].append(t["id"])

    plan: Plan = {
        "phases": [{"name": p, "tasks": phase_task_map[p]} for p in phases_seen],
        "created_at": "rule-based",
        "replans_count": 0,
    }

    print(f"[planner] Created {len(current_state['tasks'])} tasks for goal: \"{state['goal']}\"")
    for t in current_state["tasks"]:
        deps = ", ".join(t["dependencies"]) or "none"
        print(f"  {t['title']} (id={t['id']}, prio={t['priority']}, deps=[{deps}])")

    return {"tasks": current_state["tasks"], "plan": plan}


# ─────────────────────────────────────────────
# NODE: scheduler
# ─────────────────────────────────────────────
# Picks the next task using get_next logic.

def scheduler(state: AgentState) -> dict:
    """Pick the next eligible task. Pure get_next call."""
    new_state, next_task = task_manager(state, "get_next", {})

    if next_task:
        print(f"[scheduler] Next task: {next_task['title']} (id={next_task['id']})")
    else:
        print("[scheduler] No more tasks available")

    return {"current_task_id": new_state["current_task_id"]}


# ─────────────────────────────────────────────
# NODE: executor
# ─────────────────────────────────────────────
# Completes the current task: pending → in_progress → completed

def executor(state: AgentState) -> dict:
    """Execute (complete) the current task. pending → in_progress → completed."""

    ## step 1 ---> get task
    task_id = state["current_task_id"]
    if task_id is None:
        return {}

    ## find task title for logging
    title = next((t["title"] for t in state["tasks"] if t["id"] == task_id), "?")

    ## step 2 ---> Execute | transition: pending → in_progress → completed
    current_state = state
    current_state, _ = task_manager(current_state, "update", {"id": task_id, "status": "in_progress"})
    current_state, _ = task_manager(current_state, "update", {"id": task_id, "status": "completed"})

    print(f"[executor] Completed: {title}")

    return {"tasks": current_state["tasks"]}


# ─────────────────────────────────────────────
# NODE: evaluator
# ─────────────────────────────────────────────
# Checks progress. Calculates progress_pct.

def evaluator(state: AgentState) -> dict:
    """Check progress. Calculate completion percentage."""
    ## step 1 ---> calculate progress
    total = len(state["tasks"])
    if total == 0:
        return {"progress_pct": 0.0}

    completed = sum(1 for t in state["tasks"] if t["status"] == "completed")
    ## step 2 ---> calculate percentage
    pct = round((completed / total) * 100, 1)

    print(f"[evaluator] Progress: {completed}/{total} ({pct}%)")

    return {"progress_pct": pct}


# ─────────────────────────────────────────────
# ROUTING: conditional edges
# ─────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """After scheduler: if task found → executor, else → END."""
    if state["current_task_id"] is not None:
        return "executor"
    return END


def after_eval(state: AgentState) -> str:
    """After evaluator: if all done → END, else → scheduler."""
    if state["progress_pct"] >= 100.0:
        return END
    return "scheduler"


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_graph():
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    ## add nodes
    graph.add_node("planner", planner)
    graph.add_node("scheduler", scheduler)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)

    ## set entry point
    graph.set_entry_point("planner")

    ## edges: planner always goes to scheduler
    graph.add_edge("planner", "scheduler")

    ## conditional: scheduler → executor OR END
    graph.add_conditional_edges("scheduler", should_continue, {
        "executor": "executor",
        END: END,
    })

    ## edges: executor always goes to evaluator
    graph.add_edge("executor", "evaluator")

    ## conditional: evaluator → scheduler OR END
    graph.add_conditional_edges("evaluator", after_eval, {
        "scheduler": "scheduler",
        END: END,
    })

    return graph.compile()
