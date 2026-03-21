"""
Phase 3: LLM-Driven Planner
Nodes: planner (LLM) → scheduler → executor → evaluator
Only the planner node changed from Phase 2.
scheduler, executor, evaluator, edges — all UNCHANGED.
"""

import json
import re
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from state import AgentState, Task, Plan
from task_manager import task_manager

from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)


# ─────────────────────────────────────────────
# LLM SETUP (lazy — created after .env is loaded)
# ─────────────────────────────────────────────

_llm = None #global cache??

def get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model_name="claude-sonnet-4-20250514",  # type: ignore[call-arg]
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    return _llm


# ─────────────────────────────────────────────
# PLANNER PROMPT
# ─────────────────────────────────────────────

PLANNER_PROMPT = """You are a planning assistant.
Given a goal, generate a structured learning/execution plan.

Return ONLY valid JSON — no markdown fences, no explanations, no extra text.

Schema:
{{
  "phases": [
    {{ "name": "phase name" }}
  ],
  "tasks": [
    {{
      "title": "unique task title",
      "phase": "must match a phase name above",
      "priority": 1,
      "dependencies": ["title of dependency task"]
    }}
  ]
}}

Rules:
- Each task title MUST be unique
- Dependencies reference other task titles exactly (case-sensitive)
- A task cannot depend on itself
- No circular dependencies
- Priority: 1 = highest, 5 = lowest
- Order tasks logically — foundational tasks first
- Keep it practical: 4-8 tasks max
- phases should reflect logical grouping

Goal: {goal}"""


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown fences and trailing commas."""
    ## strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    ## remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    return json.loads(cleaned)

#Ensure LLM output is safe before using it
def _validate_plan(plan_data: dict) -> list[str]:
    """Validate LLM plan structure. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    ## step 1: check top-level keys
    if "phases" not in plan_data:
        errors.append("Missing 'phases' key")
    if "tasks" not in plan_data:
        errors.append("Missing 'tasks' key")
    if errors:
        return errors

    ## step 2: check phases
    phase_names = set()
    for p in plan_data["phases"]:
        if "name" not in p:
            errors.append("Phase missing 'name'")
        else:
            phase_names.add(p["name"])

    ## check tasks
    task_titles = set()
    for t in plan_data["tasks"]:
        ## required fields
        for field in ["title", "phase", "priority", "dependencies"]:
            if field not in t:
                errors.append(f"Task missing '{field}': {t}")

        title = t.get("title", "")

        ## duplicate titles
        if title in task_titles:
            errors.append(f"Duplicate task title: '{title}'")
        task_titles.add(title)

        ## phase must exist
        if t.get("phase") not in phase_names:
            errors.append(f"Task '{title}' references unknown phase: '{t.get('phase')}'")

        ## self-dependency
        if title in t.get("dependencies", []):
            errors.append(f"Task '{title}' depends on itself")

    ## check all dependencies reference existing titles
    for t in plan_data["tasks"]:
        for dep in t.get("dependencies", []):
            if dep not in task_titles:
                errors.append(f"Task '{t.get('title')}' depends on unknown task: '{dep}'")

    ## check circular dependencies (DFS)
    adj: dict[str, list[str]] = {t["title"]: t.get("dependencies", []) for t in plan_data["tasks"]}
    visited: set[str] = set()
    in_stack: set[str] = set()


    ## cycle detection using DFS
    def has_cycle(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for dep in adj.get(node, []):
            if has_cycle(dep):
                return True
        in_stack.discard(node)
        return False

    for title in adj:
        if has_cycle(title):
            errors.append(f"Circular dependency detected involving '{title}'")
            break

    return errors


# ─────────────────────────────────────────────
# NODE: planner (LLM-driven)
# ─────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    """Call LLM to generate a structured plan, then create tasks."""

    ## skip if tasks already exist (replan guard)
    if state["tasks"]:
        return {}

    ## step 1: call LLM
    prompt = PLANNER_PROMPT.format(goal=state["goal"])
    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = response.content
    if isinstance(raw, list):
        raw = "".join(str(block) for block in raw)

    print(f"[planner] LLM response received for goal: \"{state['goal']}\"")

    ## step 2: parse JSON
    try:
        plan_data = _parse_llm_json(raw)
    except json.JSONDecodeError as e:
        print(f"[planner] ERROR: Failed to parse LLM JSON: {e}")
        print(f"[planner] Raw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")

    ## step 3: validate
    errors = _validate_plan(plan_data)
    if errors:
        print(f"[planner] Validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError(f"LLM plan validation failed: {errors}")

    ## step 4: create tasks (two-pass for dependency mapping)
    ## pass 1: create all tasks WITHOUT dependencies (to get IDs)
    current_state = state
    title_to_id: dict[str, str] = {}

    for t in plan_data["tasks"]:
        current_state, created = task_manager(current_state, "create", {
            "title": t["title"],
            "phase": t["phase"],
            "priority": t["priority"],
            "dependencies": [],  ## empty for now
        })
        title_to_id[t["title"]] = created["id"]

    ## pass 2: update tasks WITH mapped dependency IDs
    for t in plan_data["tasks"]:
        if not t.get("dependencies"):
            continue
        dep_ids = [title_to_id[dep_title] for dep_title in t["dependencies"]]
        task_id = title_to_id[t["title"]]
        current_state, _ = task_manager(current_state, "update", {
            "id": task_id,
            "dependencies": dep_ids,
        })

    ## step 5: build plan metadata
    plan: Plan = {
        "phases": [{"name": p["name"], "tasks": [
            title_to_id[t["title"]]
            for t in plan_data["tasks"]
            if t["phase"] == p["name"]
        ]} for p in plan_data["phases"]],
        "created_at": "llm-generated",
        "replans_count": 0,
    }

    print(f"[planner] Created {len(current_state['tasks'])} tasks:")
    for t in current_state["tasks"]:
        dep_titles = []
        for d in t["dependencies"]:
            dep_titles.append(next(
                (tk["title"] for tk in current_state["tasks"] if tk["id"] == d), d
            ))
        deps = ", ".join(dep_titles) or "none"
        print(f"  {t['title']} (id={t['id']}, prio={t['priority']}, deps=[{deps}])")

    return {"tasks": current_state["tasks"], "plan": plan}


# ─────────────────────────────────────────────
# NODE: scheduler (UNCHANGED from Phase 2)
# ─────────────────────────────────────────────

def scheduler(state: AgentState) -> dict:
    """Pick the next eligible task. Pure get_next call."""
    new_state, next_task = task_manager(state, "get_next", {})

    if next_task:
        print(f"[scheduler] Next task: {next_task['title']} (id={next_task['id']})")
    else:
        print("[scheduler] No more tasks available")

    return {"current_task_id": new_state["current_task_id"]}


# ─────────────────────────────────────────────
# NODE: executor (UNCHANGED from Phase 2)
# ─────────────────────────────────────────────

def executor(state: AgentState) -> dict:
    """Execute (complete) the current task. pending → in_progress → completed."""

    task_id = state["current_task_id"]
    if task_id is None:
        return {}

    title = next((t["title"] for t in state["tasks"] if t["id"] == task_id), "?")

    current_state = state
    current_state, _ = task_manager(current_state, "update", {"id": task_id, "status": "in_progress"})
    current_state, _ = task_manager(current_state, "update", {"id": task_id, "status": "completed"})

    print(f"[executor] Completed: {title}")

    return {"tasks": current_state["tasks"]}


# ─────────────────────────────────────────────
# NODE: evaluator (UNCHANGED from Phase 2)
# ─────────────────────────────────────────────

def evaluator(state: AgentState) -> dict:
    """Check progress. Calculate completion percentage."""
    total = len(state["tasks"])
    if total == 0:
        return {"progress_pct": 0.0}

    completed = sum(1 for t in state["tasks"] if t["status"] == "completed")
    pct = round((completed / total) * 100, 1)

    print(f"[evaluator] Progress: {completed}/{total} ({pct}%)")

    return {"progress_pct": pct}


# ─────────────────────────────────────────────
# ROUTING: conditional edges (UNCHANGED)
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
# GRAPH BUILDER (UNCHANGED)
# ─────────────────────────────────────────────

def build_graph():
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("scheduler", scheduler)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "scheduler")

    graph.add_conditional_edges("scheduler", should_continue, {
        "executor": "executor",
        END: END,
    })

    graph.add_edge("executor", "evaluator")

    graph.add_conditional_edges("evaluator", after_eval, {
        "scheduler": "scheduler",
        END: END,
    })

    return graph.compile()
