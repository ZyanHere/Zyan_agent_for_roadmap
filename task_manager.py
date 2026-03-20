import uuid ## generate unique IDs for tasks
from typing import Any, Optional
from state import AgentState, Task

_VALID_TRANSITIONS = {
    "pending": "in_progress",
    "in_progress": "completed",
}


# created_at generator

## what it does: find the highest created_at, return the max integer, ensures stable ordering
def _next_created_at(tasks: list[Task]) -> int:
    if not tasks: 
        return 1
    return max(t["created_at"] for t in tasks) + 1


## state copier
def _copy_state(
    state: AgentState,
    tasks: Optional[list[Task]] = None,
    current_task_id: Optional[str] = ...,  # type: ignore[assignment]
) -> AgentState:   ## Creates a new state object with optional updates.
    """Return a new AgentState with optional overrides. Preserves TypedDict type."""
    return AgentState(
        goal=state["goal"],
        plan=state["plan"],
        tasks=tasks if tasks is not None else state["tasks"],
        current_task_id=state["current_task_id"] if current_task_id is ... else current_task_id, ## ... means no override provided
        progress_pct=state["progress_pct"],
    )


def task_manager(state: AgentState, action: str, payload: dict) -> tuple[AgentState, Any]:
    if action == "create":
        task: Task = {
            "id": str(uuid.uuid4())[:8],  ## unique ID
            "title": payload.get("title", ""),
            "status": "pending", ## default
            "phase": payload.get("phase", ""),
            "priority": payload.get("priority", 5),  ## lower is higher priority
            "dependencies": list(payload.get("dependencies", [])), ## always a list
            "due_date": payload.get("due_date", ""),
            "resources": list(payload.get("resources", [])),
            "created_at": _next_created_at(state["tasks"]),
        }
        new_tasks = state["tasks"] + [task] ## immutatble update pattern
        return _copy_state(state, tasks=new_tasks), task ## new state + current task

    if action == "update":
        task_id = payload["id"]
        new_tasks: list[Task] = []
        updated: Optional[Task] = None
        for t in state["tasks"]:
            if t["id"] == task_id:
                merged: Task = {**t}  #copying task to avoid mutating original
                for k, v in payload.items(): ## applying updates from payload
                    if k == "id":
                        continue
                    if k == "status":
                        allowed_next = _VALID_TRANSITIONS.get(t["status"])
                        if v != allowed_next:
                            raise ValueError(
                                f"Invalid transition: {t['status']} -> {v}"
                            )
                    merged[k] = v  ## applying cahnges
                updated = merged
                new_tasks.append(merged) ## rebuilding task list with updated task
            else:
                new_tasks.append(t)
        if updated is None:
            raise KeyError(f"Task {task_id} not found")
        return _copy_state(state, tasks=new_tasks), updated

    if action == "list":  ## read state, return tasks bsed on filters
        phase = payload.get("phase")
        status = payload.get("status")
        result = [    ## FILTER LOGIC
            t for t in state["tasks"]
            if (phase is None or t["phase"] == phase)
            and (status is None or t["status"] == status)
        ]
        return state, result

    if action == "get_next": ## decide execution later
        completed_ids = {t["id"] for t in state["tasks"] if t["status"] == "completed"} ## completed tasks
        all_ids = {t["id"] for t in state["tasks"]}

        candidates: list[Task] = []
        for t in state["tasks"]:
            if t["status"] != "pending":
                continue
            deps_met = all(
                d in completed_ids
                for d in t["dependencies"]
                if d in all_ids
            )
            blocked_by_missing = any(d not in all_ids for d in t["dependencies"])
            if blocked_by_missing or not deps_met:
                continue
            candidates.append(t)

        candidates.sort(key=lambda t: (t["priority"], t["created_at"]))
        next_task = candidates[0] if candidates else None
        new_state = _copy_state(state, current_task_id=next_task["id"] if next_task else None)
        return new_state, next_task

    raise ValueError(f"Unknown action: {action}")
