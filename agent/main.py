from state import AgentState
from task_manager import task_manager


def complete_task(state: AgentState, task_id: str) -> AgentState:
    state, _ = task_manager(state, "update", {"id": task_id, "status": "in_progress"})
    state, _ = task_manager(state, "update", {"id": task_id, "status": "completed"})
    return state


def main():
    state: AgentState = {
        "goal": "Test task ordering",
        "plan": None,
        "tasks": [],
        "current_task_id": None,
        "progress_pct": 0.0,
    }

    # Create tasks
    state, task_a = task_manager(state, "create", {"title": "Task A", "priority": 1, "dependencies": []})
    state, task_b = task_manager(state, "create", {"title": "Task B", "priority": 2, "dependencies": []})
    state, task_c = task_manager(state, "create", {
        "title": "Task C", "priority": 1,
        "dependencies": [task_a["id"], task_b["id"]],
    })
    state, task_d = task_manager(state, "create", {
        "title": "Task D", "priority": 3,
        "dependencies": [task_c["id"]],
    })

    print("=== Created Tasks ===")
    for t in state["tasks"]:
        deps = ", ".join(t["dependencies"]) or "none"
        print(f"  {t['title']} (id={t['id']}, prio={t['priority']}, deps=[{deps}])")

    # Test sequence
    steps = [
        ("get_next", "A"),
        ("complete", "A"),
        ("get_next", "B"),
        ("complete", "B"),
        ("get_next", "C"),
        ("complete", "C"),
        ("get_next", "D"),
        ("complete", "D"),
        ("get_next", "None"),
    ]

    task_map = {"A": task_a, "B": task_b, "C": task_c, "D": task_d}

    print("\n=== Test Sequence ===")
    for action, expected in steps:
        if action == "get_next":
            state, result = task_manager(state, "get_next", {})
            name = result["title"] if result else "None"
            expected_name = f"Task {expected}" if expected != "None" else "None"
            ok = "PASS" if name == expected_name else "FAIL"
            print(f"  get_next -> {name}  (expected {expected_name}) [{ok}]")
        elif action == "complete":
            tid = task_map[expected]["id"]
            state = complete_task(state, tid)
            print(f"  complete {expected}")

    # Final listing
    print("\n=== Final State ===")
    state, all_tasks = task_manager(state, "list", {})
    for t in all_tasks:
        print(f"  {t['title']}: {t['status']}")


if __name__ == "__main__":
    main()
