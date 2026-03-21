from typing import TypedDict, Optional


class Task(TypedDict):
    id: str
    title: str
    status: str  # "pending" | "in_progress" | "completed"
    phase: str
    priority: int
    dependencies: list[str]
    due_date: str
    resources: list[str]
    created_at: int  # incremental counter for ordering


class PlanPhase(TypedDict):
    name: str
    tasks: list[str]


class Plan(TypedDict):
    phases: list[PlanPhase]
    created_at: str
    replans_count: int


class AgentState(TypedDict):
    goal: str
    plan: Optional[Plan]
    tasks: list[Task]
    current_task_id: Optional[str]
    progress_pct: float
