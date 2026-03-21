"""
FastAPI bridge server: exposes the LangGraph agent over HTTP.
Start:  pip install fastapi uvicorn  &&  uvicorn server:app --reload --port 8000
"""

import io
import contextlib
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Agent Debug Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    goal: str


# ── Node regex for parsing stdout lines ───────────────────
_NODE_RE = re.compile(r"^\[(\w+)]")


def _parse_trace(output: str) -> list[dict]:
    """Convert captured stdout lines into structured trace steps."""
    steps: list[dict] = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        m = _NODE_RE.match(line)
        if m:
            node = m.group(1)
            message = line[m.end():].strip()
            steps.append({"node": node, "message": message})
        elif line.startswith("GOAL:") or line.startswith("DONE"):
            steps.append({"node": "system", "message": line})
    return steps


@app.post("/api/run-agent")
def run_agent_endpoint(req: RunRequest):
    # Import here so module-level errors don't break server startup
    from run import run_agent

    # Capture stdout prints as trace
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        final_state = run_agent(req.goal)

    trace = _parse_trace(buf.getvalue())

    # Build response matching the frontend AgentResponse shape
    tasks = []
    for t in final_state["tasks"]:
        tasks.append({
            "id": t["id"],
            "title": t["title"],
            "status": t["status"],
            "dependencies": t["dependencies"],
        })

    # Plan phases already contain task IDs from graph.py planner node
    plan = {"phases": []}
    if final_state.get("plan"):
        plan = {"phases": final_state["plan"]["phases"]}

    return {
        "goal": final_state["goal"],
        "plan": plan,
        "tasks": tasks,
        "steps": trace,
    }
