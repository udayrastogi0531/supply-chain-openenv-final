import os
import sys
import uuid
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from supply_chain_env.environment import Action, SupplyChainEnv
from supply_chain_env.grader import grade
from supply_chain_env.inference import inference_agent, run_all_tasks
from supply_chain_env.tasks import TASKS


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    session_id: str
    orders: List[float]


class SessionData(BaseModel):
    env: Any
    task: str
    seed: int
    trajectory: List[Dict[str, Any]]

    class Config:
        arbitrary_types_allowed = True


app = FastAPI(title="Supply Chain OpenEnv API", version="0.1.0")
SESSIONS: Dict[str, SessionData] = {}


def _ensure_session(session_id: str) -> SessionData:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call /reset first.")
    return SESSIONS[session_id]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "supply-chain-openenv"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    if payload.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {payload.task}")

    env = SupplyChainEnv(task=payload.task, seed=payload.seed)
    obs = env.reset(seed=payload.seed)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = SessionData(
        env=env,
        task=payload.task,
        seed=payload.seed,
        trajectory=[],
    )
    return {
        "session_id": session_id,
        "task": payload.task,
        "observation": obs.model_dump(),
    }


@app.get("/reset")
def reset_get(task: str = "easy", seed: int = 42) -> Dict[str, Any]:
    return reset(ResetRequest(task=task, seed=seed))


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    session = _ensure_session(payload.session_id)
    env = session.env
    obs, reward, done, info = env.step(Action(orders=payload.orders))
    session.trajectory.append({"reward": reward, "info": info})
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    session = _ensure_session(session_id)
    return session.env.state().model_dump()


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": cfg.name,
                "description": cfg.description,
                "demand_std": cfg.demand_std,
                "seasonal_amplitude": cfg.seasonal_amplitude,
                "disruption_prob": cfg.disruption_prob,
                "delay_prob": cfg.delay_prob,
            }
            for cfg in TASKS.values()
        ]
    }


@app.get("/baseline")
def baseline(episodes: int = 1, seed: int = 42) -> Dict[str, Any]:
    scores = run_all_tasks(episodes=max(1, episodes), seed=seed)
    return {"episodes": max(1, episodes), "seed": seed, "scores": scores}


@app.get("/grader")
def grader(session_id: str) -> Dict[str, Any]:
    session = _ensure_session(session_id)
    score = grade(session.task, session.env, session.trajectory)
    return {
        "task": session.task,
        "steps": len(session.trajectory),
        "score": score,
    }


@app.post("/autostep")
def autostep(session_id: str) -> Dict[str, Any]:
    session = _ensure_session(session_id)
    obs = session.env._get_observation()
    action = inference_agent(obs)
    step_result = step(StepRequest(session_id=session_id, orders=action.orders))
    step_result["action"] = action.orders
    return step_result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)