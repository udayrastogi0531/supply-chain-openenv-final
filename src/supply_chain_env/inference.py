from typing import Dict, List
import numpy as np

from .environment import Action, SupplyChainEnv
from .graders import FINAL_SCORE, grade_easy, grade_medium, grade_hard


def inference_agent(observation) -> Action:
    """Simple reproducible rule-based policy."""
    history = np.array(observation.demand_history[-5:]) if observation.demand_history else np.array([observation.inventory]) * 0 + 10.0
    mean_demand = history.mean(axis=0)
    safety_stock = np.maximum(1.0, history.std(axis=0))
    target_inventory = mean_demand + 1.5 * safety_stock

    orders: List[float] = []
    for i, inv in enumerate(observation.inventory):
        order_amount = max(0.0, float(target_inventory[i] - inv))
        orders.append(min(order_amount, 40.0))
    return Action(orders=orders)


def _run_task_raw(task: str, episodes: int = 1, seed: int = 42) -> float:
    scores = []
    grader = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}[task]

    for ep in range(episodes):
        env = SupplyChainEnv(task=task, seed=seed + ep)
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            action = inference_agent(obs)
            obs, reward, done, info = env.step(action)
            trajectory.append({"reward": reward, "info": info})
        episode_score = grader(env, trajectory)
        scores.append(float(episode_score))

    raw_score = float(np.mean(scores)) if scores else 0.5
    if raw_score <= 0.0:
        raw_score = 1e-6
    if raw_score >= 1.0:
        raw_score = 1 - 1e-6
    return raw_score


def run_task(task: str, episodes: int = 1, seed: int = 42) -> float:
    score = _run_task_raw(task=task, episodes=episodes, seed=seed)
    return FINAL_SCORE(score)


def run_all_tasks(episodes: int = 1, seed: int = 42) -> Dict[str, float]:
    easy = _run_task_raw("easy", episodes=episodes, seed=seed)
    medium = _run_task_raw("medium", episodes=episodes, seed=seed)
    hard = _run_task_raw("hard", episodes=episodes, seed=seed)
    return {
        "easy": FINAL_SCORE(easy),
        "medium": FINAL_SCORE(medium),
        "hard": FINAL_SCORE(hard),
    }
