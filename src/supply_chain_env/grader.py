from typing import Any, Dict, List

from .graders import grade_easy, grade_medium, grade_hard


def _safe_score(score: float) -> float:
    if score != score:
        return 0.5
    return float(max(0.01, min(0.99, score)))


def grade(task: str, env: Any, trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.5

    if task == "easy":
        return _safe_score(grade_easy(env, trajectory))
    if task == "medium":
        return _safe_score(grade_medium(env, trajectory))
    if task == "hard":
        return _safe_score(grade_hard(env, trajectory))
    raise ValueError(f"Unknown task: {task}")
