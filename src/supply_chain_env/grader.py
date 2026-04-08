from typing import Any, Dict, List

from .graders import grade_easy, grade_medium, grade_hard


def _safe_score(score: float) -> float:
    if score != score:
        return 0.5
    return float(max(0.01, min(0.99, score)))


def grade(task: str, env: Any, trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        score = 0.5
    else:
        if task == "easy":
            score = grade_easy(env, trajectory)
        elif task == "medium":
            score = grade_medium(env, trajectory)
        elif task == "hard":
            score = grade_hard(env, trajectory)
        else:
            raise ValueError(f"Unknown task: {task}")

    if score != score:
        score = 0.5
    score = max(0.01, min(0.99, score))
    return score
