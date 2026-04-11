from typing import Any, Dict, List

from .graders import FINAL_SCORE, grade_easy, grade_medium, grade_hard


def grade(task: str, env: Any, trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        score = 0.0
    else:
        if task == "easy":
            score = grade_easy(env, trajectory)
        elif task == "medium":
            score = grade_medium(env, trajectory)
        elif task == "hard":
            score = grade_hard(env, trajectory)
        else:
            raise ValueError(f"Unknown task: {task}")

    score = max(-10.0, min(10.0, score))
    return FINAL_SCORE(score)
