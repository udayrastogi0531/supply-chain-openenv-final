import math
from typing import Any, Dict, List

from .graders import SAFE_FINAL, grade_easy, grade_medium, grade_hard


def grade(task: str, env: Any, trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return SAFE_FINAL(0.5)
    else:
        if task == "easy":
            score = grade_easy(env, trajectory)
        elif task == "medium":
            score = grade_medium(env, trajectory)
        elif task == "hard":
            score = grade_hard(env, trajectory)
        else:
            raise ValueError(f"Unknown task: {task}")

    return SAFE_FINAL(score)
