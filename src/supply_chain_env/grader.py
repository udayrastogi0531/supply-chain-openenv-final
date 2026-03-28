from typing import Any, Dict, List

from .graders import grade_easy, grade_medium, grade_hard


def grade(task: str, env: Any, trajectory: List[Dict[str, Any]]) -> float:
    if task == "easy":
        return grade_easy(env, trajectory)
    if task == "medium":
        return grade_medium(env, trajectory)
    if task == "hard":
        return grade_hard(env, trajectory)
    raise ValueError(f"Unknown task: {task}")
