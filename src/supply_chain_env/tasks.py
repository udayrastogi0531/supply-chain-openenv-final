from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    demand_std: float
    seasonal_amplitude: float
    disruption_prob: float
    delay_prob: float


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        description="Stable demand, no disruptions.",
        demand_std=0.5,
        seasonal_amplitude=0.0,
        disruption_prob=0.0,
        delay_prob=0.0,
    ),
    "medium": TaskConfig(
        name="medium",
        description="Seasonal demand with moderate variability.",
        demand_std=2.0,
        seasonal_amplitude=4.0,
        disruption_prob=0.0,
        delay_prob=0.0,
    ),
    "hard": TaskConfig(
        name="hard",
        description="Noisy demand with disruptions and delayed supply.",
        demand_std=3.5,
        seasonal_amplitude=6.0,
        disruption_prob=0.07,
        delay_prob=0.06,
    ),
}


def get_task_config(task: str) -> TaskConfig:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return TASKS[task]
