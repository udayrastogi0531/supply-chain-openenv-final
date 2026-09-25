# Task Guide

The environment provides three task levels:

| Task | Demand | Disruptions | Purpose |
| --- | --- | --- | --- |
| Easy | Stable | None | Basic inventory decisions |
| Medium | Seasonal | Moderate | Planning under variability |
| Hard | Noisy | Yes | Robust decisions under uncertainty |

An agent should treat the task name as part of the environment configuration and should not assume that a strategy tuned for one level will perform identically on another.

When adding a task, document its demand assumptions, disruption behavior, expected action shape, and grading implications.
