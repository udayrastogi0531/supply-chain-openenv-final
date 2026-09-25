# Architecture

## Runtime flow

1. A client selects an inventory task.
2. The environment resets deterministic task state.
3. The agent submits product order quantities.
4. The environment advances one step and returns observation, reward, and state.
5. The grader evaluates the resulting trajectory.

## Main components

- `supply_chain_env/` — environment state, actions, observations, and task logic.
- `server/` — FastAPI/OpenEnv HTTP interface.
- `inference.py` — baseline agent entrypoint.
- `scripts/` — validation utilities.
- `tests/` — automated regression coverage.

The design keeps environment logic independent from the HTTP layer so the same environment can be exercised locally, through the server, or by the baseline agent.
