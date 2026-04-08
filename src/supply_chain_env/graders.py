import numpy as np
from typing import Any, Dict, List


def _clip01(value: float) -> float:
    eps = 0.01
    if not np.isfinite(value):
        return 0.5
    return float(max(eps, min(0.99, value)))


def _extract_series(trajectory: List[Dict[str, Any]], key: str) -> np.ndarray:
    series = [float(sum(step.get("info", {}).get(key, []))) for step in trajectory]
    return np.array(series, dtype=float)

def grade_easy(env, trajectory: List[Dict[str, Any]]) -> float:
    """Easy: reward stable service with low ordering volatility."""
    if not trajectory:
        score = 0.5
    else:
        demand = _extract_series(trajectory, "demand")
        sales = _extract_series(trajectory, "sales")
        orders = _extract_series(trajectory, "orders")

        service_level = float(sales.sum() / max(demand.sum(), 1e-6))
        volatility = float(np.std(orders) / (np.mean(orders) + 1e-6))
        smoothness = 1.0 / (1.0 + volatility)
        score = 0.75 * service_level + 0.25 * smoothness

    if score != score:
        score = 0.5
    score = max(0.01, min(0.99, score))
    return score

def grade_medium(env, trajectory: List[Dict[str, Any]]) -> float:
    """Medium: reward adaptation to trend and service quality."""
    if not trajectory:
        score = 0.5
    else:
        demand = _extract_series(trajectory, "demand")
        sales = _extract_series(trajectory, "sales")
        orders = _extract_series(trajectory, "orders")

        service_level = float(sales.sum() / max(demand.sum(), 1e-6))

        d_delta = np.diff(demand)
        o_delta = np.diff(orders)
        if len(d_delta) == 0 or len(o_delta) == 0:
            trend_follow = 0.5
        elif np.std(d_delta) < 1e-9 or np.std(o_delta) < 1e-9:
            trend_follow = 0.5
        else:
            corr = float(np.corrcoef(d_delta, o_delta)[0, 1])
            trend_follow = 0.5 if not np.isfinite(corr) else (corr + 1.0) / 2.0

        score = 0.6 * service_level + 0.4 * trend_follow

    if score != score:
        score = 0.5
    score = max(0.01, min(0.99, score))
    return score

def grade_hard(env, trajectory: List[Dict[str, Any]]) -> float:
    """Hard: reward robust service with efficient ordering under volatility."""
    if not trajectory:
        score = 0.5
    else:
        demand = _extract_series(trajectory, "demand")
        sales = _extract_series(trajectory, "sales")
        orders = _extract_series(trajectory, "orders")

        service_level = float(sales.sum() / max(demand.sum(), 1e-6))

        stockout_steps = 0
        for step in trajectory:
            d = step.get("info", {}).get("demand", [])
            s = step.get("info", {}).get("sales", [])
            if any(float(di) > float(si) for di, si in zip(d, s)):
                stockout_steps += 1
        stockout_rate = stockout_steps / len(trajectory)
        stockout_score = 1.0 - stockout_rate

        total_reward = float(sum(float(step.get("reward", 0.0)) for step in trajectory))
        total_orders = float(orders.sum())
        profit_per_order = total_reward / max(total_orders, 1e-6)
        efficiency_score = _clip01((profit_per_order + 2.0) / 8.0)

        score = 0.45 * service_level + 0.35 * stockout_score + 0.20 * efficiency_score

    if score != score:
        score = 0.5
    score = max(0.01, min(0.99, score))
    return score