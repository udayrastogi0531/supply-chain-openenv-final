import math
from typing import Any, Dict, List

import numpy as np


def SAFE_FINAL(x):
    import math

    try:
        x = float(x)
    except:
        return 0.5

    if not math.isfinite(x):
        return 0.5

    # SAFE SIGMOID (NO SATURATION TO EXACT 1)
    if x >= 0:
        z = math.exp(-x)
        s = 1 / (1 + z)
    else:
        z = math.exp(x)
        s = z / (1 + z)

    eps = 1e-4
    s = eps + (1 - 2 * eps) * s

    return float(max(eps, min(1 - eps, s)))


def FINAL_SCORE(x):
    return SAFE_FINAL(x)


def _clip_metric(value: float) -> float:
    return float(max(-10.0, min(10.0, _safe_float(value, 0.0))))


def _normalize_ratio(numerator: float, denominator: float) -> float:
    ratio = _safe_float(numerator, 0.0) / (_safe_float(denominator, 0.0) + 1e-6)
    ratio = _clip_metric(ratio)
    return float(ratio)


def _normalize_unit_interval(x: float) -> float:
    x = _clip_metric(x)
    return float(x)


def _safe_corr_to_score(a: np.ndarray, b: np.ndarray) -> float:
    corr = 0.0
    try:
        std_a = np.std(a)
        std_b = np.std(b)
    except:
        return 0.5

    if len(a) > 1 and len(b) > 1 and std_a >= 1e-9 and std_b >= 1e-9:
        try:
            corr = _safe_float(np.corrcoef(a, b)[0, 1], 0.0)
        except:
            return 0.5

    corr = _clip_metric(corr)
    trend_follow = (corr + 1.0) / (2.0 + 1e-6)
    trend_follow = _clip_metric(trend_follow)
    return float(trend_follow)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except:
        out = default
    if not np.isfinite(out):
        out = default
    return float(out)


def _safe_values(value: Any) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = value
    else:
        raw = [value]
    return [_safe_float(v, 0.0) for v in raw]


def _safe_info_sum(step: Dict[str, Any], key: str) -> float:
    values = _safe_values(step.get("info", {}).get(key, []))
    return _clip_metric(sum(values))


def _extract_series(trajectory: List[Dict[str, Any]], key: str) -> np.ndarray:
    try:
        series = [float(sum(step.get("info", {}).get(key, []))) for step in trajectory]
    except:
        return np.array([0.0], dtype=float)
    return np.array(series, dtype=float)


def _stockout_score(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return SAFE_FINAL(0.5)

    stockout_steps = 0
    for step in trajectory:
        d = _safe_values(step.get("info", {}).get("demand", []))
        s = _safe_values(step.get("info", {}).get("sales", []))
        if any(di > si for di, si in zip(d, s)):
            stockout_steps += 1

    stockout_rate = _safe_float(stockout_steps, 0.0) / (_safe_float(len(trajectory), 0.0) + 1e-6)
    stockout_rate = _clip_metric(stockout_rate)
    stockout_score = _clip_metric(1.0 - stockout_rate)
    return float(stockout_score)


def _efficiency_score(trajectory: List[Dict[str, Any]], orders: np.ndarray) -> float:
    total_reward = _clip_metric(sum(_safe_float(step.get("reward", 0.0), 0.0) for step in trajectory))
    total_orders = _clip_metric(np.sum(orders))
    profit_per_order = total_reward / (total_orders + 1e-6)
    profit_per_order = _clip_metric(profit_per_order)

    efficiency = (profit_per_order + 2.0) / (8.0 + 1e-6)
    efficiency = _clip_metric(efficiency)
    return float(efficiency)


def grade_easy(env, trajectory: List[Dict[str, Any]]) -> float:
    """Easy: reward stable service with low ordering volatility."""
    if not trajectory:
        return SAFE_FINAL(0.5)

    demand = _extract_series(trajectory, "demand")
    sales = _extract_series(trajectory, "sales")
    orders = _extract_series(trajectory, "orders")

    service_level = _normalize_ratio(np.sum(sales), np.sum(demand))
    try:
        mean_orders = _clip_metric(np.mean(orders))
    except:
        return SAFE_FINAL(0.5)
    try:
        volatility = np.std(orders) / (abs(mean_orders) + 1e-6)
    except:
        return SAFE_FINAL(0.5)
    volatility = _clip_metric(max(0.0, _safe_float(volatility, 0.0)))
    smoothness = 1.0 / (1.0 + volatility + 1e-6)
    smoothness = _normalize_unit_interval(smoothness)

    weighted_sum = _clip_metric(0.75 * service_level + 0.25 * smoothness)
    score = float(weighted_sum)
    return SAFE_FINAL(score)


def grade_medium(env, trajectory: List[Dict[str, Any]]) -> float:
    """Medium: reward adaptation to trend and service quality."""
    if not trajectory:
        return SAFE_FINAL(0.5)

    demand = _extract_series(trajectory, "demand")
    sales = _extract_series(trajectory, "sales")
    orders = _extract_series(trajectory, "orders")

    service_level = _normalize_ratio(np.sum(sales), np.sum(demand))

    try:
        d_delta = np.diff(demand)
    except:
        return SAFE_FINAL(0.5)
    try:
        o_delta = np.diff(orders)
    except:
        return SAFE_FINAL(0.5)
    trend_follow = _safe_corr_to_score(d_delta, o_delta)

    weighted_sum = _clip_metric(0.6 * service_level + 0.4 * trend_follow)
    score = float(weighted_sum)
    return SAFE_FINAL(score)


def grade_hard(env, trajectory: List[Dict[str, Any]]) -> float:
    """Hard: reward robust service with efficient ordering under volatility."""
    if not trajectory:
        return SAFE_FINAL(0.5)

    demand = _extract_series(trajectory, "demand")
    sales = _extract_series(trajectory, "sales")
    orders = _extract_series(trajectory, "orders")

    service_level = _normalize_ratio(np.sum(sales), np.sum(demand))
    stockout_score = _stockout_score(trajectory)
    efficiency_score = _efficiency_score(trajectory, orders)

    weighted_sum = _clip_metric(0.45 * service_level + 0.35 * stockout_score + 0.20 * efficiency_score)
    score = float(weighted_sum)
    return SAFE_FINAL(score)
