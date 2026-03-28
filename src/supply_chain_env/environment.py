from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from .tasks import TASKS, get_task_config

class Observation(BaseModel):
    inventory: List[float]
    time: int
    demand_history: List[List[float]]
    weather: float
    promotion: float
    sustainability_score: float

class Action(BaseModel):
    orders: List[float]

class State(BaseModel):
    inventory: List[float]
    time: int
    demand_history: List[List[float]]
    pending_orders: List[float]
    weather: float
    promotion: float
    sustainability_score: float

class SupplyChainEnv:
    def __init__(self, task: str = "easy", num_products: int = 3, episode_length: int = 365, seed: int = 42):
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.num_products = num_products
        self.episode_length = episode_length
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state: Optional[State] = None

        # Parameters
        self.holding_cost = 0.1
        self.ordering_cost = 1.0
        self.stockout_cost = 5.0
        self.sale_price = 10.0
        self.initial_inventory = 50.0

        cfg = get_task_config(task)
        self.demand_mean = [10.0 if task != "hard" else 12.0] * num_products
        self.demand_std = [cfg.demand_std] * num_products
        self.seasonal_amplitude = cfg.seasonal_amplitude
        self.disruption_prob = cfg.disruption_prob
        self.delay_prob = cfg.delay_prob

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        self._state = State(
            inventory=[self.initial_inventory] * self.num_products,
            time=0,
            demand_history=[],
            pending_orders=[0.0] * self.num_products,
            weather=1.0,
            promotion=0.0,
            sustainability_score=1.0
        )
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, Dict[str, Any]]:
        assert self._state is not None, "Environment not reset"

        if len(action.orders) != self.num_products:
            raise ValueError(f"Action size must be {self.num_products}")

        orders = [max(0.0, float(o)) for o in action.orders]

        ordering_cost = self.ordering_cost * float(sum(orders))
        holding_cost = self.holding_cost * float(sum(self._state.inventory))
        reward = -(ordering_cost + holding_cost)

        # Process pending orders in hard mode with deterministic RNG.
        if self.task == "hard":
            for i in range(self.num_products):
                if self._rng.random() < self.delay_prob and orders[i] > 0:
                    self._state.pending_orders[i] += orders[i]
                    orders[i] = 0
                else:
                    self._state.inventory[i] += self._state.pending_orders[i]
                    self._state.pending_orders[i] = 0

        # Add new orders
        for i in range(self.num_products):
            self._state.inventory[i] += orders[i]

        # Update environment signals
        self._state.weather = max(0.5, min(1.5, float(self._rng.normal(1.0, 0.15))))
        self._state.promotion = max(0.0, min(1.0, float(self._rng.beta(2, 10))))
        self._state.sustainability_score = max(0.0, min(1.0, float(self._rng.normal(0.8, 0.1))))

        # Generate demand
        demand = self._generate_demand()
        self._state.demand_history.append(demand)

        # Process sales
        sales = []
        stockout_penalty = 0.0
        for i in range(self.num_products):
            sale = min(self._state.inventory[i], demand[i])
            self._state.inventory[i] -= sale
            sales.append(sale)
            reward += self.sale_price * sale
            if demand[i] > sale:
                stockout_penalty += self.stockout_cost * (demand[i] - sale)

        reward -= stockout_penalty

        total_demand = max(sum(demand), 1e-6)
        fill_rate = sum(sales) / total_demand
        service_bonus = 2.0 * fill_rate
        inventory_penalty = 0.02 * sum(x * x for x in self._state.inventory)
        reward += service_bonus
        reward -= inventory_penalty

        sustainability_penalty = (1 - self._state.sustainability_score) * 0.5 * sum(self._state.inventory)
        reward -= sustainability_penalty

        # Update time
        self._state.time += 1
        done = self._state.time >= self.episode_length

        obs = self._get_observation()
        info = {
            "sales": sales,
            "demand": demand,
            "orders": orders,
            "components": {
                "ordering_cost": ordering_cost,
                "holding_cost": holding_cost,
                "stockout_penalty": stockout_penalty,
                "service_bonus": service_bonus,
                "inventory_penalty": inventory_penalty,
                "sustainability_penalty": sustainability_penalty,
            },
        }

        return obs, float(reward), done, info

    def state(self) -> State:
        assert self._state is not None, "Environment not reset"
        return self._state

    def _generate_demand(self) -> List[float]:
        demand = []
        for i in range(self.num_products):
            base = self.demand_mean[i]
            if self.task in ("medium", "hard") and self.seasonal_amplitude > 0:
                seasonal = self.seasonal_amplitude * np.sin(2 * np.pi * self._state.time / 365)
                base += seasonal
            if self.task == "hard":
                base *= 1 + float(self._rng.normal(0, 0.1))
            base *= self._state.weather
            base *= 1 + self._state.promotion * 0.5
            noise = float(self._rng.normal(0, self.demand_std[i]))
            if self.task == "hard" and self._rng.random() < self.disruption_prob:
                noise += float(self._rng.normal(0, 10.0))
            d = max(0.0, base + noise)
            demand.append(d)
        return demand

    def _get_observation(self) -> Observation:
        history = self._state.demand_history[-10:] if len(self._state.demand_history) > 10 else self._state.demand_history
        return Observation(
            inventory=self._state.inventory,
            time=self._state.time,
            demand_history=history,
            weather=self._state.weather,
            promotion=self._state.promotion,
            sustainability_score=self._state.sustainability_score
        )