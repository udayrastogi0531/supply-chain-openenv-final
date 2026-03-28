# Supply Chain Inventory Management Environment

A real-world OpenEnv environment simulating supply chain inventory management for AI agents.

## Description

This environment models the task of managing inventory in a retail supply chain. The agent must decide order quantities for multiple products to maximize profit while dealing with demand variability, holding costs, ordering costs, and stockout penalties.

## Tasks

- **Easy**: Constant demand with no variability
- **Medium**: Seasonal demand patterns
- **Hard**: Stochastic demand with disruptions and supplier delays

## Action Space

Continuous actions representing order quantities for each product.

- Type: `List[float]` of length `num_products`
- Range: [0, ∞) for each product

## Observation Space

Dictionary containing:
- `inventory`: Current inventory levels for each product (`List[float]`)
- `time`: Current time step (`int`)
- `demand_history`: Recent demand history (`List[List[float]]`, max 10 steps)

## Reward Function

Reward is the profit at each step:
- Revenue from sales: `sale_price * quantity_sold`
- Costs:
  - Ordering cost: `ordering_cost * order_quantity`
  - Holding cost: `holding_cost * inventory_level`
  - Stockout cost: `stockout_cost * (demand - sales)`

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Run inference agent on all tasks:
   ```bash
   python inference.py
   ```

3. Run FastAPI server:
   ```bash
   python app.py
   ```
   or via uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

4. For Hugging Face Spaces deployment, the Dockerfile is provided.

## API

```python
from supply_chain_env import SupplyChainEnv

env = SupplyChainEnv(task="easy")
obs = env.reset()
action = Action(orders=[10.0, 10.0, 10.0])
obs, reward, done, info = env.step(action)
state = env.state()
```

## Graders

Each task has a grader that evaluates agent performance on a scale of 0.0 to 1.0:
- `grade_easy`: Based on ordering accuracy for constant demand
- `grade_medium`: Based on adapting to seasonal patterns
- `grade_hard`: Based on handling disruptions and minimizing stockouts