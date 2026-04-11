import os
import random
import yaml
import pytest
from supply_chain_env import SupplyChainEnv, grade_easy, grade_medium, grade_hard
from supply_chain_env.grader import grade
from supply_chain_env.environment import Action, Observation, State


def test_openenv_yaml_exists_and_parses():
    path = os.path.join(os.path.dirname(__file__), '..', 'openenv.yaml')
    assert os.path.exists(path), "openenv.yaml must exist"
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    assert 'openenv' in data
    assert data['openenv']['name'] == 'supply-chain-env'


def run_episode(env, steps=20):
    obs = env.reset()
    trajectory = []
    done = False
    while not done and len(trajectory) < steps:
        act = Action(orders=[10.0] * env.num_products)
        next_obs, reward, done, info = env.step(act)
        trajectory.append({'obs': obs, 'reward': reward, 'info': info})
        obs = next_obs
    return trajectory


def test_env_api_methods():
    env = SupplyChainEnv(task='easy', num_products=2, episode_length=10)
    obs = env.reset()
    assert isinstance(obs, Observation)
    trajectory = run_episode(env, steps=5)
    assert len(trajectory) > 0
    s = env.state()
    assert isinstance(s, State)


def test_grader_ranges():
    for task, grader in [('easy', grade_easy), ('medium', grade_medium), ('hard', grade_hard)]:
        env = SupplyChainEnv(task=task, episode_length=5)
        traj = run_episode(env, steps=5)
        score = grader(env, [{'reward': t['reward'], 'info': t['info']} for t in traj])
        assert 0.0 < score < 1.0


def test_grader_empty_trajectory_still_strictly_bounded():
    env = SupplyChainEnv(task='easy', episode_length=5)
    assert 0.0 < grade_easy(env, []) < 1.0
    assert 0.0 < grade_medium(env, []) < 1.0
    assert 0.0 < grade_hard(env, []) < 1.0


def random_trajectory(max_steps=30, max_products=5):
    length = random.randint(0, max_steps)
    trajectory = []
    for _ in range(length):
        products = random.randint(1, max_products)
        demand = [random.uniform(-20.0, 80.0) for _ in range(products)]
        sales = [random.uniform(-20.0, 80.0) for _ in range(products)]
        orders = [random.uniform(-20.0, 80.0) for _ in range(products)]
        if random.random() < 0.03:
            demand[random.randint(0, products - 1)] = float("nan")
        if random.random() < 0.03:
            sales[random.randint(0, products - 1)] = float("inf")
        if random.random() < 0.03:
            orders[random.randint(0, products - 1)] = float("-inf")

        trajectory.append(
            {
                "reward": random.uniform(-1e6, 1e6),
                "info": {
                    "demand": demand,
                    "sales": sales,
                    "orders": orders,
                },
            }
        )
    return trajectory


def test_absolute_guarantee_score_strict_bounds():
    for task in ["easy", "medium", "hard"]:
        for _ in range(1000):
            s = grade(task, None, random_trajectory())
            assert 0.0 < s < 1.0
            assert s != 0.0
            assert s != 1.0

if __name__ == '__main__':
    pytest.main([os.path.dirname(__file__)])
