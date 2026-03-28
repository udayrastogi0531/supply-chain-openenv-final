import os
import yaml
import pytest
from supply_chain_env import SupplyChainEnv, grade_easy, grade_medium, grade_hard
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
        assert 0.0 <= score <= 1.0

if __name__ == '__main__':
    pytest.main([os.path.dirname(__file__)])
