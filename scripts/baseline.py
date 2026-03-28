#!/usr/bin/env python3
"""Baseline inference script for Supply Chain Environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from supply_chain_env import SupplyChainEnv, grade_easy, grade_medium, grade_hard
from supply_chain_env.baseline import baseline_agent
import numpy as np

def run_baseline(task: str, num_episodes: int = 5) -> float:
    """Run baseline agent and return average score."""
    env = SupplyChainEnv(task=task)
    scores = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            action = baseline_agent(obs)
            next_obs, reward, done, info = env.step(action)
            trajectory.append({
                'observation': obs,
                'action': action,
                'reward': reward,
                'info': info
            })
            obs = next_obs
        
        # Grade
        if task == "easy":
            score = grade_easy(env, trajectory)
        elif task == "medium":
            score = grade_medium(env, trajectory)
        elif task == "hard":
            score = grade_hard(env, trajectory)
        scores.append(score)
    
    return np.mean(scores)

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    print("Running baseline on easy task...")
    easy_score = run_baseline("easy")
    print(f"Easy task score: {easy_score:.3f}")
    
    print("Running baseline on medium task...")
    medium_score = run_baseline("medium")
    print(f"Medium task score: {medium_score:.3f}")
    
    print("Running baseline on hard task...")
    hard_score = run_baseline("hard")
    print(f"Hard task score: {hard_score:.3f}")