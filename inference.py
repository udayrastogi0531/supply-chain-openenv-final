import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from supply_chain_env.inference import run_all_tasks


def main() -> None:
    scores = run_all_tasks(episodes=3, seed=42)
    print("Final scores:")
    for task in ("easy", "medium", "hard"):
        print(f"{task}: {scores[task]:.4f}")


if __name__ == "__main__":
    main()
