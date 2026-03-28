import argparse
import uvicorn

from .inference import run_task


def run_baseline(task: str, num_episodes: int = 5) -> float:
    """Compatibility wrapper around inference runner."""
    return run_task(task=task, episodes=num_episodes)


def baseline_entry(task: str, num_episodes: int):
    print(f"Running baseline for task={task}, episodes={num_episodes}")
    score = run_baseline(task, num_episodes)
    print(f"Average score: {score:.3f}")
    return score


def serve(port: int = 7860):
    uvicorn.run("app:app", host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description='Supply Chain Environment run utility')
    parser.add_argument('--mode', choices=['baseline', 'serve'], default='baseline', help='Mode to run')
    parser.add_argument('--task', choices=['easy', 'medium', 'hard'], default='easy', help='Task difficulty')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes for baseline')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio serve')

    args = parser.parse_args()

    if args.mode == 'baseline':
        baseline_entry(args.task, args.episodes)
    elif args.mode == 'serve':
        serve(args.port)


if __name__ == '__main__':
    main()
