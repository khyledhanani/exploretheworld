#!/usr/bin/env python3
"""
Interactive test script for trained RL agents.

This script allows you to:
1. Select a trained model from the logs folder
2. Choose an environment to test on
3. Watch the agent perform with human rendering

Usage:
    # Interactive mode (prompts for selections)
    python test.py

    # Direct mode (specify model and environment)
    python test.py --model logs/ppo_cnn/ppo_cnn_final.zip --env MiniWorld-OneRoom-v0 --episodes 5
"""

import os
import argparse
from pathlib import Path

from utils.registry import infer_agent_type, load_agent_module


def find_final_models():
    """Find all *_final.zip and *_final.pt models in the logs folder."""
    logs_dir = Path(__file__).parent / "logs"
    models = []

    # Search for all *_final* files
    for ext in ["*.zip", "*.pt"]:
        for model_path in logs_dir.rglob(f"*_final*{ext[1:]}"):
            agent_type = model_path.parent.name
            models.append({
                "path": str(model_path),
                "agent_type": agent_type,
                "name": model_path.name,
                "display": f"{agent_type} - {model_path.name}"
            })

    return sorted(models, key=lambda x: x["agent_type"])


def select_model(models):
    """Prompt user to select a model."""
    if not models:
        print("No trained models found in logs folder!")
        print("Train a model first using: python train.py <agent> --timesteps <steps>")
        return None

    print("\n" + "=" * 60)
    print("Available Trained Models:")
    print("=" * 60)

    for i, model in enumerate(models, 1):
        print(f"{i}. {model['display']}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def select_environment():
    """Prompt user to select an environment."""
    environments = [
        {"id": "MiniWorld-OneRoom-v0", "name": "One Room", "description": "Simple room with a red box goal"},
        {"id": "MiniWorld-Maze-v0", "name": "Maze", "description": "Navigate through a maze to find the goal"},
    ]

    print("\n" + "=" * 60)
    print("Available Environments:")
    print("=" * 60)

    for i, env in enumerate(environments, 1):
        print(f"{i}. {env['name']:<15} - {env['description']}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect environment (1-{len(environments)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(environments):
                return environments[idx]["id"]
            else:
                print(f"Please enter a number between 1 and {len(environments)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def get_num_episodes():
    """Prompt user for number of episodes to run."""
    print("\n" + "=" * 60)
    default_episodes = 5

    while True:
        try:
            choice = input(f"Number of episodes to run (default: {default_episodes}): ").strip()
            if not choice:
                return default_episodes

            num = int(choice)
            if num > 0:
                return num
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def run_evaluation(model_info, env_id, n_episodes):
    """Run evaluation with the selected model and environment."""
    agent_type = infer_agent_type(model_info["path"])
    model_path = model_info["path"]
    render_fps = model_info.get("render_fps", 0.0)

    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    print(f"Model: {model_info['display']}")
    print(f"Agent Type: {agent_type}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {n_episodes}")
    print(f"Rendering: Enabled")
    if render_fps > 0:
        print(f"Render FPS: {render_fps}")
    print("=" * 60 + "\n")

    try:
        if agent_type == "dreamer":
            # Dreamer requires special handling
            from agents.dreamer import evaluate_dreamer, DreamerAgent, DreamerConfig
            import gymnasium as gym

            config = DreamerConfig(env_id=env_id)
            env = gym.make(env_id)
            obs_shape = env.observation_space.shape
            obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            action_size = env.action_space.n
            env.close()

            agent = DreamerAgent(config, obs_shape, action_size)
            agent.load(model_path)
            evaluate_dreamer(
                agent,
                env_id,
                n_episodes=n_episodes,
                render=True,
                render_fps=render_fps,
            )
        elif agent_type:
            # Use registry for standard agents
            components = load_agent_module(agent_type)
            evaluate_func = components["evaluate_func"]
            evaluate_func(
                model_path,
                env_id=env_id,
                n_episodes=n_episodes,
                render=True,
                render_fps=render_fps,
            )
        else:
            print(f"Unknown agent type for model: {model_path}")
            print("Could not determine how to load this model.")
            return False

        return True

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test trained RL agents with human rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for selections)
  python test.py

  # Direct mode (specify everything)
  python test.py --model logs/ppo_cnn/ppo_cnn_final.zip --env MiniWorld-OneRoom-v0 --episodes 5

  # List available models
  python test.py --list
        """
    )

    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--env", type=str, help="Environment ID (e.g., MiniWorld-OneRoom-v0)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=0.0,
        help="Throttle rendering to this FPS (0 disables throttling). Useful for slower playback.",
    )

    args = parser.parse_args()

    # Find available models
    models = find_final_models()

    # List mode
    if args.list:
        if not models:
            print("No trained models found in logs folder!")
            return

        print("\n" + "=" * 60)
        print("Available Trained Models:")
        print("=" * 60)
        for model in models:
            print(f"  {model['display']}")
            print(f"    Path: {model['path']}")
        print("=" * 60)
        return

    # Direct mode (non-interactive)
    if args.model and args.env:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return

        agent_type = infer_agent_type(model_path)
        model_info = {
            "path": model_path,
            "agent_type": agent_type,
            "name": os.path.basename(model_path),
            "display": f"{agent_type} - {os.path.basename(model_path)}",
            "render_fps": float(args.render_fps or 0.0),
        }

        success = run_evaluation(model_info, args.env, args.episodes)

        if success:
            print("\n" + "=" * 60)
            print("Evaluation Complete!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Evaluation Failed")
            print("=" * 60)
        return

    # Interactive mode
    print("\n" + "=" * 60)
    print("RL Agent Test Script")
    print("=" * 60)
    print("This script lets you test trained models with human rendering.")

    if not models:
        return

    # Select model
    model_info = select_model(models)
    if model_info is None:
        print("\nExiting...")
        return
    model_info["render_fps"] = float(args.render_fps or 0.0)

    # Select environment
    env_id = select_environment()
    if env_id is None:
        print("\nExiting...")
        return

    # Get number of episodes
    n_episodes = get_num_episodes()
    if n_episodes is None:
        print("\nExiting...")
        return

    # Run evaluation
    success = run_evaluation(model_info, env_id, n_episodes)

    if success:
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Evaluation Failed")
        print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
