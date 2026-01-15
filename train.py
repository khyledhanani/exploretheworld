#!/usr/bin/env python3
"""
Unified training script for all RL agents.

Usage:
    python train.py ppo --env MiniWorld-OneRoom-v0 --timesteps 500000
    python train.py recurrent --env MiniWorld-OneRoom-v0 --timesteps 500000
    python train.py qrdqn --env MiniWorld-OneRoom-v0 --timesteps 500000
    python train.py rnd --env MiniWorld-OneRoom-v0 --timesteps 1000000
    python train.py dreamer --env MiniWorld-OneRoom-v0 --steps 500000
"""

import argparse
import sys

from utils.registry import AGENT_REGISTRY, get_all_agents, load_agent_module


def create_parser():
    """Create argument parser with subcommands for each agent."""
    parser = argparse.ArgumentParser(
        description="Train RL agents for MiniWorld maze navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py ppo --timesteps 500000
  python train.py ppo --extractor vit --timesteps 500000
  python train.py ppo --extractor vit_small --features-dim 128
  python train.py recurrent --lstm-hidden 512
  python train.py qrdqn --buffer-size 200000
  python train.py rnd --intrinsic-coef 0.2
  python train.py dreamer --steps 500000
  python train.py ppo --eval logs/ppo_cnn/ppo_cnn_final.zip
        """
    )

    subparsers = parser.add_subparsers(dest="agent", help="Agent type to train")

    # Common arguments for all agents
    common_args = [
        ("--env", str, "MiniWorld-OneRoom-v0", "Environment ID"),
        ("--seed", int, 42, "Random seed"),
        ("--extractor", str, "cnn", "Feature extractor: cnn, vit, vit_small"),
        ("--features-dim", int, None, "Feature dimension (default varies by agent)"),
        ("--eval", str, None, "Path to model for evaluation"),
    ]

    # Create subparser for each agent
    for agent_name, info in AGENT_REGISTRY.items():
        subparser = subparsers.add_parser(agent_name, help=info.display_name)

        # Add common arguments
        for arg_name, arg_type, default, help_text in common_args:
            subparser.add_argument(arg_name, type=arg_type, default=default, help=help_text)

        # Add timesteps argument (different name for dreamer)
        if agent_name == "dreamer":
            subparser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
        else:
            subparser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")

        # Add agent-specific arguments
        for arg_name, arg_type, default, help_text in info.extra_args:
            subparser.add_argument(arg_name, type=arg_type, default=default, help=help_text)

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Run comparison of all agents")
    compare_parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    compare_parser.add_argument("--timesteps", type=int, default=500_000)
    compare_parser.add_argument("--seed", type=int, default=42)

    return parser


def train_agent(agent_name: str, args):
    """Train a specific agent with the given arguments."""
    components = load_agent_module(agent_name)
    config_class = components["config_class"]
    train_func = components["train_func"]
    evaluate_func = components["evaluate_func"]

    if args.eval:
        # Evaluation mode
        if agent_name == "dreamer":
            # Dreamer has special evaluation handling
            import gymnasium as gym
            from agents.dreamer import DreamerAgent, DreamerConfig

            config = DreamerConfig(env_id=args.env)
            env = gym.make(args.env)
            obs_shape = env.observation_space.shape
            obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            action_size = env.action_space.n
            env.close()

            agent = DreamerAgent(config, obs_shape, action_size)
            agent.load(args.eval)
            evaluate_func(agent, args.env, n_episodes=100)
        else:
            evaluate_func(args.eval, env_id=args.env)
    else:
        # Training mode - build config from args
        config_kwargs = {"env_id": args.env, "seed": args.seed}

        # Feature extractor config (not supported by dreamer)
        if agent_name != "dreamer":
            config_kwargs["extractor"] = args.extractor
            if args.features_dim is not None:
                config_kwargs["features_dim"] = args.features_dim

        # Map CLI args to config fields
        if agent_name == "dreamer":
            config_kwargs["total_steps"] = args.steps
        else:
            config_kwargs["total_timesteps"] = getattr(args, "timesteps", 1_000_000)

        # Agent-specific config mapping
        if agent_name == "ppo":
            config_kwargs["n_envs"] = args.n_envs
        elif agent_name == "recurrent":
            config_kwargs["n_envs"] = args.n_envs
            config_kwargs["lstm_hidden_size"] = args.lstm_hidden
        elif agent_name == "qrdqn":
            config_kwargs["buffer_size"] = args.buffer_size
            config_kwargs["n_quantiles"] = args.n_quantiles
        elif agent_name == "rnd":
            config_kwargs["n_envs"] = args.n_envs
            config_kwargs["intrinsic_reward_coef"] = args.intrinsic_coef

        config = config_class(**config_kwargs)
        train_func(config)


def run_comparison(env_id: str, timesteps: int, seed: int):
    """Run all agents and compare results."""
    import json
    import os
    from datetime import datetime

    results = {}

    agents = [
        ("ppo", "PPO + CNN"),
        ("recurrent", "RecurrentPPO"),
        ("qrdqn", "QR-DQN"),
        ("rnd", "PPO + RND"),
    ]

    for agent_name, display_name in agents:
        print(f"\n{'='*60}")
        print(f"Training {display_name}...")
        print(f"{'='*60}\n")

        try:
            components = load_agent_module(agent_name)
            config_class = components["config_class"]
            train_func = components["train_func"]
            evaluate_func = components["evaluate_func"]
            info = components["info"]

            # Build config
            config_kwargs = {"env_id": env_id, "seed": seed}

            if agent_name == "rnd":
                config_kwargs["total_timesteps"] = timesteps * 2  # RND needs more steps
            else:
                config_kwargs["total_timesteps"] = timesteps

            config = config_class(**config_kwargs)
            model = train_func(config)

            # Evaluate
            model_path = os.path.join(config.log_dir, f"{agent_name}_final.zip")
            if agent_name == "ppo":
                model_path = os.path.join(config.log_dir, "ppo_cnn_final.zip")
            elif agent_name == "recurrent":
                model_path = os.path.join(config.log_dir, "recurrent_ppo_final.zip")
            elif agent_name == "rnd":
                model_path = os.path.join(config.log_dir, "ppo_rnd_final.zip")

            eval_results = evaluate_func(model_path, env_id=env_id)
            results[display_name] = eval_results

        except Exception as e:
            print(f"Error training {display_name}: {e}")
            results[display_name] = {"error": str(e)}

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Agent':<20} {'Success Rate':<15} {'Mean Reward':<15}")
    print("-" * 60)

    for agent_name, result in results.items():
        if "error" in result:
            print(f"{agent_name:<20} {'ERROR':<15} {'':<15}")
        else:
            sr = f"{result['success_rate']:.2%}"
            mr = f"{result['mean_reward']:.2f}"
            print(f"{agent_name:<20} {sr:<15} {mr:<15}")

    # Save results
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/comparison_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to logs/comparison_{timestamp}.json")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.agent is None:
        parser.print_help()
        sys.exit(1)

    if args.agent == "compare":
        run_comparison(args.env, args.timesteps, args.seed)
    else:
        train_agent(args.agent, args)


if __name__ == "__main__":
    main()
