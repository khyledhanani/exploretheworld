"""Generic evaluation utilities for all agents."""

import time
from typing import Dict, Optional, Callable, Any
import numpy as np

from utils.env_utils import make_vec_env


def compute_episode_stats(
    episode_rewards: list,
    episode_lengths: list,
    successes: list,
) -> Dict[str, float]:
    """Compute evaluation statistics from episode data."""
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = np.mean(successes)

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(mean_length),
        "success_rate": float(success_rate),
    }


def print_eval_results(results: Dict[str, float]) -> None:
    """Print evaluation results in standard format."""
    print("\nEvaluation Results:")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")


def evaluate_sb3_agent(
    model,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    frame_stack: int = 4,
    render: bool = False,
    render_fps: float = 0.0,
    deterministic: bool = True,
    is_recurrent: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Generic evaluation function for Stable Baselines 3 agents.

    Args:
        model: Trained SB3 model (PPO, RecurrentPPO, QRDQN, etc.)
        env_id: Environment ID to evaluate on
        n_episodes: Number of evaluation episodes
        frame_stack: Number of frames to stack
        render: Whether to render the environment
        render_fps: Target FPS for rendering (0 = no throttling)
        deterministic: Whether to use deterministic actions
        is_recurrent: Whether the model is recurrent (e.g., RecurrentPPO)
        seed: Random seed for evaluation environment

    Returns:
        Dictionary with evaluation metrics
    """
    # Create evaluation environment
    env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        frame_stack=frame_stack,
        time_penalty=0.0,  # No penalty for evaluation
        use_subproc=False,
        render_mode="human" if render else None,
        seed=seed,
    )

    episode_rewards = []
    episode_lengths = []
    successes = []

    # For recurrent models, track LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool) if is_recurrent else None

    obs = env.reset()
    episode_reward = 0
    episode_length = 0

    # Setup frame rate limiting
    frame_dt = (1.0 / float(render_fps)) if render and render_fps > 0 else None
    next_frame_t = time.perf_counter()

    while len(episode_rewards) < n_episodes:
        # Get action from model
        if is_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]
        episode_length += 1

        if is_recurrent:
            episode_starts = done

        # Handle rendering with optional FPS limiting
        if render:
            env.render()
            if frame_dt is not None:
                next_frame_t += frame_dt
                sleep_s = next_frame_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_t = time.perf_counter()

        if done[0]:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(float(info[0].get("success", episode_reward > 0)))

            episode_reward = 0
            episode_length = 0

            if is_recurrent:
                lstm_states = None

            obs = env.reset()

    env.close()

    results = compute_episode_stats(episode_rewards, episode_lengths, successes)
    print_eval_results(results)

    return results
