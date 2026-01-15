"""
PPO CNN with FrameStack 
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utils.env_utils import make_vec_env
from utils.callbacks import MetricsCallback
from utils.feature_extractors import get_extractor
from utils.evaluator import evaluate_sb3_agent
import miniworld


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    n_envs: int = 8
    frame_stack: int = 4
    time_penalty: float = -0.001

    extractor: str = "cnn"
    features_dim: int = 256

    # hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # architecture (normally just go 2 layers 256 dim)
    net_arch: tuple = (256, 256)

    total_timesteps: int = 1_000_000
    seed: int = 42

    log_dir: str = "logs/ppo_cnn"
    save_freq: int = 50000


def create_ppo_agent(config: PPOConfig) -> PPO:
    """Create PPO agent with specified configuration."""

    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        time_penalty=config.time_penalty,
        frame_stack=config.frame_stack,
        random_seed_per_episode=True,
    )

    extractor_class = get_extractor(config.extractor)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        net_arch=dict(pi=list(config.net_arch), vf=list(config.net_arch)),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.log_dir,
        seed=config.seed,
        verbose=1,
    )

    return model


def train_ppo(config: Optional[PPOConfig] = None) -> PPO:
    """
    Train PPO agent.
    """

    if config is None:
        config = PPOConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Extractor: {config.extractor} (features_dim={config.features_dim})")
    print(f"Parallel envs: {config.n_envs}")
    print(f"Frame stack: {config.frame_stack}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 60)

    model = create_ppo_agent(config)

    callbacks = CallbackList([
        MetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=config.save_freq // config.n_envs,
            save_path=os.path.join(config.log_dir, "checkpoints"),
            name_prefix="ppo_cnn",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(config.log_dir, "ppo_cnn_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    return model


def evaluate_ppo(
    model_path: str,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evals
    """

    model = PPO.load(model_path)

    return evaluate_sb3_agent(
        model=model,
        env_id=env_id,
        n_episodes=n_episodes,
        frame_stack=4,
        render=render,
        render_fps=render_fps,
        deterministic=True,
        is_recurrent=False,
        seed=seed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO + CNN agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_ppo(args.eval)
    else:
        config = PPOConfig(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
        )
        train_ppo(config)
