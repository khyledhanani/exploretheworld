"""
QR-DQN (Quantile Regression DQN) Agent

Distributional value-based method that models the full return distribution.
Very sample efficient for discrete action spaces like MiniWorld navigation.
Uses n-step returns to propagate sparse rewards through the maze.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utils.env_utils import make_vec_env
from utils.callbacks import MetricsCallback
from utils.feature_extractors import get_extractor
from utils.evaluator import evaluate_sb3_agent


@dataclass
class QRDQNConfig:
    """Configuration for QR-DQN agent."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    # DQN typically uses single env with replay buffer
    n_envs: int = 1
    frame_stack: int = 4
    time_penalty: float = -0.001

    # Feature extractor: "cnn", "vit", "vit_small"
    extractor: str = "cnn"
    # QR-DQN uses larger feature dim
    features_dim: int = 512

    # QR-DQN hyperparameters
    learning_rate: float = 1e-4
    buffer_size: int = 500_000
    learning_starts: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    # Hard update
    tau: float = 1.0
    target_update_interval: int = 10_000
    train_freq: int = 4
    gradient_steps: int = 1

    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # QR-DQN specific
    n_quantiles: int = 200

    # Training
    total_timesteps: int = 1_000_000
    seed: int = 42

    # Logging
    log_dir: str = "logs/qrdqn"
    save_freq: int = 50000


def create_qrdqn_agent(config: QRDQNConfig) -> QRDQN:
    """Create QR-DQN agent with specified configuration."""

    # Create vectorized environment with random seeds per episode
    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        time_penalty=config.time_penalty,
        frame_stack=config.frame_stack,
        # Single env for DQN
        use_subproc=False,
        random_seed_per_episode=True,
    )

    # Policy kwargs with configurable feature extractor
    extractor_class = get_extractor(config.extractor)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        n_quantiles=config.n_quantiles,
        net_arch=[256, 256],
    )

    # Create QR-DQN agent
    model = QRDQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        target_update_interval=config.target_update_interval,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.log_dir,
        seed=config.seed,
        verbose=1,
    )

    return model


def train_qrdqn(config: Optional[QRDQNConfig] = None) -> QRDQN:
    """Train QR-DQN agent."""

    if config is None:
        config = QRDQNConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("QR-DQN Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Extractor: {config.extractor} (features_dim={config.features_dim})")
    print(f"Buffer size: {config.buffer_size:,}")
    print(f"Quantiles: {config.n_quantiles}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 60)

    model = create_qrdqn_agent(config)

    # Setup callbacks
    callbacks = CallbackList([
        MetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=config.save_freq,
            save_path=os.path.join(config.log_dir, "checkpoints"),
            name_prefix="qrdqn",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(config.log_dir, "qrdqn_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    return model


def evaluate_qrdqn(
    model_path: str,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
) -> Dict[str, float]:
    """Evaluate a trained QR-DQN model."""

    model = QRDQN.load(model_path)

    return evaluate_sb3_agent(
        model=model,
        env_id=env_id,
        n_episodes=n_episodes,
        frame_stack=4,
        render=render,
        render_fps=render_fps,
        deterministic=True,
        is_recurrent=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train QR-DQN agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--n-quantiles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_qrdqn(args.eval)
    else:
        config = QRDQNConfig(
            env_id=args.env,
            total_timesteps=args.timesteps,
            buffer_size=args.buffer_size,
            n_quantiles=args.n_quantiles,
            seed=args.seed,
        )
        train_qrdqn(config)
