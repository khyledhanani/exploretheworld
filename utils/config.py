"""Base configuration classes for RL agents."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BaseAgentConfig:
    """Base configuration shared by all agents."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    n_envs: int = 8
    frame_stack: int = 4
    time_penalty: float = -0.001

    # Feature extractor: "cnn", "vit", "vit_small"
    extractor: str = "cnn"
    features_dim: int = 256

    # Common hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 42

    # Training
    total_timesteps: int = 1_000_000

    # Logging
    log_dir: str = "logs/agent"
    save_freq: int = 50000


@dataclass
class PPOBaseConfig(BaseAgentConfig):
    """Base configuration for PPO-based agents (PPO, RecurrentPPO, RND)."""

    # PPO hyperparameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    net_arch: Tuple[int, ...] = (256, 256)
