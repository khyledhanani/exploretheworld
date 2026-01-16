"""
RecurrentPPO (CNN-LSTM) Agent

For true partial observability where the agent needs to remember past observations
(corridors visited, turn history, etc.). LSTM maintains hidden state across timesteps.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, List

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utils.env_utils import make_vec_env
from utils.callbacks import MetricsCallback
from utils.feature_extractors import get_extractor
from utils.evaluator import evaluate_sb3_agent
from utils.wandb_utils import init_wandb, get_wandb_callback, finish_wandb


@dataclass
class RecurrentPPOConfig:
    """Configuration for RecurrentPPO agent."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    n_envs: int = 8
    # LSTM handles temporal info, less need for frame stacking
    frame_stack: int = 1
    time_penalty: float = -0.001

    # Feature extractor: "cnn", "vit", "vit_small"
    extractor: str = "cnn"
    features_dim: int = 256

    # RecurrentPPO hyperparameters
    # Slightly lower for LSTM stability
    learning_rate: float = 2.5e-4
    # Shorter rollouts work well with LSTM
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # LSTM configuration
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    shared_lstm: bool = True
    enable_critic_lstm: bool = False

    # Network architecture
    # Smaller after LSTM
    net_arch: tuple = (256,)

    # Training
    total_timesteps: int = 1_000_000
    seed: int = 42

    # Logging
    log_dir: str = "logs/recurrent_ppo"
    save_freq: int = 50000
    wandb_enabled: bool = True
    wandb_project: str = "worldmodels"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


def create_recurrent_ppo_agent(config: RecurrentPPOConfig) -> RecurrentPPO:
    """Create RecurrentPPO agent with specified configuration."""

    # Create vectorized environment with random seeds per episode
    env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        time_penalty=config.time_penalty,
        frame_stack=config.frame_stack,
        random_seed_per_episode=True,
    )

    # sb3-contrib constraint: shared LSTM requires enable_critic_lstm=False
    enable_critic_lstm = config.enable_critic_lstm
    if config.shared_lstm and enable_critic_lstm:
        enable_critic_lstm = False

    # Policy kwargs with configurable extractor and LSTM config
    extractor_class = get_extractor(config.extractor)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        lstm_hidden_size=config.lstm_hidden_size,
        n_lstm_layers=config.n_lstm_layers,
        shared_lstm=config.shared_lstm,
        enable_critic_lstm=enable_critic_lstm,
        net_arch=dict(pi=list(config.net_arch), vf=list(config.net_arch)),
    )

    # Create RecurrentPPO agent
    model = RecurrentPPO(
        policy="CnnLstmPolicy",
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


def train_recurrent_ppo(config: Optional[RecurrentPPOConfig] = None) -> RecurrentPPO:
    """Train RecurrentPPO agent."""

    if config is None:
        config = RecurrentPPOConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("RecurrentPPO (LSTM) Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Extractor: {config.extractor} (features_dim={config.features_dim})")
    print(f"Parallel envs: {config.n_envs}")
    print(f"LSTM hidden size: {config.lstm_hidden_size}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 60)

    run = init_wandb(config, "recurrent_ppo", log_dir=config.log_dir)
    model = create_recurrent_ppo_agent(config)

    # Setup callbacks
    callbacks = [
        MetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=config.save_freq // config.n_envs,
            save_path=os.path.join(config.log_dir, "checkpoints"),
            name_prefix="recurrent_ppo",
        ),
    ]
    wandb_callback = get_wandb_callback(config, log_dir=config.log_dir)
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    try:
        # Train
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(config.log_dir, "recurrent_ppo_final")
        model.save(final_path)
        print(f"Model saved to {final_path}")
    finally:
        finish_wandb(run)

    return model


def evaluate_recurrent_ppo(
    model_path: str,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
) -> Dict[str, float]:
    """Evaluate a trained RecurrentPPO model."""

    model = RecurrentPPO.load(model_path)

    return evaluate_sb3_agent(
        model=model,
        env_id=env_id,
        n_episodes=n_episodes,
        # LSTM handles temporal context
        frame_stack=1,
        render=render,
        render_fps=render_fps,
        deterministic=True,
        is_recurrent=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RecurrentPPO agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_recurrent_ppo(args.eval, env_id=args.env)
    else:
        config = RecurrentPPOConfig(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            lstm_hidden_size=args.lstm_hidden,
            seed=args.seed,
        )
        train_recurrent_ppo(config)
