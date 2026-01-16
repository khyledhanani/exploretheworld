"""
RecurrentPPO + RND Agent

Combines:
- RecurrentPPO (LSTM) for handling partial observability and temporal dependencies
- RND (Random Network Distillation) for curiosity-driven exploration

Best for: Sparse reward environments where the agent needs both memory
(to track where it's been) and exploration incentives (to discover new areas).
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd
import gymnasium as gym

from utils.env_utils import make_vec_env
from utils.callbacks import MetricsCallback
from utils.feature_extractors import get_extractor
from utils.evaluator import evaluate_sb3_agent
from utils.wandb_utils import init_wandb, get_wandb_callback, finish_wandb
import miniworld


@dataclass
class RecurrentPPORNDConfig:
    """Configuration for RecurrentPPO + RND agent."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    n_envs: int = 8
    # LSTM handles temporal info, minimal frame stacking needed
    frame_stack: int = 1
    # No time penalty - RND provides exploration incentive
    time_penalty: float = 0.0

    # Feature extractor: "cnn", "vit", "vit_small"
    extractor: str = "cnn"
    features_dim: int = 256

    # RecurrentPPO hyperparameters
    learning_rate: float = 2.5e-4
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

    # RND configuration
    rnd_lr: float = 1e-4
    rnd_output_dim: int = 512
    intrinsic_reward_coef: float = 1.0
    intrinsic_reward_coef_min: float = 0.01
    intrinsic_decay_rate: float = 0.99995
    intrinsic_gamma: float = 0.99
    normalize_intrinsic: bool = True
    update_proportion: float = 0.25

    # Network architecture
    net_arch: tuple = (256,)

    # Training
    total_timesteps: int = 2_000_000
    seed: int = 42

    # Logging
    log_dir: str = "logs/recurrent_ppo_rnd"
    save_freq: int = 50000
    wandb_enabled: bool = True
    wandb_project: str = "worldmodels"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class RNDNetwork(nn.Module):
    """
    Random Network Distillation module.
    Contains both target (fixed) and predictor (trained) networks.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        output_dim: int = 512,
        device: str = "auto"
    ):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        n_input_channels = observation_space.shape[0]

        # Target network (random, fixed)
        self.target = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Predictor network (trained)
        self.predictor = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Compute flatten size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flatten_size = self.target(sample).shape[1]

        # Output projection layers
        self.target_head = nn.Linear(flatten_size, output_dim)
        self.predictor_head = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute target and predictor outputs."""
        obs = obs.to(self.device)

        with torch.no_grad():
            target_features = self.target_head(self.target(obs))

        predictor_features = self.predictor_head(self.predictor(obs))

        return target_features, predictor_features

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward as prediction error."""
        target_features, predictor_features = self.forward(obs)
        intrinsic_reward = torch.mean((target_features - predictor_features) ** 2, dim=1)
        return intrinsic_reward


class RNDRewardWrapper(VecEnvWrapper):
    """Wrapper that adds RND intrinsic rewards to environment rewards with decay."""

    def __init__(
        self,
        venv: VecEnv,
        rnd_network: RNDNetwork,
        intrinsic_coef: float = 1.0,
        intrinsic_coef_min: float = 0.01,
        decay_rate: float = 0.99995,
        normalize_intrinsic: bool = True,
        gamma: float = 0.99,
    ):
        super().__init__(venv)
        self.rnd = rnd_network
        self.intrinsic_coef_initial = intrinsic_coef
        self.intrinsic_coef_min = intrinsic_coef_min
        self.decay_rate = decay_rate
        self.normalize_intrinsic = normalize_intrinsic
        self.gamma = gamma

        # Track steps for decay
        self.total_steps = 0
        self._current_coef = intrinsic_coef

        if normalize_intrinsic:
            self.reward_rms = RunningMeanStd(shape=())

    @property
    def current_intrinsic_coef(self) -> float:
        """Get the current (decayed) intrinsic reward coefficient."""
        return self._current_coef

    def _update_coef(self, n_steps: int = 1):
        """Update the intrinsic coefficient with decay."""
        self.total_steps += n_steps
        # Exponential decay with floor
        decayed = self.intrinsic_coef_initial * (self.decay_rate ** self.total_steps)
        self._current_coef = max(decayed, self.intrinsic_coef_min)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Update decay (n_envs steps happened)
        self._update_coef(n_steps=self.num_envs)

        # Compute intrinsic rewards
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            intrinsic_rewards = self.rnd.compute_intrinsic_reward(obs_tensor)
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        # Normalize intrinsic rewards
        if self.normalize_intrinsic:
            self.reward_rms.update(intrinsic_rewards)
            intrinsic_rewards = intrinsic_rewards / np.sqrt(self.reward_rms.var + 1e-8)

        # Combine rewards with DECAYED coefficient
        combined_rewards = rewards + self._current_coef * intrinsic_rewards

        # Store info for logging
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = intrinsic_rewards[i]
            info["intrinsic_coef"] = self._current_coef

        return obs, combined_rewards, dones, infos


class RecurrentRNDCallback(BaseCallback):
    """Callback to train RND predictor network during RecurrentPPO training."""

    def __init__(
        self,
        rnd_network: RNDNetwork,
        rnd_wrapper: RNDRewardWrapper,
        learning_rate: float = 1e-4,
        update_proportion: float = 0.25,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rnd = rnd_network
        self.rnd_wrapper = rnd_wrapper
        self.update_proportion = update_proportion

        # Optimizer for predictor network only
        predictor_params = list(self.rnd.predictor.parameters()) + \
                          list(self.rnd.predictor_head.parameters())
        self.optimizer = torch.optim.Adam(predictor_params, lr=learning_rate)

        self.intrinsic_rewards = []
        self._wandb = None
        self._wandb_checked = False

    def _get_wandb(self):
        """Lazy load wandb if available."""
        if not self._wandb_checked:
            self._wandb_checked = True
            try:
                import wandb
                if wandb.run is not None:
                    self._wandb = wandb
            except ImportError:
                pass
        return self._wandb

    def _on_step(self) -> bool:
        # Log intrinsic rewards and coefficient
        current_coef = None
        for info in self.locals.get("infos", []):
            if "intrinsic_reward" in info:
                self.intrinsic_rewards.append(info["intrinsic_reward"])
            if "intrinsic_coef" in info:
                current_coef = info["intrinsic_coef"]

        if self.n_calls % 1000 == 0 and len(self.intrinsic_rewards) > 0:
            mean_intrinsic = np.mean(self.intrinsic_rewards[-1000:])
            self.logger.record("rnd/mean_intrinsic_reward", mean_intrinsic)

            # Log current coefficient
            if current_coef is not None:
                self.logger.record("rnd/intrinsic_coef", current_coef)

            # Also log directly to wandb
            wandb = self._get_wandb()
            if wandb is not None:
                log_dict = {
                    "rnd/mean_intrinsic_reward": mean_intrinsic,
                    "rnd/max_intrinsic_reward": np.max(self.intrinsic_rewards[-1000:]),
                    "rnd/min_intrinsic_reward": np.min(self.intrinsic_rewards[-1000:]),
                }
                if current_coef is not None:
                    log_dict["rnd/intrinsic_coef"] = current_coef
                wandb.log(log_dict, step=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        """Train RND on collected rollout data from RecurrentRolloutBuffer."""
        rollout_buffer = self.model.rollout_buffer

        # RecurrentRolloutBuffer stores observations with shape:
        # (buffer_size, n_seq, *obs_shape) where n_seq is the sequence length
        # We need to flatten and sample from these
        obs = rollout_buffer.observations

        # Flatten all dimensions except observation shape
        # Original shape: (buffer_size, n_seq, *obs_shape)
        original_shape = obs.shape
        obs_shape = original_shape[2:]  # Get the actual observation dimensions
        total_samples = original_shape[0] * original_shape[1]
        obs_flat = obs.reshape(total_samples, *obs_shape)

        # Sample subset of observations
        n_samples = int(total_samples * self.update_proportion)
        indices = np.random.choice(
            total_samples,
            size=min(n_samples, total_samples),
            replace=False
        )

        obs_sample = obs_flat[indices]

        # Train RND predictor
        obs_tensor = torch.FloatTensor(obs_sample).to(self.rnd.device)

        self.optimizer.zero_grad()
        target_features, predictor_features = self.rnd(obs_tensor)
        loss = F.mse_loss(predictor_features, target_features.detach())
        loss.backward()
        self.optimizer.step()

        self.logger.record("rnd/predictor_loss", loss.item())

        # Also log directly to wandb
        wandb = self._get_wandb()
        if wandb is not None:
            wandb.log({
                "rnd/predictor_loss": loss.item(),
            }, step=self.num_timesteps)


def create_recurrent_ppo_rnd_agent(
    config: RecurrentPPORNDConfig
) -> Tuple[RecurrentPPO, RNDNetwork, RNDRewardWrapper]:
    """Create RecurrentPPO + RND agent with specified configuration."""

    # Create base vectorized environment with random seeds per episode
    base_env = make_vec_env(
        env_id=config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        time_penalty=config.time_penalty,
        frame_stack=config.frame_stack,
        random_seed_per_episode=True,
    )

    # Create RND network
    rnd_network = RNDNetwork(
        observation_space=base_env.observation_space,
        output_dim=config.rnd_output_dim,
    )

    # Wrap environment with RND rewards (with decay)
    rnd_wrapper = RNDRewardWrapper(
        base_env,
        rnd_network=rnd_network,
        intrinsic_coef=config.intrinsic_reward_coef,
        intrinsic_coef_min=config.intrinsic_reward_coef_min,
        decay_rate=config.intrinsic_decay_rate,
        normalize_intrinsic=config.normalize_intrinsic,
        gamma=config.intrinsic_gamma,
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
        env=rnd_wrapper,
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

    return model, rnd_network, rnd_wrapper


def train_recurrent_ppo_rnd(
    config: Optional[RecurrentPPORNDConfig] = None
) -> Tuple[RecurrentPPO, RNDNetwork]:
    """Train RecurrentPPO + RND agent."""

    if config is None:
        config = RecurrentPPORNDConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("RecurrentPPO + RND (LSTM + Curiosity) Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Extractor: {config.extractor} (features_dim={config.features_dim})")
    print(f"Parallel envs: {config.n_envs}")
    print(f"LSTM hidden size: {config.lstm_hidden_size}")
    print(f"Intrinsic reward: {config.intrinsic_reward_coef} -> {config.intrinsic_reward_coef_min} (decay={config.intrinsic_decay_rate})")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 60)

    run = init_wandb(config, "recurrent_ppo_rnd", log_dir=config.log_dir)
    model, rnd_network, rnd_wrapper = create_recurrent_ppo_rnd_agent(config)

    # Setup callbacks
    callbacks = [
        MetricsCallback(verbose=1),
        RecurrentRNDCallback(
            rnd_network=rnd_network,
            rnd_wrapper=rnd_wrapper,
            learning_rate=config.rnd_lr,
            update_proportion=config.update_proportion,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=config.save_freq // config.n_envs,
            save_path=os.path.join(config.log_dir, "checkpoints"),
            name_prefix="recurrent_ppo_rnd",
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

        # Save final model and RND network
        final_path = os.path.join(config.log_dir, "recurrent_ppo_rnd_final")
        model.save(final_path)

        rnd_path = os.path.join(config.log_dir, "rnd_network.pt")
        torch.save(rnd_network.state_dict(), rnd_path)

        print(f"Model saved to {final_path}")
        print(f"RND network saved to {rnd_path}")
    finally:
        finish_wandb(run)

    return model, rnd_network


def evaluate_recurrent_ppo_rnd(
    model_path: str,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
) -> Dict[str, float]:
    """Evaluate a trained RecurrentPPO + RND model (without intrinsic rewards)."""

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

    parser = argparse.ArgumentParser(description="Train RecurrentPPO + RND agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--intrinsic-coef", type=float, default=1.0,
                        help="Initial intrinsic reward coefficient")
    parser.add_argument("--intrinsic-coef-min", type=float, default=0.01,
                        help="Minimum intrinsic reward coefficient after decay")
    parser.add_argument("--decay-rate", type=float, default=0.99995,
                        help="Per-step decay rate for intrinsic coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_recurrent_ppo_rnd(args.eval, env_id=args.env)
    else:
        config = RecurrentPPORNDConfig(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            lstm_hidden_size=args.lstm_hidden,
            intrinsic_reward_coef=args.intrinsic_coef,
            intrinsic_reward_coef_min=args.intrinsic_coef_min,
            intrinsic_decay_rate=args.decay_rate,
            seed=args.seed,
        )
        train_recurrent_ppo_rnd(config)
