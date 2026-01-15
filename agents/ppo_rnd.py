"""
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd
import gymnasium as gym

from utils.env_utils import make_vec_env
from utils.callbacks import MetricsCallback
from utils.feature_extractors import get_extractor
from utils.evaluator import evaluate_sb3_agent
import miniworld


@dataclass
class PPORNDConfig:
    """Configuration for PPO + RND agent."""

    # Environment
    env_id: str = "MiniWorld-OneRoom-v0"
    n_envs: int = 8
    frame_stack: int = 4
    # No time penalty - RND provides exploration
    time_penalty: float = 0.0

    extractor: str = "cnn"
    features_dim: int = 256

    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    rnd_lr: float = 1e-4
    rnd_output_dim: int = 512
    intrinsic_reward_coef: float = 0.1
    intrinsic_gamma: float = 0.99
    normalize_intrinsic: bool = True
    update_proportion: float = 0.25

    # Network architecture
    net_arch: tuple = (256, 256)

    # Training
    # RND needs more steps
    total_timesteps: int = 2_000_000
    seed: int = 42

    log_dir: str = "logs/ppo_rnd"
    save_freq: int = 50000


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
    """Wrapper that adds RND intrinsic rewards to environment rewards."""

    def __init__(
        self,
        venv: VecEnv,
        rnd_network: RNDNetwork,
        intrinsic_coef: float = 0.1,
        normalize_intrinsic: bool = True,
        gamma: float = 0.99,
    ):
        super().__init__(venv)
        self.rnd = rnd_network
        self.intrinsic_coef = intrinsic_coef
        self.normalize_intrinsic = normalize_intrinsic
        self.gamma = gamma

        if normalize_intrinsic:
            self.reward_rms = RunningMeanStd(shape=())

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Compute intrinsic rewards
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            intrinsic_rewards = self.rnd.compute_intrinsic_reward(obs_tensor)
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        # Normalize intrinsic rewards
        if self.normalize_intrinsic:
            self.reward_rms.update(intrinsic_rewards)
            intrinsic_rewards = intrinsic_rewards / np.sqrt(self.reward_rms.var + 1e-8)

        # Combine rewards
        combined_rewards = rewards + self.intrinsic_coef * intrinsic_rewards

        # Store intrinsic reward in info for logging
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = intrinsic_rewards[i]

        return obs, combined_rewards, dones, infos


class RNDCallback(BaseCallback):
    """Callback to train RND predictor network during PPO training."""

    def __init__(
        self,
        rnd_network: RNDNetwork,
        learning_rate: float = 1e-4,
        update_proportion: float = 0.25,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rnd = rnd_network
        self.update_proportion = update_proportion

        # Optimizer for predictor network only
        predictor_params = list(self.rnd.predictor.parameters()) + \
                          list(self.rnd.predictor_head.parameters())
        self.optimizer = torch.optim.Adam(predictor_params, lr=learning_rate)

        self.intrinsic_rewards = []

    def _on_step(self) -> bool:
        # Log intrinsic rewards
        for info in self.locals.get("infos", []):
            if "intrinsic_reward" in info:
                self.intrinsic_rewards.append(info["intrinsic_reward"])

        if self.n_calls % 1000 == 0 and len(self.intrinsic_rewards) > 0:
            mean_intrinsic = np.mean(self.intrinsic_rewards[-1000:])
            self.logger.record("rnd/mean_intrinsic_reward", mean_intrinsic)

        return True

    def _on_rollout_end(self) -> None:
        """Train RND on collected rollout data."""
        rollout_buffer = self.model.rollout_buffer

        # Sample subset of observations
        n_samples = int(rollout_buffer.buffer_size * rollout_buffer.n_envs * self.update_proportion)
        indices = np.random.choice(
            rollout_buffer.buffer_size * rollout_buffer.n_envs,
            size=min(n_samples, rollout_buffer.buffer_size * rollout_buffer.n_envs),
            replace=False
        )

        # Get observations
        obs = rollout_buffer.observations.reshape(-1, *rollout_buffer.observations.shape[2:])
        obs_sample = obs[indices]

        # Train RND predictor
        obs_tensor = torch.FloatTensor(obs_sample).to(self.rnd.device)

        self.optimizer.zero_grad()
        target_features, predictor_features = self.rnd(obs_tensor)
        loss = F.mse_loss(predictor_features, target_features.detach())
        loss.backward()
        self.optimizer.step()

        self.logger.record("rnd/predictor_loss", loss.item())


def create_ppo_rnd_agent(config: PPORNDConfig) -> Tuple[PPO, RNDNetwork]:
    """Create PPO + RND agent with specified configuration."""

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

    # Wrap environment with RND rewards
    env = RNDRewardWrapper(
        base_env,
        rnd_network=rnd_network,
        intrinsic_coef=config.intrinsic_reward_coef,
        normalize_intrinsic=config.normalize_intrinsic,
        gamma=config.intrinsic_gamma,
    )

    # Policy kwargs with configurable feature extractor
    extractor_class = get_extractor(config.extractor)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(features_dim=config.features_dim),
        net_arch=dict(pi=list(config.net_arch), vf=list(config.net_arch)),
    )

    # Create PPO agent
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

    return model, rnd_network


def train_ppo_rnd(config: Optional[PPORNDConfig] = None) -> Tuple[PPO, RNDNetwork]:
    """Train PPO + RND agent."""

    if config is None:
        config = PPORNDConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    print("=" * 60)
    print("PPO + RND (Curiosity-Driven) Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Extractor: {config.extractor} (features_dim={config.features_dim})")
    print(f"Parallel envs: {config.n_envs}")
    print(f"Intrinsic reward coefficient: {config.intrinsic_reward_coef}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print("=" * 60)

    model, rnd_network = create_ppo_rnd_agent(config)

    # Setup callbacks
    callbacks = CallbackList([
        MetricsCallback(verbose=1),
        RNDCallback(
            rnd_network=rnd_network,
            learning_rate=config.rnd_lr,
            update_proportion=config.update_proportion,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=config.save_freq // config.n_envs,
            save_path=os.path.join(config.log_dir, "checkpoints"),
            name_prefix="ppo_rnd",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model and RND network
    final_path = os.path.join(config.log_dir, "ppo_rnd_final")
    model.save(final_path)

    rnd_path = os.path.join(config.log_dir, "rnd_network.pt")
    torch.save(rnd_network.state_dict(), rnd_path)

    print(f"Model saved to {final_path}")
    print(f"RND network saved to {rnd_path}")

    return model, rnd_network


def evaluate_ppo_rnd(
    model_path: str,
    env_id: str = "MiniWorld-OneRoom-v0",
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
) -> Dict[str, float]:
    """Evaluate a trained PPO + RND model (without intrinsic rewards)."""

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
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO + RND agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--intrinsic-coef", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_ppo_rnd(args.eval, env_id=args.env)
    else:
        config = PPORNDConfig(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            intrinsic_reward_coef=args.intrinsic_coef,
            seed=args.seed,
        )
        train_ppo_rnd(config)
