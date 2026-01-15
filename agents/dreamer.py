"""
Mbased on dreamer -  model-based RL that learns a latent dynamics model from images and trains
an actor-critic entirely in imagination. Particularly powerful for visual
navigation with sparse rewards.

- Encoder: CNN or vit that maps observations to latent representations
- RSSM: Recurrent State-Space Model for latent dynamics
- Decoder: Reconstructs observations (optional, for auxiliary loss)
- Reward Model: Predicts rewards from latent states
- Actor-Critic: Policy and value function operating in latent space
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import miniworld.envs 
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DreamerConfig:
    """Configuration for Dreamer agent."""
    env_id: str = "MiniWorld-OneRoom-v0"
    action_repeat: int = 1
    time_penalty: float = 0.0

    # World model dimensions
    # Stochastic latent dimension
    stoch_size: int = 32
    # Deterministic (GRU) hidden dimension
    deter_size: int = 256
    # MLP hidden dimension
    hidden_size: int = 256
    # Base CNN channel depth
    cnn_depth: int = 32

    batch_size: int = 50
    # Sequence length for training
    batch_length: int = 50
    world_lr: float = 3e-4
    kl_scale: float = 1.0
    # Balance between prior and posterior KL
    kl_balance: float = 0.8
    # KL free bits
    free_nats: float = 1.0

    # Actor-critic
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    imagination_horizon: int = 15
    gamma: float = 0.99
    lambda_gae: float = 0.95
    actor_entropy: float = 1e-3
    # 'reinforce' or 'dynamics'
    actor_grad: str = "reinforce"

    # Replay buffer
    buffer_size: int = 1_000_000
    prefill_steps: int = 5000

    # Training
    total_steps: int = 1_000_000
    train_every: int = 5
    train_steps: int = 100
    seed: int = 42

    # Logging
    log_dir: str = "logs/dreamer"
    save_freq: int = 50000
    eval_freq: int = 10000


class ConvEncoder(nn.Module):
    """CNN encoder for visual observations."""

    def __init__(self, obs_shape: Tuple[int, ...], depth: int = 32):
        super().__init__()
        channels = obs_shape[0]

        self.convs = nn.Sequential(
            nn.Conv2d(channels, depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.output_size = self.convs(dummy).shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        return self.convs(obs)


class ConvDecoder(nn.Module):
    """CNN decoder for observation reconstruction."""

    def __init__(self, latent_size: int, obs_shape: Tuple[int, ...], depth: int = 32):
        super().__init__()
        self.obs_shape = obs_shape
        channels = obs_shape[0]

        self.fc = nn.Linear(latent_size, 32 * depth)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(depth, channels, 6, stride=2),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(x.size(0), -1, 1, 1)
        return self.deconvs(x)


class RSSM(nn.Module):
    """
    Recurrent State-Space Model.

    State consists of:
    - Deterministic hidden state h (GRU output)
    - Stochastic latent z (sampled from learned distribution)

    Prior: p(z_t | h_t) - predict z from h alone (imagination)
    Posterior: q(z_t | h_t, o_t) - infer z from h and observation (training)
    """

    def __init__(
        self,
        embed_size: int,
        action_size: int,
        stoch_size: int = 32,
        deter_size: int = 256,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size

        # GRU for deterministic state
        self.gru_cell = nn.GRUCell(stoch_size + action_size, deter_size)

        # Prior: p(z_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )

        # Posterior: q(z_t | h_t, o_t)
        self.posterior_fc = nn.Sequential(
            nn.Linear(deter_size + embed_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter_size, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
        }

    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Single observation step (for training).
        Returns prior and posterior distributions.
        """
        # Deterministic step
        x = torch.cat([prev_state["stoch"], action], dim=-1)
        deter = self.gru_cell(x, prev_state["deter"])

        # Prior
        prior_params = self.prior_fc(deter)
        prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1

        # Posterior
        posterior_params = self.posterior_fc(torch.cat([deter, embed], dim=-1))
        post_mean, post_std = torch.chunk(posterior_params, 2, dim=-1)
        post_std = F.softplus(post_std) + 0.1

        # Sample from posterior
        post_dist = td.Normal(post_mean, post_std)
        stoch = post_dist.rsample()

        state = {"deter": deter, "stoch": stoch}
        prior = {"mean": prior_mean, "std": prior_std}
        posterior = {"mean": post_mean, "std": post_std}

        return state, prior, posterior

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single imagination step (for actor-critic training).
        Samples from prior without observation.
        """
        # Deterministic step
        x = torch.cat([prev_state["stoch"], action], dim=-1)
        deter = self.gru_cell(x, prev_state["deter"])

        # Prior
        prior_params = self.prior_fc(deter)
        prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1

        # Sample from prior
        prior_dist = td.Normal(prior_mean, prior_std)
        stoch = prior_dist.rsample()

        return {"deter": deter, "stoch": stoch}

    def get_feature(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate deterministic and stochastic parts."""
        return torch.cat([state["deter"], state["stoch"]], dim=-1)


class RewardModel(nn.Module):
    """Predict rewards from latent states."""

    def __init__(self, latent_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.fc(latent).squeeze(-1)


class TerminalModel(nn.Module):
    """Predict episode termination from latent states."""

    def __init__(self, latent_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(latent)).squeeze(-1)


class Actor(nn.Module):
    """Policy network operating in latent space."""

    def __init__(self, latent_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.action_size = action_size

        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, latent: torch.Tensor) -> td.Distribution:
        logits = self.fc(latent)
        return td.Categorical(logits=logits)


class Critic(nn.Module):
    """Value network operating in latent space."""

    def __init__(self, latent_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.fc(latent).squeeze(-1)


class ReplayBuffer:
    """Simple replay buffer storing episodes."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, terminal: bool):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = terminal

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_length: int) -> Dict[str, torch.Tensor]:
        """Sample batch of sequences."""
        if self.size < seq_length:
            raise ValueError(f"Not enough data to sample: size={self.size}, seq_length={seq_length}")

        # Map a "logical" index (0=oldest, size-1=newest) to a physical index in the ring buffer.
        def _phys(logical_idx: int) -> int:
            # Oldest element is at (idx - size) mod capacity.
            return (self.idx - self.size + logical_idx) % self.capacity

        # Rejection sample valid sequence starts that do not cross terminals (except possibly at last step).
        starts: List[int] = []
        max_start = self.size - seq_length
        max_tries = max(1000, batch_size * 200)
        tries = 0

        while len(starts) < batch_size and tries < max_tries:
            tries += 1
            start = random.randint(0, max_start)
            idxs = [(_phys(start + t)) for t in range(seq_length)]
            if np.any(self.terminals[idxs[:-1]]):
                continue
            starts.append(start)

        if len(starts) == 0:
            raise RuntimeError("Could not sample any valid sequences (too many terminals / insufficient contiguous segments).")

        # Gather sequences using physical indices.
        obs_batch = []
        action_batch = []
        reward_batch = []
        terminal_batch = []

        for start in starts:
            idxs = [(_phys(start + t)) for t in range(seq_length)]
            obs_batch.append(self.observations[idxs])
            action_batch.append(self.actions[idxs])
            reward_batch.append(self.rewards[idxs])
            terminal_batch.append(self.terminals[idxs])

        obs_batch = np.stack(obs_batch, axis=0)
        action_batch = np.stack(action_batch, axis=0)
        reward_batch = np.stack(reward_batch, axis=0)
        terminal_batch = np.stack(terminal_batch, axis=0)

        # Keep observations as uint8 so the encoder can consistently normalize.
        return {
            "observations": torch.from_numpy(obs_batch),
            "actions": torch.from_numpy(action_batch).long(),
            "rewards": torch.from_numpy(reward_batch).float(),
            "terminals": torch.from_numpy(terminal_batch).float(),
        }


class DreamerAgent:
    """
    Dreamer agent combining world model with actor-critic in imagination.
    """

    def __init__(self, config: DreamerConfig, obs_shape: Tuple[int, ...], action_size: int):
        self.config = config
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        latent_size = config.deter_size + config.stoch_size

        # World model components
        self.encoder = ConvEncoder(obs_shape, config.cnn_depth).to(self.device)
        self.rssm = RSSM(
            self.encoder.output_size,
            action_size,
            config.stoch_size,
            config.deter_size,
            config.hidden_size,
        ).to(self.device)
        self.reward_model = RewardModel(latent_size, config.hidden_size).to(self.device)
        self.terminal_model = TerminalModel(latent_size, config.hidden_size).to(self.device)

        # Actor-critic
        self.actor = Actor(latent_size, action_size, config.hidden_size).to(self.device)
        self.critic = Critic(latent_size, config.hidden_size).to(self.device)

        # Optimizers
        world_params = (
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.terminal_model.parameters())
        )
        self.world_optimizer = torch.optim.Adam(world_params, lr=config.world_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # State for acting
        self.state = None
        self._last_action = 0

    def reset(self):
        """Reset agent state for new episode."""
        self.state = self.rssm.initial_state(1, self.device)
        self._last_action = 0

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select action given observation."""
        with torch.no_grad():
            # Keep obs as uint8 when possible; ConvEncoder handles normalization.
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            embed = self.encoder(obs_tensor)

            # Update state with observation
            action_onehot = torch.zeros(1, self.action_size, device=self.device)
            if hasattr(self, '_last_action'):
                action_onehot[0, self._last_action] = 1.0

            self.state, _, _ = self.rssm.observe_step(self.state, action_onehot, embed)

            # Get action from policy
            feature = self.rssm.get_feature(self.state)
            action_dist = self.actor(feature)

            if training:
                action = action_dist.sample().item()
            else:
                action = action_dist.probs.argmax().item()

            self._last_action = action
            return action

    def train_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train world model on batch of sequences."""
        obs = batch["observations"].to(self.device)  # (B, T, C, H, W)
        actions = batch["actions"].to(self.device)  # (B, T)
        rewards = batch["rewards"].to(self.device)  # (B, T)
        terminals = batch["terminals"].to(self.device)  # (B, T)

        B, T = obs.shape[:2]

        # One-hot encode actions
        actions_onehot = F.one_hot(actions, self.action_size).float()

        # Encode all observations
        obs_flat = obs.view(B * T, *self.obs_shape)
        embed_flat = self.encoder(obs_flat)
        embed = embed_flat.view(B, T, -1)

        # Roll through RSSM
        state = self.rssm.initial_state(B, self.device)
        priors, posteriors = [], []
        states = []

        for t in range(T):
            state, prior, posterior = self.rssm.observe_step(
                state, actions_onehot[:, t], embed[:, t]
            )
            priors.append(prior)
            posteriors.append(posterior)
            states.append(state)

        # Stack states
        features = torch.stack([self.rssm.get_feature(s) for s in states], dim=1)

        # Reconstruction loss (reward prediction)
        pred_rewards = self.reward_model(features)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # Terminal prediction loss
        pred_terminals = self.terminal_model(features)
        terminal_loss = F.binary_cross_entropy(pred_terminals, terminals)

        # KL divergence loss
        kl_loss = 0.0
        for prior, post in zip(priors, posteriors):
            prior_dist = td.Normal(prior["mean"], prior["std"])
            post_dist = td.Normal(post["mean"], post["std"])

            # KL balancing
            kl_post = td.kl_divergence(post_dist, prior_dist).sum(-1)
            kl_prior = td.kl_divergence(
                td.Normal(post["mean"].detach(), post["std"].detach()),
                prior_dist
            ).sum(-1)

            kl = self.config.kl_balance * kl_post + (1 - self.config.kl_balance) * kl_prior
            kl = torch.clamp(kl, min=self.config.free_nats).mean()
            kl_loss += kl

        kl_loss /= T

        # Total loss
        loss = reward_loss + terminal_loss + self.config.kl_scale * kl_loss

        self.world_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.terminal_model.parameters()),
            100.0
        )
        self.world_optimizer.step()

        return {
            "world/loss": loss.item(),
            "world/reward_loss": reward_loss.item(),
            "world/terminal_loss": terminal_loss.item(),
            "world/kl_loss": kl_loss.item(),
        }

    def train_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train actor-critic by imagining trajectories."""
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        B, T = obs.shape[:2]
        H = self.config.imagination_horizon

        # Get starting states from real data
        with torch.no_grad():
            obs_flat = obs.view(B * T, *self.obs_shape)
            embed_flat = self.encoder(obs_flat)
            embed = embed_flat.view(B, T, -1)

            # Roll through RSSM on real data and pick a random starting timestep per batch element.
            start_t = torch.randint(0, T, (B,), device=self.device)
            state = self.rssm.initial_state(B, self.device)
            actions_onehot = F.one_hot(actions, self.action_size).float()

            deters: List[torch.Tensor] = []
            stochs: List[torch.Tensor] = []

            for t in range(T):
                state, _, _ = self.rssm.observe_step(state, actions_onehot[:, t], embed[:, t])
                deters.append(state["deter"])
                stochs.append(state["stoch"])

            deter_seq = torch.stack(deters, dim=1)  # (B, T, D_deter)
            stoch_seq = torch.stack(stochs, dim=1)  # (B, T, D_stoch)
            batch_idx = torch.arange(B, device=self.device)
            state = {
                "deter": deter_seq[batch_idx, start_t],
                "stoch": stoch_seq[batch_idx, start_t],
            }

        # Imagine trajectories
        imagined_features = []
        imagined_actions = []
        imagined_action_log_probs = []

        for h in range(H):
            feature = self.rssm.get_feature(state)
            imagined_features.append(feature)

            # Sample action from actor
            action_dist = self.actor(feature)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            imagined_actions.append(action)
            imagined_action_log_probs.append(action_log_prob)

            # One-hot for RSSM
            action_onehot = F.one_hot(action, self.action_size).float()

            # Imagine next state
            with torch.no_grad():
                state = self.rssm.imagine_step(state, action_onehot)

        imagined_features = torch.stack(imagined_features, dim=1)  # (B, H, D)
        imagined_action_log_probs = torch.stack(imagined_action_log_probs, dim=1)  # (B, H)

        # Predict rewards and values
        with torch.no_grad():
            imagined_rewards = self.reward_model(imagined_features)
            imagined_terminals = self.terminal_model(imagined_features)

        imagined_values = self.critic(imagined_features)

        # Compute lambda returns
        with torch.no_grad():
            returns = self._compute_lambda_returns(
                imagined_rewards,
                imagined_values.detach(),
                imagined_terminals,
                self.config.gamma,
                self.config.lambda_gae,
            )

        # Critic loss
        critic_loss = F.mse_loss(imagined_values[:, :-1], returns[:, :-1])

        # Actor loss (REINFORCE style)
        advantages = (returns - imagined_values).detach()
        actor_loss = -(imagined_action_log_probs[:, :-1] * advantages[:, :-1]).mean()

        # Entropy bonus
        entropy = -imagined_action_log_probs[:, :-1].mean()
        actor_loss -= self.config.actor_entropy * entropy

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        return {
            "actor/loss": actor_loss.item(),
            "critic/loss": critic_loss.item(),
            "actor/entropy": entropy.item(),
            "actor/mean_value": imagined_values.mean().item(),
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float,
        lambda_: float,
    ) -> torch.Tensor:
        """Compute GAE-style lambda returns."""
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)

        # Bootstrap from final value
        returns[:, -1] = values[:, -1]

        for t in reversed(range(H - 1)):
            cont = 1.0 - terminals[:, t]
            returns[:, t] = rewards[:, t] + gamma * cont * (
                (1 - lambda_) * values[:, t + 1] + lambda_ * returns[:, t + 1]
            )

        return returns

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "terminal_model": self.terminal_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.terminal_model.load_state_dict(checkpoint["terminal_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


def train_dreamer(config: Optional[DreamerConfig] = None) -> DreamerAgent:
    """Train Dreamer agent."""

    if config is None:
        config = DreamerConfig()

    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    # Create environment
    env = gym.make(config.env_id)
    obs_shape = env.observation_space.shape
    # Transpose to channels-first
    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    action_size = env.action_space.n

    print("=" * 60)
    print("Dreamer World Model Training")
    print("=" * 60)
    print(f"Environment: {config.env_id}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_size}")
    print(f"Total steps: {config.total_steps:,}")
    print("=" * 60)

    # Create agent and replay buffer
    agent = DreamerAgent(config, obs_shape, action_size)
    buffer = ReplayBuffer(config.buffer_size, obs_shape, action_size)

    # Prefill buffer with random actions
    print("Prefilling replay buffer...")
    obs, _ = env.reset(seed=config.seed)
    for _ in tqdm(range(config.prefill_steps)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Transpose observation to channels-first
        obs_transposed = np.transpose(obs, (2, 0, 1))
        buffer.add(obs_transposed, action, reward + config.time_penalty, done)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Training loop
    print("\nStarting training...")
    obs, _ = env.reset()
    agent.reset()

    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_rewards = []

    for step in tqdm(range(config.total_steps)):
        # Act in environment
        obs_transposed = np.transpose(obs, (2, 0, 1))
        action = agent.act(obs_transposed, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs_transposed, action, reward + config.time_penalty, done)

        episode_reward += reward
        episode_length += 1

        if done:
            episode_rewards.append(episode_reward)
            writer.add_scalar("train/episode_reward", episode_reward, step)
            writer.add_scalar("train/episode_length", episode_length, step)
            episode_count += 1

            obs, _ = env.reset()
            agent.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Train
        if step % config.train_every == 0 and buffer.size > config.batch_size * config.batch_length:
            for _ in range(config.train_steps):
                batch = buffer.sample(config.batch_size, config.batch_length)

                # Train world model
                world_metrics = agent.train_world_model(batch)

                # Train actor-critic
                ac_metrics = agent.train_actor_critic(batch)

            # Log metrics
            for k, v in {**world_metrics, **ac_metrics}.items():
                writer.add_scalar(k, v, step)

        # Evaluate
        if step % config.eval_freq == 0 and step > 0:
            eval_rewards = evaluate_dreamer(agent, config.env_id, n_episodes=10)
            mean_reward = np.mean(eval_rewards)
            writer.add_scalar("eval/mean_reward", mean_reward, step)
            print(f"\nStep {step}: Eval mean reward = {mean_reward:.2f}")

        # Save
        if step % config.save_freq == 0 and step > 0:
            agent.save(os.path.join(config.log_dir, f"dreamer_{step}.pt"))

    # Save final model
    agent.save(os.path.join(config.log_dir, "dreamer_final.pt"))
    print(f"Model saved to {config.log_dir}/dreamer_final.pt")

    env.close()
    writer.close()

    return agent


def evaluate_dreamer(
    agent: DreamerAgent,
    env_id: str,
    n_episodes: int = 100,
    render: bool = False,
    render_fps: float = 0.0,
) -> List[float]:
    """Evaluate Dreamer agent."""

    env = gym.make(env_id, render_mode="human" if render else None)
    episode_rewards = []
    frame_dt = (1.0 / float(render_fps)) if render and render_fps and render_fps > 0 else None
    import time
    next_frame_t = time.perf_counter()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        agent.reset()
        episode_reward = 0
        done = False

        while not done:
            obs_transposed = np.transpose(obs, (2, 0, 1))
            action = agent.act(obs_transposed, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if render:
                env.render()
                if frame_dt is not None:
                    next_frame_t += frame_dt
                    sleep_s = next_frame_t - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_frame_t = time.perf_counter()

        episode_rewards.append(episode_reward)

    env.close()

    if len(episode_rewards) > 0:
        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(f"  Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"  Success Rate: {np.mean([r > 0 for r in episode_rewards]):.2%}")

    return episode_rewards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Dreamer agent")
    parser.add_argument("--env", type=str, default="MiniWorld-OneRoom-v0")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", type=str, help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        config = DreamerConfig(env_id=args.env)
        env = gym.make(args.env)
        obs_shape = env.observation_space.shape
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        action_size = env.action_space.n
        env.close()

        agent = DreamerAgent(config, obs_shape, action_size)
        agent.load(args.eval)
        evaluate_dreamer(agent, args.env, n_episodes=100, render=True)
    else:
        config = DreamerConfig(
            env_id=args.env,
            total_steps=args.steps,
            seed=args.seed,
        )
        train_dreamer(config)
