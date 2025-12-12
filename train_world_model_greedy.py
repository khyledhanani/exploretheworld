import argparse
import os
from collections import deque

import gymnasium as gym
import miniworld
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from worldmodel import WorldModel


# ============================================================================
# Config
# ============================================================================

CONFIG = {
    "env_name": "MiniWorld-OneRoom-v0",
    "obs_size": (64, 64),
    "action_dim": 3,
    "embedding_dim": 128,
    "hidden_dim": 200,
    "stochastic_dim": 64,
    # Training hyperparameters
    "batch_size": 16,
    "seq_length": 16,
    "learning_rate": 3e-4,
    "num_collection_episodes": 20,  # smaller default for a quick sanity check
    "num_training_steps": 2000,
    "collect_every_n_steps": 50,
    # Loss weights / regularization
    "lambda_rec": 10.0,
    "lambda_kl_start": 0.0,
    "lambda_kl_end": 0.10,
    "kl_anneal_steps": 4000,
    "lambda_reward": 1.0,
    "lambda_value": 1.0,
    "free_nats": 1.0,
    # N-step returns
    "n_step": 5,
    "gamma": 0.99,
    # Exploration (epsilon-greedy heuristic)
    "epsilon": 0.3,
}


# ============================================================================
# Replay buffer & utilities (ported from notebook)
# ============================================================================


class ReplayBuffer:
    """Replay buffer for storing trajectories of (obs, action, reward, next_obs, done)."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs,
                "done": float(done),
            }
        )

    def add_trajectory(self, trajectory):
        for transition in trajectory:
            self.add(**transition)

    def sample_sequences(self, batch_size, seq_length, action_dim):
        """Sample sequences of length seq_length.

        Returns tensors:
          obs:   (B, T, 3, 64, 64)
          action (one-hot): (B, T, action_dim)
          reward: (B, T)
          done:   (B, T)
        """
        if len(self.buffer) < seq_length:
            return None

        max_start = len(self.buffer) - seq_length
        starts = np.random.randint(0, max_start + 1, size=batch_size)

        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        for start in starts:
            seq = [self.buffer[start + i] for i in range(seq_length)]
            obs_seq.append([s["obs"] for s in seq])
            action_seq.append([s["action"] for s in seq])
            reward_seq.append([s["reward"] for s in seq])
            done_seq.append([s["done"] for s in seq])

        # Convert to tensors
        obs_tensor = torch.stack(
            [
                torch.stack(
                    [torch.tensor(o, dtype=torch.float32) for o in obs], dim=0
                )
                for obs in obs_seq
            ],
            dim=0,
        )  # (B, T, 3, 64, 64)

        action_tensor = torch.stack(
            [
                torch.stack(
                    [
                        F.one_hot(
                            torch.tensor(a, dtype=torch.long), num_classes=action_dim
                        ).float()
                        for a in actions
                    ],
                    dim=0,
                )
                for actions in action_seq
            ],
            dim=0,
        )  # (B, T, A)

        reward_tensor = torch.stack(
            [torch.tensor(r, dtype=torch.float32) for r in reward_seq], dim=0
        )  # (B, T)

        done_tensor = torch.stack(
            [torch.tensor(d, dtype=torch.float32) for d in done_seq], dim=0
        )  # (B, T)

        return obs_tensor, action_tensor, reward_tensor, done_tensor

    def __len__(self):
        return len(self.buffer)


def preprocess_obs(obs, target_size=(64, 64)):
    """Preprocess MiniWorld obs (H, W, 3) -> (3, 64, 64) float32 in [0,1]."""
    if isinstance(obs, np.ndarray):
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        img = Image.fromarray(obs)
    else:
        img = obs

    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    return img_array


def heuristic_policy(env, epsilon=0.3):
    """Simple heuristic: mostly move forward (action 0), sometimes random."""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return 0


def collect_trajectory(env, policy_fn, epsilon, max_steps=500):
    """Collect a single trajectory using the heuristic policy."""
    obs, info = env.reset()
    obs = preprocess_obs(obs)

    trajectory = []
    total_reward = 0.0

    for _ in range(max_steps):
        action = policy_fn(env, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_proc = preprocess_obs(next_obs)
        done = terminated or truncated

        trajectory.append(
            {
                "obs": obs.copy(),
                "action": int(action),
                "reward": float(reward),
                "next_obs": next_obs_proc.copy(),
                "done": float(done),
            }
        )

        total_reward += float(reward)
        obs = next_obs_proc

        if done:
            break

    return trajectory, total_reward, len(trajectory)


def compute_n_step_returns(rewards, dones, values, gamma=0.99, n_step=5):
    """Vectorized n-step returns: G_t = r_t + ... + γ^n V_{t+n}."""
    batch_size, seq_length = rewards.shape
    device = rewards.device
    values = values.detach()

    discounts = gamma ** torch.arange(n_step + 1, device=device, dtype=torch.float32)
    returns = torch.zeros_like(rewards)

    for t in range(seq_length):
        end_idx = min(t + n_step, seq_length)
        n_rewards = rewards[:, t:end_idx]
        n_dones = dones[:, t:end_idx]
        n_actual = n_rewards.shape[1]

        disc = discounts[:n_actual].unsqueeze(0)
        done_mask = torch.cumprod(1.0 - n_dones, dim=1)
        masked_rewards = n_rewards * done_mask

        reward_sum = (masked_rewards * disc).sum(dim=1)

        if t + n_step < seq_length:
            alive = (1.0 - n_dones).cumprod(dim=1)[:, -1]
            bootstrap = discounts[n_step] * values[:, t + n_step] * alive
        else:
            alive = (1.0 - n_dones).cumprod(dim=1)[:, -1]
            bootstrap = discounts[n_actual] * values[:, -1] * alive

        returns[:, t] = reward_sum + bootstrap

    return returns


# ============================================================================
# Reward shaping wrapper (MiniWorld distance shaping)
# ============================================================================


class ShapedRewardWrapper(gym.Wrapper):
    """Adds distance-based potential shaping to MiniWorld envs."""

    def __init__(self, env, shaping_scale=0.1):
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self.prev_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            agent_pos = self.env.unwrapped.agent.pos
            goal_pos = self.env.unwrapped.box.pos
            self.prev_dist = np.linalg.norm(agent_pos - goal_pos)
        except AttributeError:
            self.prev_dist = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.prev_dist is not None:
            try:
                agent_pos = self.env.unwrapped.agent.pos
                goal_pos = self.env.unwrapped.box.pos
                curr_dist = np.linalg.norm(agent_pos - goal_pos)

                shaping = (self.prev_dist - curr_dist) * self.shaping_scale
                reward = reward + shaping

                self.prev_dist = curr_dist
                info["distance_to_goal"] = curr_dist
                info["shaping_reward"] = shaping
            except AttributeError:
                pass

        return obs, reward, terminated, truncated, info


# ============================================================================
# Main training loop
# ============================================================================


def train(config: dict, device: torch.device, output_path: str):
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Env setup
    env_base = gym.make(config["env_name"], render_mode="rgb_array")
    env = ShapedRewardWrapper(env_base, shaping_scale=0.1)

    print(f"\nEnvironment: {config['env_name']} (with distance-based reward shaping)")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # World model
    model = WorldModel(
        action_dim=config["action_dim"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        stochastic_dim=config["stochastic_dim"],
        action_space_size=config["action_dim"],
    ).to(device)

    print("\nWorld Model initialized:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    replay_buffer = ReplayBuffer(capacity=50_000)

    # ------------------------------------------------------------------
    # Initial data collection with heuristic policy
    # ------------------------------------------------------------------
    print("\nCollecting initial trajectories with heuristic greedy policy...")
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(
        range(config["num_collection_episodes"]), desc="Collecting", leave=False
    ):
        trajectory, total_reward, traj_len = collect_trajectory(
            env, heuristic_policy, epsilon=config["epsilon"], max_steps=500
        )
        replay_buffer.add_trajectory(trajectory)
        episode_rewards.append(total_reward)
        episode_lengths.append(traj_len)

    print(f"Collected {len(replay_buffer)} transitions")
    print(
        f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
    )

    # ------------------------------------------------------------------
    # Training loop (KL annealing + free bits)
    # ------------------------------------------------------------------
    model.train()

    print("\nStarting world model training...")
    print(
        f"KL weight: {config['lambda_kl_start']:.3f} -> {config['lambda_kl_end']:.3f} "
        f"over {config['kl_anneal_steps']} steps"
    )
    print(f"Free nats: {config['free_nats']:.1f}")

    for step in tqdm(range(config["num_training_steps"]), desc="Training"):
        # Periodically add fresh data from the same heuristic policy
        if step % config["collect_every_n_steps"] == 0 and step > 0:
            trajectory, _, _ = collect_trajectory(
                env, heuristic_policy, epsilon=config["epsilon"], max_steps=500
            )
            replay_buffer.add_trajectory(trajectory)

        # KL annealing schedule
        if step < config["kl_anneal_steps"]:
            kl_weight = config["lambda_kl_start"] + (
                (config["lambda_kl_end"] - config["lambda_kl_start"]) * step
                / config["kl_anneal_steps"]
            )
        else:
            kl_weight = config["lambda_kl_end"]

        batch = replay_buffer.sample_sequences(
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            action_dim=config["action_dim"],
        )
        if batch is None:
            continue

        obs_seq, action_seq, reward_seq, done_seq = batch
        obs_seq = obs_seq.to(device)
        action_seq = action_seq.to(device)
        reward_seq = reward_seq.to(device)
        done_seq = done_seq.to(device)

        B, T = obs_seq.shape[:2]

        # Roll RSSM over sequence
        h_prev, z_prev = None, None
        all_outputs = []
        for t in range(T):
            obs_t = obs_seq[:, t]
            if t == 0:
                action_prev = torch.zeros(B, config["action_dim"], device=device)
            else:
                action_prev = action_seq[:, t - 1]

            outputs = model(obs_t, action_prev, h_prev, z_prev, use_posterior=True)
            all_outputs.append(outputs)
            h_prev = outputs["h_t"]
            z_prev = outputs["z_t"]

        o_hat_seq = torch.stack([o["o_hat_t"] for o in all_outputs], dim=1)
        r_hat_seq = torch.stack([o["r_hat_t"] for o in all_outputs], dim=1)
        v_hat_seq = torch.stack([o["v_hat_t"] for o in all_outputs], dim=1)

        with torch.no_grad():
            value_targets = compute_n_step_returns(
                reward_seq,
                done_seq,
                v_hat_seq.detach(),
                gamma=config["gamma"],
                n_step=config["n_step"],
            )

        # Losses
        recon_loss = F.mse_loss(o_hat_seq, obs_seq)
        reward_loss = F.mse_loss(r_hat_seq, reward_seq)
        value_loss = F.mse_loss(v_hat_seq, value_targets)

        kl_loss_raw = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)

        for t in range(T):
            prior_dist = all_outputs[t]["prior_dist"]
            post_dist = all_outputs[t]["post_dist"]
            if prior_dist is None or post_dist is None:
                continue

            kl_per_dim = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
            kl_t_raw = kl_per_dim.mean()
            kl_loss_raw += kl_t_raw

            free_nats = config["free_nats"]
            kl_per_dim_clamped = torch.clamp(kl_per_dim - free_nats, min=0.0)
            kl_t = kl_per_dim_clamped.mean()
            kl_loss += kl_t

        kl_loss /= T
        kl_loss_raw /= T

        total_loss = (
            config["lambda_rec"] * recon_loss
            + config["lambda_reward"] * reward_loss
            + config["lambda_value"] * value_loss
            + kl_weight * kl_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if (step + 1) % 500 == 0 or step == 0:
            print(
                f"\nStep {step+1}/{config['num_training_steps']}\n"
                f"  Total loss: {total_loss.item():.4f}\n"
                f"  Recon: {recon_loss.item():.4f}, Reward: {reward_loss.item():.4f}\n"
                f"  Value: {value_loss.item():.4f}\n"
                f"  KL (raw/clamped): {kl_loss_raw.item():.4f}/{kl_loss.item():.4f}, "
                f"KL weight: {kl_weight:.4f}"
            )

    # Save checkpoint
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)
    print(f"\nSaved world model checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train world model on MiniWorld-OneRoom with a heuristic greedy policy."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="worldmodel_checkpoint.pth",
        help="Where to save the trained world model.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(CONFIG, device, args.checkpoint)


if __name__ == "__main__":
    main()
