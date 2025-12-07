#!/usr/bin/env python3
"""
Complete World Model Training and MCTS Planning Script
Combined from world_model_complete.ipynb
"""

# ============================================================================
# Imports and Setup
# ============================================================================

# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(800, 600))
# display.start()

import os
# Make sure we are NOT forcing pyglet headless; clear any leftovers
os.environ.pop("PYGLET_HEADLESS", None)
os.environ.pop("MINIWORLD_HEADLESS", None)

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict, deque, OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import miniworld
from tqdm import tqdm
import random
from PIL import Image
import sys

# Import models from separate files
from worldmodel import WorldModel
from MTCS import MCTS

# Set device - M4 Mac should use MPS!
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: CUDA")
else:
    device = torch.device('cpu')
    print(f"Using device: CPU (SLOW!)")

print(f"Device: {device}")

# ============================================================================
# Configuration
# ============================================================================

config = {
    'env_name': 'MiniWorld-OneRoom-v0',
    'obs_size': (64, 64),
    'action_dim': 3,  # OneRoom has 3 actions
    'embedding_dim': 128,
    'hidden_dim': 200,
    'stochastic_dim': 64,
    
    # Training hyperparameters
    'batch_size': 64,
    'seq_length': 30,
    'learning_rate': 3e-4,
    'num_collection_episodes': 100,
    'num_training_steps': 30000,
    'collect_every_n_steps': 250,
    
    # Loss weights / regularization balance
    'lambda_rec': 10.0,
    'lambda_kl_start': 0.0,
    'lambda_kl_end': 1.0,
    'kl_anneal_steps': 15000,
    'lambda_reward': 1.0,
    'lambda_value': 1.0,
    'free_nats': 0.0,
    
    # N-step returns
    'n_step': 5,
    'gamma': 0.99,
    
    # Exploration
    'epsilon': 0.3,
}

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Replay buffer for storing trajectories.
    Stores: (o_t, a_t, r_t, o_{t+1}, done_t)
    Optimized for sequence sampling using list instead of deque for slicing speed.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = [] 
    
    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
        })
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def add_trajectory(self, trajectory):
        """Add a full trajectory"""
        for transition in trajectory:
            self.add(**transition)
    
    def sample_sequences(self, batch_size, seq_length):
        """
        Sample sequences of length seq_length from the buffer.
        Returns sequences of (obs, action, reward, done)
        """
        if len(self.buffer) < seq_length:
            return None
        
        # Sample random starting indices
        max_start = len(self.buffer) - seq_length
        starts = np.random.randint(0, max_start, size=batch_size)
        
        # Pre-allocate arrays for speed
        obs_sample = self.buffer[0]['obs']
        obs_shape = obs_sample.shape
        
        obs_seq = np.zeros((batch_size, seq_length, *obs_shape), dtype=np.float32)
        action_seq = np.zeros((batch_size, seq_length), dtype=np.int64)
        reward_seq = np.zeros((batch_size, seq_length), dtype=np.float32)
        done_seq = np.zeros((batch_size, seq_length), dtype=np.float32)
        
        for i, start in enumerate(starts):
            # Fast slice 
            seq = self.buffer[start : start + seq_length]
            
            for t, item in enumerate(seq):
                obs_seq[i, t] = item['obs']
                action_seq[i, t] = item['action']
                reward_seq[i, t] = item['reward']
                done_seq[i, t] = item['done']
        
        # Convert to tensors
        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
        action_tensor = F.one_hot(torch.tensor(action_seq, dtype=torch.long), config['action_dim']).float()
        reward_tensor = torch.tensor(reward_seq, dtype=torch.float32)
        done_tensor = torch.tensor(done_seq, dtype=torch.float32)
        
        return obs_tensor, action_tensor, reward_tensor, done_tensor
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_obs(obs):
    """
    Preprocess observation to (3, 64, 64) tensor.
    MiniWorld returns (H, W, 3) numpy array.
    """
    if isinstance(obs, np.ndarray):
        # Convert to PIL Image if needed
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        img = Image.fromarray(obs)
    else:
        img = obs
    
    # Resize to 64x64
    img = img.resize((64, 64), Image.LANCZOS)
    
    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Convert HWC to CHW: (64, 64, 3) -> (3, 64, 64)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def heuristic_policy(env, epsilon=0.3):
    """
    Simple heuristic policy: prefers moving forward (action 0), 
    occasionally takes random actions.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return 0  # move_forward


def collect_trajectory(env, policy_fn, max_steps=500):
    """
    Collect a single trajectory using the exploration policy.
    
    Returns:
        trajectory: list of (obs, action, reward, next_obs, done)
    """
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    
    trajectory = []
    total_reward = 0
    
    for step in range(max_steps):
        # Get action from policy
        action = policy_fn(env, config['epsilon'])
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_obs(next_obs)
        done = terminated or truncated
        
        # Store transition
        trajectory.append({
            'obs': obs.copy(),
            'action': action,
            'reward': float(reward),
            'next_obs': next_obs.copy(),
            'done': float(done),
        })
        
        total_reward += reward
        obs = next_obs
        
        if done:
            break
    
    return trajectory, total_reward, len(trajectory)


def compute_n_step_returns(rewards, dones, values, gamma=0.99, n_step=5):
    """
    Optimized vectorized n-step returns computation.
    G_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V_{t+n}
    
    Args:
        rewards: (batch, seq) tensor of rewards
        dones: (batch, seq) tensor of done flags
        values: (batch, seq) tensor of predicted values (for bootstrapping) - should be detached
        gamma: discount factor
        n_step: number of steps for n-step return
    
    Returns:
        returns: (batch, seq) tensor of n-step returns
    """
    batch_size, seq_length = rewards.shape
    device = rewards.device
    values = values.detach()
    
    # Pre-compute discount factors
    discounts = gamma ** torch.arange(n_step + 1, device=device, dtype=torch.float32)
    
    returns = torch.zeros_like(rewards)
    
    # Vectorized computation per timestep
    for t in range(seq_length):
        # Get rewards for next n steps
        end_idx = min(t + n_step, seq_length)
        n_rewards = rewards[:, t:end_idx]  # (B, n_actual)
        n_dones = dones[:, t:end_idx]  # (B, n_actual)
        
        n_actual = n_rewards.shape[1]
        disc = discounts[:n_actual].unsqueeze(0)  # (1, n_actual)
        
        # Mask out rewards after done (cumulative product to stop after first done)
        done_mask = torch.cumprod(1.0 - n_dones, dim=1)  # (B, n_actual)
        masked_rewards = n_rewards * done_mask
        
        # Sum discounted rewards
        reward_sum = (masked_rewards * disc).sum(dim=1)  # (B,)
        
        # Bootstrap value
        if t + n_step < seq_length:
            alive = (1.0 - n_dones).cumprod(dim=1)[:, -1] 
            bootstrap = discounts[n_step] * values[:, t + n_step] * alive
        else:
            alive = (1.0 - n_dones).cumprod(dim=1)[:, -1]
            bootstrap = discounts[n_actual] * values[:, -1] * alive
    
        returns[:, t] = reward_sum + bootstrap
    return returns

# ============================================================================
# Reward Shaping Wrapper
# ============================================================================

class ShapedRewardWrapper(gym.Wrapper):
    """
    Adds distance-based reward shaping to MiniWorld environments.
    
    Uses potential-based shaping: reward += scale * (prev_dist - curr_dist)
    This encourages the agent to move closer to the goal while preserving
    the optimal policy (potential-based shaping is theoretically sound).
    """
    def __init__(self, env, shaping_scale=0.1):
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self.prev_dist = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Get initial distance to goal (the red box in OneRoom)
        try:
            agent_pos = self.env.unwrapped.agent.pos
            goal_pos = self.env.unwrapped.box.pos
            self.prev_dist = np.linalg.norm(agent_pos - goal_pos)
        except AttributeError:
            # Fallback if environment doesn't have expected attributes
            self.prev_dist = None
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add distance-based reward shaping
        if self.prev_dist is not None:
            try:
                agent_pos = self.env.unwrapped.agent.pos
                goal_pos = self.env.unwrapped.box.pos
                curr_dist = np.linalg.norm(agent_pos - goal_pos)
                
                # Potential-based shaping: reward for getting closer
                shaping = (self.prev_dist - curr_dist) * self.shaping_scale
                reward = reward + shaping
                
                self.prev_dist = curr_dist
                
                # Add debug info
                info['distance_to_goal'] = curr_dist
                info['shaping_reward'] = shaping
            except AttributeError:
                pass
        
        return obs, reward, terminated, truncated, info

# ============================================================================
# Initialize Environment and Model
# ============================================================================

# Initialize environment with reward shaping
env_base = gym.make(config['env_name'], render_mode='rgb_array')
env = ShapedRewardWrapper(env_base, shaping_scale=0.1)

print(f"Environment: {config['env_name']} (with distance-based reward shaping)")
print(f"  Shaping scale: 0.1 (reward += 0.1 * distance_improvement)")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Initialize world model
model = WorldModel(
    action_dim=config['action_dim'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    stochastic_dim=config['stochastic_dim'],
    action_space_size=config['action_dim'],
).to(device)

print(f"\nWorld Model initialized:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Initialize replay buffer (larger capacity for bigger batches)
replay_buffer = ReplayBuffer(capacity=50000)

print("\nInitialization complete!")

# ============================================================================
# Data Collection Phase
# ============================================================================

# Initial data collection
print("Collecting initial trajectories...")
episode_rewards = []
episode_lengths = []

for episode in tqdm(range(config['num_collection_episodes']), desc="Collecting"):
    trajectory, total_reward, traj_length = collect_trajectory(
        env, heuristic_policy, max_steps=500
    )
    replay_buffer.add_trajectory(trajectory)
    episode_rewards.append(total_reward)
    episode_lengths.append(traj_length)

print(f"\nCollected {len(replay_buffer)} transitions")
print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")

# ============================================================================
# Training Loop
# ============================================================================

# Training loop with KL annealing and free bits
model.train()
losses_history = {
    'total': [],
    'recon': [],
    'reward': [],
    'kl': [],
    'kl_raw': [],
    'value': [],
    'kl_weight': [],
    'posterior_std': [],
    'prior_std': [],
}

print("Starting training with KL annealing and free bits...")
print(f"KL weight: {config['lambda_kl_start']:.3f} -> {config['lambda_kl_end']:.3f} over {config['kl_anneal_steps']} steps")
print(f"Free nats: {config['free_nats']:.1f}")

for step in tqdm(range(config['num_training_steps']), desc="Training"):
    # Collect new data periodically
    if step % config['collect_every_n_steps'] == 0 and step > 0:
        trajectory, _, _ = collect_trajectory(env, heuristic_policy, max_steps=500)
        replay_buffer.add_trajectory(trajectory)
    
    # KL annealing schedule
    if step < config['kl_anneal_steps']:
        kl_weight = config['lambda_kl_start'] + (config['lambda_kl_end'] - config['lambda_kl_start']) * (step / config['kl_anneal_steps'])
    else:
        kl_weight = config['lambda_kl_end']
    
    # Sample batch of sequences
    batch = replay_buffer.sample_sequences(
        batch_size=config['batch_size'],
        seq_length=config['seq_length']
    )
    
    if batch is None:
        continue
    
    obs_seq, action_seq, reward_seq, done_seq = batch
    obs_seq = obs_seq.to(device)
    action_seq = action_seq.to(device)
    reward_seq = reward_seq.to(device)
    done_seq = done_seq.to(device)
    
    # Reshape for processing: (B*T, ...)
    B, T = obs_seq.shape[:2]
    obs_flat = obs_seq.view(B * T, 3, 64, 64)
    action_flat = action_seq.view(B * T, config['action_dim'])
    reward_flat = reward_seq.view(B * T)
    
    # Initialize states
    h_prev = None
    z_prev = None
    
    # Optimized Forward pass through sequence using batching
    outputs = model.forward_sequence(obs_seq, action_seq, h_prev, z_prev)
    
    o_hat_seq = outputs['o_hat_t']
    r_hat_seq = outputs['r_hat_t']
    v_hat_seq = outputs['v_hat_t']
    
    # Compute n-step returns for value targets (detach values for bootstrapping)
    with torch.no_grad():
        value_targets = compute_n_step_returns(
            reward_seq, done_seq, v_hat_seq.detach(),
            gamma=config['gamma'],
            n_step=config['n_step']
        )
    
    # Compute losses
    recon_loss = F.mse_loss(o_hat_seq, obs_seq)
    reward_loss = F.mse_loss(r_hat_seq, reward_seq)
    value_loss = F.mse_loss(v_hat_seq, value_targets)
    
    # KL loss with FREE BITS constraint (sum over sequence, mean over batch)
    kl_loss = torch.tensor(0.0, device=device)
    kl_loss_raw = torch.tensor(0.0, device=device)
    
    if outputs['prior_dist'] is not None and outputs['post_dist'] is not None:
        # Per-dimension KL divergence
        kl_per_dim = torch.distributions.kl.kl_divergence(
            outputs['post_dist'],
            outputs['prior_dist']
        ) # (B, T, Z)
        
        # Raw KL (for logging)
        kl_loss_raw = kl_per_dim.sum(dim=-1).mean()
        
        # Free bits: don't penalize the first `free_nats` nats per dim
        free_nats = config['free_nats']
        kl_per_dim_clamped = torch.clamp(kl_per_dim - free_nats, min=0.0)
        kl_loss = kl_per_dim_clamped.sum(dim=-1).mean()
        
        # Track std for diagnosing posterior collapse
        # Average over B and T
        post_std_mean = outputs['post_dist'].stddev.mean().item()
        prior_std_mean = outputs['prior_dist'].stddev.mean().item()
        posterior_stds.append(post_std_mean)
        prior_stds.append(prior_std_mean)
    
    # Total loss
    total_loss = (
        config['lambda_rec'] * recon_loss +
        kl_weight * kl_loss +
        config['lambda_reward'] * reward_loss +
        config['lambda_value'] * value_loss
    )
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Log losses
    losses_history['total'].append(total_loss.item())
    losses_history['recon'].append(recon_loss.item())
    losses_history['reward'].append(reward_loss.item())
    losses_history['kl'].append(kl_loss.item())
    losses_history['kl_raw'].append(kl_loss_raw.item())
    losses_history['value'].append(value_loss.item())
    losses_history['kl_weight'].append(kl_weight)
    losses_history['posterior_std'].append(np.mean(posterior_stds) if posterior_stds else 0)
    losses_history['prior_std'].append(np.mean(prior_stds) if prior_stds else 0)
    
    # Print progress (less frequently to reduce overhead)
    if (step + 1) % 1000 == 0:
        print(f"\nStep {step + 1}/{config['num_training_steps']}")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Recon: {recon_loss.item():.4f}, Reward: {reward_loss.item():.4f}")
        print(f"  KL (raw/clamped): {kl_loss_raw.item():.4f}/{kl_loss.item():.4f}, KL weight: {kl_weight:.4f}")
        print(f"  Value: {value_loss.item():.4f}")
        print(f"  Post/Prior std: {np.mean(posterior_stds):.4f}/{np.mean(prior_stds):.4f}")

print("\nTraining complete!")

# ============================================================================
# Plot Training Losses
# ============================================================================

# Plot training losses with enhanced diagnostics
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# Row 1: Main losses
axes[0, 0].plot(losses_history['total'])
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

axes[0, 1].plot(losses_history['recon'])
axes[0, 1].set_title('Reconstruction Loss')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True)

axes[0, 2].plot(losses_history['reward'])
axes[0, 2].set_title('Reward Prediction Loss')
axes[0, 2].set_xlabel('Step')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True)

# Row 2: KL diagnostics
axes[1, 0].plot(losses_history['kl_raw'], label='Raw KL', alpha=0.7)
axes[1, 0].plot(losses_history['kl'], label='Clamped KL (free bits)', alpha=0.7)
axes[1, 0].set_title('KL Divergence Loss')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(losses_history['kl_weight'])
axes[1, 1].set_title('KL Weight (Annealing Schedule)')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].grid(True)

axes[1, 2].plot(losses_history['value'])
axes[1, 2].set_title('Value Prediction Loss')
axes[1, 2].set_xlabel('Step')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True)

# Row 3: Posterior collapse diagnostics
axes[2, 0].plot(losses_history['posterior_std'], label='Posterior', alpha=0.7)
axes[2, 0].plot(losses_history['prior_std'], label='Prior', alpha=0.7)
axes[2, 0].set_title('Latent Distribution Std Devs')
axes[2, 0].set_xlabel('Step')
axes[2, 0].set_ylabel('Std Dev')
axes[2, 0].legend()
axes[2, 0].grid(True)
axes[2, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='Collapse threshold')

# Histogram of final losses
axes[2, 1].hist(losses_history['recon'][-1000:], bins=30, alpha=0.7)
axes[2, 1].set_title('Recent Reconstruction Loss Distribution')
axes[2, 1].set_xlabel('Loss')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].grid(True)

# Histogram of KL
axes[2, 2].hist(losses_history['kl_raw'][-1000:], bins=30, alpha=0.7, label='Raw')
axes[2, 2].hist(losses_history['kl'][-1000:], bins=30, alpha=0.5, label='Clamped')
axes[2, 2].set_title('Recent KL Loss Distribution')
axes[2, 2].set_xlabel('Loss')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].legend()
axes[2, 2].grid(True)

plt.tight_layout()
plt.savefig('training_losses.png')
print("Training losses plot saved to training_losses.png")

# Print summary statistics
print("\n=== Training Summary ===")
print(f"Final reconstruction loss: {losses_history['recon'][-1]:.6f}")
print(f"Final KL (raw/clamped): {losses_history['kl_raw'][-1]:.6f} / {losses_history['kl'][-1]:.6f}")
print(f"Final posterior std: {losses_history['posterior_std'][-1]:.6f}")
print(f"Final prior std: {losses_history['prior_std'][-1]:.6f}")
print(f"\nPosterior collapse check:")
if losses_history['posterior_std'][-1] < 0.1:
    print("  ⚠️ WARNING: Posterior may have collapsed (std < 0.1)")
elif losses_history['posterior_std'][-1] > 0.5:
    print("  ✅ GOOD: Posterior is active (std > 0.5)")
else:
    print("  ⚠️ MARGINAL: Posterior std is low but not collapsed")

# ============================================================================
# Visualize Reconstructions
# ============================================================================

# Visualize reconstructions with detailed statistics
model.eval()
with torch.no_grad():
    # Sample a sequence
    batch = replay_buffer.sample_sequences(batch_size=1, seq_length=10)
    if batch is not None:
        obs_seq, action_seq, reward_seq, done_seq = batch
        obs_seq = obs_seq.to(device)
        action_seq = action_seq.to(device)
        
        # Reconstruct and collect latents / predictions
        h_prev = None
        z_prev = None
        reconstructions = []
        latents_h = []
        latents_z = []
        rewards_hat = []
        values_hat = []
        
        for t in range(min(10, obs_seq.shape[1])):
            obs_t = obs_seq[:, t]
            if t == 0:
                action_prev = torch.zeros(1, config['action_dim'], device=device)
            else:
                action_prev = action_seq[:, t-1]
            
            outputs = model(obs_t, action_prev, h_prev, z_prev, use_posterior=True)
            reconstructions.append(outputs['o_hat_t'])
            latents_h.append(outputs['h_t'].detach().cpu())
            latents_z.append(outputs['z_t'].detach().cpu())
            rewards_hat.append(outputs['r_hat_t'].detach().cpu())
            values_hat.append(outputs['v_hat_t'].detach().cpu())
            h_prev = outputs['h_t']
            z_prev = outputs['z_t']
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for t in range(min(10, len(reconstructions))):
            # Original
            orig = obs_seq[0, t].cpu().numpy().transpose(1, 2, 0)
            axes[0, t].imshow(np.clip(orig, 0, 1))
            axes[0, t].set_title(f'Original t={t}')
            axes[0, t].axis('off')
            
            # Reconstructed
            recon = reconstructions[t][0].cpu().numpy().transpose(1, 2, 0)
            axes[1, t].imshow(np.clip(recon, 0, 1))
            axes[1, t].set_title(f'Reconstructed t={t}')
            axes[1, t].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstructions.png')
        print("Reconstructions plot saved to reconstructions.png")
        
        # Print detailed statistics for the first frame
        print("\n=== Reconstruction Statistics (t=0) ===")
        orig_0 = obs_seq[0, 0].cpu().numpy()
        recon_0 = reconstructions[0][0].cpu().numpy()
        
        print(f"Original image:")
        print(f"  Mean: {orig_0.mean():.4f}, Std: {orig_0.std():.4f}")
        print(f"  Min: {orig_0.min():.4f}, Max: {orig_0.max():.4f}")
        
        print(f"\nReconstructed image:")
        print(f"  Mean: {recon_0.mean():.4f}, Std: {recon_0.std():.4f}")
        print(f"  Min: {recon_0.min():.4f}, Max: {recon_0.max():.4f}")
        
        # Per-channel statistics
        print(f"\nPer-channel (RGB) statistics:")
        for c, color in enumerate(['Red', 'Green', 'Blue']):
            print(f"  {color} - Orig: {orig_0[c].mean():.4f}, Recon: {recon_0[c].mean():.4f}")
        
        # MSE
        mse = np.mean((orig_0 - recon_0) ** 2)
        print(f"\nMSE: {mse:.6f}")
        print(f"RMSE: {np.sqrt(mse):.6f}")
        
        # Check if reconstruction is constant
        if recon_0.std() < 0.01:
            print("\n⚠️ WARNING: Reconstruction has very low variance - likely outputting constant values!")
        else:
            print(f"\n✅ Reconstruction has variance: {recon_0.std():.4f}")

# ============================================================================
# Save Model
# ============================================================================

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'losses_history': losses_history,
}, 'worldmodel_checkpoint.pth')
print("Model saved to worldmodel_checkpoint.pth")

# ============================================================================
# MCTS Helper Functions
# ============================================================================

def get_root_state(model, obs, h_prev, z_prev, action_prev, device):
    """
    Compute the root latent state (h_t, z_t) for MCTS from a real observation.
    
    This uses the encoder + RSSM posterior (the "representation function" in MuZero terms).
    
    Args:
        model: WorldModel instance
        obs: (3, 64, 64) numpy array or tensor - current observation
        h_prev: (hidden_dim,) tensor - previous deterministic state (or None for t=0)
        z_prev: (stochastic_dim,) tensor - previous stochastic state (or None for t=0)
        action_prev: int - previous action taken (or None for t=0)
        device: torch device
    
    Returns:
        h_t: (hidden_dim,) tensor - current deterministic state
        z_t: (stochastic_dim,) tensor - current stochastic state (from posterior)
    """
    model.eval()
    
    with torch.no_grad():
        # Prepare observation
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 3 else obs
            obs_tensor = obs_tensor.to(device)
        
        # Initialize states if None (first timestep)
        if h_prev is None:
            h_prev = torch.zeros(model.hidden_dim, device=device)
        if z_prev is None:
            z_prev = torch.zeros(model.stochastic_dim, device=device)
        
        # Prepare previous action as one-hot
        if action_prev is None:
            action_one_hot = torch.zeros(1, model.action_dim, device=device)
        else:
            action_one_hot = F.one_hot(
                torch.tensor([action_prev], device=device), 
                num_classes=model.action_dim
            ).float()
        
        # Step 1: RSSM prior to get h_t (deterministic state update)
        h_t, _, _ = model.rssm.prior(
            h_prev.unsqueeze(0),
            z_prev.unsqueeze(0),
            action_one_hot
        )
        h_t = h_t.squeeze(0)
        
        # Step 2: Encode observation
        e_t = model.encoder(obs_tensor)
        
        # Step 3: RSSM posterior to get z_t (corrected stochastic state)
        z_t, _ = model.rssm.posterior(h_t.unsqueeze(0), e_t)
        z_t = z_t.squeeze(0)
    
    return h_t, z_t


def mcts_policy(model, obs, h_prev, z_prev, action_prev, device, 
                num_simulations=50, c_puct=1.0, discount=0.99, temperature=1.0):
    """
    Use MCTS to select an action given the current observation.
    
    Args:
        model: WorldModel instance
        obs: (3, 64, 64) numpy array - current observation
        h_prev: previous deterministic state (or None)
        z_prev: previous stochastic state (or None)
        action_prev: previous action (or None)
        device: torch device
        num_simulations: number of MCTS simulations
        c_puct: exploration constant
        discount: reward discount factor
        temperature: action selection temperature (0 = greedy, 1 = proportional)
    
    Returns:
        action: int - selected action
        policy_target: np.array - visit count distribution (for training)
        value: float - root value estimate
        h_t: tensor - current deterministic state (to pass to next step)
        z_t: tensor - current stochastic state (to pass to next step)
    """
    # Get root latent state from real observation
    h_t, z_t = get_root_state(model, obs, h_prev, z_prev, action_prev, device)
    
    # Run MCTS from this root state
    action, policy_target, value = MCTS(
        world_model=model,
        root_h=h_t,
        root_z=z_t,
        c_puct=c_puct,
        num_simulations=num_simulations,
        discount=discount,
        action_space_size=model.action_dim,
        temperature=temperature,
    )
    
    return action, policy_target, value, h_t, z_t


print("MCTS helper functions defined!")
print("  - get_root_state(): Encode observation → latent state")
print("  - mcts_policy(): Full MCTS planning from observation")

# ============================================================================
# Test MCTS Planning
# ============================================================================

def run_mcts_episode(env, model, device, max_steps=500, num_simulations=50, 
                     c_puct=1.0, discount=0.99, temperature=1.0, render=False):
    """
    Run a single episode using MCTS for action selection.
    
    Returns:
        trajectory: list of (obs, action, reward, next_obs, done, policy_target, value)
        total_reward: float
        episode_length: int
    """
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    
    trajectory = []
    total_reward = 0
    
    # Initialize recurrent states
    h_prev = None
    z_prev = None
    action_prev = None
    
    for step in range(max_steps):
        # Use MCTS to select action
        action, policy_target, value, h_t, z_t = mcts_policy(
            model=model,
            obs=obs,
            h_prev=h_prev,
            z_prev=z_prev,
            action_prev=action_prev,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=discount,
            temperature=temperature,
        )
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_obs(next_obs)
        done = terminated or truncated
        
        # Store transition with MCTS policy target
        trajectory.append({
            'obs': obs.copy(),
            'action': action,
            'reward': float(reward),
            'next_obs': next_obs.copy(),
            'done': float(done),
            'policy_target': policy_target.copy(),
            'value': value,
        })
        
        total_reward += reward
        
        # Update for next step
        obs = next_obs
        h_prev = h_t
        z_prev = z_t
        action_prev = action
        
        if done:
            break
    
    return trajectory, total_reward, len(trajectory)


# Quick test: run one episode with MCTS
print("Testing MCTS planning on a single episode...")
model.eval()

test_traj, test_reward, test_length = run_mcts_episode(
    env=env,
    model=model,
    device=device,
    max_steps=200,
    num_simulations=25,
    temperature=1.0,
)

print(f"\nMCTS Test Episode:")
print(f"  Total reward: {test_reward:.2f}")
print(f"  Episode length: {test_length}")
print(f"  Sample policy targets (first 5 steps):")
for i in range(min(5, len(test_traj))):
    print(f"    t={i}: action={test_traj[i]['action']}, policy={test_traj[i]['policy_target']}, value={test_traj[i]['value']:.4f}")

# ============================================================================
# MCTS Replay Buffer
# ============================================================================

class MCTSReplayBuffer:
    """
    Replay buffer that also stores MCTS policy targets and value estimates.
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done, policy_target=None, value_target=None):
        """Add a single transition with optional MCTS targets"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'policy_target': policy_target,
            'value_target': value_target,
        })
    
    def add_trajectory(self, trajectory):
        """Add a full trajectory (list of dicts)"""
        for t in trajectory:
            self.add(
                obs=t['obs'],
                action=t['action'],
                reward=t['reward'],
                next_obs=t['next_obs'],
                done=t['done'],
                policy_target=t.get('policy_target'),
                value_target=t.get('value'),
            )
    
    def sample_sequences(self, batch_size, seq_length, action_dim):
        """
        Sample sequences of length seq_length.
        Returns sequences with MCTS targets if available.
        """
        if len(self.buffer) < seq_length:
            return None
        
        max_start = len(self.buffer) - seq_length
        starts = np.random.randint(0, max_start, size=batch_size)
        
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        policy_target_seq = []
        value_target_seq = []
        has_mcts_targets = []
        
        for start in starts:
            seq = [self.buffer[start + i] for i in range(seq_length)]
            
            obs_seq.append([s['obs'] for s in seq])
            action_seq.append([s['action'] for s in seq])
            reward_seq.append([s['reward'] for s in seq])
            done_seq.append([s['done'] for s in seq])
            
            # MCTS targets (may be None for warm-up data)
            policy_targets = [s['policy_target'] for s in seq]
            value_targets = [s['value_target'] for s in seq]
            
            # Check if this sequence has MCTS targets
            has_targets = all(p is not None for p in policy_targets)
            has_mcts_targets.append(has_targets)
            
            if has_targets:
                policy_target_seq.append(policy_targets)
                value_target_seq.append(value_targets)
            else:
                # Placeholder - uniform policy, zero value
                policy_target_seq.append([np.ones(action_dim) / action_dim for _ in seq])
                value_target_seq.append([0.0 for _ in seq])
        
        # Convert to tensors
        obs_tensor = torch.stack([
            torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]) 
            for obs in obs_seq
        ])
        action_tensor = torch.stack([
            torch.stack([F.one_hot(torch.tensor(a, dtype=torch.long), action_dim).float() for a in action]) 
            for action in action_seq
        ])
        reward_tensor = torch.stack([torch.tensor(r, dtype=torch.float32) for r in reward_seq])
        done_tensor = torch.stack([torch.tensor(d, dtype=torch.float32) for d in done_seq])
        policy_target_tensor = torch.stack([
            torch.stack([torch.tensor(p, dtype=torch.float32) for p in policy]) 
            for policy in policy_target_seq
        ])
        value_target_tensor = torch.stack([torch.tensor(v, dtype=torch.float32) for v in value_target_seq])
        has_mcts_tensor = torch.tensor(has_mcts_targets, dtype=torch.bool)
        
        return {
            'obs': obs_tensor,
            'action': action_tensor,
            'reward': reward_tensor,
            'done': done_tensor,
            'policy_target': policy_target_tensor,
            'value_target': value_target_tensor,
            'has_mcts_targets': has_mcts_tensor,
        }
    
    def __len__(self):
        return len(self.buffer)


print("MCTSReplayBuffer defined!")

# ============================================================================
# MuZero-style Training Step
# ============================================================================

def muzero_training_step(model, optimizer, batch, device, config, step):
    """
    Single training step with MuZero-style losses:
    - Reconstruction loss
    - Reward prediction loss
    - KL divergence loss
    - Value prediction loss (using n-step returns OR MCTS value targets)
    - Policy distillation loss (match MCTS visit distribution)
    """
    model.train()
    
    obs_seq = batch['obs'].to(device)
    action_seq = batch['action'].to(device)
    reward_seq = batch['reward'].to(device)
    done_seq = batch['done'].to(device)
    policy_target = batch['policy_target'].to(device)
    value_target = batch['value_target'].to(device)
    has_mcts = batch['has_mcts_targets']
    
    B, T = obs_seq.shape[:2]
    
    # KL annealing
    if step < config['kl_anneal_steps']:
        kl_weight = config['lambda_kl_start'] + \
            (config['lambda_kl_end'] - config['lambda_kl_start']) * (step / config['kl_anneal_steps'])
    else:
        kl_weight = config['lambda_kl_end']
    
    # Forward pass through sequence
    h_prev = None
    z_prev = None
    all_outputs = []
    
    for t in range(T):
        obs_t = obs_seq[:, t]
        action_prev = torch.zeros(B, config['action_dim'], device=device) if t == 0 else action_seq[:, t-1]
        
        outputs = model(obs_t, action_prev, h_prev, z_prev, use_posterior=True)
        all_outputs.append(outputs)
        
        h_prev = outputs['h_t']
        z_prev = outputs['z_t']
    
    # Stack outputs
    o_hat_seq = torch.stack([o['o_hat_t'] for o in all_outputs], dim=1)
    r_hat_seq = torch.stack([o['r_hat_t'] for o in all_outputs], dim=1)
    v_hat_seq = torch.stack([o['v_hat_t'] for o in all_outputs], dim=1)
    policy_logits_seq = torch.stack([o['policy_logits'] for o in all_outputs], dim=1)
    
    # ========== LOSSES ==========
    
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(o_hat_seq, obs_seq)
    
    # 2. Reward prediction loss
    reward_loss = F.mse_loss(r_hat_seq, reward_seq)
    
    # 3. KL loss with free bits
    kl_loss = torch.tensor(0.0, device=device)
    for t in range(T):
        if all_outputs[t]['prior_dist'] is not None and all_outputs[t]['post_dist'] is not None:
            kl_per_dim = torch.distributions.kl.kl_divergence(
                all_outputs[t]['post_dist'],
                all_outputs[t]['prior_dist']
            )
            kl_per_dim_clamped = torch.clamp(kl_per_dim - config['free_nats'], min=0.0)
            kl_loss += kl_per_dim_clamped.sum(dim=-1).mean()
    kl_loss = kl_loss / T
    
    # 4. Value loss - use n-step returns computed from rewards
    with torch.no_grad():
        n_step_returns = compute_n_step_returns(
            reward_seq, done_seq, v_hat_seq.detach(),
            gamma=config['gamma'], n_step=config['n_step']
        )
    value_loss = F.mse_loss(v_hat_seq, n_step_returns)
    
    # 5. Policy distillation loss (only for samples with MCTS targets)
    policy_loss = torch.tensor(0.0, device=device)
    if has_mcts.any():
        # Flatten for cross-entropy: (B*T, action_dim)
        policy_logits_flat = policy_logits_seq.view(-1, config['action_dim'])
        policy_target_flat = policy_target.view(-1, config['action_dim'])
        
        # Create mask for samples with MCTS targets
        mask = has_mcts.unsqueeze(1).expand(-1, T).reshape(-1).to(device)
        
        if mask.sum() > 0:
            # Cross-entropy loss: -sum(target * log_softmax(logits))
            log_probs = F.log_softmax(policy_logits_flat[mask], dim=-1)
            policy_loss = -(policy_target_flat[mask] * log_probs).sum(dim=-1).mean()
    
    # Total loss
    total_loss = (
        config['lambda_rec'] * recon_loss +
        kl_weight * kl_loss +
        config['lambda_reward'] * reward_loss +
        config['lambda_value'] * value_loss +
        config.get('lambda_policy', 1.0) * policy_loss
    )
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_loss': kl_loss.item(),
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss,
        'kl_weight': kl_weight,
    }


print("MuZero training step defined!")

# ============================================================================
# MCTS Training Configuration
# ============================================================================

# Configuration for MCTS training
mcts_config = config.copy()
mcts_config.update({
    # MCTS parameters
    'num_simulations': 100,
    'c_puct': 1.0,
    'temperature_start': 1.0,
    'temperature_end': 0.1,
    'temperature_decay_episodes': 200,
    
    # Training schedule
    'mcts_episodes': 200,
    'training_steps_per_episode': 100,
    'eval_every': 20,
    
    # Policy distillation weight
    'lambda_policy': 1.0,
})

print("MCTS Training Configuration:")
for k in ['num_simulations', 'c_puct', 'temperature_start', 'temperature_end', 
          'mcts_episodes', 'training_steps_per_episode', 'lambda_policy']:
    print(f"  {k}: {mcts_config[k]}")

# ============================================================================
# MCTS Training Loop
# ============================================================================

# Initialize new replay buffer for MCTS data
mcts_replay_buffer = MCTSReplayBuffer(capacity=50000)

# Copy existing warm-up data (without MCTS targets)
print("Copying warm-up data to MCTS replay buffer...")
for item in replay_buffer.buffer:
    mcts_replay_buffer.add(
        obs=item['obs'],
        action=item['action'],
        reward=item['reward'],
        next_obs=item['next_obs'],
        done=item['done'],
        policy_target=None,
        value_target=None,
    )
print(f"Copied {len(mcts_replay_buffer)} transitions")

# Training history
mcts_training_history = {
    'episode_rewards': [],
    'episode_lengths': [],
    'total_loss': [],
    'recon_loss': [],
    'reward_loss': [],
    'kl_loss': [],
    'value_loss': [],
    'policy_loss': [],
}

# Reset optimizer (optional - can continue from previous)
optimizer = optim.Adam(model.parameters(), lr=mcts_config['learning_rate'])

print("\n" + "="*60)
print("Starting MCTS Training Loop")
print("="*60)

global_step = 0

for episode in tqdm(range(mcts_config['mcts_episodes']), desc="MCTS Episodes"):
    # Temperature annealing
    temp_progress = min(1.0, episode / mcts_config['temperature_decay_episodes'])
    temperature = mcts_config['temperature_start'] + \
        (mcts_config['temperature_end'] - mcts_config['temperature_start']) * temp_progress
    
    # ========== COLLECT EPISODE WITH MCTS ==========
    trajectory, total_reward, episode_length = run_mcts_episode(
        env=env,
        model=model,
        device=device,
        max_steps=500,
        num_simulations=mcts_config['num_simulations'],
        c_puct=mcts_config['c_puct'],
        discount=mcts_config['gamma'],
        temperature=temperature,
    )
    
    # Add to replay buffer
    mcts_replay_buffer.add_trajectory(trajectory)
    
    mcts_training_history['episode_rewards'].append(total_reward)
    mcts_training_history['episode_lengths'].append(episode_length)
    
    # ========== TRAINING STEPS ==========
    for _ in range(mcts_config['training_steps_per_episode']):
        batch = mcts_replay_buffer.sample_sequences(
            batch_size=mcts_config['batch_size'],
            seq_length=mcts_config['seq_length'],
            action_dim=mcts_config['action_dim'],
        )
        
        if batch is None:
            continue
        
        losses = muzero_training_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            config=mcts_config,
            step=global_step,
        )
        
        mcts_training_history['total_loss'].append(losses['total_loss'])
        mcts_training_history['recon_loss'].append(losses['recon_loss'])
        mcts_training_history['reward_loss'].append(losses['reward_loss'])
        mcts_training_history['kl_loss'].append(losses['kl_loss'])
        mcts_training_history['value_loss'].append(losses['value_loss'])
        mcts_training_history['policy_loss'].append(losses['policy_loss'])
        
        global_step += 1
    
    # ========== EVALUATION ==========
    if (episode + 1) % mcts_config['eval_every'] == 0:
        # Run evaluation episode (greedy, no exploration)
        model.eval()
        eval_traj, eval_reward, eval_length = run_mcts_episode(
            env=env,
            model=model,
            device=device,
            max_steps=500,
            num_simulations=mcts_config['num_simulations'],
            temperature=0.0,  # Greedy
        )
        
        recent_train_rewards = mcts_training_history['episode_rewards'][-mcts_config['eval_every']:]
        
        print(f"\n[Episode {episode + 1}]")
        print(f"  Train reward (last {mcts_config['eval_every']}): {np.mean(recent_train_rewards):.2f} ± {np.std(recent_train_rewards):.2f}")
        print(f"  Eval reward: {eval_reward:.2f} (length: {eval_length})")
        print(f"  Temperature: {temperature:.3f}")
        print(f"  Losses - Total: {losses['total_loss']:.4f}, Policy: {losses['policy_loss']:.4f}")
        print(f"  Buffer size: {len(mcts_replay_buffer)}")

print("\n" + "="*60)
print("MCTS Training Complete!")
print("="*60)

# ============================================================================
# Plot MCTS Training Results
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Episode rewards
axes[0, 0].plot(mcts_training_history['episode_rewards'])
axes[0, 0].set_title('Episode Rewards (MCTS)')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].grid(True)

# Episode lengths
axes[0, 1].plot(mcts_training_history['episode_lengths'])
axes[0, 1].set_title('Episode Lengths')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Steps')
axes[0, 1].grid(True)

# Total loss
axes[0, 2].plot(mcts_training_history['total_loss'])
axes[0, 2].set_title('Total Loss')
axes[0, 2].set_xlabel('Training Step')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True)

# Policy loss (distillation)
axes[1, 0].plot(mcts_training_history['policy_loss'])
axes[1, 0].set_title('Policy Distillation Loss')
axes[1, 0].set_xlabel('Training Step')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True)

# Value loss
axes[1, 1].plot(mcts_training_history['value_loss'])
axes[1, 1].set_title('Value Loss')
axes[1, 1].set_xlabel('Training Step')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True)

# Reconstruction loss
axes[1, 2].plot(mcts_training_history['recon_loss'])
axes[1, 2].set_title('Reconstruction Loss')
axes[1, 2].set_xlabel('Training Step')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig('mcts_training_results.png')
print("MCTS training results plot saved to mcts_training_results.png")

# Summary statistics
print("\n=== MCTS Training Summary ===")
print(f"Total episodes: {len(mcts_training_history['episode_rewards'])}")
print(f"Total training steps: {len(mcts_training_history['total_loss'])}")
print(f"\nReward statistics:")
print(f"  Mean: {np.mean(mcts_training_history['episode_rewards']):.2f}")
print(f"  Std: {np.std(mcts_training_history['episode_rewards']):.2f}")
print(f"  Max: {np.max(mcts_training_history['episode_rewards']):.2f}")
print(f"  Last 10 mean: {np.mean(mcts_training_history['episode_rewards'][-10:]):.2f}")

# ============================================================================
# Save MCTS-trained Model
# ============================================================================

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': mcts_config,
    'mcts_training_history': mcts_training_history,
    'losses_history': losses_history,
}, 'worldmodel_mcts_checkpoint.pth')

print("MCTS-trained model saved to worldmodel_mcts_checkpoint.pth")

# ============================================================================
# Final Evaluation
# ============================================================================

def evaluate_policy(env, policy_fn, num_episodes=10, max_steps=500):
    """Evaluate a policy over multiple episodes."""
    rewards = []
    lengths = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        obs = preprocess_obs(obs)
        total_reward = 0
        
        # For MCTS policy
        h_prev, z_prev, action_prev = None, None, None
        
        for step in range(max_steps):
            action = policy_fn(obs, h_prev, z_prev, action_prev)
            
            # Handle MCTS policy return values
            if isinstance(action, tuple):
                action, _, _, h_prev, z_prev = action
                action_prev = action
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = preprocess_obs(next_obs)
            done = terminated or truncated
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        rewards.append(total_reward)
        lengths.append(step + 1)
    
    return np.array(rewards), np.array(lengths)


# Define policies
def random_policy(obs, h_prev=None, z_prev=None, action_prev=None):
    return env.action_space.sample()

def mcts_greedy_policy(obs, h_prev=None, z_prev=None, action_prev=None):
    return mcts_policy(
        model=model,
        obs=obs,
        h_prev=h_prev,
        z_prev=z_prev,
        action_prev=action_prev,
        device=device,
        num_simulations=50,
        temperature=0.0,  # Greedy
    )


print("Evaluating policies...")
model.eval()

# Evaluate random policy
print("\nEvaluating Random Policy...")
random_rewards, random_lengths = evaluate_policy(env, random_policy, num_episodes=20)

# Evaluate MCTS policy
print("Evaluating MCTS Policy...")
mcts_rewards, mcts_lengths = evaluate_policy(env, mcts_greedy_policy, num_episodes=20)

# Results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"\nRandom Policy:")
print(f"  Reward: {random_rewards.mean():.2f} ± {random_rewards.std():.2f}")
print(f"  Length: {random_lengths.mean():.1f} ± {random_lengths.std():.1f}")

print(f"\nMCTS Policy (greedy):")
print(f"  Reward: {mcts_rewards.mean():.2f} ± {mcts_rewards.std():.2f}")
print(f"  Length: {mcts_lengths.mean():.1f} ± {mcts_lengths.std():.1f}")

improvement = mcts_rewards.mean() - random_rewards.mean()
print(f"\nImprovement over random: {improvement:+.2f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Rewards comparison
x = np.arange(2)
width = 0.35
axes[0].bar(x, [random_rewards.mean(), mcts_rewards.mean()], 
            yerr=[random_rewards.std(), mcts_rewards.std()], capsize=5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Random', 'MCTS'])
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Policy Comparison: Rewards')
axes[0].grid(True, axis='y')

# Episode lengths comparison
axes[1].bar(x, [random_lengths.mean(), mcts_lengths.mean()],
            yerr=[random_lengths.std(), mcts_lengths.std()], capsize=5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Random', 'MCTS'])
axes[1].set_ylabel('Episode Length')
axes[1].set_title('Policy Comparison: Episode Length')
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('evaluation_comparison.png')
print("Evaluation comparison plot saved to evaluation_comparison.png")

print("\nScript complete!")


