# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(800, 600))
# display.start()

import os
# os.environ.pop("PYGLET_HEADLESS", None)
# os.environ.pop("MINIWORLD_HEADLESS", None)
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import argparse
import miniworld
import sys
from contextlib import contextmanager

from worldmodel import WorldModel
from MCTS import MCTS


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_config():
    """Returns the training configuration."""
    return {
        'env_name': 'MiniWorld-OneRoom-v0',
        'obs_size': (64, 64),
        'action_dim': 3,
        'embedding_dim': 128,
        'hidden_dim': 200,
        'stochastic_dim': 64,
        
        # Training hyperparameters
        'batch_size': 32,
        'seq_length': 16,
        'learning_rate': 3e-4,
        'num_collection_episodes': 100,
        'num_training_steps': 5000,
        'collect_every_n_steps': 50,
        
        # Loss weights
        'lambda_rec': 10.0,
        'lambda_kl_start': 0.0,
        'lambda_kl_end': 0.10,
        'kl_anneal_steps': 4000,
        'lambda_reward': 1.0,
        'lambda_value': 10.0,  # Increased from 1.0 to force value learning with sparse rewards
        'lambda_policy': 1.0,
        'free_nats': 1.0,
        'use_huber_value_loss': True,  # Use Huber loss for more robust value learning
        
        # N-step returns
        'n_step': 5,
        'gamma': 0.99,
        
        # Exploration
        'epsilon': 0.3,
        
        # Reward shaping
        'use_reward_shaping': True,
        'shaping_scale': 0.1,
        'goal_bonus': 10.0,
        'near_goal_threshold': 0.8,
        
        # MCTS parameters
        'num_simulations': 50,
        'c_puct': 1.0,
        'temperature_start': 1.0,
        'temperature_end': 0.1,
        'temperature_decay_episodes': 200,
        
        # MCTS training schedule
        'mcts_episodes': 800,
        'training_steps_per_episode': 100,
        'eval_every': 10,
    }



def get_device():
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    return device



class ReplayBuffer:
    """Replay buffer for storing trajectories."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
        })
    
    def add_trajectory(self, trajectory):
        """Add a full trajectory"""
        for transition in trajectory:
            self.add(**transition)
    
    def sample_sequences(self, batch_size, seq_length, action_dim):
        """Sample sequences of length seq_length from the buffer."""
        if len(self.buffer) < seq_length:
            return None
        
        max_start = len(self.buffer) - seq_length

        valid_starts = []
        attempts = 0
        max_attempts = max(200, batch_size * 50)
        while len(valid_starts) < batch_size and attempts < max_attempts:
            s = int(np.random.randint(0, max_start))
            # Disallow a terminal in the middle of the sequence (allow at last step).
            crosses = any(float(self.buffer[s + i]['done']) > 0.5 for i in range(seq_length - 1))
            if not crosses:
                valid_starts.append(s)
            attempts += 1

        if len(valid_starts) < batch_size:
            # Fill remainder with random (state-reset logic in training will handle).
            filler = np.random.randint(0, max_start, size=(batch_size - len(valid_starts)))
            starts = np.array(valid_starts + list(filler), dtype=np.int64)
        else:
            starts = np.array(valid_starts, dtype=np.int64)
        
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        
        for start in starts:
            seq = [self.buffer[start + i] for i in range(seq_length)]
            
            obs_seq.append([s['obs'] for s in seq])
            action_seq.append([s['action'] for s in seq])
            reward_seq.append([s['reward'] for s in seq])
            done_seq.append([s['done'] for s in seq])
        
        # Convert to tensors
        obs_tensor = torch.stack([torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]) for obs in obs_seq])
        action_tensor = torch.stack([torch.stack([F.one_hot(torch.tensor(a, dtype=torch.long), action_dim).float() for a in action]) for action in action_seq])
        reward_tensor = torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in reward_seq])
        done_tensor = torch.stack([torch.tensor(done, dtype=torch.float32) for done in done_seq])
        
        return obs_tensor, action_tensor, reward_tensor, done_tensor
    
    def __len__(self):
        return len(self.buffer)


class MCTSReplayBuffer:
    """Replay buffer that also stores MCTS policy targets and value estimates."""
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
        """Sample sequences with MCTS targets if available."""
        if len(self.buffer) < seq_length:
            return None
        
        max_start = len(self.buffer) - seq_length

        valid_starts = []
        attempts = 0
        max_attempts = max(200, batch_size * 50)
        while len(valid_starts) < batch_size and attempts < max_attempts:
            s = int(np.random.randint(0, max_start))
            crosses = any(float(self.buffer[s + i]['done']) > 0.5 for i in range(seq_length - 1))
            if not crosses:
                valid_starts.append(s)
            attempts += 1

        if len(valid_starts) < batch_size:
            filler = np.random.randint(0, max_start, size=(batch_size - len(valid_starts)))
            starts = np.array(valid_starts + list(filler), dtype=np.int64)
        else:
            starts = np.array(valid_starts, dtype=np.int64)
        
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
            
            policy_targets = [s['policy_target'] for s in seq]
            value_targets = [s['value_target'] for s in seq]

            timestep_has = [p is not None for p in policy_targets]
            has_mcts_targets.append(timestep_has)

            filled_policy = [
                (p if p is not None else (np.ones(action_dim, dtype=np.float32) / action_dim))
                for p in policy_targets
            ]
            filled_value = [
                (v if v is not None else 0.0)
                for v in value_targets
            ]
            policy_target_seq.append(filled_policy)
            value_target_seq.append(filled_value)
        
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



class ShapedRewardWrapper(gym.Wrapper):
    """Adds distance-based reward shaping to MiniWorld environments."""
    def __init__(self, env, shaping_scale=0.1, goal_bonus=10.0, near_goal_threshold=0.8):
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self.goal_bonus = goal_bonus
        self.near_goal_threshold = near_goal_threshold
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
                
                if curr_dist > self.near_goal_threshold:
                    shaping = (self.prev_dist - curr_dist) * self.shaping_scale
                    reward = reward + shaping
                
                if terminated:
                    reward += self.goal_bonus
                
                self.prev_dist = curr_dist
                info['distance_to_goal'] = curr_dist
            except AttributeError:
                pass
        
        return obs, reward, terminated, truncated, info


def preprocess_obs(obs):
    """Preprocess observation to (3, 64, 64) tensor."""
    if isinstance(obs, np.ndarray):
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        img = Image.fromarray(obs)
    else:
        img = obs
    
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def heuristic_policy(env, epsilon=0.3):
    """Simple heuristic policy: prefers moving forward, occasionally random."""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return 0  # move_forward


def compute_n_step_returns(rewards, dones, values, gamma=0.99, n_step=5):
    """Compute n-step returns for value targets."""
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


def collect_trajectory(env, policy_fn, config, max_steps=500):
    """Collect a single trajectory using the exploration policy."""
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    
    trajectory = []
    total_reward = 0
    
    for step in range(max_steps):
        action = policy_fn(env, config['epsilon'])
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_obs(next_obs)
        done = terminated or truncated
        
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



def get_root_state(model, obs, h_prev, z_prev, action_prev, device):
    """Compute the root latent state (h_t, z_t) for MCTS from a real observation."""
    model.eval()
    
    with torch.no_grad():
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 3 else obs
            obs_tensor = obs_tensor.to(device)
        
        if h_prev is None:
            h_prev = torch.zeros(model.hidden_dim, device=device)
        if z_prev is None:
            z_prev = torch.zeros(model.stochastic_dim, device=device)
        
        if action_prev is None:
            action_one_hot = torch.zeros(1, model.action_dim, device=device)
        else:
            action_one_hot = F.one_hot(
                torch.tensor([action_prev], device=device),
                num_classes=model.action_dim
            ).float()
        
        h_t, _, _ = model.rssm.prior(
            h_prev.unsqueeze(0),
            z_prev.unsqueeze(0),
            action_one_hot
        )
        h_t = h_t.squeeze(0)
        
        e_t = model.encoder(obs_tensor)
        
        z_t, _ = model.rssm.posterior(h_t.unsqueeze(0), e_t)
        z_t = z_t.squeeze(0)
    
    return h_t, z_t


def mcts_policy(model, obs, h_prev, z_prev, action_prev, device,
                num_simulations=50, c_puct=1.0, discount=0.99, temperature=1.0):
    """Use MCTS to select an action given the current observation."""
    h_t, z_t = get_root_state(model, obs, h_prev, z_prev, action_prev, device)
    
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


def run_mcts_episode(env, model, device, config, max_steps=500,
                     num_simulations=50, c_puct=1.0, discount=0.99, temperature=1.0):
    """Run a single episode using MCTS for action selection."""
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    
    trajectory = []
    total_reward = 0
    
    h_prev = None
    z_prev = None
    action_prev = None
    
    for step in range(max_steps):
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
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_obs(next_obs)
        done = terminated or truncated
        
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
        
        obs = next_obs
        h_prev = h_t
        z_prev = z_t
        action_prev = action
        
        if done:
            break
    
    return trajectory, total_reward, len(trajectory)


def plot_warmup_losses(losses_history, save_path='warmup_losses.png'):
    """Plot warmup training losses with diagnostics."""
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
    axes[2, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.3)
    
    # Rolling average for smoothed view
    if len(losses_history['total']) > 100:
        window = 100
        smoothed_total = np.convolve(losses_history['total'], 
                                     np.ones(window)/window, mode='valid')
        axes[2, 1].plot(smoothed_total)
        axes[2, 1].set_title(f'Total Loss (smoothed, window={window})')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].grid(True)
    
    # KL vs Reconstruction trade-off
    axes[2, 2].scatter(losses_history['kl'], losses_history['recon'], 
                       alpha=0.3, s=1)
    axes[2, 2].set_title('KL vs Reconstruction Trade-off')
    axes[2, 2].set_xlabel('KL Loss')
    axes[2, 2].set_ylabel('Reconstruction Loss')
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Warmup losses plot saved to {save_path}")
    plt.close()


def train_world_model_warmup(model, optimizer, replay_buffer, env, device, config):
    """Warm-up training phase: train world model on heuristic data."""
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
    
    print("Starting warm-up training...")
    
    for step in tqdm(range(config['num_training_steps']), desc="Training"):
        # Collect new data periodically using the passed environment
        if step % config['collect_every_n_steps'] == 0 and step > 0:
            trajectory, _, _ = collect_trajectory(env, heuristic_policy, config, max_steps=500)
            replay_buffer.add_trajectory(trajectory)
        
        # KL annealing
        if step < config['kl_anneal_steps']:
            kl_weight = config['lambda_kl_start'] + \
                (config['lambda_kl_end'] - config['lambda_kl_start']) * (step / config['kl_anneal_steps'])
        else:
            kl_weight = config['lambda_kl_end']
        
        # Sample batch
        batch = replay_buffer.sample_sequences(
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            action_dim=config['action_dim']
        )
        
        if batch is None:
            continue
        
        obs_seq, action_seq, reward_seq, done_seq = batch
        obs_seq = obs_seq.to(device)
        action_seq = action_seq.to(device)
        reward_seq = reward_seq.to(device)
        done_seq = done_seq.to(device)
        
        B, T = obs_seq.shape[:2]
        
        # Forward pass through sequence
        h_prev = None
        z_prev = None
        all_outputs = []
        
        for t in range(T):
            obs_t = obs_seq[:, t]
            action_t = action_seq[:, t]
            
            if t == 0:
                action_prev = torch.zeros(B, config['action_dim'], device=device)
            else:
                done_prev = done_seq[:, t - 1].unsqueeze(-1)  # (B, 1)
                if h_prev is not None:
                    h_prev = h_prev * (1.0 - done_prev)
                if z_prev is not None:
                    z_prev = z_prev * (1.0 - done_prev)
                action_prev = action_seq[:, t - 1] * (1.0 - done_prev)
            
            outputs = model(obs_t, action_prev, h_prev, z_prev, use_posterior=True)
            all_outputs.append(outputs)
            
            h_prev = outputs['h_t']
            z_prev = outputs['z_t']
        
        # Stack outputs
        o_hat_seq = torch.stack([o['o_hat_t'] for o in all_outputs], dim=1)
        r_hat_seq = torch.stack([o['r_hat_t'] for o in all_outputs], dim=1)
        v_hat_seq = torch.stack([o['v_hat_t'] for o in all_outputs], dim=1)
        
        # Compute n-step returns
        with torch.no_grad():
            value_targets = compute_n_step_returns(
                reward_seq, done_seq, v_hat_seq.detach(),
                gamma=config['gamma'],
                n_step=config['n_step']
            )
        
        # Compute losses
        recon_loss = F.mse_loss(o_hat_seq, obs_seq)
        reward_loss = F.mse_loss(r_hat_seq, reward_seq)
        
        # Use Huber loss for value function for more robust learning
        if config.get('use_huber_value_loss', False):
            value_loss = F.smooth_l1_loss(v_hat_seq, value_targets)
        else:
            value_loss = F.mse_loss(v_hat_seq, value_targets)
        
        # KL loss with free bits
        kl_loss_raw = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        posterior_stds = []
        prior_stds = []
        
        for t in range(T):
            if all_outputs[t]['prior_dist'] is not None and all_outputs[t]['post_dist'] is not None:
                kl_per_dim = torch.distributions.kl.kl_divergence(
                    all_outputs[t]['post_dist'],
                    all_outputs[t]['prior_dist']
                )

                kl_per_sample = kl_per_dim.sum(dim=-1)  # (B,)
                kl_t_raw = kl_per_sample.mean()
                kl_loss_raw += kl_t_raw

                kl_t = torch.clamp(kl_per_sample - config['free_nats'], min=0.0).mean()
                kl_loss += kl_t
                
                posterior_stds.append(all_outputs[t]['post_dist'].stddev.mean().item())
                prior_stds.append(all_outputs[t]['prior_dist'].stddev.mean().item())
        
        kl_loss = kl_loss / T
        kl_loss_raw = kl_loss_raw / T
        
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
        
        if (step + 1) % 1000 == 0:
            print(f"\nStep {step + 1}/{config['num_training_steps']}")
            print(f"  Total: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                  f"Reward: {reward_loss.item():.4f}")
            print(f"  KL (raw/clamped): {kl_loss_raw.item():.4f}/{kl_loss.item():.4f}, "
                  f"KL weight: {kl_weight:.4f}")
            print(f"  Value: {value_loss.item():.4f}")
    
    return losses_history


def muzero_training_step(model, optimizer, batch, device, config, step):
    """Single MuZero-style training step with policy distillation."""
    model.train()
    
    obs_seq = batch['obs'].to(device)
    action_seq = batch['action'].to(device)
    reward_seq = batch['reward'].to(device)
    done_seq = batch['done'].to(device)
    policy_target = batch['policy_target'].to(device)
    value_target = batch['value_target'].to(device)
    has_mcts = batch['has_mcts_targets'].to(device)
    
    B, T = obs_seq.shape[:2]
    
    # KL annealing
    if step < config['kl_anneal_steps']:
        kl_weight = config['lambda_kl_start'] + \
            (config['lambda_kl_end'] - config['lambda_kl_start']) * (step / config['kl_anneal_steps'])
    else:
        kl_weight = config['lambda_kl_end']
    
    # Forward pass
    h_prev = None
    z_prev = None
    all_outputs = []
    
    for t in range(T):
        obs_t = obs_seq[:, t]
        if t == 0:
            action_prev = torch.zeros(B, config['action_dim'], device=device)
        else:
            # Reset RSSM state on episode boundaries inside the batch.
            done_prev = done_seq[:, t - 1].unsqueeze(-1)  # (B, 1)
            if h_prev is not None:
                h_prev = h_prev * (1.0 - done_prev)
            if z_prev is not None:
                z_prev = z_prev * (1.0 - done_prev)
            action_prev = action_seq[:, t - 1] * (1.0 - done_prev)
        
        outputs = model(obs_t, action_prev, h_prev, z_prev, use_posterior=True)
        all_outputs.append(outputs)
        
        h_prev = outputs['h_t']
        z_prev = outputs['z_t']
    
    # Stack outputs
    o_hat_seq = torch.stack([o['o_hat_t'] for o in all_outputs], dim=1)
    r_hat_seq = torch.stack([o['r_hat_t'] for o in all_outputs], dim=1)
    v_hat_seq = torch.stack([o['v_hat_t'] for o in all_outputs], dim=1)
    policy_logits_seq = torch.stack([o['policy_logits'] for o in all_outputs], dim=1)
    
    # Compute losses
    recon_loss = F.mse_loss(o_hat_seq, obs_seq)
    reward_loss = F.mse_loss(r_hat_seq, reward_seq)
    
    # KL loss
    kl_loss = torch.tensor(0.0, device=device)
    for t in range(T):
        if all_outputs[t]['prior_dist'] is not None and all_outputs[t]['post_dist'] is not None:
            kl_per_dim = torch.distributions.kl.kl_divergence(
                all_outputs[t]['post_dist'],
                all_outputs[t]['prior_dist']
            )
            kl_per_sample = kl_per_dim.sum(dim=-1)  # (B,)
            kl_loss += torch.clamp(kl_per_sample - config['free_nats'], min=0.0).mean()
    kl_loss = kl_loss / T
    
    # Value loss with MCTS value targets when available (per-timestep)
    with torch.no_grad():
        # Compute n-step returns as baseline
        n_step_returns = compute_n_step_returns(
            reward_seq, done_seq, v_hat_seq.detach(),
            gamma=config['gamma'], n_step=config['n_step']
        )

        value_targets = torch.where(has_mcts, value_target, n_step_returns)
    
    # Use Huber loss for more robust value learning with sparse rewards
    if config.get('use_huber_value_loss', False):
        value_loss = F.smooth_l1_loss(v_hat_seq, value_targets)
    else:
        value_loss = F.mse_loss(v_hat_seq, value_targets)
    
    # Policy distillation loss
    policy_loss = torch.tensor(0.0, device=device)
    if has_mcts.any():
        policy_logits_flat = policy_logits_seq.reshape(-1, config['action_dim'])
        policy_target_flat = policy_target.reshape(-1, config['action_dim'])
        mask = has_mcts.reshape(-1)

        if mask.sum() > 0:
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
        # Add value diagnostics
        'value_pred_mean': v_hat_seq.mean().item(),
        'value_pred_std': v_hat_seq.std().item(),
        'value_target_mean': value_targets.mean().item(),
        'value_target_std': value_targets.std().item(),
        'reward_mean': reward_seq.mean().item(),
        'has_mcts_frac': has_mcts.float().mean().item(),
    }


def plot_combined_summary(warmup_losses, mcts_history, save_path='training_summary.png'):
    """Plot combined summary of warmup and MCTS training."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Episode metrics
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(mcts_history['episode_rewards'], alpha=0.6, label='Episode Reward')
    if len(mcts_history['episode_rewards']) > 10:
        window = min(20, len(mcts_history['episode_rewards']) // 5)
        smoothed = np.convolve(mcts_history['episode_rewards'], 
                              np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(mcts_history['episode_rewards'])), 
                smoothed, 'r-', linewidth=2, label=f'{window}-ep moving avg')
    ax1.set_title('MCTS Episode Rewards', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(mcts_history['episode_rewards'], bins=30, alpha=0.7, edgecolor='black')
    ax2.set_title('Reward Distribution', fontsize=12)
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Middle row: Loss evolution across both phases
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(warmup_losses['recon'], alpha=0.5, label='Warmup', color='blue')
    if len(mcts_history['recon_loss']) > 0:
        warmup_steps = len(warmup_losses['recon'])
        ax3.plot(range(warmup_steps, warmup_steps + len(mcts_history['recon_loss'])),
                mcts_history['recon_loss'], alpha=0.5, label='MCTS', color='orange')
        ax3.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='MCTS Start')
    ax3.set_title('Reconstruction Loss', fontsize=12)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(warmup_losses['value'], alpha=0.5, label='Warmup', color='blue')
    if len(mcts_history['value_loss']) > 0:
        warmup_steps = len(warmup_losses['value'])
        ax4.plot(range(warmup_steps, warmup_steps + len(mcts_history['value_loss'])),
                mcts_history['value_loss'], alpha=0.5, label='MCTS', color='orange')
        ax4.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='MCTS Start')
    ax4.set_title('Value Loss', fontsize=12)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    if len(mcts_history['policy_loss']) > 0:
        ax5.plot(mcts_history['policy_loss'], alpha=0.7, color='green')
        ax5.set_title('Policy Distillation Loss', fontsize=12)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Loss')
        ax5.grid(True, alpha=0.3)
    
    # Bottom row: KL and diagnostics
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(warmup_losses['kl'], label='Warmup KL', alpha=0.7, color='blue')
    if len(mcts_history['kl_loss']) > 0:
        warmup_steps = len(warmup_losses['kl'])
        ax6.plot(range(warmup_steps, warmup_steps + len(mcts_history['kl_loss'])),
                mcts_history['kl_loss'], alpha=0.7, label='MCTS KL', color='orange')
        ax6.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5)
    ax6.set_title('KL Divergence', fontsize=12)
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(warmup_losses['posterior_std'], label='Posterior', alpha=0.7)
    ax7.plot(warmup_losses['prior_std'], label='Prior', alpha=0.7)
    ax7.axhline(y=0.1, color='r', linestyle='--', alpha=0.3, label='Collapse threshold')
    ax7.set_title('Latent Std Dev (Warmup)', fontsize=12)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Std Dev')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 2])
    if len(mcts_history['episode_lengths']) > 0:
        ax8.plot(mcts_history['episode_lengths'], alpha=0.6)
        if len(mcts_history['episode_lengths']) > 10:
            window = min(20, len(mcts_history['episode_lengths']) // 5)
            smoothed = np.convolve(mcts_history['episode_lengths'], 
                                  np.ones(window)/window, mode='valid')
            ax8.plot(range(window-1, len(mcts_history['episode_lengths'])), 
                    smoothed, 'r-', linewidth=2)
        ax8.set_title('Episode Lengths', fontsize=12)
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Steps')
        ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Training Summary: Warmup + MCTS', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Combined training summary saved to {save_path}")
    plt.close()


def plot_mcts_training(training_history, save_path='mcts_training.png'):
    """Plot MCTS training results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(training_history['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards (MCTS)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Add rolling average if enough data
    if len(training_history['episode_rewards']) > 10:
        window = 10
        smoothed = np.convolve(training_history['episode_rewards'], 
                              np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(training_history['episode_rewards'])), 
                       smoothed, 'r-', alpha=0.7, linewidth=2, label=f'{window}-ep avg')
        axes[0, 0].legend()
    
    # Episode lengths
    axes[0, 1].plot(training_history['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Total loss
    axes[0, 2].plot(training_history['total_loss'])
    axes[0, 2].set_title('Total Loss')
    axes[0, 2].set_xlabel('Training Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    
    # Policy loss (distillation)
    axes[1, 0].plot(training_history['policy_loss'])
    axes[1, 0].set_title('Policy Distillation Loss')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Value loss
    axes[1, 1].plot(training_history['value_loss'])
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    # Reconstruction loss
    axes[1, 2].plot(training_history['recon_loss'])
    axes[1, 2].set_title('Reconstruction Loss')
    axes[1, 2].set_xlabel('Training Step')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"MCTS training plot saved to {save_path}")
    plt.close()


def train_with_mcts(model, optimizer, mcts_replay_buffer, env, device, config):
    """MCTS training loop."""
    training_history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'total_loss': [],
        'recon_loss': [],
        'reward_loss': [],
        'kl_loss': [],
        'value_loss': [],
        'policy_loss': [],
        'value_pred_mean': [],
        'value_target_mean': [],
        'reward_mean': [],
    }
    
    print("\n" + "="*60)
    print("Starting MCTS Training Loop")
    print("="*60)
    
    global_step = 0
    
    for episode in tqdm(range(config['mcts_episodes']), desc="MCTS Episodes"):
        # Temperature annealing
        temp_progress = min(1.0, episode / config['temperature_decay_episodes'])
        temperature = config['temperature_start'] + \
            (config['temperature_end'] - config['temperature_start']) * temp_progress
        
        # Collect episode with MCTS
        trajectory, total_reward, episode_length = run_mcts_episode(
            env=env,
            model=model,
            device=device,
            config=config,
            max_steps=500,
            num_simulations=config['num_simulations'],
            c_puct=config['c_puct'],
            discount=config['gamma'],
            temperature=temperature,
        )
        
        mcts_replay_buffer.add_trajectory(trajectory)
        
        training_history['episode_rewards'].append(total_reward)
        training_history['episode_lengths'].append(episode_length)
        
        # Training steps
        for _ in range(config['training_steps_per_episode']):
            batch = mcts_replay_buffer.sample_sequences(
                batch_size=config['batch_size'],
                seq_length=config['seq_length'],
                action_dim=config['action_dim'],
            )
            
            if batch is None:
                continue
            
            losses = muzero_training_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                config=config,
                step=global_step,
            )
            
            training_history['total_loss'].append(losses['total_loss'])
            training_history['recon_loss'].append(losses['recon_loss'])
            training_history['reward_loss'].append(losses['reward_loss'])
            training_history['kl_loss'].append(losses['kl_loss'])
            training_history['value_loss'].append(losses['value_loss'])
            training_history['policy_loss'].append(losses['policy_loss'])
            training_history['value_pred_mean'].append(losses['value_pred_mean'])
            training_history['value_target_mean'].append(losses['value_target_mean'])
            training_history['reward_mean'].append(losses['reward_mean'])
            
            global_step += 1
        
        # Evaluation
        if (episode + 1) % config['eval_every'] == 0:
            model.eval()
            eval_traj, eval_reward, eval_length = run_mcts_episode(
                env=env,
                model=model,
                device=device,
                config=config,
                max_steps=500,
                num_simulations=config['num_simulations'],
                temperature=0.0,
            )
            
            recent_train_rewards = training_history['episode_rewards'][-config['eval_every']:]
            
            print(f"\n[Episode {episode + 1}]")
            print(f"  Train reward (last {config['eval_every']}): "
                  f"{np.mean(recent_train_rewards):.2f} ± {np.std(recent_train_rewards):.2f}")
            print(f"  Eval reward: {eval_reward:.2f} (length: {eval_length})")
            print(f"  Temperature: {temperature:.3f}")
            print(f"  Losses - Total: {losses['total_loss']:.4f}, "
                  f"Policy: {losses['policy_loss']:.4f}, Value: {losses['value_loss']:.4f}")
            print(f"  Value Pred: {losses['value_pred_mean']:.4f} ± {losses['value_pred_std']:.4f}")
            print(f"  Value Target: {losses['value_target_mean']:.4f} ± {losses['value_target_std']:.4f}")
            print(f"  Reward Mean: {losses['reward_mean']:.4f}, MCTS coverage: {losses['has_mcts_frac']:.2%}")
            print(f"  Buffer size: {len(mcts_replay_buffer)}")
    
    print("\n" + "="*60)
    print("MCTS Training Complete!")
    print("="*60)
    
    return training_history


def main():
    parser = argparse.ArgumentParser(description='Train World Model with MCTS')
    parser.add_argument('--warmup-only', action='store_true',
                        help='Only run warm-up training, skip MCTS')
    parser.add_argument('--load-warmup', type=str,
                        help='Load warm-up checkpoint and skip to MCTS training')
    parser.add_argument('--eval-only', type=str,
                        help='Load checkpoint and only run evaluation')
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    device = get_device()
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Initialize environment for warmup (with reward shaping)
    print("\nInitializing environment...")
    with suppress_output():
        env_base = gym.make(config['env_name'], render_mode='rgb_array')
    env_warmup = ShapedRewardWrapper(
        env_base,
        shaping_scale=config['shaping_scale'],
        goal_bonus=config['goal_bonus'],
        near_goal_threshold=config['near_goal_threshold']
    ) if config['use_reward_shaping'] else env_base
    print(f"Environment: {config['env_name']}")
    print(f"Action space: {env_warmup.action_space}")
    if config['use_reward_shaping']:
        print(f"Using reward shaping for warmup (scale={config['shaping_scale']}, bonus={config['goal_bonus']})")
    else:
        print("Using raw rewards (no shaping) for warmup phase")
    
    # Initialize world model
    print("\nInitializing world model...")
    model = WorldModel(
        action_dim=config['action_dim'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        stochastic_dim=config['stochastic_dim'],
        action_space_size=config['action_dim'],
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Evaluation only
    if args.eval_only:
        print(f"\nLoading checkpoint from {args.eval_only}...")
        checkpoint = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded!")
        
        print("\nRunning evaluation (without reward shaping)...")
        # Create environment without reward shaping for evaluation
        with suppress_output():
            env_eval = gym.make(config['env_name'], render_mode='rgb_array')
        
        model.eval()
        eval_rewards = []
        eval_lengths = []
        
        for i in range(20):
            traj, reward, length = run_mcts_episode(
                env=env_eval,
                model=model,
                device=device,
                config=config,
                max_steps=500,
                num_simulations=50,
                temperature=0.0,
            )
            eval_rewards.append(reward)
            eval_lengths.append(length)
            print(f"  Episode {i+1}: reward={reward:.2f}, length={length}")
        
        print(f"\nEvaluation Results:")
        print(f"  Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  Mean length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
        
        env_eval.close()
        env_warmup.close()
        return
    
    # Warm-up training or load checkpoint
    losses_history = None  # Initialize for later use in combined plot
    if args.load_warmup:
        print(f"\nLoading warm-up checkpoint from {args.load_warmup}...")
        checkpoint = torch.load(args.load_warmup, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Try to load losses history if available
        if 'losses_history' in checkpoint:
            losses_history = checkpoint['losses_history']
            print("Loaded warmup losses history from checkpoint")
        print("Warm-up checkpoint loaded!")
        
        # Initialize MCTS replay buffer
        mcts_replay_buffer = MCTSReplayBuffer(capacity=50000)
    else:
        # Warm-up phase: collect initial data
        print("\n" + "="*60)
        print("Phase 1: Warm-up Data Collection")
        print("="*60)
        
        replay_buffer = ReplayBuffer(capacity=50000)
        
        print("Collecting initial trajectories...")
        episode_rewards = []
        episode_lengths = []
        
        for episode in tqdm(range(config['num_collection_episodes']), desc="Collecting"):
            trajectory, total_reward, traj_length = collect_trajectory(
                env_warmup, heuristic_policy, config, max_steps=500
            )
            replay_buffer.add_trajectory(trajectory)
            episode_rewards.append(total_reward)
            episode_lengths.append(traj_length)
        
        print(f"\nCollected {len(replay_buffer)} transitions")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        
        # Warm-up training
        print("\n" + "="*60)
        print("Phase 2: Warm-up Training")
        print("="*60)
        
        losses_history = train_world_model_warmup(
            model, optimizer, replay_buffer, env_warmup, device, config
        )
        
        print("\nTraining complete!")
        
        # Plot warmup losses
        plot_warmup_losses(losses_history, save_path='warmup_losses.png')
        
        # Save warm-up checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'losses_history': losses_history,
        }, 'worldmodel_checkpoint.pth')
        print("Warm-up model saved to worldmodel_checkpoint.pth")
        
        if args.warmup_only:
            print("\nWarm-up only mode. Exiting.")
            env_warmup.close()
            return
        
        # Initialize MCTS replay buffer with warm-up data
        print("\nCopying warm-up data to MCTS replay buffer...")
        mcts_replay_buffer = MCTSReplayBuffer(capacity=50000)
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
    
    # MCTS training phase
    print("\n" + "="*60)
    print("Phase 3: MCTS Training")
    print("="*60)
    
    # Create new environment WITH reward shaping for MCTS
    print("Creating new environment for MCTS...")
    with suppress_output():
        env_mcts_base = gym.make(config['env_name'], render_mode='rgb_array')
    env_mcts = ShapedRewardWrapper(
        env_mcts_base,
        shaping_scale=config['shaping_scale'],
        goal_bonus=config['goal_bonus'],
        near_goal_threshold=config['near_goal_threshold']
    ) if config['use_reward_shaping'] else env_mcts_base
    
    if config['use_reward_shaping']:
        print(f"Using reward shaping for MCTS (scale={config['shaping_scale']}, bonus={config['goal_bonus']})")
    else:
        print("Using raw rewards (no shaping) for MCTS phase")
    
    # Close warmup environment if it exists
    if 'env_warmup' in locals():
        env_warmup.close()
    
    mcts_training_history = train_with_mcts(
        model, optimizer, mcts_replay_buffer, env_mcts, device, config
    )
    
    # Plot MCTS training results
    plot_mcts_training(mcts_training_history, save_path='mcts_training.png')
    
    # Plot combined summary if we have warmup losses
    if 'losses_history' in locals() and losses_history is not None:
        plot_combined_summary(losses_history, mcts_training_history, 
                            save_path='training_summary.png')
    
    # Save MCTS checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'mcts_training_history': mcts_training_history,
    }, 'worldmodel_mcts_checkpoint.pth')
    print("\nMCTS-trained model saved to worldmodel_mcts_checkpoint.pth")
    
    # Final summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total MCTS episodes: {len(mcts_training_history['episode_rewards'])}")
    print(f"Mean reward: {np.mean(mcts_training_history['episode_rewards']):.2f}")
    print(f"Std reward: {np.std(mcts_training_history['episode_rewards']):.2f}")
    print(f"Max reward: {np.max(mcts_training_history['episode_rewards']):.2f}")
    print(f"Last 10 episodes mean: "
          f"{np.mean(mcts_training_history['episode_rewards'][-10:]):.2f}")
    
    env_mcts.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

