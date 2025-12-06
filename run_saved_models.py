#!/usr/bin/env python3
"""
Run saved world model checkpoints on rendered environment.

This script loads saved checkpoints and runs them for 5 episodes
with full rendering to visualize the agent's behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import miniworld
from PIL import Image
import sys
import os

# Import world model from worldmodel.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from worldmodel import WorldModel

# Set device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: CUDA")
else:
    device = torch.device('cpu')
    print(f"Using device: CPU")

print(f"Device: {device}\n")


def preprocess_obs(obs):
    """
    Preprocess observation to (3, 64, 64) tensor.
    MiniWorld returns (H, W, 3) numpy array.
    """
    if isinstance(obs, np.ndarray):
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


def get_root_state(model, obs, h_prev=None, z_prev=None, action_prev=None, device=None):
    """
    Encode observation into latent state using the world model.
    
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
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Prepare observation
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 64, 64)
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
            h_prev.unsqueeze(0),  # (1, hidden_dim)
            z_prev.unsqueeze(0),  # (1, stochastic_dim)
            action_one_hot        # (1, action_dim)
        )
        h_t = h_t.squeeze(0)  # (hidden_dim,)
        
        # Step 2: Encode observation
        e_t = model.encoder(obs_tensor)  # (1, embedding_dim)
        
        # Step 3: RSSM posterior to get z_t (corrected stochastic state)
        z_t, _ = model.rssm.posterior(h_t.unsqueeze(0), e_t)
        z_t = z_t.squeeze(0)  # (stochastic_dim,)
    
    return h_t, z_t


def direct_policy_action(model, h_t, z_t, temperature=1.0):
    """
    Select action directly from policy_prior_head.
    
    Args:
        model: WorldModel instance
        h_t: (hidden_dim,) tensor - current deterministic state
        z_t: (stochastic_dim,) tensor - current stochastic state
        temperature: sampling temperature (1.0 = normal, 0.0 = greedy)
    
    Returns:
        action: int - selected action
        probs: np.array - action probabilities
    """
    model.eval()
    
    with torch.no_grad():
        _, policy_probs = model.policy_prior_head(
            h_t.unsqueeze(0), z_t.unsqueeze(0)
        )
        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        
        if temperature == 0.0:
            # Greedy
            action = np.argmax(policy_probs)
        else:
            # Sample with temperature
            logits = np.log(policy_probs + 1e-8) / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            action = np.random.choice(len(probs), p=probs)
    
    return action, policy_probs


def run_episode(model, env, use_direct_policy=True, temperature=1.0, max_steps=500):
    """
    Run a single episode with the loaded model.
    
    Args:
        model: WorldModel instance
        env: gymnasium environment
        use_direct_policy: if True, use direct policy; if False, would use MCTS (not implemented here)
        temperature: action sampling temperature
        max_steps: maximum steps per episode
    
    Returns:
        total_reward: float
        episode_length: int
        frames: list of rendered frames
    """
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    
    # Initialize latent states
    h_prev = None
    z_prev = None
    action_prev = None
    
    total_reward = 0.0
    frames = []
    
    for step in range(max_steps):
        # Render and store frame
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())
        
        # Get latent state from observation
        h_t, z_t = get_root_state(model, obs, h_prev, z_prev, action_prev, device)
        
        # Select action
        if use_direct_policy:
            action, action_probs = direct_policy_action(model, h_t, z_t, temperature)
        else:
            # MCTS would go here (not implemented for simplicity)
            action, action_probs = direct_policy_action(model, h_t, z_t, temperature)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = preprocess_obs(next_obs)
        done = terminated or truncated
        
        total_reward += reward
        
        # Update states for next step
        h_prev = h_t
        z_prev = z_t
        action_prev = action
        obs = next_obs
        
        if done:
            break
    
    # Final render
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())
    
    return total_reward, step + 1, frames


def load_checkpoint(checkpoint_path):
    """
    Load a saved checkpoint.
    
    Returns:
        model: WorldModel instance
        config: dict - configuration
        checkpoint_info: dict - additional checkpoint info
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False to allow numpy objects
    # (Safe since these are user's own checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config', {})
    
    # Get model parameters from config
    action_dim = config.get('action_dim', 3)
    embedding_dim = config.get('embedding_dim', 128)
    hidden_dim = config.get('hidden_dim', 200)
    stochastic_dim = config.get('stochastic_dim', 64)
    action_space_size = config.get('action_dim', 3)  # Usually same as action_dim
    
    # Create model
    model = WorldModel(
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        stochastic_dim=stochastic_dim,
        action_space_size=action_space_size,
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model loaded successfully!")
    print(f"  Config: action_dim={action_dim}, stochastic_dim={stochastic_dim}")
    
    # Get additional info
    checkpoint_info = {
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'has_losses': 'losses_history' in checkpoint,
        'has_mcts_history': 'mcts_training_history' in checkpoint,
    }
    
    return model, config, checkpoint_info


def main():
    """Main function to run saved models."""
    
    # Check which checkpoints exist
    checkpoints = {
        # 'worldmodel': 'worldmodel_checkpoint.pth',
        'worldmodel_mcts': 'worldmodel_mcts_checkpoint.pth',
    }
    
    available = {}
    for name, path in checkpoints.items():
        if os.path.exists(path):
            available[name] = path
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")
    
    if not available:
        print("\n❌ No checkpoints found! Please train a model first.")
        return
    
    print(f"\n{'='*60}")
    print("Running Saved Models on Rendered Environment")
    print(f"{'='*60}\n")
    
    # Run each available checkpoint
    for checkpoint_name, checkpoint_path in available.items():
        print(f"\n{'='*60}")
        print(f"Checkpoint: {checkpoint_name}")
        print(f"{'='*60}\n")
        
        # Load model
        try:
            model, config, checkpoint_info = load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            continue
        
        # Create environment with rendering
        env_name = config.get('env_name', 'MiniWorld-OneRoom-v0')
        env = gym.make(env_name, render_mode='human')
        
        print(f"\nEnvironment: {env_name}")
        print(f"Action space: {env.action_space}")
        print(f"Running 5 episodes with rendering...\n")
        
        # Run 5 episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(5):
            print(f"Episode {episode + 1}/5...", end=" ", flush=True)
            
            reward, length, frames = run_episode(
                model, env, 
                use_direct_policy=True,
                temperature=1,  # Slightly greedy for evaluation
                max_steps=180
            )
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            
            print(f"Reward: {reward:.2f}, Length: {length}")
        
        env.close()
        
        # Print summary
        print(f"\n{'─'*60}")
        print(f"Summary for {checkpoint_name}:")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Mean length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"  Best reward: {np.max(episode_rewards):.2f}")
        print(f"  Worst reward: {np.min(episode_rewards):.2f}")
        print(f"{'─'*60}\n")
    
    print(f"\n{'='*60}")
    print("All checkpoints evaluated!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

