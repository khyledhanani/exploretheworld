import argparse
import time
import gymnasium as gym
import miniworld
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from worldmodel import WorldModel
from MCTS import MCTS


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


def run_episode(env, model, device, config, max_steps=500):
    """
    Run one episode using MCTS for action selection.
    
    Returns:
        total_reward: cumulative reward for the episode
        episode_length: number of steps taken
    """
    obs, info = env.reset()
    obs_proc = preprocess_obs(obs)
    
    # Render the initial state
    env.render()
    
    # Initialize latent state using the encoder
    with torch.no_grad():
        obs_tensor = torch.tensor(obs_proc, dtype=torch.float32, device=device).unsqueeze(0)
        e_t = model.encoder(obs_tensor)
        
        # Initialize h and z to zeros for the first step
        h_t = torch.zeros(1, config["hidden_dim"], device=device)
        z_t = torch.zeros(1, config["stochastic_dim"], device=device)
        
        # Get the posterior state from the initial observation
        action_prev = torch.zeros(1, config["action_dim"], device=device)
        h_t, z_t_prior, _ = model.rssm.prior(h_t, z_t, action_prev)
        z_t, _ = model.rssm.posterior(h_t, e_t)
    
    total_reward = 0.0
    episode_length = 0
    
    for step in range(max_steps):
        # Run MCTS to select action
        with torch.no_grad():
            action, policy_target, root_value = MCTS(
                world_model=model,
                root_h=h_t.squeeze(0),
                root_z=z_t.squeeze(0),
                c_puct=config["c_puct"],
                num_simulations=config["num_simulations"],
                discount=config["gamma"],
                action_space_size=config["action_dim"],
                temperature=config["temperature"],
                dirichlet_alpha=config["dirichlet_alpha"],
            )
        
        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_proc = preprocess_obs(next_obs)
        done = terminated or truncated
        
        # Render after each step
        env.render()
        
        # Add a small delay to make rendering visible
        time.sleep(0.05)  # 50ms delay
        
        total_reward += float(reward)
        episode_length += 1
        
        # Update latent state for next step
        if not done:
            with torch.no_grad():
                next_obs_tensor = torch.tensor(next_obs_proc, dtype=torch.float32, device=device).unsqueeze(0)
                e_t = model.encoder(next_obs_tensor)
                
                # One-hot encode the action
                action_tensor = torch.tensor([action], dtype=torch.long, device=device)
                action_one_hot = F.one_hot(action_tensor, num_classes=config["action_dim"]).float()
                
                # Update RSSM state
                h_t, z_t_prior, _ = model.rssm.prior(h_t, z_t, action_one_hot)
                z_t, _ = model.rssm.posterior(h_t, e_t)
        
        if done:
            break
    
    return total_reward, episode_length


def main():
    parser = argparse.ArgumentParser(
        description="Run worldmodel_mcts_checkpoint on MiniWorld-OneRoom-v0 with human rendering."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="worldmodel_mcts_checkpoint.pth",
        help="Path to the MCTS checkpoint file.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=50,
        help="Number of MCTS simulations per action.",
    )
    parser.add_argument(
        "--c_puct",
        type=float,
        default=1.0,
        help="PUCT exploration constant.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for action selection (0.0 = greedy).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    print("\nModel configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Initialize model
    model = WorldModel(
        action_dim=config["action_dim"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        stochastic_dim=config["stochastic_dim"],
        action_space_size=config["action_dim"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("\nModel loaded successfully!")

    # MCTS configuration
    mcts_config = {
        "action_dim": config["action_dim"],
        "hidden_dim": config["hidden_dim"],
        "stochastic_dim": config["stochastic_dim"],
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "gamma": config.get("gamma", 0.99),
        "temperature": args.temperature,
        "dirichlet_alpha": 0.3,
    }
    
    print("\nMCTS configuration:")
    print(f"  num_simulations: {args.num_simulations}")
    print(f"  c_puct: {args.c_puct}")
    print(f"  temperature: {args.temperature}")
    print(f"  gamma: {mcts_config['gamma']}")

    # Create environment with human rendering
    print(f"\nCreating environment: {config['env_name']} with human rendering...")
    env = gym.make(config["env_name"], render_mode="human")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes...\n")
    episode_rewards = []
    episode_lengths = []

    for episode_idx in range(args.num_episodes):
        print(f"Episode {episode_idx + 1}/{args.num_episodes}")
        
        total_reward, episode_length = run_episode(
            env, model, device, mcts_config, max_steps=args.max_steps
        )
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print()

    # Summary statistics
    print("=" * 60)
    print("Summary Statistics:")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    print(f"  Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
