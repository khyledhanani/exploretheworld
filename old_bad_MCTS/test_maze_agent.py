"""
Test/Visualize trained RecurrentPPO agent in MiniWorld Maze

Usage:
    python test_maze_agent.py --model ./models/improved_recurrent_ppo_maze/final_model
"""

import argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import RecurrentPPO


def test_agent(
    model_path: str,
    n_episodes: int = 5,
    max_steps: int = 1000,
    render: bool = True,
    seed: int = 42,
):
    """
    Load and test a trained RecurrentPPO agent in MiniWorld Maze.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        n_episodes: Number of episodes to run
        max_steps: Max steps per episode
        render: Whether to render the environment
        seed: Random seed
    """
    
    print(f"Loading model from: {model_path}")
    model = RecurrentPPO.load(model_path)
    
    # Create environment with rendering
    print("Creating environment...")
    import miniworld
    
    if render:
        env = gym.make('MiniWorld-Maze-v0', render_mode='human')
    else:
        env = gym.make('MiniWorld-Maze-v0')
    
    # Wrap with same observation size as training
    from recurrent_ppo_maze_improved import MiniWorldObsWrapper
    env = MiniWorldObsWrapper(env, obs_size=(64, 64))
    
    # Stats tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nRunning {n_episodes} episodes...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        
        # IMPORTANT: RecurrentPPO needs LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Predict action (with LSTM state tracking)
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True  # Use deterministic policy for evaluation
            )
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            episode_starts = np.zeros((1,), dtype=bool)  # Only first step is True
            
            if render:
                env.render()
        
        # Check if goal was reached (reward > 0 on termination usually means success)
        success = terminated and reward > 0
        if success:
            success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        status = "✓ SUCCESS" if success else "✗ Failed"
        print(f"Episode {episode + 1}/{n_episodes}: {status}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print()
    
    env.close()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Success Rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test trained RecurrentPPO agent in MiniWorld Maze'
    )
    
    parser.add_argument('--model', type=str, 
                        default='./models/improved_recurrent_ppo_maze/final_model',
                        help='Path to saved model (without .zip)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per episode (default: 1000)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster, stats only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        seed=args.seed,
    )

