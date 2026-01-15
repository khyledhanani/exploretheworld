"""Environment utilities for MiniWorld maze environments."""

import gymnasium as gym
import miniworld.envs  # Register MiniWorld environments
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from typing import Callable, Optional
import numpy as np


class TimeStepPenaltyWrapper(gym.Wrapper):
    """Add small negative reward per timestep to discourage dithering."""

    def __init__(self, env: gym.Env, penalty: float = -0.001):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.penalty
        return obs, reward, terminated, truncated, info


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class RandomSeedWrapper(gym.Wrapper):
    """
    Randomize the environment seed on each reset.
    This forces the agent to learn a general policy instead of
    memorizing solutions for specific spawn positions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        # Generate a new random seed for each episode
        kwargs['seed'] = int(self._rng.integers(0, 2**31))
        return self.env.reset(**kwargs)


def make_env(
    env_id: str = "MiniWorld-OneRoom-v0",
    seed: Optional[int] = 0,
    time_penalty: float = 0.0,
    normalize_obs: bool = False,
    render_mode: Optional[str] = None,
    random_seed_per_episode: bool = False,
) -> Callable[[], gym.Env]:
    """
    Create a callable that returns a wrapped MiniWorld environment.

    Args:
        env_id: Gymnasium environment ID
        seed: Random seed (None for random initial seed)
        time_penalty: Per-step penalty (0 to disable)
        normalize_obs: Whether to normalize observations to [0, 1]
        render_mode: Render mode ('human', 'rgb_array', or None)
        random_seed_per_episode: If True, randomize seed on each episode reset
                                 (improves generalization during training)

    Returns:
        Callable that creates the environment
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        
        # Initial reset with provided seed (or random if None)
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()

        if time_penalty != 0.0:
            env = TimeStepPenaltyWrapper(env, penalty=time_penalty)

        if normalize_obs:
            env = NormalizeObservationWrapper(env)

        # Add random seed wrapper for training diversity
        if random_seed_per_episode:
            env = RandomSeedWrapper(env)

        env = Monitor(env)
        return env

    return _init


def make_vec_env(
    env_id: str = "MiniWorld-OneRoom-v0",
    n_envs: int = 8,
    seed: Optional[int] = 0,
    time_penalty: float = -0.001,
    normalize_obs: bool = False,
    frame_stack: int = 4,
    use_subproc: bool = True,
    render_mode: Optional[str] = None,
    random_seed_per_episode: bool = False,
) -> VecFrameStack:
    """
    Create vectorized environment with frame stacking.

    Args:
        env_id: Gymnasium environment ID
        n_envs: Number of parallel environments
        seed: Base random seed (None for random)
        time_penalty: Per-step penalty
        normalize_obs: Whether to normalize observations
        frame_stack: Number of frames to stack
        use_subproc: Use subprocess vectorization (faster but more memory)
        render_mode: Render mode ('human', 'rgb_array', or None)
        random_seed_per_episode: If True, randomize seed on each episode reset
                                 (RECOMMENDED for training to improve generalization)

    Returns:
        Vectorized environment with frame stacking
    """
    import platform
    import sys
    
    # Disable SubprocVecEnv on macOS due to OpenGL/multiprocessing issues
    # MiniWorld uses OpenGL which doesn't work well with fork() on macOS
    if platform.system() == "Darwin":
        use_subproc = False
        if n_envs > 1:
            print(f"Note: Using DummyVecEnv instead of SubprocVecEnv on macOS (n_envs={n_envs})")
    
    env_fns = [
        make_env(
            env_id=env_id,
            seed=(seed + i) if seed is not None else None,
            time_penalty=time_penalty,
            normalize_obs=normalize_obs,
            render_mode=render_mode,
            random_seed_per_episode=random_seed_per_episode,
        )
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Transpose to channels-first for PyTorch CNN
    vec_env = VecTransposeImage(vec_env)

    # Stack frames
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    return vec_env
