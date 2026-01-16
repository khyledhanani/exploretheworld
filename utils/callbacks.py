"""Training callbacks for logging and evaluation."""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional
import gymnasium as gym


class MetricsCallback(BaseCallback):
    """
    Callback to log custom metrics during training.
    Tracks success rate, episode length, and reward statistics.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
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
        # Check for episode completion in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                # MiniWorld sets 'success' in info when goal is reached
                success = info.get("success", info["episode"]["r"] > 0)
                self.episode_successes.append(float(success))

        # Log every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            recent_successes = self.episode_successes[-100:]

            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)

            self.logger.record("rollout/mean_reward", mean_reward)
            self.logger.record("rollout/mean_length", mean_length)
            self.logger.record("rollout/success_rate", success_rate)

            # Also log directly to wandb
            wandb = self._get_wandb()
            if wandb is not None:
                wandb.log({
                    "rollout/mean_reward": mean_reward,
                    "rollout/mean_length": mean_length,
                    "rollout/success_rate": success_rate,
                    "rollout/num_episodes": len(self.episode_rewards),
                }, step=self.num_timesteps)

        return True


class VideoRecorderCallback(BaseCallback):
    """
    Callback to record evaluation videos periodically.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        video_folder: str,
        video_freq: int = 10000,
        video_length: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.video_freq = video_freq
        self.video_length = video_length

        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        try:
            import imageio
        except ImportError:
            if self.verbose > 0:
                print("imageio not installed, skipping video recording")
            return

        frames = []
        obs = self.eval_env.reset()

        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)

            # Get render frame
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)

            if done[0]:
                break

        if frames:
            video_path = os.path.join(
                self.video_folder, f"video_step_{self.n_calls}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=30)
            if self.verbose > 0:
                print(f"Saved video to {video_path}")


class SuccessRateCallback(BaseCallback):
    """
    Callback that stops training when target success rate is achieved.
    """

    def __init__(
        self,
        target_success_rate: float = 0.95,
        window_size: int = 100,
        min_episodes: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.min_episodes = min_episodes
        self.successes = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                success = info.get("success", info["episode"]["r"] > 0)
                self.successes.append(float(success))

        if len(self.successes) >= self.min_episodes:
            recent_success_rate = np.mean(self.successes[-self.window_size:])
            if recent_success_rate >= self.target_success_rate:
                if self.verbose > 0:
                    print(f"Target success rate {self.target_success_rate} achieved!")
                return False  # Stop training

        return True
