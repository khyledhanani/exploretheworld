"""Shared feature extractors for all agents."""

import math
from typing import Literal

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Registry of available feature extractors
EXTRACTOR_REGISTRY = {}


def register_extractor(name: str):
    """Decorator to register a feature extractor."""
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_extractor(name: str):
    """Get a feature extractor class by name."""
    if name not in EXTRACTOR_REGISTRY:
        available = list(EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown extractor: {name}. Available: {available}")
    return EXTRACTOR_REGISTRY[name]


@register_extractor("cnn")
class NatureCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor based on Nature DQN architecture.
    Designed for MiniWorld 60x80 RGB images (with optional frame stacking).

    This is the shared feature extractor used by PPO, RecurrentPPO, QR-DQN, and PPO+RND.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


@register_extractor("vit")
class ViTExtractor(BaseFeaturesExtractor):
    """
    Vision Transformer (ViT) feature extractor for visual RL.

    Splits image into patches, projects to embeddings, and processes with transformer.
    Designed for MiniWorld 60x80 RGB images (with optional frame stacking).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)

        n_channels, height, width = observation_space.shape

        # Compute number of patches
        assert height % patch_size == 0 or width % patch_size == 0, \
            f"Image dims ({height}x{width}) should be divisible by patch_size ({patch_size})"

        # Pad if necessary to make divisible
        self.pad_h = (patch_size - height % patch_size) % patch_size
        self.pad_w = (patch_size - width % patch_size) % patch_size
        padded_h = height + self.pad_h
        padded_w = width + self.pad_w

        self.num_patches = (padded_h // patch_size) * (padded_w // patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding (conv with kernel=stride=patch_size)
        self.patch_embed = nn.Conv2d(
            n_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.ReLU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize patch embedding
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.size(0), -1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Pad if necessary
        if self.pad_h > 0 or self.pad_w > 0:
            observations = nn.functional.pad(
                observations, (0, self.pad_w, 0, self.pad_h)
            )

        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.patch_embed(observations)
        x = x.flatten(2).transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Take [CLS] token output
        x = self.norm(x[:, 0])

        # Project to features_dim
        return self.head(x)


@register_extractor("vit_small")
class ViTSmallExtractor(ViTExtractor):
    """Smaller ViT variant for faster training."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            patch_size=10,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            mlp_ratio=2.0,
        )
