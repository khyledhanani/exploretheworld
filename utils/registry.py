"""Agent registry for unified training and evaluation dispatch."""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    name: str
    display_name: str
    module: str
    config_class: str
    train_func: str
    evaluate_func: str
    # CLI argument definitions: (name, type, default, help)
    extra_args: List[tuple] = None

    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = []


# Registry of all available agents
AGENT_REGISTRY: Dict[str, AgentInfo] = {
    "ppo": AgentInfo(
        name="ppo",
        display_name="PPO + CNN",
        module="agents.ppo_cnn",
        config_class="PPOConfig",
        train_func="train_ppo",
        evaluate_func="evaluate_ppo",
        extra_args=[
            ("--n-envs", int, 8, "Number of parallel environments"),
        ],
    ),
    "recurrent": AgentInfo(
        name="recurrent",
        display_name="RecurrentPPO",
        module="agents.recurrent_ppo",
        config_class="RecurrentPPOConfig",
        train_func="train_recurrent_ppo",
        evaluate_func="evaluate_recurrent_ppo",
        extra_args=[
            ("--n-envs", int, 8, "Number of parallel environments"),
            ("--lstm-hidden", int, 256, "LSTM hidden size"),
        ],
    ),
    "qrdqn": AgentInfo(
        name="qrdqn",
        display_name="QR-DQN",
        module="agents.qrdqn",
        config_class="QRDQNConfig",
        train_func="train_qrdqn",
        evaluate_func="evaluate_qrdqn",
        extra_args=[
            ("--buffer-size", int, 500_000, "Replay buffer size"),
            ("--n-quantiles", int, 200, "Number of quantiles"),
        ],
    ),
    "rnd": AgentInfo(
        name="rnd",
        display_name="PPO + RND",
        module="agents.ppo_rnd",
        config_class="PPORNDConfig",
        train_func="train_ppo_rnd",
        evaluate_func="evaluate_ppo_rnd",
        extra_args=[
            ("--n-envs", int, 8, "Number of parallel environments"),
            ("--intrinsic-coef", float, 0.1, "Intrinsic reward coefficient"),
        ],
    ),
    "dreamer": AgentInfo(
        name="dreamer",
        display_name="Dreamer",
        module="agents.dreamer",
        config_class="DreamerConfig",
        train_func="train_dreamer",
        evaluate_func="evaluate_dreamer",
        extra_args=[],
    ),
    "recurrent_rnd": AgentInfo(
        name="recurrent_rnd",
        display_name="RecurrentPPO + RND",
        module="agents.recurrent_ppo_rnd",
        config_class="RecurrentPPORNDConfig",
        train_func="train_recurrent_ppo_rnd",
        evaluate_func="evaluate_recurrent_ppo_rnd",
        extra_args=[
            ("--n-envs", int, 8, "Number of parallel environments"),
            ("--lstm-hidden", int, 256, "LSTM hidden size"),
            ("--intrinsic-coef", float, 1.0, "Initial intrinsic reward coefficient"),
        ],
    ),
}


def get_agent_info(agent_name: str) -> Optional[AgentInfo]:
    """Get agent info by name."""
    return AGENT_REGISTRY.get(agent_name)


def get_all_agents() -> List[str]:
    """Get list of all registered agent names."""
    return list(AGENT_REGISTRY.keys())


def infer_agent_type(model_path: str) -> Optional[str]:
    """Infer agent type from model path."""
    path_lower = model_path.lower()

    # Order matters: check more specific patterns first
    patterns = [
        ("recurrent_ppo_rnd", "recurrent_rnd"),
        ("recurrent_rnd", "recurrent_rnd"),
        ("ppo_rnd", "rnd"),
        ("rnd", "rnd"),
        ("recurrent", "recurrent"),
        ("qrdqn", "qrdqn"),
        ("dreamer", "dreamer"),
        ("ppo", "ppo"),
    ]

    for pattern, agent_type in patterns:
        if pattern in path_lower:
            return agent_type

    return None


def load_agent_module(agent_name: str):
    """Dynamically load an agent module and return its components."""
    import importlib

    info = get_agent_info(agent_name)
    if info is None:
        raise ValueError(f"Unknown agent: {agent_name}")

    module = importlib.import_module(info.module)
    config_class = getattr(module, info.config_class)
    train_func = getattr(module, info.train_func)
    evaluate_func = getattr(module, info.evaluate_func)

    return {
        "module": module,
        "config_class": config_class,
        "train_func": train_func,
        "evaluate_func": evaluate_func,
        "info": info,
    }
