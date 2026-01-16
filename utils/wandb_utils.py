"""Utilities for optional Weights & Biases logging."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {"config": str(config)}


def _is_wandb_enabled(config: Any) -> bool:
    return bool(getattr(config, "wandb_enabled", False))


def init_wandb(
    config: Any,
    agent_name: str,
    log_dir: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """Initialize a wandb run if enabled. Returns the run or None."""
    if not _is_wandb_enabled(config):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping wandb logging.")
        return None

    run_config = _config_to_dict(config)
    if extra_config:
        run_config.update(extra_config)

    project = getattr(config, "wandb_project", "worldmodels")
    entity = getattr(config, "wandb_entity", None)
    group = getattr(config, "wandb_group", None)
    name = getattr(config, "wandb_run_name", None)
    tags = getattr(config, "wandb_tags", None)
    mode = getattr(config, "wandb_mode", "online")

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        tags=tags,
        config=run_config,
        dir=log_dir,
        mode=mode,
        job_type=agent_name,
    )


def get_wandb_callback(config: Any, log_dir: Optional[str] = None):
    """Return SB3 WandbCallback if available and enabled."""
    if not _is_wandb_enabled(config):
        return None

    try:
        from wandb.integration.sb3 import WandbCallback
    except Exception:
        return None

    return WandbCallback(
        verbose=1,
        model_save_path=log_dir,
        gradient_save_freq=0,  # Don't save gradients (expensive)
    )


def log_wandb(run, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    wandb.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    wandb.finish()
