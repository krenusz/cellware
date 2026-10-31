"""Typed experiment configuration.

Replaces the 50-flag argparse block of the legacy scripts. Configs are plain dataclasses,
loadable from YAML, so every run is reproducible from a single file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    size: int = 10
    max_episode_steps: int = 128
    level: int = 1


@dataclass
class PPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    k_epochs: int = 20
    entropy_coef: float = 1e-3
    target_kl: float = 1e-3
    use_adv_norm: bool = True
    use_grad_clip: bool = True
    use_lr_decay: bool = True
    hidden_width: int = 32


@dataclass
class EncoderConfig:
    lr: float = 3e-4
    k_epochs: int = 20
    embed_dim: int = 8
    predictor_embed_dim: int = 64
    num_heads: int = 8
    depth: int = 5
    dropout: float = 0.1


@dataclass
class VertexConfig:
    """Parameters of the 2D vertex model energy (Farhadifar et al. 2007 form)."""

    area_stiffness: float = 1.0  # K_A
    target_area: float = 1.0  # A_0
    line_tension: float = -0.12  # Lambda, in units of K_A * A_0**1.5
    contractility: float = 0.04  # Gamma, in units of K_A * A_0


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 0
    num_agents: int = 4
    num_epochs: int = 1000
    collective: bool = False
    device: str = "cpu"
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return _from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _from_dict(cls: type, data: dict[str, Any]):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
