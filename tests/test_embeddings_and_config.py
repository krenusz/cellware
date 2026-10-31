import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cellware.config import ExperimentConfig
from cellware.embeddings import state_action_reward_embedding

LEGACY_EMB = (
    Path(__file__).parents[1] / "legacy" / "jepa_evolution_network" / "state_action_reward_emb.py"
)


def _load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_emb", LEGACY_EMB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_matches_legacy_numpy():
    legacy = _load_legacy()
    rng = np.random.default_rng(0)
    states = rng.integers(0, 10, size=(3, 6, 4)).astype(float)
    cur = rng.integers(0, 10, size=(3, 6, 4)).astype(float)
    actions = rng.integers(0, 5, size=(3, 6)).astype(float)
    rewards = rng.integers(0, 90, size=(3, 6)).astype(float)
    expected = legacy.state_action_reward_embedding(states, actions, rewards, cur)
    got = state_action_reward_embedding(
        torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(cur)
    )
    assert np.allclose(got.numpy(), expected)


def test_config_roundtrip_and_typo_rejection(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: test\ncollective: true\nppo:\n  k_epochs: 5\n", encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.collective is True and cfg.ppo.k_epochs == 5
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"colective": True})
