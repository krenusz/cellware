import math

import torch

from cellware.tissue import (
    BIAS_A,
    BIAS_B,
    DIVIDE,
    FATE_A,
    FATE_B,
    REST,
    STEM,
    TissueConfig,
    TissueEnv,
    shift,
)


def full(env, action):
    return torch.full(env.occ.shape, action, dtype=torch.long)


def test_shift_semantics():
    t = torch.arange(9).view(1, 3, 3)
    assert shift(t, 0, 1)[0, 0].tolist() == [1, 2, 0]  # out[r, c] = t[r, c + 1]
    assert shift(t, 1, 0)[0, :, 0].tolist() == [3, 6, 0]


def test_starts_from_one_stem_cell():
    env = TissueEnv(TissueConfig(), batch_size=3)
    assert env.counts().tolist() == [[1, 0, 0]] * 3
    assert env.observe().shape == (3, 12, 12, env.obs_dim)


def test_division_fills_tissue_and_daughters_share_pair_ids():
    cfg = TissueConfig(fate_noise=0.0, drift_dt=0.0)  # keep cells uncommitted
    env = TissueEnv(cfg, seed=0)
    sizes = []
    for _ in range(12):
        env.step(full(env, DIVIDE))
        sizes.append(env.occ.sum().item())
    assert sizes == sorted(sizes) and sizes[-1] > 20
    assert (env.fate[env.occ] == STEM).all()
    # most-recent sisters share an id: every id appears at most twice
    _, counts = torch.unique(env.pair[env.occ & (env.pair >= 0)], return_counts=True)
    assert counts.max() <= 2 and (counts == 2).any()


def test_refractory_period_blocks_immediate_redivision():
    env = TissueEnv(TissueConfig(fate_noise=0.0, drift_dt=0.0, refractory=3), seed=0)
    env.step(full(env, DIVIDE))
    assert env.occ.sum() == 2
    env.step(full(env, DIVIDE))  # both cells were just reset to age 1 < 3
    assert env.occ.sum() == 2


def test_memory_is_inherited_or_resampled():
    on = TissueEnv(TissueConfig(memory_noise=0.0, fate_noise=0.0, drift_dt=0.0), seed=0)
    on.mem[on.occ] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    on.step(full(on, DIVIDE))
    assert torch.allclose(on.mem[on.occ], torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4))

    off = TissueEnv(TissueConfig(inherit_memory=False, fate_noise=0.0, drift_dt=0.0), seed=0)
    off.mem[off.occ] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    off.step(full(off, DIVIDE))
    daughters = off.mem[off.occ]
    assert not torch.allclose(daughters[0], daughters[1])


def test_bias_commits_irreversibly():
    env = TissueEnv(TissueConfig(fate_noise=0.0), seed=0)
    for _ in range(5):
        env.step(full(env, BIAS_B))
    assert env.counts().tolist() == [[0, 0, 1]]
    x_before = env.x.clone()
    env.step(full(env, BIAS_A))  # committed cells no longer act
    assert env.counts().tolist() == [[0, 0, 1]]
    assert torch.equal(env.x, x_before)


def test_stem_state_is_unstable_without_renewal():
    env = TissueEnv(TissueConfig(fate_noise=0.0), seed=0)
    env.x[env.occ] = 0.05
    for _ in range(40):
        env.step(full(env, REST))
    assert env.counts()[0, FATE_A] == 1


def test_morphogen_comes_from_a_cells_and_decays_away():
    env = TissueEnv(TissueConfig(fate_noise=0.0), seed=0)
    for _ in range(4):
        env.step(full(env, BIAS_A))
    assert env.counts()[0, FATE_A] == 1
    for _ in range(30):
        env.step(full(env, REST))
    centre = env.conc[0, 6, 6].item()
    assert centre > env.conc[0, 6, 9].item() > env.conc[0, 0, 0].item() >= 0
    off = TissueEnv(TissueConfig(fate_noise=0.0, morphogen=False), seed=0)
    for _ in range(4):
        off.step(full(off, BIAS_A))
    assert (off.observe()[..., 6] == 0).all()  # knocked-out channel is invisible


def test_score_rewards_target_composition():
    cfg = TissueConfig(target_size=4, target_props=(0.25, 0.5, 0.25))
    env = TissueEnv(cfg)
    env.occ[0, 0, :4] = True
    env.fate[0, 0, :4] = torch.tensor([STEM, FATE_A, FATE_A, FATE_B])
    env.occ[0, 6, 6] = False
    assert math.isclose(env.score().item(), 0.0, abs_tol=1e-6)


def test_totipotency_window_blocks_early_commitment():
    env = TissueEnv(TissueConfig(fate_noise=0.0, min_commit_size=3), seed=0)
    for _ in range(6):
        env.step(full(env, BIAS_A))
    assert env.counts().tolist() == [[1, 0, 0]]  # pushed hard, but still uncommitted
    assert env.x[env.occ].item() < env.cfg.commit_threshold
    env.step(full(env, DIVIDE))
    env.step(full(env, DIVIDE))  # refractory 2: second division round
    env.step(full(env, DIVIDE))
    assert env.occ.sum() >= 3
    env.step(full(env, BIAS_A))
    assert env.counts()[0, FATE_A] >= 1  # window open: cells near the threshold commit


def test_no_global_readout_by_default():
    local = TissueEnv(TissueConfig())
    glob = TissueEnv(TissueConfig(global_readout=True))
    assert local.obs_dim == glob.obs_dim - 2
    assert local.observe().shape[-1] == local.obs_dim
    assert glob.observe().shape[-1] == glob.obs_dim


def test_pattern_accuracy_rewards_a_core_b_rim():
    cfg = TissueConfig(spatial_target=True, core_radius=1.0)
    env = TissueEnv(cfg)
    env.fate[0, 6, 6] = FATE_A  # founding site is in the core
    env.occ[0, 6, 9] = True
    env.fate[0, 6, 9] = FATE_B  # 3 sites away: rim
    assert env.pattern_accuracy().item() == 1.0
    env.fate[0, 6, 6] = FATE_B
    assert env.pattern_accuracy().item() == 0.5
    plain = TissueEnv(TissueConfig(core_radius=1.0))
    plain.occ, plain.fate = env.occ.clone(), env.fate.clone()
    assert env.score().item() == plain.score().item() - 0.5


def test_episode_reward_telescopes_to_score_change():
    env = TissueEnv(TissueConfig(), batch_size=4, seed=1)
    start = env.score()
    total = torch.zeros(4)
    done = False
    while not done:
        actions = torch.randint(0, 5, env.occ.shape, generator=env.generator)
        _, r, done = env.step(actions)
        total += r
    assert torch.allclose(total, env.score() - start, atol=1e-5)
    m = env.metrics()
    assert 0 <= m["prop_a"] <= 1 and m["size"] >= 1


def test_cue_target_sets_founder_memory_and_swaps_targets():
    env = TissueEnv(TissueConfig(cue_target=True), batch_size=64, seed=0)
    cue = env.mem[:, 6, 6, 0]
    assert set(cue.tolist()) == {-1.0, 1.0}
    assert torch.allclose(env.target_props[cue > 0], torch.tensor([0.1, 0.6, 0.3]))
    assert torch.allclose(env.target_props[cue < 0], torch.tensor([0.1, 0.3, 0.6]))
