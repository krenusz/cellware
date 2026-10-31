"""Stem-cell tissue on a lattice: cells as learning agents deciding fate and division.

Each lattice site holds at most one cell. A tissue starts from a single stem cell and grows by
division. Every cell has:

- a **fate coordinate** ``x`` moving on a pitchfork landscape ``dx = (x - x**3) dt``. The stem
  state ``x = 0`` is unstable, so stemness has to be actively maintained; crossing
  ``+/- commit_threshold`` commits the cell irreversibly to fate A or B (cf. geometric
  Waddington landscapes, Rand et al. 2021, Saez et al. 2022);
- a **heritable hidden memory** ``m`` copied to both daughters at division with small noise
  (cf. LARRY and DARLIN lineage tracing: sisters share fate bias beyond their current state);
- three **signalling channels**, each of which can be knocked out:
  contact (Notch-like counts of neighbour fates), a diffusing morphogen secreted by A cells,
  and gap-junction coupling of a bioelectric potential ``v``.

Only uncommitted stem cells act. Actions: 0 rest, 1 divide (needs an empty neighbour site and
a refractory period), 2 bias toward A, 3 bias toward B, 4 self-renew (pull back toward x = 0).

The tissue-level goal is a target size and target fate proportions. The per-step reward is the
change in a score, so an episode's total reward equals the final score minus the initial one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

STEM, FATE_A, FATE_B = 0, 1, 2
REST, DIVIDE, BIAS_A, BIAS_B, RENEW = range(5)
NUM_ACTIONS = 5
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class TissueConfig:
    height: int = 12
    width: int = 12
    horizon: int = 40
    memory_dim: int = 4
    memory_noise: float = 0.1
    inherit_memory: bool = True
    contact: bool = True
    morphogen: bool = True
    gap_junction: bool = True
    refractory: int = 2
    commit_threshold: float = 0.7
    bias_step: float = 0.3
    drift_dt: float = 0.2
    fate_noise: float = 0.05
    diffusion: float = 0.2
    decay: float = 0.1
    secretion: float = 1.0
    gap_coupling: float = 0.3
    target_size: int = 60
    target_props: tuple[float, float, float] = (0.1, 0.6, 0.3)  # stem, A, B
    # Spatial target: A cells inside ``core_radius`` of the founding site, B cells outside it.
    spatial_target: bool = False
    core_radius: float = 3.4
    # Expose global tissue size and episode time to every cell. Real cells have no such
    # readout; it was on in the first knockout study and let cells ignore local signals.
    global_readout: bool = False


def shift(t: torch.Tensor, dr: int, dc: int, fill=0) -> torch.Tensor:
    """``out[..., r, c] = t[..., r + dr, c + dc]`` over the last two dims; ``fill`` outside."""
    h, w = t.shape[-2:]
    out = torch.full_like(t, fill)
    rs, re = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    out[..., rs:re, cs:ce] = t[..., rs + dr : re + dr, cs + dc : ce + dc]
    return out


class TissueEnv:
    """``batch_size`` independent tissues simulated as tensors of shape ``(B, H, W, ...)``."""

    def __init__(self, cfg: TissueConfig, batch_size: int = 1, seed: int | None = None):
        self.cfg = cfg
        self.batch_size = batch_size
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.target_props = torch.tensor(cfg.target_props)
        rows = torch.arange(cfg.height).view(-1, 1) - cfg.height // 2
        cols = torch.arange(cfg.width).view(1, -1) - cfg.width // 2
        self.core = (rows**2 + cols**2).float().sqrt() <= cfg.core_radius  # (H, W)
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> torch.Tensor:
        c, b = self.cfg, self.batch_size
        shape = (b, c.height, c.width)
        self.occ = torch.zeros(shape, dtype=torch.bool)
        self.fate = torch.zeros(shape, dtype=torch.long)
        self.x = torch.zeros(shape)
        self.mem = torch.zeros(*shape, c.memory_dim)
        self.age = torch.zeros(shape, dtype=torch.long)
        self.pair = torch.full(shape, -1, dtype=torch.long)
        self.conc = torch.zeros(shape)
        self.v = torch.zeros(shape)
        self.occ[:, c.height // 2, c.width // 2] = True
        self.age[:, c.height // 2, c.width // 2] = c.refractory
        self.t = 0
        self._next_pair = 0
        self._score = self.score()
        return self.observe()

    @property
    def agents(self) -> torch.Tensor:
        """Sites holding an uncommitted stem cell, the cells that act."""
        return self.occ & (self.fate == STEM)

    def counts(self) -> torch.Tensor:
        """``(B, 3)`` numbers of stem, A and B cells."""
        return torch.stack([(self.occ & (self.fate == f)).sum((1, 2)) for f in range(3)], -1)

    def score(self) -> torch.Tensor:
        counts = self.counts().float()
        n = counts.sum(-1)
        props = counts / n.clamp(min=1).unsqueeze(-1)
        size_err = (n - self.cfg.target_size).abs() / self.cfg.target_size
        prop_err = (props - self.target_props).abs().sum(-1)
        score = -(size_err + prop_err)
        if self.cfg.spatial_target:
            score = score - (1.0 - self.pattern_accuracy())
        return score

    def pattern_accuracy(self) -> torch.Tensor:
        """``(B,)`` fraction of committed cells in their target region (A core, B rim)."""
        a = self.occ & (self.fate == FATE_A)
        b = self.occ & (self.fate == FATE_B)
        correct = (a & self.core).sum((1, 2)) + (b & ~self.core).sum((1, 2))
        committed = (a | b).sum((1, 2))
        return correct.float() / committed.clamp(min=1).float()

    # ----------------------------------------------------------- observation
    def _neighbour_counts(self) -> torch.Tensor:
        """``(B, H, W, 4)`` fraction of the 4 neighbours that are stem, A, B, empty."""
        feats = []
        for f in (STEM, FATE_A, FATE_B):
            mask = (self.occ & (self.fate == f)).float()
            feats.append(sum(shift(mask, dr, dc) for dr, dc in DIRECTIONS) / 4)
        empty = (~self.occ).float()
        feats.append(sum(shift(empty, dr, dc) for dr, dc in DIRECTIONS) / 4)
        return torch.stack(feats, -1)

    def observe(self) -> torch.Tensor:
        """Per-site observation ``(B, H, W, D)``; only meaningful where a cell exists."""
        c = self.cfg
        nb = self._neighbour_counts()
        can_divide = (nb[..., 3] > 0) & (self.age >= c.refractory)
        contact = nb[..., :3] * float(c.contact)
        morph = (self.conc * float(c.morphogen)).unsqueeze(-1)
        volt = (self.v * float(c.gap_junction)).unsqueeze(-1)
        local = [self.x, (self.age.float() / c.refractory).clamp(max=2.0), can_divide.float()]
        if c.global_readout:
            local += [
                torch.full_like(self.x, self.t / c.horizon),
                self.occ.sum((1, 2), keepdim=True).float().expand_as(self.x) / c.target_size,
            ]
        scalars = torch.stack(local, -1)
        return torch.cat([scalars, contact, morph, volt, self.mem], -1)

    @property
    def obs_dim(self) -> int:
        return (5 if self.cfg.global_readout else 3) + 3 + 1 + 1 + self.cfg.memory_dim

    # ------------------------------------------------------------------ step
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Apply per-site ``actions`` ``(B, H, W)``; non-agent sites are ignored.

        Returns the next observation, the per-tissue reward ``(B,)`` and whether the
        episode is over.
        """
        c = self.cfg
        acting = self.agents
        a = torch.where(acting, actions, torch.full_like(actions, REST))

        # 1. fate landscape: pitchfork drift + chosen bias + noise
        noise = torch.randn(self.x.shape, generator=self.generator) * c.fate_noise
        drift = c.drift_dt * (self.x - self.x**3)
        bias = c.bias_step * ((a == BIAS_A).float() - (a == BIAS_B).float())
        new_x = self.x + drift + bias + noise
        new_x = torch.where(a == RENEW, 0.5 * new_x, new_x)
        self.x = torch.where(acting, new_x, self.x)

        # 2. irreversible commitment
        self.fate = torch.where(acting & (self.x > c.commit_threshold), FATE_A, self.fate)
        self.fate = torch.where(acting & (self.x < -c.commit_threshold), FATE_B, self.fate)

        # 3. division into an empty neighbour site (random direction priority each step)
        dividers = self.agents & (a == DIVIDE) & (self.age >= c.refractory)
        self._divide(dividers)

        # 4. ageing, morphogen and bioelectric fields
        self.age = torch.where(self.occ, self.age + 1, self.age)
        self._update_morphogen()
        self._update_voltage()

        self.t += 1
        score = self.score()
        reward = score - self._score
        self._score = score
        return self.observe(), reward, self.t >= c.horizon

    def _divide(self, dividers: torch.Tensor) -> None:
        c = self.cfg
        placed = torch.zeros_like(dividers)
        order = torch.randperm(4, generator=self.generator).tolist()
        for k in order:
            dr, dc = DIRECTIONS[k]
            target_empty = shift(~self.occ, dr, dc, fill=False)
            place = dividers & ~placed & target_empty
            if not place.any():
                continue
            placed |= place
            ids = self._next_pair + torch.arange(place.numel()).view_as(place)
            self._next_pair += place.numel()
            # daughter attributes, moved from the parent site to the target site
            born = shift(place, -dr, -dc, fill=False)
            par_x = shift(self.x, -dr, -dc)
            par_mem = shift(self.mem.movedim(-1, 0), -dr, -dc).movedim(0, -1)
            par_ids = shift(ids, -dr, -dc, fill=-1)
            if c.inherit_memory:
                d_mem = par_mem + c.memory_noise * torch.randn(
                    par_mem.shape, generator=self.generator
                )
            else:
                d_mem = torch.randn(par_mem.shape, generator=self.generator)
            d_x = par_x + c.fate_noise * torch.randn(par_x.shape, generator=self.generator)
            self.occ |= born
            self.fate = torch.where(born, STEM, self.fate)
            self.x = torch.where(born, d_x, self.x)
            self.mem = torch.where(born.unsqueeze(-1), d_mem, self.mem)
            self.age = torch.where(born | place, 0, self.age)
            self.pair = torch.where(born, par_ids, self.pair)
            self.pair = torch.where(place, ids, self.pair)

    def _update_morphogen(self) -> None:
        c = self.cfg
        padded = F.pad(self.conc.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        lap = (
            padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
        ) - 4 * self.conc
        source = (self.occ & (self.fate == FATE_A)).float() * c.secretion
        self.conc = (self.conc + c.diffusion * lap - c.decay * self.conc + c.decay * source).clamp(
            min=0
        )

    def _update_voltage(self) -> None:
        c = self.cfg
        rest = torch.zeros_like(self.v)
        rest = torch.where(self.fate == FATE_A, 1.0, rest)
        rest = torch.where(self.fate == FATE_B, -1.0, rest)
        v = torch.where(self.occ, self.v + 0.5 * (rest - self.v), torch.zeros_like(self.v))
        if c.gap_junction:
            occ = self.occ.float()
            nb_sum = sum(shift(v * occ, dr, dc) for dr, dc in DIRECTIONS)
            nb_n = sum(shift(occ, dr, dc) for dr, dc in DIRECTIONS)
            coupled = v + c.gap_coupling * (nb_sum / nb_n.clamp(min=1) - v) * (nb_n > 0)
            v = torch.where(self.occ, coupled, v)
        self.v = v

    # --------------------------------------------------------------- metrics
    def metrics(self) -> dict[str, float]:
        counts = self.counts().float()
        n = counts.sum(-1)
        props = (counts / n.clamp(min=1).unsqueeze(-1)).mean(0)
        return {
            "score": self.score().mean().item(),
            "size": n.mean().item(),
            "prop_stem": props[0].item(),
            "prop_a": props[1].item(),
            "prop_b": props[2].item(),
            "pattern_accuracy": self.pattern_accuracy().mean().item(),
            "sister_concordance": self.sister_concordance(),
            "segregation": self.segregation(),
        }

    def sister_concordance(self) -> float:
        """Among most-recent sister pairs where both committed, the fraction with equal fate."""
        same = total = 0
        committed = self.occ & (self.fate != STEM)
        for b in range(self.batch_size):
            ids = self.pair[b][committed[b]]
            fates = self.fate[b][committed[b]]
            if ids.numel() == 0:
                continue
            uniq, inv, cnt = torch.unique(ids, return_inverse=True, return_counts=True)
            for j in torch.nonzero(cnt == 2).flatten().tolist():
                f = fates[inv == j]
                total += 1
                same += int(f[0] == f[1])
        return same / total if total else float("nan")

    def segregation(self) -> float:
        """Mean excess of same-fate neighbours over the tissue-wide fate share (0 = mixed)."""
        vals = []
        for f in (FATE_A, FATE_B):
            mask = (self.occ & (self.fate == f)).float()
            occ = self.occ.float()
            same = sum(shift(mask, dr, dc) for dr, dc in DIRECTIONS)
            nb = sum(shift(occ, dr, dc) for dr, dc in DIRECTIONS)
            local = (same / nb.clamp(min=1))[mask.bool() & (nb > 0)]
            share = mask.sum() / occ.sum().clamp(min=1)
            if local.numel():
                vals.append((local.mean() - share).item())
        return sum(vals) / len(vals) if vals else float("nan")
