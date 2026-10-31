"""2D vertex model of an epithelial tissue, written from the published energy.

The tissue is a polygonal tiling: cells are faces, cell-cell junctions are edges and
tricellular contacts are vertices. Mechanical equilibrium is a minimum of

    E = sum_cells K_A / 2 * (A_c - A_0)**2      (area elasticity)
      + sum_edges Lambda * l_ij                 (line tension)
      + sum_cells Gamma / 2 * P_c**2            (actomyosin contractility)

following Farhadifar et al., Curr. Biol. 17:2095 (2007), and the review by
Fletcher et al., Biophys. J. 106:2291 (2014). Implemented independently in PyTorch;
gradients come from autograd, so the energy is differentiable with respect to both
vertex positions and mechanical parameters.

Topological transitions (T1 swaps, divisions, extrusion) are not handled yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cellware.config import VertexConfig


@dataclass
class Tissue:
    """Vertex positions plus cell -> vertex connectivity.

    ``cells`` is a ``(C, K)`` long tensor of vertex indices in counter-clockwise order,
    padded with -1 for cells with fewer than ``K`` vertices.
    """

    vertices: torch.Tensor  # (V, 2) float
    cells: torch.Tensor  # (C, K) long, -1 padded

    @property
    def num_cells(self) -> int:
        return self.cells.shape[0]

    def edges(self) -> torch.Tensor:
        """Unique undirected edges ``(E, 2)``, each shared junction counted once."""
        nxt = _next_vertex(self.cells)
        mask = self.cells >= 0
        pairs = torch.stack([self.cells[mask], nxt[mask]], dim=-1)
        pairs = torch.sort(pairs, dim=-1).values
        return torch.unique(pairs, dim=0)


def _next_vertex(cells: torch.Tensor) -> torch.Tensor:
    """For each slot, the index of the following vertex of the same cell (cyclic)."""
    mask = cells >= 0
    counts = mask.sum(dim=1, keepdim=True)
    slot = torch.arange(cells.shape[1], device=cells.device).expand_as(cells)
    nxt_slot = (slot + 1) % counts.clamp(min=1)
    return torch.gather(cells, 1, nxt_slot)


def cell_areas(vertices: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Signed shoelace areas ``(C,)``; positive for counter-clockwise cells."""
    mask = cells >= 0
    p = vertices[cells.clamp(min=0)]
    q = vertices[_next_vertex(cells).clamp(min=0)]
    cross = p[..., 0] * q[..., 1] - q[..., 0] * p[..., 1]
    return 0.5 * (cross * mask).sum(dim=1)


def cell_perimeters(vertices: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    mask = cells >= 0
    p = vertices[cells.clamp(min=0)]
    q = vertices[_next_vertex(cells).clamp(min=0)]
    return (torch.linalg.vector_norm(q - p, dim=-1) * mask).sum(dim=1)


def energy(
    tissue: Tissue, params: VertexConfig, vertices: torch.Tensor | None = None
) -> torch.Tensor:
    v = tissue.vertices if vertices is None else vertices
    areas = cell_areas(v, tissue.cells)
    perims = cell_perimeters(v, tissue.cells)
    edges = tissue.edges()
    lengths = torch.linalg.vector_norm(v[edges[:, 1]] - v[edges[:, 0]], dim=-1)
    return (
        0.5 * params.area_stiffness * ((areas - params.target_area) ** 2).sum()
        + params.line_tension * lengths.sum()
        + 0.5 * params.contractility * (perims**2).sum()
    )


def relax(
    tissue: Tissue,
    params: VertexConfig,
    max_iter: int = 500,
    grad_tol: float = 1e-9,
) -> tuple[Tissue, list[float]]:
    """Find the nearest mechanical equilibrium (energy minimum) with L-BFGS.

    Stops when the largest force on any vertex drops below ``grad_tol``.
    Returns the relaxed tissue and the energy after each iteration.
    """
    v = tissue.vertices.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [v],
        max_iter=max_iter,
        tolerance_grad=grad_tol,
        tolerance_change=0.0,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    trace: list[float] = []

    def closure() -> torch.Tensor:
        opt.zero_grad()
        e = energy(tissue, params, v)
        e.backward()
        trace.append(e.item())
        return e

    opt.step(closure)
    return Tissue(v.detach(), tissue.cells), trace


def hexagonal_tissue(nx: int, ny: int, cell_area: float = 1.0) -> Tissue:
    """A patch of ``nx * ny`` regular hexagons (pointy-top) with shared vertices."""
    radius = math.sqrt(2.0 * cell_area / (3.0 * math.sqrt(3.0)))
    keys: dict[tuple[int, int], int] = {}
    vertices: list[tuple[float, float]] = []
    cells: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            cx = math.sqrt(3.0) * radius * (i + 0.5 * (j % 2))
            cy = 1.5 * radius * j
            cell = []
            for k in range(6):
                angle = math.radians(30 + 60 * k)
                x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
                key = (round(x * 1e6), round(y * 1e6))
                if key not in keys:
                    keys[key] = len(vertices)
                    vertices.append((x, y))
                cell.append(keys[key])
            cells.append(cell)
    return Tissue(
        torch.tensor(vertices, dtype=torch.float64),
        torch.tensor(cells, dtype=torch.long),
    )
