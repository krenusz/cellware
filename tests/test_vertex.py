import math

import torch

from cellware.config import VertexConfig
from cellware.physics import cell_areas, cell_perimeters, energy, hexagonal_tissue, relax


def test_regular_hexagon_geometry():
    tissue = hexagonal_tissue(1, 1, cell_area=2.0)
    assert torch.allclose(
        cell_areas(tissue.vertices, tissue.cells), torch.tensor([2.0], dtype=torch.float64)
    )
    side = math.sqrt(2 * 2.0 / (3 * math.sqrt(3)))
    expected = torch.tensor([6 * side], dtype=torch.float64)
    assert torch.allclose(cell_perimeters(tissue.vertices, tissue.cells), expected)


def test_patch_shares_vertices_and_edges():
    tissue = hexagonal_tissue(3, 3)
    assert tissue.vertices.shape[0] < 9 * 6
    # every edge is counted once even when two cells share it
    assert tissue.edges().shape[0] < 9 * 6
    assert (cell_areas(tissue.vertices, tissue.cells) > 0).all()


def test_gradient_matches_finite_difference():
    params = VertexConfig()
    tissue = hexagonal_tissue(2, 2)
    v = tissue.vertices + 0.05 * torch.randn_like(tissue.vertices, dtype=torch.float64)
    v.requires_grad_(True)
    energy(tissue, params, v).backward()
    eps = 1e-6
    for idx in [(0, 0), (3, 1), (7, 0)]:
        vp, vm = v.detach().clone(), v.detach().clone()
        vp[idx] += eps
        vm[idx] -= eps
        fd = (energy(tissue, params, vp) - energy(tissue, params, vm)) / (2 * eps)
        assert torch.isclose(v.grad[idx], fd, rtol=1e-4, atol=1e-8)


def test_single_cell_relaxes_to_target_area_without_tension():
    params = VertexConfig(target_area=1.5, line_tension=0.0, contractility=0.0)
    tissue = hexagonal_tissue(1, 1, cell_area=1.0)
    relaxed, trace = relax(tissue, params)
    assert trace[-1] < trace[0]
    assert abs(cell_areas(relaxed.vertices, relaxed.cells).item() - 1.5) < 1e-6


def test_relaxation_lowers_energy_of_perturbed_patch():
    torch.manual_seed(0)
    params = VertexConfig()
    tissue = hexagonal_tissue(4, 4)
    tissue.vertices = tissue.vertices + 0.03 * torch.randn_like(tissue.vertices)
    relaxed, trace = relax(tissue, params)
    assert trace[-1] < trace[0]
    assert (cell_areas(relaxed.vertices, relaxed.cells) > 0).all()


def test_energy_is_differentiable_in_parameters():
    tissue = hexagonal_tissue(2, 2)
    tension = torch.tensor(-0.12, dtype=torch.float64, requires_grad=True)
    params = VertexConfig(line_tension=tension)  # tensors work as parameters too
    energy(tissue, params).backward()
    # dE/dLambda is the total junction length
    assert torch.isclose(
        tension.grad,
        tissue.edges().shape[0]
        * torch.linalg.vector_norm(
            tissue.vertices[tissue.edges()[0, 1]] - tissue.vertices[tissue.edges()[0, 0]]
        ),
        rtol=1e-6,
    )
