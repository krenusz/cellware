# Cellware

**Multi-agent reinforcement learning for simulating stem-cell fate decisions and tissue patterning.**

Cellware models cells as learning agents. Each cell senses its local environment, decides what
to do (move, divide, commit to a fate), and passes what it has learned to its daughters. The aim
is to ask which local rules and which kinds of cell-cell signalling let identical cells form a
correctly proportioned, patterned tissue, and recover when disturbed. Simulated outcomes will be
checked against public single-cell and spatial atlases.

> **Status: v0.2 in development (branch `modernize`).** The 2023-24 research code is kept
> unchanged apart from bug fixes in [`legacy/`](legacy/). The new `cellware` package is being
> built step by step; see the roadmap below.

## Background

Cellware began in 2023 as a multi-agent PPO simulation of primitive cells in a shared "petri
dish", with a centralised JEPA encoder, reproduction by weight inheritance and an optional
shared-memory ("collective") mode. Interim results comparing individual and collective learning
were presented to the computational group of the Levin Lab (Tufts University) in November 2023.

The next version reframes the project around stem-cell differentiation:

| Simulator piece | Biological meaning |
|---|---|
| Agent | A single cell |
| Internal state | A low-dimensional fate landscape (Waddington-style) tilted by signals |
| Inherited hidden memory | Heritable fate bias, copied at division (cf. LARRY lineage tracing, DARLIN) |
| Fate-selection network | Commitment to a lineage or self-renewal |
| Signalling channels | Notch-style contact, diffusing morphogens, gap-junction (bioelectric) coupling |
| Vertex-model body | Cell shape and mechanics in an epithelium |

## What is in the package today

- `cellware.envs.PetriDish`: batched, Ray-free PyTorch port of the original environment, tested
  to reproduce the legacy dynamics exactly.
- `cellware.physics`: a 2D vertex model of epithelial mechanics (area elasticity, line tension,
  contractility; Farhadifar et al. 2007), differentiable in vertex positions and parameters, with
  L-BFGS relaxation to mechanical equilibrium.
- `cellware.embeddings`: the state/action/reward sin-cos embeddings, ported to PyTorch.
- `cellware.config`: typed, YAML-loadable experiment configs (unknown keys are rejected).

## Install

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/krenusz/cellware.git
cd cellware
uv sync --extra dev
uv run pytest
```

The project pins CPU PyTorch wheels for local development; train on a GPU machine by installing
the matching CUDA build of PyTorch there.

### Quick example

```python
import torch
from cellware.config import VertexConfig
from cellware.physics import hexagonal_tissue, relax, cell_areas

tissue = hexagonal_tissue(6, 6)
tissue.vertices += 0.03 * torch.randn_like(tissue.vertices)
relaxed, energies = relax(tissue, VertexConfig())
print(energies[0], "->", energies[-1])
print(cell_areas(relaxed.vertices, relaxed.cells))
```

## Roadmap

1. **Foundation** (this branch): package layout, configs, tests, CI, Ray-free environment,
   vertex mechanics.
2. **Cell model**: fate landscape state, inherited memory at division, three signalling channels,
   shared-parameter PPO over many cells.
3. **First validation**: fate proportions (haematopoiesis datasets) and clonal fate bias (LARRY).
4. **Mechanics coupling**: cells deciding fate while the vertex model handles shape and forces.
5. **Preprint and open release.**

## Running the legacy code

```bash
uv sync --extra legacy
cd legacy/jepa_evolution_network
uv run python PPO_universal_main_selfreplicate.py --agent_number 2 --use_collective true
```

Boolean flags now take `true`/`false` (previously any value, even `False`, parsed as true).

## License

MIT, see [LICENSE](LICENSE).

## Contact

Bence Krénusz, bence.krenusz@gmail.com
