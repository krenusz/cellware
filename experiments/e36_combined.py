"""E36: E34 (full arm separate) + full-arm co-presence in the loss + 10 simulations per evaluation,
optionally E35's signed instruction, warm-started from the best earlier fit.
See the E36 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e36_combined.py seed signed|unsigned
"""

from __future__ import annotations

import json
import os
import sys

import e29_ceiling as e29
import e30_refit as e30
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import numpy as np
from clone_state_model import OUT

CO = ("full-G", "full-M")


def targets(cnt):
    t = e30.targets(cnt)
    co = e32.copresence(cnt)
    t.update({f"co_{p}": co[p] for p in CO})
    return t


def extra_se(obs):
    return {f"co_{p}": float(np.sqrt(max(obs[f"co_{p}"], 4))) for p in CO}


def logit(v):
    v = float(np.clip(v, 1e-4, 1 - 1e-4))
    return float(np.log(v / (1 - v)))


def warm_x(src, signed):
    """Map a fitted E33/E35 parameter set to the 21-dim E36 space (rho_a 0.99, nu_full = nu)."""
    p = json.load(open(os.path.join(OUT, f"{src}.json")))["params"]
    base = list(e29.warm_start(0.0, src)[:17])  # 16 E26b params + nu, as in E33
    if signed:  # tau_G, tau_M slots hold the sigmoid offsets from tau_full
        t0 = p["tau"]["full"]
        base[3] = logit((p["tau"]["G"] - t0) / 8)
        base[4] = logit((t0 - p["tau"]["M"]) / 8)
    x = [*base, logit(p["w"]), logit(p["kappa"] / 2), base[16], logit(0.99)]
    return np.clip(x, -3, 3)


def unpack_signed(x):
    prm = e34.unpack(x)
    t0 = prm["tau"]["full"]
    prm["tau"]["G"] = float(t0 + 8 / (1 + np.exp(-x[3])))
    prm["tau"]["M"] = float(t0 - 8 / (1 + np.exp(-x[4])))
    return prm


if __name__ == "__main__":
    seed, mode = int(sys.argv[1]), sys.argv[2]
    signed = mode == "signed"
    e34.e33.unpack_base = e33.unpack  # e34.unpack builds on the E33 parameters
    e33.simulate = e34.simulate
    e33.unpack = unpack_signed if signed else e34.unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = targets, extra_se, 10
    src = os.environ.get("E36_WARM", "e33_s3")
    x0 = warm_x(src, signed) if seed < 3 else None
    e33.main(seed, maxiter=500, dim=21, name=f"e36{mode[0]}", x0=x0)
