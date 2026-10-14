"""Record bispectrum multipoles from upstream FolpsD's two bispectrum classes.

Inputs reuse the power-spectrum golden tables (FOLPS no-A_full, and fkpt HS for
a genuinely scale-dependent f(k)). Usage: python tests/make_golden_bk.py
"""
import os
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")

GOLDEN = Path(__file__).resolve().parent / "golden"

B1 = 1.645
BPARS = [B1, -0.46, -4.0 / 7.0 * (B1 - 1.0), 0.0, 0.0, 0.0, 0.0, 1.0]   # b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b
K_EV = [0.02 + 0.01 * i for i in range(16)]
SUGIYAMA = ("B000", "B110", "B220", "B202", "B022", "B112")
CAL_A, CAL_AP = 1.02, 0.015   # arbitrary non-trivial beyond-EdS constants


def inputs(name, with_fk):
    import jax.numpy as jnp
    import numpy as np
    g = np.load(GOLDEN / f"{name}.npz")
    k, pkl, pklnw = g["table_00"], g["table_01"], g["table_now_01"]
    f0 = float(g["table_29"])                 # last entry of the A_full=False wiggle table
    rows = [k, pkl, pklnw]
    if with_fk:
        rows += [g["table_02"] * f0, np.full_like(k, CAL_A), np.full_like(k, CAL_AP)]
    return jnp.asarray(np.array(rows)), f0


def triplets():
    import numpy as np
    k = np.asarray(K_EV)
    return np.vstack([k, k, k]).T             # equilateral


def sugiyama(cls, kpk, f0, damping):
    import jax
    import jax.numpy as jnp
    import numpy as np
    k = np.asarray(K_EV)
    pairs = jnp.asarray(np.vstack([k, k]).T)
    calc = cls(model="FOLPSD")
    run = jax.jit(lambda bp: calc.Sugiyama_Bell(f0, bp, kpk, pairs, 1.0, 1.0, precision=[8, 10, 10],
                                                damping=damping, multipoles=list(SUGIYAMA),
                                                renormalize=True, interpolation_method="linear",
                                                bias_scheme="folps"))
    try:
        return np.asarray(run(jnp.asarray(BPARS)))
    finally:
        cls._tables_cache = {}   # class-level cache keeps jit tracers; desilike clears it the same way


def scoccimarro(cls, kpk, f0, damping):
    import jax.numpy as jnp
    import numpy as np
    calc = cls(model="FOLPSD")
    cls._tables_cache = {}
    return np.asarray(calc.Scoccimarro_Bell(jnp.asarray(triplets()), f0, jnp.asarray(BPARS), 1.0, 1.0, kpk,
                                            precision=[8, 8], damping=damping, interpolation_method="linear",
                                            multipoles=["B0", "B2", "B4"], bias_scheme="folps"))


def compute_all():
    import folps
    out = {}
    kpk3, f0 = inputs("folps_noafull", with_fk=False)
    kpk_fk, f0_fk = inputs("fkpt_hs_fr5", with_fk=True)
    for damping in ("lor", "exp"):
        out[f"plain_sugiyama_{damping}"] = sugiyama(folps.BispectrumCalculator, kpk3, f0, damping)
        out[f"plain_scoccimarro_{damping}"] = scoccimarro(folps.BispectrumCalculator, kpk3, f0, damping)
        out[f"fk_sugiyama_{damping}"] = sugiyama(folps.BispectrumCalculator_fk, kpk_fk, f0_fk, damping)
        out[f"fk_scoccimarro_{damping}"] = scoccimarro(folps.BispectrumCalculator_fk, kpk_fk, f0_fk, damping)
    return out


if __name__ == "__main__":
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    out = compute_all()
    np.savez(GOLDEN / "bispectrum.npz", **out)
    for key, v in out.items():
        print(f"{key}: {v.shape} finite={np.all(np.isfinite(v))}")
