"""Cross-check the FFTLog (FOLPS) and quadrature (fkpt) loop engines on identical inputs.

Both engines get the same P_lin, no-wiggle P_lin, f(k) (FOLPS's Eisenstein-Hu fit with
massive neutrinos), f0, EdS kernel constants and A_full=False layout; the same shared
RSD code then turns either table into multipoles. Each engine is also run at higher
resolution so a disagreement can be attributed to whichever engine is not converged.

Usage: python crosscheck/engines.py   (writes crosscheck/engines.npz, prints a summary)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
from make_golden import COSMO, PARS, load_inputs   # noqa: E402

from folpsfkpt import LoopTable                      # noqa: E402
from folpsfkpt.rsd import pk_multipoles              # noqa: E402
from folpsfkpt.table import A_FULL_TERMS, CORE_TERMS  # noqa: E402

K_RANGE = (0.01, 0.3)                               # comparison range, h/Mpc
K_CMP = np.geomspace(*K_RANGE, 60)                   # for multipoles
FFTLOG = {"nfftlog128": dict(nfftlog=128), "nfftlog256": dict(nfftlog=256)}
# kmin / kmax / Nk_kernel put fkpt's output on FOLPS's fixed grid (geomspace(1e-3, 0.5, 120)),
# so loop columns compare point by point; "fine" refines only fkpt's integration.
_GRID = dict(kmin=1e-3, kmax=0.5, Nk_kernel=120)
QUAD = {"default": dict(_GRID, nquadSteps=300, NQ=10, NR=10),
        "fine": dict(_GRID, nquadSteps=600, NQ=20, NR=20)}


def common_growth(k):
    from folps import f_over_f0_EH
    _, f_over_f0, _ = f_over_f0_EH(zev=COSMO["z"], k=np.asarray(k), OmM0=COSMO["Omega_m"],
                                   h=COSMO["h"], fnu=COSMO["fnu"])
    f0 = COSMO["f0"]
    return np.asarray(f_over_f0) * f0, f0


def run_fftlog(k, pk, pknow, fk, f0, nfftlog):
    from folps import MatrixCalculator, NonLinearPowerSpectrumCalculator
    mm = MatrixCalculator(nfftlog=nfftlog, A_full=False).get_mmatrices()
    # pkttlin = P_lin f(k)^2 is FOLPS's route for an external f(k)
    kw = dict(pkttlin=jnp.asarray(pk * fk**2), f0=f0)
    nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="fk", **kw)
    table, table_now = nl.calculate_loop_table(k=jnp.asarray(k), pklin=jnp.asarray(pk),
                                               pknow=jnp.asarray(pknow), cosmo=None, **kw)
    return LoopTable.from_folps_tuples(table, table_now)


def run_quadrature(k, pk, pknow, fk, f0, **res):
    from fkptjax.kfuncs_to_tables import Kfuncs_to_tables
    tw, tn = Kfuncs_to_tables(k, pk, pknow, z=COSMO["z"], Om=COSMO["Omega_m"], fk=fk, f0=f0,
                              beyond_eds=False, model="LCDM", return_kernel_constants=False, **res)
    return LoopTable.from_folps_tuples(tw, tn)


def on_grid(lt, name, which="wiggle"):
    k = np.asarray(lt.k)
    assert np.allclose(k, np.geomspace(1e-3, 0.5, 120), rtol=1e-12), "engines must share the output grid"
    sel = (k >= K_RANGE[0]) & (k <= K_RANGE[1])
    return np.asarray(getattr(getattr(lt, which), name))[sel]


def column_diff(a, b, name):
    """max |a - b| over K_RANGE, relative to max |b| there (columns cross zero)."""
    x, y = on_grid(a, name), on_grid(b, name)
    scale = np.max(np.abs(y))
    return np.max(np.abs(x - y)) / scale if scale > 0 else 0.0


def poles(lt, model, damping):
    return np.asarray(pk_multipoles(lt, jnp.asarray(K_CMP), jnp.asarray(PARS), model=model, damping=damping))


def main():
    k, pk, pknow = load_inputs()
    fk, f0 = common_growth(k)
    tables = {f"fftlog_{n}": run_fftlog(k, pk, pknow, fk, f0, **o) for n, o in FFTLOG.items()}
    tables.update({f"quad_{n}": run_quadrature(k, pk, pknow, fk, f0, **o) for n, o in QUAD.items()})

    pairs = {"engines (default)": ("quad_default", "fftlog_nfftlog128"),
             "engines (fine)": ("quad_fine", "fftlog_nfftlog256"),
             "fftlog resolution": ("fftlog_nfftlog256", "fftlog_nfftlog128"),
             "quadrature resolution": ("quad_fine", "quad_default")}
    names = [n for n in CORE_TERMS if n not in ("I3uuud_1B", "I4uuuu_1B")]   # zero in both without TNS
    out = {"k_cmp": K_CMP, "columns": np.array(names)}
    print(f"{'column':12s}" + "".join(f"{p:>24s}" for p in pairs))
    for n in names:
        row = [column_diff(tables[a], tables[b], n) for a, b in pairs.values()]
        out[f"coldiff_{n}"] = np.array(row)
        print(f"{n:12s}" + "".join(f"{v:24.2e}" for v in row))

    print("\nscalars:")
    for s in ("sigma2_nw", "delta_sigma2_nw"):
        print(f"  {s}: " + ", ".join(f"{t}={float(getattr(lt, s)):.8g}" for t, lt in tables.items()))
    for t, lt in tables.items():
        print(f"  sigma2w {t}: {float(lt.wiggle.sigma2w):.8g}  (no-wiggle {float(lt.nowiggle.sigma2w):.8g})")

    print("\nmultipoles, max |dP_ell| / max |P_ell| over k = 0.01-0.3 h/Mpc:")
    for model, damping in [("FOLPSD", "lor"), ("EFT", None)]:
        P = {t: poles(lt, model, damping) for t, lt in tables.items()}
        for t, v in P.items():
            out[f"poles_{model}_{t}"] = v
        for label, (a, b) in pairs.items():
            d = [np.max(np.abs(P[a][i] - P[b][i])) / np.max(np.abs(P[b][i])) for i in range(3)]
            out[f"polediff_{model}_{label}"] = np.array(d)
            print(f"  {model:6s} {label:22s} P0 {d[0]:.2e}  P2 {d[1]:.2e}  P4 {d[2]:.2e}")
    np.savez(HERE / "engines.npz", **out)


if __name__ == "__main__":
    main()
