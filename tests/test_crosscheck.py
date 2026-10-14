"""FFTLog (FOLPS) and quadrature (fkpt) engines agree on identical inputs.

Same P_lin, no-wiggle P_lin, f(k), f0, EdS kernel constants and output grid; same
shared RSD code. Tolerances sit ~2-3x above the measured differences
(crosscheck/engines.py), which are at the level of each engine's own resolution
change, except for the P13-type UV cutoff noted below.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crosscheck"))
import engines as e   # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tables():
    k, pk, pknow = e.load_inputs()
    fk, f0 = e.common_growth(k)
    return (e.run_fftlog(k, pk, pknow, fk, f0, 128),
            e.run_quadrature(k, pk, pknow, fk, f0, **e.QUAD["default"]))


def test_shared_scalars_identical(tables):
    a, b = tables
    for s in ("sigma2_nw", "delta_sigma2_nw", "f0"):
        assert float(getattr(a, s)) == pytest.approx(float(getattr(b, s)), rel=1e-12)
    assert float(a.wiggle.sigma2w) == pytest.approx(float(b.wiggle.sigma2w), rel=1e-7)


def test_loop_columns_agree(tables):
    """Max column difference over k = 0.01-0.3 h/Mpc, relative to the column's max there.
    Measured: <= 2.8% (I3uuu_3) at default resolution, <= 0.8% with both engines refined."""
    a, b = tables
    names = [n for n in e.CORE_TERMS if n not in ("I3uuud_1B", "I4uuuu_1B")]
    worst = {n: e.column_diff(b, a, n) for n in names}
    assert max(worst.values()) < 0.05, worst


@pytest.mark.parametrize("model, damping, tol", [("FOLPSD", "lor", (5e-4, 1.5e-3, 4e-3)),
                                                 ("EFT", None, (6e-4, 2e-3, 5e-3))])
def test_multipoles_agree(tables, model, damping, tol):
    """Measured (default resolution): FOLPSD 1.7e-4 / 4.7e-4 / 1.2e-3, EFT 2.4e-4 / 6.4e-4 / 1.9e-3."""
    a, b = tables
    pa, pb = e.poles(a, model, damping), e.poles(b, model, damping)
    for ell, t in enumerate(tol):
        d = np.max(np.abs(pb[ell] - pa[ell])) / np.max(np.abs(pa[ell]))
        assert d < t, (2 * ell, d)


def test_p13_uv_cutoff_explains_residual():
    """fkpt integrates to p_max = 16 k_max(output); with k_max = 0.5 that truncates P13-type
    terms (~1% low at k = 0.45). Raising k_max -> p_max = 32 h/Mpc brings them to ~0.1%."""
    k, pk, pknow = e.load_inputs()
    fk, f0 = e.common_growth(k)
    a = e.run_fftlog(k, pk, pknow, fk, f0, 256)
    pts = np.log([0.1, 0.3])
    ratio = lambda lt: np.interp(pts, np.log(np.asarray(lt.k)),
                                 np.asarray(lt.wiggle.sigma3_pk) / np.asarray(lt.wiggle.pk_lin))
    ref = ratio(a)
    short = e.run_quadrature(k, pk, pknow, fk, f0, kmin=1e-3, kmax=0.5, Nk_kernel=120, nquadSteps=600, NQ=20, NR=20)
    long = e.run_quadrature(k, pk, pknow, fk, f0, kmin=1e-3, kmax=2.0, Nk_kernel=242, nquadSteps=600, NQ=20, NR=20)
    assert np.all(ratio(short) / ref - 1 < -2e-3)                 # truncated: clearly low
    assert np.all(np.abs(ratio(long) / ref - 1) < 1.5e-3)         # wide enough: agrees
