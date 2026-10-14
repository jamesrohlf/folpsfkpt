"""LoopTable must round-trip the upstream layouts and reproduce the golden outputs.

All cases run in ONE process on purpose: it checks that the engines do not leak
FOLPS's module-level A_full / use_TNS_model state into each other.
"""
import os
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from folpsfkpt import FFTLogEngine, LoopTable, QuadratureEngine
from make_golden import CASES, COSMO

GOLDEN = Path(__file__).resolve().parent / "golden"
RTOL = 1e-7   # reruns of the golden script on GPU agree to ~1e-9


def load_golden(name):
    g = np.load(GOLDEN / f"{name}.npz")
    table = tuple(g[n] for n in sorted(g.files) if n.startswith("table_") and not n.startswith("table_now_"))
    table_now = tuple(g[n] for n in sorted(g.files) if n.startswith("table_now_"))
    return g, table, table_now


def assert_tuples_close(got, ref, rtol=RTOL):
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        b = np.asarray(b)
        np.testing.assert_allclose(np.asarray(a), b, rtol=rtol, atol=1e-12 * np.max(np.abs(b)) if b.size else 0,
                                   err_msg=f"table entry {i}")


@pytest.mark.parametrize("name", list(CASES))
def test_roundtrip_exact(name):
    _, table, table_now = load_golden(name)
    lt = LoopTable.from_folps_tuples(table, table_now)
    assert lt.A_full == (len(table) == 36)
    back, back_now = lt.to_folps_tuples()
    for a, b in zip(back + back_now, table + table_now):
        assert np.array_equal(np.asarray(a), b)


def test_fkpt_and_folps_name_clash_resolved():
    """Slot 10 is the bs2^2 term in both codes; it must land in P_bs2bs2."""
    _, table, table_now = load_golden("folps_noafull")
    lt = LoopTable.from_folps_tuples(table, table_now)
    assert np.array_equal(lt.wiggle.P_b2bs2, table[9])
    assert np.array_equal(lt.wiggle.P_bs2bs2, table[10])


def test_pytree_flags_are_static():
    _, table, table_now = load_golden("folps_afull")
    lt = LoopTable.from_folps_tuples(table, table_now, use_TNS_model=True)
    out = jax.jit(lambda t: t)(lt)
    assert out.A_full is True and out.use_TNS_model is True
    assert np.array_equal(np.asarray(out.wiggle.P_dd), table[3])


def folps_poles(lt, k, pars, model="FOLPSD"):
    from folps import RSDMultipolesPowerSpectrumCalculator
    table, table_now = lt.to_folps_tuples()
    return RSDMultipolesPowerSpectrumCalculator(model=model, use_TNS_model=lt.use_TNS_model).get_rsd_pkell(
        kobs=jnp.asarray(k), qpar=1.0, qper=1.0, pars=jnp.asarray(pars),
        table=table, table_now=table_now, bias_scheme="folps", damping="lor")


def fkpt_poles(lt, k, pars):
    from fkptjax.rsd import tables_to_poles
    table, table_now = lt.to_folps_tuples()
    x, w = np.polynomial.legendre.leggauss(6)
    mu, wmu = 0.5 * (x + 1.0), 0.5 * w
    return tables_to_poles(1.0, jnp.asarray(k)[:, None], jnp.asarray(mu)[None, :], jnp.asarray(pars),
                           table + table_now, mu, wmu, ncols=len(table), bias_scheme="folps",
                           IR_resummation=True, damping=None, ells=(0, 2, 4), A_full=False, use_TNS_model=False)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_engine_reproduces_golden(name):
    g, table, table_now = load_golden(name)
    k, pk, pknow, pars = g["k_in"], g["pk_in"], g["pknow_in"], g["pars"]
    opts = dict(CASES[name])
    if opts.pop("engine") == "folps":
        model = opts.pop("model", "FOLPSD")
        lt = FFTLogEngine(**opts)(jnp.asarray(k), jnp.asarray(pk), jnp.asarray(pknow), **COSMO)
        poles = folps_poles(lt, k, pars, model=model)
    else:
        lt = QuadratureEngine(**opts)(k, pk, pknow, **COSMO)
        poles = fkpt_poles(lt, k, pars)
    got, got_now = lt.to_folps_tuples()
    assert_tuples_close(got, table)
    assert_tuples_close(got_now, table_now)
    np.testing.assert_allclose(np.asarray(poles), g["poles"], rtol=RTOL, atol=1e-12 * np.max(np.abs(g["poles"])))
