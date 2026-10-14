"""Shared sigma^2 integrals: one implementation, same numbers eager and traced, every builder."""
import os

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from folpsfkpt.preprocess import extrapolate_pklin, ir_bao_grid, ir_damping_scalars, sigma2_v
from make_golden import CASES, COSMO, fkpt_tables_to_poles
from test_table import RTOL, assert_tuples_close, load_golden

BUILDER_CASES = [n for n, c in CASES.items() if c["engine"] == "fkpt_builder"]


def inputs():
    g, _, _ = load_golden("folps_noafull")
    k_ext, pk_ext = extrapolate_pklin(jnp.asarray(g["k_in"]), jnp.asarray(g["pk_in"]))
    _, pknow_ext = extrapolate_pklin(jnp.asarray(g["k_in"]), jnp.asarray(g["pknow_in"]))
    return k_ext, pk_ext, pknow_ext


def test_eager_and_traced_agree():
    k, pk, pknow = inputs()
    fk = 1.0 + 0.01 * jnp.tanh(k)
    eager = (sigma2_v(k, pk, fk), *ir_damping_scalars(k, pknow))
    grid = ir_bao_grid()
    traced = jax.jit(lambda k, pk, pknow, fk: (sigma2_v(k, pk, fk), *ir_damping_scalars(k, pknow, grid=grid)))(k, pk, pknow, fk)
    assert isinstance(eager[0], float)   # eager calls take the numpy fast path
    np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-12)


def test_folps_scalars_match_golden():
    """FOLPS passes its own pknow; the LoopTable's IR scalars come from the shared function."""
    g, table, table_now = load_golden("folps_noafull")
    k_ext, pknow_ext = extrapolate_pklin(jnp.asarray(g["k_in"]), jnp.asarray(g["pknow_in"]))
    s2, ds2 = ir_damping_scalars(k_ext, pknow_ext)
    np.testing.assert_allclose([s2, ds2], [table_now[-3], table_now[-2]], rtol=RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("name", BUILDER_CASES)
def test_fkptjax_builder_reproduces_golden(name):
    from fkptjax import kfuncs_to_tables
    g, table, table_now = load_golden(name)
    opts = dict(CASES[name]); opts.pop("engine"); builder = getattr(kfuncs_to_tables, opts.pop("builder"))
    out = builder(g["k_in"], g["pk_in"], g["pknow_in"], z=COSMO["z"], Om=COSMO["Omega_m"],
                  return_kernel_constants=False, **opts)
    assert_tuples_close(out[0], table)
    assert_tuples_close(out[1], table_now)
    np.testing.assert_allclose(fkpt_tables_to_poles(g["k_in"], out[0], out[1]), g["poles"], rtol=RTOL,
                               atol=1e-12 * np.max(np.abs(g["poles"])))


@pytest.mark.slow
def test_jitted_jax_builder_reproduces_golden():
    """Traced pk / pk_now: exercises the jax branch of the shared integrals."""
    from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax, build_jax_static_ctx
    g, table, table_now = load_golden("fkptjax_jax_mu0")
    k = jnp.asarray(g["k_in"])
    # the builder's own kmin/kmax defaults, made concrete (under jit it cannot derive them)
    kmin, kmax = float(min(1e-3, g["k_in"].min())), float(max(0.5, g["k_in"].max()))
    ctx = build_jax_static_ctx(k, kmin=kmin, kmax=kmax, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10)
    opts = dict(CASES["fkptjax_jax_mu0"]); opts.pop("engine"); opts.pop("builder")
    run = jax.jit(lambda pk, pknow: Kfuncs_to_tables_jax(k, pk, pknow, z=COSMO["z"], Om=COSMO["Omega_m"],
                                                          return_kernel_constants=False, static_ctx=ctx,
                                                          kmin=kmin, kmax=kmax, **opts)[:2])
    tw, tn = run(jnp.asarray(g["pk_in"]), jnp.asarray(g["pknow_in"]))
    assert_tuples_close(tw, table)
    assert_tuples_close(tn, table_now)
