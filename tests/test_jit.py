"""Engines and RSD code must trace under jax.jit and give the eager (golden) result."""
import os

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from folpsfkpt import FFTLogEngine, pk_multipoles
from make_golden import COSMO
from test_table import RTOL, assert_tuples_close, load_golden


def test_fftlog_engine_and_poles_under_jit():
    """ir_bao_grid once converted a traced Bessel grid to numpy, so the engine failed inside jit."""
    g, table, table_now = load_golden("folps_afull")
    engine = FFTLogEngine(A_full=True)

    @jax.jit
    def run(k, pk, pknow, pars, cosmo):
        lt = engine(k, pk, pknow, **cosmo)
        return lt.to_folps_tuples(), pk_multipoles(lt, k, pars, model="FOLPSD", damping="lor")

    (got, got_now), poles = run(jnp.asarray(g["k_in"]), jnp.asarray(g["pk_in"]), jnp.asarray(g["pknow_in"]),
                                jnp.asarray(g["pars"]), {n: jnp.asarray(v) for n, v in COSMO.items()})
    assert_tuples_close(got, table)
    assert_tuples_close(got_now, table_now)
    np.testing.assert_allclose(np.asarray(poles), g["poles"], rtol=RTOL, atol=1e-12 * np.max(np.abs(g["poles"])))
