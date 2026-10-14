"""Shared postprocessing reproduces both upstream RSD routes and decouples model from engine."""
import os

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from folpsfkpt import LoopTable
from folpsfkpt.rsd import pk_multipoles, pkmu, project_to_multipoles
from test_table import RTOL, load_golden


def golden_table(name):
    g, table, table_now = load_golden(name)
    return g, LoopTable.from_folps_tuples(table, table_now)


def fkpt_nodes():
    x, w = np.polynomial.legendre.leggauss(6)
    return 0.5 * (x + 1.0), 0.5 * w


def assert_poles_close(got, ref):
    np.testing.assert_allclose(np.asarray(got), ref, rtol=RTOL, atol=1e-12 * np.max(np.abs(ref)))


@pytest.mark.parametrize("name", ["folps_afull", "folps_noafull"])
def test_folpsd_route_reproduces_folps_golden(name):
    g, lt = golden_table(name)
    poles = pk_multipoles(lt, jnp.asarray(g["k_in"]), jnp.asarray(g["pars"]), model="FOLPSD", damping="lor")
    assert_poles_close(poles, g["poles"])


def test_tns_route_reproduces_upstream_golden():
    """Upstream reached TNS via a global set by MatrixCalculator; now the flag rides on the table."""
    g, table, table_now = load_golden("folps_tns")
    lt = LoopTable.from_folps_tuples(table, table_now, use_TNS_model=True)
    poles = pk_multipoles(lt, jnp.asarray(g["k_in"]), jnp.asarray(g["pars"]), model="TNS", damping="lor")
    assert_poles_close(poles, g["poles"])


@pytest.mark.parametrize("name", ["fkpt_lcdm", "fkpt_hs_fr5"])
def test_eft_route_reproduces_fkpt_golden(name):
    g, lt = golden_table(name)
    mu, wmu = fkpt_nodes()
    grid = pkmu(lt, jnp.asarray(g["k_in"])[:, None], jnp.asarray(mu)[None, :], jnp.asarray(g["pars"]),
                model="EFT", damping=None)
    assert_poles_close(project_to_multipoles(grid, mu, wmu), g["poles"])


@pytest.mark.parametrize("model, damping, match", [
    ("EFT", "lor", "no FoG damping"),
    ("FOLPSD", None, "needs damping"),
    ("TNS", None, "use_TNS_model=True"),
    ("SPT", None, "model must be"),
    ("FOLPSD", "gauss", "damping must be"),
])
def test_inconsistent_options_raise(model, damping, match):
    g, lt = golden_table("fkpt_lcdm")
    with pytest.raises(ValueError, match=match):
        pk_multipoles(lt, jnp.asarray(g["k_in"]), jnp.asarray(g["pars"]), model=model, damping=damping)


@pytest.mark.parametrize("damping", ["lor", "exp", "vdg"])
def test_fkpt_tables_run_through_folpsd(damping):
    """Previously impossible: fkpt loop terms with FOLPS-D damping."""
    g, lt = golden_table("fkpt_lcdm")
    k, pars = jnp.asarray(g["k_in"]), jnp.asarray(g["pars"])
    damped = np.asarray(pk_multipoles(lt, k, pars, model="FOLPSD", damping=damping))
    eft = np.asarray(pk_multipoles(lt, k, pars, model="EFT", damping=None))
    assert np.all(np.isfinite(damped))
    sel = (g["k_in"] > 0.05) & (g["k_in"] < 0.2)
    assert np.max(np.abs(damped[0, sel] / eft[0, sel] - 1)) > 1e-3   # damping actually applied
