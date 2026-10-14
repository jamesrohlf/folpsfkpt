"""FolpsD must not carry A_full / use_TNS_model as module state any more."""
import os

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import folps.folps as _folps
from folps import MatrixCalculator, RSDMultipolesPowerSpectrumCalculator
from folps.folps import _matrices_flags, _table_has_A_full
from test_table import load_golden


def test_module_has_no_flag_globals():
    MatrixCalculator(A_full=False, use_TNS_model=True)
    assert not hasattr(_folps, "A_full_status")
    assert not hasattr(_folps, "use_TNS_model_status")


@pytest.mark.parametrize("name, expected", [("folps_afull", True), ("folps_noafull", False), ("fkpt_lcdm", False)])
def test_table_length_gives_A_full(name, expected):
    _, table, table_now = load_golden(name)
    assert _table_has_A_full(table) is expected and _table_has_A_full(table_now) is expected
    calc = RSDMultipolesPowerSpectrumCalculator(model="EFT")
    k = jnp.asarray(table[0][5:10])
    assert _table_has_A_full(calc.interp_table(k, table)) is expected
    assert _table_has_A_full(calc.interp_table(k, table_now)) is expected


def test_old_matrices_without_flags():
    mm = {k: v for k, v in MatrixCalculator(A_full=True).get_mmatrices().items() if k in ("M22matrices", "M13vectors")}
    with pytest.warns(UserWarning, match="use_TNS_model"):
        assert _matrices_flags(mm) == (True, False)


def test_tns_default_follows_model():
    assert RSDMultipolesPowerSpectrumCalculator(model="TNS").use_TNS_model is True
    assert RSDMultipolesPowerSpectrumCalculator(model="FOLPSD").use_TNS_model is False
    assert RSDMultipolesPowerSpectrumCalculator(model="FOLPSD", use_TNS_model=True).use_TNS_model is True
