"""The merged FolpsD bispectrum class reproduces both upstream classes."""
import os

os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import folps
import make_golden_bk as m

GOLDEN = np.load(m.GOLDEN / "bispectrum.npz")
RTOL = 1e-7


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=1e-12 * np.max(np.abs(b)))


@pytest.fixture(scope="module")
def plain_inputs():
    return m.inputs("folps_noafull", with_fk=False)


@pytest.fixture(scope="module")
def fk_inputs():
    return m.inputs("fkpt_hs_fr5", with_fk=True)


@pytest.mark.parametrize("damping", ["lor", "exp"])
def test_plain_sugiyama(plain_inputs, damping):
    close(m.sugiyama(folps.BispectrumCalculator, *plain_inputs, damping), GOLDEN[f"plain_sugiyama_{damping}"])


def test_plain_scoccimarro_lor(plain_inputs):
    close(m.scoccimarro(folps.BispectrumCalculator, *plain_inputs, "lor"), GOLDEN["plain_scoccimarro_lor"])


def test_plain_scoccimarro_now_honours_damping(plain_inputs):
    """Upstream's constant-f Scoccimarro_B024 hardcoded damping='lor'; the fk class passed it through."""
    exp = m.scoccimarro(folps.BispectrumCalculator, *plain_inputs, "exp")
    assert np.array_equal(GOLDEN["plain_scoccimarro_exp"], GOLDEN["plain_scoccimarro_lor"])   # the old bug
    assert not np.allclose(exp, GOLDEN["plain_scoccimarro_lor"], rtol=1e-6)


@pytest.mark.parametrize("basis", ["sugiyama", "scoccimarro"])
@pytest.mark.parametrize("damping", ["lor", "exp"])
def test_fk_class(fk_inputs, basis, damping):
    run = getattr(m, basis)
    close(run(folps.BispectrumCalculator_fk, *fk_inputs, damping), GOLDEN[f"fk_{basis}_{damping}"])


@pytest.mark.parametrize("basis", ["sugiyama", "scoccimarro"])
def test_fk_with_constant_growth_equals_plain(plain_inputs, basis):
    kpk3, f0 = plain_inputs
    ones = jnp.ones_like(kpk3[0])[None]
    kpk6 = jnp.vstack([kpk3, f0 * ones, ones, 0 * ones])           # f(k) = f, calA = 1, calAp = 0
    run = getattr(m, basis)
    close(run(folps.BispectrumCalculator_fk, kpk6, f0, "lor"), GOLDEN[f"plain_{basis}_lor"])


def test_jit_cache_does_not_leak(plain_inputs, fk_inputs):
    """Upstream cached jit tracers in a class-level dict; callers had to clear it after every call."""
    folps.BispectrumCalculator._tables_cache = {}
    k = np.asarray(m.K_EV)
    pairs = jnp.asarray(np.vstack([k, k]).T)
    for cls, (kpk, f0) in [(folps.BispectrumCalculator_fk, fk_inputs), (folps.BispectrumCalculator, plain_inputs)] * 2:
        calc = cls(model="FOLPSD")
        jax.jit(lambda bp: calc.Sugiyama_Bell(f0, bp, kpk, pairs, 1.0, 1.0, precision=[8, 10, 10]))(jnp.asarray(m.BPARS))
    assert all(isinstance(t, np.ndarray) for tabs in folps.BispectrumCalculator._tables_cache.values() for t in tabs)


def test_plain_warns_and_ignores_growth_rows(plain_inputs, fk_inputs):
    """desilike's fkpt bispectrum passes f(k) / calA rows to the plain class, which drops them."""
    kpk_fk, f0 = fk_inputs
    with pytest.warns(UserWarning, match="ignoring the f\\(k\\) / calA rows"):
        with_rows = m.sugiyama(folps.BispectrumCalculator, kpk_fk, f0, "lor")
    close(with_rows, m.sugiyama(folps.BispectrumCalculator, kpk_fk[:3], f0, "lor"))


def test_option_errors(plain_inputs):
    kpk3, f0 = plain_inputs
    with pytest.raises(ValueError, match="needs an f\\(k\\) row"):
        m.sugiyama(folps.BispectrumCalculator_fk, kpk3, f0, "lor")
    calc = folps.BispectrumCalculator(model="FOLPSD")
    k = np.asarray(m.K_EV)
    with pytest.raises(ValueError, match="mg_kernel_fn requires"):
        calc.Sugiyama_Bell(f0, jnp.asarray(m.BPARS), kpk3, jnp.asarray(np.vstack([k, k]).T), 1.0, 1.0,
                           mg_kernel_fn=lambda kf, k1, k2: (1., 0., 1., 0.))
    folps.BispectrumCalculator._tables_cache = {}
