"""Tree-level bispectrum multipoles (Sugiyama basis), with constant growth or with fkpt's f(k).

FolpsD now has one bispectrum class. ``scale_dependent_growth=True`` (formerly the separate
``BispectrumCalculator_fk``) reads f(k) and the beyond-EdS kernel constants calA / calAp from
extra rows of the input array; here they come from a Hu-Sawicki f(R) quadrature table.

    python examples/06_bispectrum.py
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, load_linear, savefig

from folps import BispectrumCalculator
from folpsfkpt import QuadratureEngine

k, pk, pknow = load_linear()
# b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b
B1 = 1.645
bpars = jnp.array([B1, -0.46, -4. / 7. * (B1 - 1.), 0., 0., 0., 0., 1.])
kk = np.linspace(0.02, 0.17, 16)
pairs = jnp.asarray(np.vstack([kk, kk]).T)                 # k1 = k2 configurations
multipoles = ["B000", "B202"]


def sugiyama(calc, k_pk_pknw, f0):
    run = jax.jit(lambda bp: calc.Sugiyama_Bell(f0, bp, k_pk_pknw, pairs, 1.0, 1.0, precision=[8, 10, 10],
                                                damping="lor", multipoles=multipoles, renormalize=True,
                                                interpolation_method="linear", bias_scheme="folps"))
    return np.asarray(run(bpars))


# constant growth rate: rows (k, P_lin, P_nowiggle)
b_const = sugiyama(BispectrumCalculator(model="FOLPSD"), jnp.asarray(np.array([k, pk, pknow])), COSMO["f0"])

# scale-dependent growth from an f(R) table: rows (k, P_lin, P_nw, f(k), calA, calAp)
table = QuadratureEngine(model="HS", fR0_HS=1e-5, rescale_PS=True, beyond_eds=True)(k, pk, pknow, **COSMO)
f0 = float(table.f0)
calA, calAp = (float(c) for c in table.kernel_constants[:2])
ones = np.ones_like(np.asarray(table.k))
rows = np.array([table.k, table.wiggle.pk_lin, table.nowiggle.pk_lin, np.asarray(table.wiggle.f_over_f0) * f0,
                 calA * ones, calAp * ones])
b_fk = sugiyama(BispectrumCalculator(model="FOLPSD", scale_dependent_growth=True), jnp.asarray(rows), f0)
print(f"beyond-EdS constants for f(R): calA = {calA:.4f}, calAp = {calAp:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, i, name in zip(axes, range(2), multipoles):
    ax.plot(kk, kk**2 * b_const[i], label=r"$\Lambda$CDM, constant $f$")
    ax.plot(kk, kk**2 * b_fk[i], "--", label=r"f(R) $10^{-5}$, $f(k)$ + beyond-EdS")
    ax.set(xlabel=r"$k_1 = k_2\ [h\,{\rm Mpc}^{-1}]$", ylabel=rf"$k^2 B_{{{name[1:]}}}$")
axes[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, "06_bispectrum.png")
