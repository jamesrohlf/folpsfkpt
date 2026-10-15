"""Modified gravity with the quadrature (fkpt) engine: Hu-Sawicki f(R) and nDGP against LCDM.

fkpt solves the scale-dependent growth ODE of the chosen model, so f(k) and the loop kernels
carry the modification. ``rescale_PS=True`` rescales the input (LCDM) P_lin to the MG linear
spectrum at z first. The multipoles then go through the same RSD code as any other table.

    python examples/03_modified_gravity.py
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, PARS, load_linear, savefig

from folpsfkpt import QuadratureEngine, pk_multipoles

k, pk, pknow = load_linear()
kobs = np.linspace(0.01, 0.25, 50)


def poles(**options):
    table = QuadratureEngine(**options)(k, pk, pknow, **COSMO)
    # fkpt's own convention is undamped EFT; FoG damping is opt-in (model="FOLPSD", damping=...)
    return np.asarray(pk_multipoles(table, jnp.asarray(kobs), jnp.asarray(PARS), model="EFT", damping=None)), table


ref, table_lcdm = poles(model="LCDM")
models = {r"f(R), $|f_{R0}|=10^{-6}$": dict(model="HS", fR0_HS=1e-6, rescale_PS=True),
          r"f(R), $|f_{R0}|=10^{-5}$": dict(model="HS", fR0_HS=1e-5, rescale_PS=True),
          r"nDGP, $H_0 r_c = 1$": dict(model="NDGP", r_c=1.0, rescale_PS=True)}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for label, options in models.items():
    p, table = poles(**options)
    for ell, ax in zip(range(3), axes[:2]):
        # difference over the peak amplitude: a plain ratio blows up where P_2 crosses zero (k ~ 0.26)
        ax.plot(kobs, (p[ell] - ref[ell]) / np.max(np.abs(ref[ell])), label=label)
    axes[2].plot(np.asarray(table.k), np.asarray(table.wiggle.f_over_f0) * float(table.f0), label=label)
axes[2].plot(np.asarray(table_lcdm.k), np.asarray(table_lcdm.wiggle.f_over_f0) * float(table_lcdm.f0), "k:",
             label=r"$\Lambda$CDM")
axes[0].set(ylabel=r"$(P_0 - P_0^{\Lambda{\rm CDM}}) / \max|P_0^{\Lambda{\rm CDM}}|$")
axes[1].set(ylabel=r"$(P_2 - P_2^{\Lambda{\rm CDM}}) / \max|P_2^{\Lambda{\rm CDM}}|$")
axes[2].set(ylabel=r"$f(k)$", xscale="log", xlim=(1e-3, 0.5))
for ax in axes:
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
axes[2].legend(fontsize=7)
fig.suptitle("Quadrature engine: modified gravity vs LCDM, same nuisances, z = 0.3")
fig.tight_layout()
savefig(fig, "03_modified_gravity.png")
