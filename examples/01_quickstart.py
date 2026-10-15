"""Quickstart: linear P(k) -> loop table -> redshift-space multipoles P_0, P_2, P_4.

    python examples/01_quickstart.py
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, PARS, load_linear, savefig

from folpsfkpt import FFTLogEngine, pk_multipoles

k, pk, pknow = load_linear()

# 1. Engine: FOLPS's FFTLog matrices are built once here (a few seconds), then reused for every call.
engine = FFTLogEngine(A_full=True)

# 2. Loop table: all one-loop terms for the wiggle and no-wiggle spectra, as named fields.
table = engine(jnp.asarray(k), jnp.asarray(pk), jnp.asarray(pknow), **COSMO)
print(f"loop table on {table.k.shape[0]} k-points, A_full={table.A_full}, "
      f"Sigma^2={float(table.sigma2_nw):.2f}, f0={float(table.f0):.3f}")

# 3. Multipoles at the observed k, here for the FOLPS-D model with Lorentzian FoG damping.
kobs = np.linspace(0.01, 0.3, 60)
poles = pk_multipoles(table, jnp.asarray(kobs), jnp.asarray(PARS), model="FOLPSD", damping="lor")

fig, ax = plt.subplots(figsize=(5, 3.6))
for ell, p in zip((0, 2, 4), np.asarray(poles)):
    ax.plot(kobs, kobs * p, label=rf"$\ell={ell}$")
ax.set(xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"$k\,P_\ell(k)\ [h^{-2}{\rm Mpc}^2]$",
       title="FOLPS-D (FFTLog engine), z = 0.3")
ax.legend()
savefig(fig, "01_quickstart.png")
