"""Shared inputs for the examples: a linear power spectrum, its no-wiggle version, a cosmology and nuisances.

The spectrum is FOLPS's test input (z = 0.3, Planck-like with massive neutrinos), so every example
runs without a Boltzmann code. To use your own, replace ``load_linear`` with e.g. CLASS / CAMB /
cosmoprimo output on a log-spaced k grid from ~1e-4 to a few h/Mpc.
"""
import os
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")   # FOLPS's jax backend: needed for jit / gradients

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)        # the loop integrals need double precision

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# Cosmology of the input spectrum. f0 is the growth rate at z (large scales); fnu = Omega_nu / Omega_m.
COSMO = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114, f0=0.6880638641959066, fnu=0.004453689063655854)

# Power-spectrum nuisances, FOLPS order:
# b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p
B1 = 1.645
PARS = np.array([B1, -0.46, -4. / 7. * (B1 - 1.), 32. / 315. * (B1 - 1.),
                 3.0, -28.9, 0.0, 0.0, 0.08, -8.1, 1. / 0.0002118763, 1.0])
PAR_NAMES = ["b1", "b2", "bs2", "b3nl", "alpha0", "alpha2", "alpha4", "ctilde",
             "alphashot0", "alphashot2", "PshotP", "X_FoG_p"]


def load_linear():
    """(k [h/Mpc], P_lin [(Mpc/h)^3], P_nowiggle) on the input grid."""
    import jax.numpy as jnp
    from folps import extrapolate_pklin, get_pknow_jax
    k, pk = np.loadtxt(HERE / "data" / "pk_lin_z0.3.txt", unpack=True)
    # FOLPS's no-wiggle spectrum: extrapolate, then remove the BAO with its DST filter
    k_ext, pk_ext = extrapolate_pklin(k=k, pk=pk)
    k_now, pknow = get_pknow_jax(k=jnp.asarray(k_ext), pk=jnp.asarray(pk_ext), h=COSMO["h"])
    return k, pk, np.interp(k, np.asarray(k_now), np.asarray(pknow))


def savefig(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path.relative_to(HERE.parent)}")
