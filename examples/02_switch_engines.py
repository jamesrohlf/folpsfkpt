"""Same input, two loop engines: FOLPS's FFTLog matrices vs fkpt's Q/R quadrature.

Both engines return a LoopTable with the same named terms, and the same RSD code turns either
into multipoles, so switching engine is a one-line change. To compare like with like, the
quadrature engine is given FOLPS's f(k) and output grid (fkpt otherwise solves its own growth
ODE, which differs slightly from FOLPS's Eisenstein-Hu fit for massive neutrinos).

    python examples/02_switch_engines.py
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, PARS, load_linear, savefig

from folpsfkpt import ENGINES, pk_multipoles

k, pk, pknow = load_linear()

# f(k) for massive neutrinos, as FOLPS computes it internally
from folps import f_over_f0_EH
_, f_over_f0, _ = f_over_f0_EH(zev=COSMO["z"], k=k, OmM0=COSMO["Omega_m"], h=COSMO["h"], fnu=COSMO["fnu"])
fk = np.asarray(f_over_f0) * COSMO["f0"]

engines = {
    # FFTLog engine. A_full=False to match fkpt's table layout.
    "fftlog": ENGINES["fftlog"](A_full=False),
    # Quadrature engine: options go to fkptjax's Kfuncs_to_tables. kmin/kmax/Nk_kernel = FOLPS's output grid.
    "quadrature": ENGINES["quadrature"](model="LCDM", fk=fk, f0=COSMO["f0"], kmin=1e-3, kmax=0.5, Nk_kernel=120),
}
kobs = np.linspace(0.01, 0.3, 60)
poles = {}
for name, engine in engines.items():
    table = engine(jnp.asarray(k) if name == "fftlog" else k, jnp.asarray(pk) if name == "fftlog" else pk,
                   jnp.asarray(pknow) if name == "fftlog" else pknow, **COSMO)
    poles[name] = np.asarray(pk_multipoles(table, jnp.asarray(kobs), jnp.asarray(PARS), model="FOLPSD",
                                           damping="lor"))

fig, (ax, rx) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
for i, ell in enumerate((0, 2, 4)):
    line, = ax.plot(kobs, kobs * poles["fftlog"][i], label=rf"$\ell={ell}$ FFTLog")
    ax.plot(kobs, kobs * poles["quadrature"][i], "--", color=line.get_color(), label=rf"$\ell={ell}$ quadrature")
    diff = (poles["quadrature"][i] - poles["fftlog"][i]) / np.max(np.abs(poles["fftlog"][i]))
    rx.plot(kobs, diff, color=line.get_color())
    print(f"P_{ell}: max |quadrature - fftlog| / max |P_{ell}| = {np.max(np.abs(diff)):.1e}")
ax.set(ylabel=r"$k\,P_\ell(k)$", title="Loop engines on identical inputs")
ax.legend(fontsize=7, ncol=2)
rx.set(xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"$\Delta P_\ell / \max|P_\ell|$")
savefig(fig, "02_switch_engines.png")
