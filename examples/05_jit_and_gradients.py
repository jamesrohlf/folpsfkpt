"""Compile the whole pipeline with jax.jit and differentiate it, e.g. for samplers, Fisher forecasts
or emulator training.

LoopTable is a jax pytree (A_full / use_TNS_model are static metadata), so it can be returned
from and passed into jitted functions. FOLPS's pipeline contains custom_vjp functions, so use
reverse mode (jax.grad / jax.jacrev), not jax.jacfwd.

    python examples/05_jit_and_gradients.py
"""
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, PAR_NAMES, PARS, load_linear, savefig

from folpsfkpt import FFTLogEngine, pk_multipoles

k, pk, pknow = load_linear()
kobs = jnp.linspace(0.01, 0.3, 60)
engine = FFTLogEngine(A_full=True)


def multipoles(theta, pars):
    """P_ell for a P_lin amplitude rescaling, f0 and AP parameters theta = (A, f0, qpar, qper)."""
    amp, f0, qpar, qper = theta
    table = engine(jnp.asarray(k), amp * jnp.asarray(pk), amp * jnp.asarray(pknow), **dict(COSMO, f0=f0))
    return pk_multipoles(table, kobs, pars, model="FOLPSD", damping="lor", qpar=qpar, qper=qper)


theta = jnp.array([1.0, COSMO["f0"], 1.0, 1.0])
pars = jnp.asarray(PARS)

fast = jax.jit(multipoles)
fast(theta, pars).block_until_ready()                      # first call compiles
t0 = time.perf_counter()
for _ in range(20):
    fast(theta * (1 + 1e-3 * np.random.randn(4)), pars).block_until_ready()
print(f"jitted P_ell: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms per evaluation (CPU)")

jac = jax.jit(jax.jacrev(multipoles, argnums=(0, 1)))
d_theta, d_pars = jac(theta, pars)                         # shapes (3, nk, 4) and (3, nk, 12)
print("d P_ell / d theta:", d_theta.shape, " d P_ell / d nuisances:", d_pars.shape)

# check one derivative against finite differences
eps = 1e-4
fd = (fast(theta.at[1].add(eps), pars) - fast(theta.at[1].add(-eps), pars)) / (2 * eps)
print(f"dP0/df0: autodiff vs finite difference, max rel. diff "
      f"{float(jnp.max(jnp.abs(d_theta[0, :, 1] - fd[0])) / jnp.max(jnp.abs(fd[0]))):.1e}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for i, name in enumerate(["A (P_lin amplitude)", "f0", "qpar", "qper"]):
    axes[0].plot(kobs, kobs * d_theta[0, :, i], label=name)
for i in (0, 1, 4, 10):
    axes[1].plot(kobs, kobs * d_pars[0, :, i] * (abs(PARS[i]) or 1.), label=f"{PAR_NAMES[i]} (x |value|)")
axes[0].set(title=r"$k\,\partial P_0/\partial\theta$")
axes[1].set(title=r"$k\,\partial P_0/\partial$ nuisance")
for ax in axes:
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "05_gradients.png")
