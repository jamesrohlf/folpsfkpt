"""Inside the LoopTable: named loop terms instead of FOLPS's positional 30/36-entry tuples.

    python examples/07_loop_table.py
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, load_linear, savefig

from folpsfkpt import FFTLogEngine, LoopTable
from folpsfkpt.table import A_FULL_TERMS, CORE_TERMS

k, pk, pknow = load_linear()
table = FFTLogEngine(A_full=True)(jnp.asarray(k), jnp.asarray(pk), jnp.asarray(pknow), **COSMO)

print("core terms:  ", ", ".join(CORE_TERMS))
print("A_full terms:", ", ".join(A_FULL_TERMS))
print(f"scalars: sigma2w = {float(table.wiggle.sigma2w):.3f}, Sigma^2 = {float(table.sigma2_nw):.3f}, "
      f"delta Sigma^2 = {float(table.delta_sigma2_nw):.3f}, f0 = {float(table.f0):.4f}")

# Lossless round trip to the positional layout FOLPS / fkptjax / desilike use
t, t_now = table.to_folps_tuples()
back = LoopTable.from_folps_tuples(t, t_now)
print(f"positional layout: {len(t)} + {len(t_now)} entries; round trip exact: "
      f"{all(np.array_equal(a, b) for a, b in zip(back.to_folps_tuples()[0], t))}")

kk = np.asarray(table.k)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for name in ("pk_lin", "P_dd", "P_dt", "P_tt", "P_b1b2", "P_b2b2", "P_bs2bs2"):
    ax.loglog(kk, np.abs(np.asarray(getattr(table.wiggle, name))), label=name)
ax.set(xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"$|P(k)|\ [h^{-3}{\rm Mpc}^3]$", ylim=(1e-1, 1e5),
       title="Some loop terms (wiggle spectrum)")
ax.legend(fontsize=7, ncol=2)
savefig(fig, "07_loop_table.png")
