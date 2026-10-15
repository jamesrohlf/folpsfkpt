"""RSD models and options on one loop table: EFT, FOLPS-D (lor / exp / vdg FoG damping), TNS, AP.

The model and damping are explicit; inconsistent combinations raise (e.g. model="EFT" with a
damping, or model="TNS" on a table built without use_TNS_model=True).

    python examples/04_rsd_models.py
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from common import COSMO, PARS, load_linear, savefig

from folpsfkpt import FFTLogEngine, pk_multipoles

k, pk, pknow = load_linear()
args = (jnp.asarray(k), jnp.asarray(pk), jnp.asarray(pknow))
kobs = jnp.linspace(0.01, 0.3, 60)
pars = jnp.asarray(PARS)

table = FFTLogEngine(A_full=True)(*args, **COSMO)
# TNS needs the Delta P(k, mu) pieces removed when the matrices are built: a separate table.
table_tns = FFTLogEngine(A_full=True, use_TNS_model=True)(*args, **COSMO)

runs = {
    "EFT (no damping)": (table, dict(model="EFT", damping=None)),
    "FOLPS-D, lor": (table, dict(model="FOLPSD", damping="lor")),
    "FOLPS-D, exp": (table, dict(model="FOLPSD", damping="exp")),
    "FOLPS-D, vdg": (table, dict(model="FOLPSD", damping="vdg")),
    "TNS, lor": (table_tns, dict(model="TNS", damping="lor")),
    "FOLPS-D, lor, AP q=(1.03, 0.98)": (table, dict(model="FOLPSD", damping="lor", qpar=1.03, qper=0.98)),
    "FOLPS-D, lor, classpt biases": (table, dict(model="FOLPSD", damping="lor", bias_scheme="classpt")),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for label, (t, options) in runs.items():
    p = np.asarray(pk_multipoles(t, kobs, pars, **options))
    for ax, ell in zip(axes, (0, 1)):
        ax.plot(kobs, kobs * p[ell], label=label)
axes[0].set(ylabel=r"$k\,P_0(k)$")
axes[1].set(ylabel=r"$k\,P_2(k)$")
for ax in axes:
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
axes[1].legend(fontsize=7)
fig.suptitle("Same loop table, different RSD models (note: the same nuisance values mean different things per model)")
fig.tight_layout()
savefig(fig, "04_rsd_models.png")

# Inconsistent options fail loudly instead of being ignored:
for bad in (dict(model="EFT", damping="lor"), dict(model="TNS", damping="lor")):
    try:
        pk_multipoles(table, kobs, pars, **bad)
    except ValueError as exc:
        print(f"{bad} -> ValueError: {exc}")
