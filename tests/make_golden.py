"""Record reference outputs from the unmodified upstream FOLPS and fkptjax.

Every refactoring step of folpsfkpt must reproduce these files. Each case runs
in its own process because FOLPS keeps A_full / use_TNS_model in module-level
globals, so cases sharing a process could contaminate each other.

Usage:
    python tests/make_golden.py            # all cases
    python tests/make_golden.py CASE       # one case, in this process
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"

# Cosmology and nuisances from FolpsD's folps/test_folps_jax.py.
COSMO = dict(z=0.3, h=0.6711, Omega_m=0.3211636237981114,
             f0=0.6880638641959066, fnu=0.004453689063655854)
B1 = 1.645
PARS = [B1, -0.46, -4.0 / 7.0 * (B1 - 1.0), 32.0 / 315.0 * (B1 - 1.0),
        3.0, -28.9, 0.0, 0.0, 0.08, -8.1, 1.0 / 0.0002118763, 1.0]

CASES = {
    "folps_afull": dict(engine="folps", A_full=True),
    "folps_noafull": dict(engine="folps", A_full=False),
    "fkpt_lcdm": dict(engine="fkpt", model="LCDM"),
    "fkpt_hs_fr5": dict(engine="fkpt", model="HS", fR0_HS=1e-5, rescale_PS=True),
}


def load_inputs():
    import jax.numpy as jnp
    import numpy as np
    from folps import extrapolate_pklin, get_pknow_jax

    k, pk = np.loadtxt(GOLDEN / "inputpkT.txt", unpack=True)
    k_ext, pk_ext = extrapolate_pklin(k=k, pk=pk)
    k_now, pknow = get_pknow_jax(k=jnp.asarray(k_ext), pk=jnp.asarray(pk_ext), h=COSMO["h"])
    pknow = np.interp(k, np.asarray(k_now), np.asarray(pknow))
    return k, pk, pknow


def run_folps(k, pk, pknow, A_full):
    import jax.numpy as jnp
    import numpy as np
    from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator,
                       RSDMultipolesPowerSpectrumCalculator)

    mm = MatrixCalculator(A_full=A_full).get_mmatrices()
    nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="fk", **COSMO)
    table, table_now = nl.calculate_loop_table(k=jnp.asarray(k), pklin=jnp.asarray(pk),
                                               pknow=jnp.asarray(pknow), cosmo=None, **COSMO)
    poles = RSDMultipolesPowerSpectrumCalculator(model="FOLPSD").get_rsd_pkell(
        kobs=jnp.asarray(k), qpar=1.0, qper=1.0, pars=jnp.asarray(PARS),
        table=table, table_now=table_now, bias_scheme="folps", damping="lor")
    return table, table_now, np.asarray(poles)


def run_fkpt(k, pk, pknow, **mg):
    import jax.numpy as jnp
    import numpy as np
    from fkptjax.pipelines import full_method_poles

    x, w = np.polynomial.legendre.leggauss(6)
    mu, wmu = 0.5 * (x + 1.0), 0.5 * w   # project_to_poles sums (2l+1) w P L over mu in [0, 1]
    poles, state = full_method_poles(
        k=k, pk=pk, pk_now=pknow, jac=1.0, kap=jnp.asarray(k)[:, None], muap=jnp.asarray(mu)[None, :],
        pars=jnp.asarray(PARS), mu=mu, wmu=wmu, z=COSMO["z"], Om=COSMO["Omega_m"], **mg)
    return state.table_w, state.table_now, np.asarray(poles)


def run_case(name):
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)

    opts = dict(CASES[name])
    engine = opts.pop("engine")
    k, pk, pknow = load_inputs()
    if engine == "folps":
        table, table_now, poles = run_folps(k, pk, pknow, **opts)
    else:
        table, table_now, poles = run_fkpt(k, pk, pknow, **opts)
    out = dict(k_in=k, pk_in=pk, pknow_in=pknow, poles=poles, pars=np.asarray(PARS))
    out.update({f"table_{i:02d}": np.asarray(c) for i, c in enumerate(table)})
    out.update({f"table_now_{i:02d}": np.asarray(c) for i, c in enumerate(table_now)})
    np.savez(GOLDEN / f"{name}.npz", **out)
    finite = all(np.all(np.isfinite(v)) for v in out.values())
    print(f"{name}: {len(table)} + {len(table_now)} table entries, poles {poles.shape}, finite={finite}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_case(sys.argv[1])
    else:
        for name in CASES:
            subprocess.run([sys.executable, __file__, name], check=True)
