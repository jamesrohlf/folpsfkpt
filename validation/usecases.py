"""Use cases run point by point through either the upstream codes or the combined package.

``upstream`` calls FolpsD / fkptjax exactly as a user of origin/main would (put the upstream
worktrees first on PYTHONPATH); ``new`` goes through folpsfkpt (LoopTable, the two engines,
the shared RSD code) on top of the unified-pipeline branches. The two can not share a process:
both import the same module names.

Usage: python validation/usecases.py {upstream,new} CASE [N]   -> data/<impl>_<CASE>.npz
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).resolve().parent / "data"
KOBS = np.linspace(0.01, 0.3, 50)
BK_K = np.linspace(0.02, 0.17, 16)
NMU = 6

# Each case: engine options, RSD model / damping, and which MG parameter (if any) it varies.
CASES = {
    "folps_afull_lor": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor"),
    "folps_noafull_exp": dict(kind="folps", A_full=False, model="FOLPSD", damping="exp"),
    "folps_afull_vdg": dict(kind="folps", A_full=True, model="FOLPSD", damping="vdg"),
    "folps_eft": dict(kind="folps", A_full=True, model="EFT", damping=None),
    "folps_tns": dict(kind="folps", A_full=True, use_TNS_model=True, model="TNS", damping="lor"),
    "fkpt_lcdm": dict(kind="fkpt", opts=dict(model="LCDM")),
    "fkpt_hs": dict(kind="fkpt", opts=dict(model="HS", rescale_PS=True), mg="fR0_HS"),
    "fkpt_mu0_jax": dict(kind="fkpt", builder="Kfuncs_to_tables_jax",
                         opts=dict(model="HDKI", mg_variant="mu_OmDE"), mg="mu0"),
    "bk_folps_sugiyama": dict(kind="bk", source="folps", basis="sugiyama", damping="lor"),
    "bk_fkpt_sugiyama": dict(kind="bk", source="fkpt", basis="sugiyama", damping="exp"),
    "bk_fkpt_scoccimarro": dict(kind="bk", source="fkpt", basis="scoccimarro", damping="lor"),
}


class Point:
    """Inputs of point i: cosmology, linear spectra, nuisances and MG parameters."""

    def __init__(self, inp, i):
        self.k, self.pk = inp["k"], inp["pk"][i]
        cosmo = dict(zip(inp["cosmo_names"], inp["cosmo"][i]))
        self.cosmo = dict(z=cosmo["z"], h=cosmo["h"], Omega_m=cosmo["Omega_m"], f0=float(inp["f0"][i]),
                          fnu=float(inp["fnu"][i]))
        self.pars, self.bpars = inp["pk_pars"][i], inp["bk_pars"][i]
        self.qpar, self.qper = inp["ap"][i]
        mg = dict(zip(inp["mg_names"], inp["mg"][i]))
        self.mg = dict(fR0_HS=10 ** mg["log10_fR0"], mu0=mg["mu0"])
        self._pknow = None

    @property
    def pknow(self):
        """FOLPS's no-wiggle spectrum, computed by the implementation under test."""
        if self._pknow is None:
            from folps import extrapolate_pklin, get_pknow_jax
            k_ext, pk_ext = extrapolate_pklin(k=self.k, pk=self.pk)
            k_now, pknow = get_pknow_jax(k=jnp.asarray(k_ext), pk=jnp.asarray(pk_ext), h=self.cosmo["h"])
            self._pknow = np.interp(self.k, np.asarray(k_now), np.asarray(pknow))
        return self._pknow


def mu_nodes():
    """FOLPS's mu nodes on [0, 1]: positive half of a 2*NMU Gauss-Legendre rule, weights symmetrised."""
    x, w = np.polynomial.legendre.leggauss(2 * NMU)
    return x[NMU:], (w[NMU:] + w[NMU - 1::-1]) / 2.


# ---------------------------------------------------------------- FOLPS power spectrum

def make_folps(impl, A_full, model, damping, use_TNS_model=False):
    if impl == "upstream":
        from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator,
                           RSDMultipolesPowerSpectrumCalculator)
        mm = MatrixCalculator(A_full=A_full, use_TNS_model=use_TNS_model).get_mmatrices()
        rsd = RSDMultipolesPowerSpectrumCalculator(model=model)

        def run(k, pk, pknow, pars, qpar, qper, cosmo):
            nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels="fk", **cosmo)
            table, table_now = nl.calculate_loop_table(k=k, pklin=pk, pknow=pknow, cosmo=None, **cosmo)
            poles = rsd.get_rsd_pkell(kobs=jnp.asarray(KOBS), qpar=qpar, qper=qper, pars=pars, table=table,
                                      table_now=table_now, bias_scheme="folps", damping=damping, nmu=NMU)
            return table, table_now, poles
    else:
        from folpsfkpt import FFTLogEngine, pk_multipoles
        engine = FFTLogEngine(A_full=A_full, use_TNS_model=use_TNS_model)

        def run(k, pk, pknow, pars, qpar, qper, cosmo):
            lt = engine(k, pk, pknow, **cosmo)
            poles = pk_multipoles(lt, jnp.asarray(KOBS), pars, model=model, damping=damping, qpar=qpar, qper=qper,
                                  nmu=NMU)
            return (*lt.to_folps_tuples(), poles)

    run = jax.jit(run)

    def point(p):
        table, table_now, poles = run(jnp.asarray(p.k), jnp.asarray(p.pk), jnp.asarray(p.pknow), jnp.asarray(p.pars),
                                      p.qpar, p.qper, {n: jnp.asarray(v) for n, v in p.cosmo.items()})
        return dict(poles=poles, table=jnp.concatenate([jnp.atleast_1d(t) for t in table]),
                    table_now=jnp.concatenate([jnp.atleast_1d(t) for t in table_now]))
    return point


# ---------------------------------------------------------------- fkpt power spectrum

def fkpt_tables(impl, p, builder, opts, mg):
    """Loop tables (and kernel constants) from fkptjax's builder, upstream or through QuadratureEngine."""
    from fkptjax import kfuncs_to_tables
    kw = dict(opts, **({mg: p.mg[mg]} if mg else {}))
    if builder == "Kfuncs_to_tables_jax":
        kw.update(kmin=1e-3, kmax=0.5)
    if impl == "new" and builder is None:
        from folpsfkpt import QuadratureEngine
        lt = QuadratureEngine(**kw)(p.k, p.pk, p.pknow, **p.cosmo)
        return lt.to_folps_tuples() + (lt.kernel_constants,)
    out = getattr(kfuncs_to_tables, builder or "Kfuncs_to_tables")(p.k, p.pk, p.pknow, z=p.cosmo["z"],
                                                                 Om=p.cosmo["Omega_m"], return_kernel_constants=True,
                                                                 **kw)
    return tuple(out[0]), tuple(out[1]), out[2]


def make_fkpt(impl, opts, builder=None, mg=None):
    mu, wmu = mu_nodes()
    if impl == "upstream":
        from fkptjax.rsd import tables_to_poles

        def to_poles(table, table_now, pars, qpar, qper):
            # fkptjax takes kap / muap from the caller; build them with FOLPS's AP mapping, as desilike does
            F = qpar / qper
            nu = jnp.sqrt(1. + mu[None, :] ** 2 * (1. / F ** 2 - 1.))
            kap, muap = jnp.asarray(KOBS)[:, None] / qper * nu, mu[None, :] / F / nu
            return tables_to_poles(1. / (qpar * qper ** 2), kap, muap, pars, tuple(table) + tuple(table_now), mu, wmu,
                                   ncols=len(table), bias_scheme="folps", IR_resummation=True, damping=None,
                                   ells=(0, 2, 4), A_full=False, use_TNS_model=False)
    else:
        from folpsfkpt import LoopTable, pk_multipoles

        def to_poles(table, table_now, pars, qpar, qper):
            return pk_multipoles(LoopTable.from_folps_tuples(table, table_now), jnp.asarray(KOBS), pars, model="EFT",
                                 damping=None, qpar=qpar, qper=qper, nmu=NMU)
    to_poles = jax.jit(to_poles)

    def point(p):
        table, table_now, _ = fkpt_tables(impl, p, builder, opts, mg)
        poles = to_poles(tuple(map(jnp.asarray, table)), tuple(map(jnp.asarray, table_now)), jnp.asarray(p.pars),
                         p.qpar, p.qper)
        return dict(poles=poles, table=np.concatenate([np.atleast_1d(t) for t in table]),
                    table_now=np.concatenate([np.atleast_1d(t) for t in table_now]))
    return point


# ---------------------------------------------------------------- bispectrum

def make_bk(impl, source, basis, damping):
    import folps
    if source == "folps":
        cls = folps.BispectrumCalculator
    else:
        cls = folps.BispectrumCalculator_fk if impl == "upstream" else folps.BispectrumCalculator
    kwargs = {} if (impl == "upstream" or source == "folps") else dict(scale_dependent_growth=True)
    calc = cls(model="FOLPSD", **kwargs)

    if basis == "sugiyama":
        pairs = jnp.asarray(np.vstack([BK_K, BK_K]).T)

        def run(bpars, kpk, f0, qpar, qper):
            return calc.Sugiyama_Bell(f0, bpars, kpk, pairs, qpar, qper, precision=[8, 10, 10], damping=damping,
                                      multipoles=["B000", "B110", "B220", "B202", "B022", "B112"], renormalize=True,
                                      interpolation_method="linear", bias_scheme="folps")
        run = jax.jit(run)
    else:
        triplets = jnp.asarray(np.vstack([BK_K, BK_K, BK_K]).T)

        def run(bpars, kpk, f0, qpar, qper):
            return calc.Scoccimarro_Bell(triplets, f0, bpars, qpar, qper, kpk, precision=[8, 8], damping=damping,
                                         interpolation_method="linear", multipoles=["B0", "B2", "B4"],
                                         bias_scheme="folps")

    def point(p):
        if source == "folps":
            kpk, f0 = jnp.asarray(np.array([p.k, p.pk, p.pknow])), p.cosmo["f0"]
        else:   # desilike's fkpt route: fkpt HS tables + beyond-EdS constants (full_shape.py:3483)
            table, table_now, kcs = fkpt_tables(impl, p, None, dict(model="HS", rescale_PS=True), "fR0_HS")
            f0 = float(table[-1])
            ones = np.ones_like(np.asarray(table[0]))
            kpk = jnp.asarray(np.array([table[0], table[1], table_now[1], np.asarray(table[2]) * f0,
                                        ones * float(kcs[0]), ones * float(kcs[1])]))
        out = run(jnp.asarray(p.bpars), kpk, f0, p.qpar, p.qper)
        return dict(bk=out)
    return point


def make(impl, case):
    c = dict(CASES[case])
    kind = c.pop("kind")
    if kind == "folps":
        return make_folps(impl, **c)
    if kind == "fkpt":
        return make_fkpt(impl, **c)
    return make_bk(impl, **c)


def main(impl, case, n=None):
    inp = np.load(DATA / "inputs.npz")
    n = int(n or len(inp["pk"]))
    point = make(impl, case)
    results, failed, t0 = {}, [], time.time()
    for i in range(n):
        try:
            out = point(Point(inp, i))
            for name, v in out.items():
                results.setdefault(name, []).append(np.asarray(v))
        except Exception as exc:   # record, keep going: a failure in one code but not the other is a finding
            failed.append(i)
            print(f"point {i}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            for name in results:
                results[name].append(np.full_like(results[name][0], np.nan))
        if i in (0, 9) or (i + 1) % 200 == 0:
            print(f"{impl} {case}: {i + 1}/{n} points, {time.time() - t0:.0f} s", flush=True)
    np.savez(DATA / f"{impl}_{case}.npz", failed=np.array(failed, dtype=int), seconds=time.time() - t0,
             **{name: np.array(v) for name, v in results.items()})
    print(f"{impl} {case}: done, {len(failed)} failed, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
