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
    # ---- options the cases above leave at their defaults
    "opt_folps_eager": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor", jit=False),
    "opt_bias_classpt": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor", bias_scheme="classpt"),
    "opt_bias_desi": dict(kind="folps", A_full=False, model="FOLPSD", damping="exp", bias_scheme="desi"),
    "opt_no_ir": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor", IR_resummation=False),
    "opt_eds_kernels": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor", kernels="EdS"),
    "opt_loopctr_gtns": dict(kind="folps", A_full=True, model="FOLPSD", damping="lor", damping_method="loop+ctr",
                             use_GTNS=True),
    "opt_all_damped": dict(kind="folps", A_full=True, model="FOLPSD", damping="vdg", damping_method="all"),
    "opt_tns_noafull": dict(kind="folps", A_full=False, use_TNS_model=True, model="TNS", damping="lor"),
    # TNS tables evaluated with FOLPSD: desilike's remove_DeltaP=True; upstream relied on the module global
    "opt_tns_tables_folpsd": dict(kind="folps", A_full=True, use_TNS_model=True, model="FOLPSD", damping="lor"),
    "marg_folpsd": dict(kind="marg", A_full=True, model="FOLPSD", damping="lor"),
    "marg_tns_tables_folpsd": dict(kind="marg", A_full=True, use_TNS_model=True, model="FOLPSD", damping="lor"),
    "marg_eft_noafull": dict(kind="marg", A_full=False, model="EFT", damping=None),
    "fkpt_rescale_jax": dict(kind="fkpt", builder="Kfuncs_to_tables_rescale_jax",
                             opts=dict(model="HDKI", mg_variant="mu_OmDE"), mg="mu0"),
    "fkpt_ndgp": dict(kind="fkpt", opts=dict(model="NDGP"), mg="r_c"),
    "fkpt_hs_beyond_eds": dict(kind="fkpt", opts=dict(model="HS", beyond_eds=True), mg="fR0_HS"),
    "fkpt_hs_full_kernels": dict(kind="fkpt", opts=dict(model="HS", beyond_eds=True, fkpt_approximation=False),
                                 mg="fR0_HS"),
    "fkpt_growth_index_jax": dict(kind="fkpt", builder="Kfuncs_to_tables_jax",
                                  opts=dict(model="PHENOM", mg_variant="growth_index"), mg="gamma_0"),
    "bk_folps_scoccimarro": dict(kind="bk", source="folps", basis="scoccimarro", damping="lor"),
    "bk_fkpt_mg_kernel_fn": dict(kind="bk", source="fkpt", basis="sugiyama", damping="lor", mg_kernel=True),
    # ---- derivatives: jax.jacfwd through the whole pipeline
    "grad_folps": dict(kind="grad_folps", A_full=True, model="FOLPSD", damping="lor"),
    "grad_folps_tns": dict(kind="grad_folps", A_full=True, use_TNS_model=True, model="TNS", damping="lor"),
    "grad_fkpt_jax": dict(kind="grad_fkpt", opts=dict(model="HDKI", mg_variant="mu_OmDE")),
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
        u = (mg["mu0"] + 0.3) / 0.9   # the mu0 column, rescaled to [0, 1], seeds the other MG cases
        self.mg = dict(fR0_HS=10 ** mg["log10_fR0"], mu0=mg["mu0"], r_c=10 ** (2. * u - 0.5), gamma_0=0.45 + 0.2 * u)
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

def folps_pipeline(impl, A_full, model, damping, use_TNS_model=False, kernels="fk", **rsd):
    """(k, pk, pknow, pars, qpar, qper, cosmo) -> (table, table_now, poles); rsd: bias_scheme, IR_resummation, ..."""
    rsd.setdefault("bias_scheme", "folps")
    if impl == "upstream":
        from folps import (MatrixCalculator, NonLinearPowerSpectrumCalculator,
                           RSDMultipolesPowerSpectrumCalculator)
        mm = MatrixCalculator(A_full=A_full, use_TNS_model=use_TNS_model).get_mmatrices()
        calc = RSDMultipolesPowerSpectrumCalculator(model=model)

        def run(k, pk, pknow, pars, qpar, qper, cosmo):
            nl = NonLinearPowerSpectrumCalculator(mmatrices=mm, kernels=kernels, **cosmo)
            table, table_now = nl.calculate_loop_table(k=k, pklin=pk, pknow=pknow, cosmo=None, **cosmo)
            poles = calc.get_rsd_pkell(kobs=jnp.asarray(KOBS), qpar=qpar, qper=qper, pars=pars, table=table,
                                       table_now=table_now, damping=damping, nmu=NMU, **rsd)
            return table, table_now, poles
    else:
        from folpsfkpt import FFTLogEngine, pk_multipoles
        engine = FFTLogEngine(A_full=A_full, use_TNS_model=use_TNS_model, kernels=kernels)

        def run(k, pk, pknow, pars, qpar, qper, cosmo):
            lt = engine(k, pk, pknow, **cosmo)
            poles = pk_multipoles(lt, jnp.asarray(KOBS), pars, model=model, damping=damping, qpar=qpar, qper=qper,
                                  nmu=NMU, **rsd)
            return (*lt.to_folps_tuples(), poles)
    return run


def make_folps(impl, jit=True, **options):
    run = folps_pipeline(impl, **options)
    if jit:
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
    to_poles = jax.jit(fkpt_to_poles(impl))

    def point(p):
        table, table_now, _ = fkpt_tables(impl, p, builder, opts, mg)
        poles = to_poles(tuple(map(jnp.asarray, table)), tuple(map(jnp.asarray, table_now)), jnp.asarray(p.pars),
                         p.qpar, p.qper)
        return dict(poles=poles, table=np.concatenate([np.atleast_1d(t) for t in table]),
                    table_now=np.concatenate([np.atleast_1d(t) for t in table_now]))
    return point


# ---------------------------------------------------------------- bispectrum

def fkpt_to_poles(impl):
    """(table, table_now, pars, qpar, qper) -> EFT poles: fkptjax's own projection upstream, folpsfkpt's here."""
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
    return to_poles


def make_marg(impl, A_full, model, damping, use_TNS_model=False):
    """FOLPS's analytic-marginalisation helpers on the case's tables (TNS flag: global upstream, explicit here)."""
    import folps
    run_tables = jax.jit(folps_pipeline(impl, A_full, model, damping, use_TNS_model=use_TNS_model))
    flag = {} if impl == "upstream" else dict(use_TNS_model=use_TNS_model)

    @jax.jit
    def marg(table, table_now, pars, qpar, qper):
        kw = dict(kobs=jnp.asarray(KOBS), qpar=qpar, qper=qper, pars=pars, table=table, table_now=table_now,
                  bias_scheme="folps", damping=damping, nmu=NMU, model=model, **flag)
        return folps.get_rsd_pkell_marg_const(**kw), folps.get_rsd_pkell_marg_derivatives(**kw)

    def point(p):
        table, table_now, _ = run_tables(jnp.asarray(p.k), jnp.asarray(p.pk), jnp.asarray(p.pknow),
                                         jnp.asarray(p.pars), p.qpar, p.qper,
                                         {n: jnp.asarray(v) for n, v in p.cosmo.items()})
        const, deriv = marg(table, table_now, jnp.asarray(p.pars), p.qpar, p.qper)
        return dict(const=jnp.asarray(const), deriv=jnp.stack([jnp.asarray(d) for d in deriv]))
    return point


def make_grad_folps(impl, **options):
    """d poles / d (P_lin amplitude, f0, Omega_m, fnu, qpar, qper) and d poles / d nuisances.

    Reverse mode: FOLPS's pipeline contains custom_vjp functions, which jacfwd can not differentiate.
    """
    run = folps_pipeline(impl, **options)

    def poles(theta, pars, k, pk, pknow, cosmo):
        amp, f0, om, fnu, qpar, qper = theta
        cosmo = dict(cosmo, f0=f0, Omega_m=om, fnu=fnu)
        return run(k, amp * pk, amp * pknow, pars, qpar, qper, cosmo)[2]
    jac = jax.jit(jax.jacrev(poles, argnums=(0, 1)))

    def point(p):
        c = p.cosmo
        theta = jnp.array([1., c["f0"], c["Omega_m"], c["fnu"], p.qpar, p.qper])
        d_theta, d_pars = jac(theta, jnp.asarray(p.pars), jnp.asarray(p.k), jnp.asarray(p.pk), jnp.asarray(p.pknow),
                              {n: jnp.asarray(v) for n, v in c.items()})
        return dict(d_theta=jnp.moveaxis(d_theta, -1, 0), d_pars=jnp.moveaxis(d_pars, -1, 0))
    return point


def make_grad_fkpt(impl, opts):
    """d poles / d (mu0, qpar, qper) and d nuisances, through Kfuncs_to_tables_jax.

    The k grid must stay concrete (the builder sets up its quadrature from it): closed over, static_ctx prebuilt.
    """
    from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax
    to_poles = fkpt_to_poles(impl)
    jacs = {}

    def get_jac(k):
        from fkptjax.kfuncs_to_tables import build_jax_static_ctx
        grid = dict(kmin=1e-3, kmax=0.5, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10)
        ctx = build_jax_static_ctx(k, **grid)   # fkptjax's documented route for jit: concrete static pieces

        def poles(theta, pars, pk, pknow, z, om):
            mu0, qpar, qper = theta
            table, table_now = Kfuncs_to_tables_jax(k, pk, pknow, z=z, Om=om, static_ctx=ctx, **grid,
                                                    return_kernel_constants=False, mu0=mu0, **opts)[:2]
            return to_poles(tuple(table), tuple(table_now), pars, qpar, qper)
        return jax.jit(jax.jacfwd(poles, argnums=(0, 1)))

    def point(p):
        key = p.k.tobytes()
        if key not in jacs:
            jacs[key] = get_jac(p.k)
        jac = jacs[key]
        theta = jnp.array([p.mg["mu0"], p.qpar, p.qper])
        d_theta, d_pars = jac(theta, jnp.asarray(p.pars), jnp.asarray(p.pk), jnp.asarray(p.pknow), p.cosmo["z"],
                              p.cosmo["Omega_m"])
        return dict(d_theta=jnp.moveaxis(d_theta, -1, 0), d_pars=jnp.moveaxis(d_pars, -1, 0))
    return point


def toy_mg_kernel(kf, k1, k2):
    """Smooth stand-in for fkptjax's A_B_grid ODE: exercises FolpsD's mg_kernel_fn branch, not fkptjax's solver."""
    x = (k1 * k2) / (kf ** 2 + k1 ** 2 + k2 ** 2)
    return 1. + 0.03 * x, 0.01 * jnp.tanh(kf), 1. - 0.02 * x, 0.005 * jnp.tanh(k1 + k2)


def make_bk(impl, source, basis, damping, mg_kernel=False):
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
                                      interpolation_method="linear", bias_scheme="folps",
                                      **(dict(mg_kernel_fn=toy_mg_kernel) if mg_kernel else {}))
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
    return dict(fkpt=make_fkpt, bk=make_bk, marg=make_marg, grad_folps=make_grad_folps,
                grad_fkpt=make_grad_fkpt)[kind](impl, **c)


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
