"""Shared postprocessing: LoopTable -> P(k, mu) -> P_ell(k), any engine, any RSD model.

The RSD model and FoG damping are explicit options here, independent of the loop
engine. Upstream they were coupled: fkptjax hardcodes FOLPS's ``EFT`` model,
which drops any damping silently. Here inconsistent combinations raise instead.
"""
import numpy as np

MODELS = ("EFT", "TNS", "FOLPSD")
DAMPINGS = (None, "lor", "exp", "vdg")


def check_rsd_options(table, model, damping):
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if damping not in DAMPINGS:
        raise ValueError(f"damping must be one of {DAMPINGS}, got {damping!r}")
    if model == "EFT" and damping is not None:
        raise ValueError("model='EFT' has no FoG damping; pass damping=None or use model='FOLPSD'")
    if model == "FOLPSD" and damping is None:
        raise ValueError("model='FOLPSD' needs damping='lor', 'exp' or 'vdg'")
    if model == "TNS" and not table.use_TNS_model:
        raise ValueError("model='TNS' needs a table built with use_TNS_model=True")


def _folps_calculator(table, model):
    import folps.folps as _folps
    from folps import RSDMultipolesPowerSpectrumCalculator
    # FOLPS RSD still reads module globals; set them from the table's own flags.
    _folps.A_full_status, _folps.use_TNS_model_status = table.A_full, table.use_TNS_model
    return RSDMultipolesPowerSpectrumCalculator(model=model)


def pkmu(table, kap, muap, pars, *, model="FOLPSD", damping="lor", bias_scheme="folps",
         IR_resummation=True, damping_method=None, use_GTNS=None):
    """P(k, mu) at (already AP-distorted) ``kap``, ``muap``; no AP Jacobian applied."""
    check_rsd_options(table, model, damping)
    calc = _folps_calculator(table, model)
    t, t_now = table.to_folps_tuples()
    pars = calc.set_bias_scheme(pars, bias_scheme=bias_scheme)
    return calc.get_rsd_pkmu(kap, muap, pars, t, t_now, IR_resummation, damping,
                             damping_method=damping_method, use_GTNS=use_GTNS)


def pk_multipoles(table, k, pars, *, model="FOLPSD", damping="lor", qpar=1., qper=1., ells=(0, 2, 4),
                  nmu=6, bias_scheme="folps", IR_resummation=True, damping_method=None, use_GTNS=None):
    """P_ell(k) with FOLPS's AP mapping and mu quadrature (nmu positive Gauss-Legendre nodes)."""
    check_rsd_options(table, model, damping)
    calc = _folps_calculator(table, model)
    t, t_now = table.to_folps_tuples()
    return calc.get_rsd_pkell(kobs=k, qpar=qpar, qper=qper, pars=pars, table=t, table_now=t_now,
                              bias_scheme=bias_scheme, damping=damping, nmu=nmu, ells=tuple(ells),
                              IR_resummation=IR_resummation, damping_method=damping_method,
                              use_GTNS=use_GTNS)


def project_to_multipoles(pkmu_grid, mu, wmu, ells=(0, 2, 4)):
    """(2 ell + 1) sum_mu w L_ell(mu) P(k, mu), for nodes/weights on mu in [0, 1] with sum(w) = 1."""
    from numpy.polynomial.legendre import Legendre
    return np.array([(2 * ell + 1) * np.sum(wmu * np.asarray(pkmu_grid) * Legendre.basis(ell)(mu), axis=-1)
                     for ell in ells])
