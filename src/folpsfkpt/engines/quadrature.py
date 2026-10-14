"""Quadrature loop engine: fkptjax's Q/R integrals with its own growth ODE."""
from ..table import LoopTable


class QuadratureEngine:
    """Wraps fkptjax ``Kfuncs_to_tables``. Produces the A_full=False layout only.

    ``options`` (model, mg_variant, fR0_HS, rescale_PS, beyond_eds, Nk_kernel, ...)
    are forwarded to ``Kfuncs_to_tables`` unchanged.
    """
    A_full = False
    use_TNS_model = False

    def __init__(self, **options):
        self.options = options

    def __call__(self, k, pk, pknow, *, z, Omega_m, **unused_cosmo):
        from fkptjax.kfuncs_to_tables import Kfuncs_to_tables
        table_w, table_now, kcs = Kfuncs_to_tables(k, pk, pknow, z=z, Om=Omega_m,
                                                   return_kernel_constants=True, **self.options)
        return LoopTable.from_folps_tuples(table_w, table_now, kernel_constants=kcs)
