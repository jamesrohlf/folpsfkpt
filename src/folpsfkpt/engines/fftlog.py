"""FFTLog loop engine: FOLPS's precomputed M22/M13 matrices."""
from ..table import LoopTable


class FFTLogEngine:
    """Wraps FOLPS ``NonLinearPowerSpectrumCalculator``; matrices are built once at setup.

    Note: FOLPS's ``MatrixCalculator`` sets module globals (A_full, use_TNS_model)
    that its table builder reads, so this engine re-asserts them before each call.
    """

    def __init__(self, A_full=True, use_TNS_model=False, kernels="fk", rbao=104., nfftlog=128):
        from folps import MatrixCalculator
        self.A_full, self.use_TNS_model = A_full, use_TNS_model
        self.kernels, self.rbao = kernels, rbao
        self._matrix_calc = MatrixCalculator(nfftlog=nfftlog, A_full=A_full, use_TNS_model=use_TNS_model)
        self.mmatrices = self._matrix_calc.get_mmatrices()

    def __call__(self, k, pk, pknow=None, **cosmo):
        import folps.folps as _folps
        from folps import NonLinearPowerSpectrumCalculator
        _folps.A_full_status, _folps.use_TNS_model_status = self.A_full, self.use_TNS_model
        nl = NonLinearPowerSpectrumCalculator(mmatrices=self.mmatrices, kernels=self.kernels,
                                              rbao=self.rbao, **cosmo)
        table, table_now = nl.calculate_loop_table(k=k, pklin=pk, pknow=pknow, cosmo=None, **cosmo)
        return LoopTable.from_folps_tuples(table, table_now, use_TNS_model=self.use_TNS_model)
