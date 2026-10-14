"""Shared preprocessing: extrapolation of P_lin and the sigma^2 integrals.

One implementation, in FolpsD, called by FOLPS's table builder and by all of
fkptjax's table builders; re-exported here as the pipeline's entry point.
The IR scalars must be computed from the no-wiggle spectrum *after* any growth
rescaling (fkpt's rescale_PS), so they sit after the growth stage.
"""
from folps import extrapolate_pklin, ir_bao_grid, ir_damping_scalars, sigma2_v

__all__ = ["extrapolate_pklin", "ir_bao_grid", "ir_damping_scalars", "sigma2_v"]
