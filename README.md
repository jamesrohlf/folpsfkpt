# folpsfkpt

Shared pipeline for FOLPS and fkpt: common preprocessing and RSD/bispectrum
postprocessing, with switchable growth (EH + massive neutrinos | MG growth ODE)
and loop engine (FFTLog matrices | Q/R quadrature).

Design notes: https://claude.ai/code/artifact/127b7fff-b0d1-4a12-a01b-e3f25ae2b797

## Regression references

`tests/make_golden.py` records loop tables and P_ell from unmodified upstream
cosmodesi/FolpsD @ 8389414 and gqcristhian/fkptjax_muMG @ 1caa757. Reruns agree
to ~1e-9 relative (GPU, not bit-exact), so compare with rtol ~1e-7.

## Engine cross-check (`crosscheck/engines.py`, `tests/test_crosscheck.py`)

FFTLog (FOLPS) vs quadrature (fkpt) with identical P_lin, P_nw, f(k), f0, EdS kernel
constants and output grid, through the same RSD code. Over k = 0.01-0.3 h/Mpc the
multipoles agree to 1.7e-4 (P0), 4.7e-4 (P2), 1.2e-3 (P4) of max |P_ell| (FOLPS-D,
lor). Loop columns agree to <= 2.8% at default resolution, <= 0.8% with both engines
refined; differences track each engine's own resolution change, except P13-type terms,
which fkpt truncates at p_max = 16 k_max(output). With k_max = 0.5 they are ~1% low at
k = 0.45; with k_max = 2 (p_max = 32 h/Mpc) ~0.1%. The full suite runs on CPU with
`JAX_PLATFORMS=cpu` (the shared login-node GPU can fail to allocate FFT plans).
