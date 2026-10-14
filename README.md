# folpsfkpt

Shared pipeline for FOLPS and fkpt: common preprocessing and RSD/bispectrum
postprocessing, with switchable growth (EH + massive neutrinos | MG growth ODE)
and loop engine (FFTLog matrices | Q/R quadrature).

Design notes: https://claude.ai/code/artifact/127b7fff-b0d1-4a12-a01b-e3f25ae2b797

## Regression references

`tests/make_golden.py` records loop tables and P_ell from unmodified upstream
cosmodesi/FolpsD @ 8389414 and gqcristhian/fkptjax_muMG @ 1caa757. Reruns agree
to ~1e-9 relative (GPU, not bit-exact), so compare with rtol ~1e-7.
