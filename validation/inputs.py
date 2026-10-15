"""Draw the validation points and compute their linear power spectra with CLASS.

One Latin hypercube of N cosmologies (shared by every use case, so a disagreement can be
traced to a cosmology) plus, per use case, an independent hypercube of nuisance / MG
parameters. Writes validation/data/inputs.npz.

Usage: python validation/inputs.py [N] [NPROC]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import qmc

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261014
K = np.geomspace(1e-4, 2.0, 1000)   # same span and size as tests/golden/inputpkT.txt

COSMO_RANGES = dict(h=(0.62, 0.74), Omega_m=(0.26, 0.37), omega_b=(0.020, 0.0245), n_s=(0.92, 1.0),
                    sigma8=(0.70, 0.90), m_ncdm=(0.06, 0.30), z=(0.1, 1.6))
# FOLPS pars order: b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG_p
PK_RANGES = [(1.0, 2.5), (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-50., 50.), (-50., 50.), (-50., 50.),
             (-5.0, 5.0), (-1.0, 1.0), (-10., 10.), (1e3, 1e4), (0.0, 3.0)]
# bispectrum pars: b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b
BK_RANGES = [(1.0, 2.5), (-2.0, 2.0), (-1.0, 1.0), (-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 3.0)]
AP_RANGES = [(0.95, 1.05), (0.95, 1.05)]   # qpar, qper
MG_RANGES = dict(log10_fR0=(-6.5, -4.0), mu0=(-0.3, 0.6))


def lhs(ranges, n, seed):
    lo, hi = np.array(ranges).T
    return qmc.scale(qmc.LatinHypercube(d=len(ranges), seed=seed).random(n), lo, hi)


def linear(row):
    from cosmoprimo import Cosmology
    p = dict(zip(COSMO_RANGES, row))
    z = p.pop("z")
    c = Cosmology(engine="class", m_ncdm=[p.pop("m_ncdm")], **p)
    pk = c.get_fourier().pk_interpolator(of="delta_cb")(K, z=z)
    f0 = c.get_fourier().sigma8_z(z, of="theta_cb") / c.get_fourier().sigma8_z(z, of="delta_cb")
    fnu = c["Omega_ncdm_tot"] / c["Omega_m"]
    return pk, f0, fnu


def main(n=1000, nproc=32):
    cosmo = lhs(list(COSMO_RANGES.values()), n, SEED)
    with Pool(nproc) as pool:
        out = pool.map(linear, cosmo, chunksize=4)
    pk, f0, fnu = (np.array(x) for x in zip(*out))
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "inputs.npz", k=K, pk=pk, f0=f0, fnu=fnu, cosmo=cosmo, cosmo_names=list(COSMO_RANGES),
             pk_pars=lhs(PK_RANGES, n, SEED + 1), bk_pars=lhs(BK_RANGES, n, SEED + 2),
             ap=lhs(AP_RANGES, n, SEED + 4), mg=lhs(list(MG_RANGES.values()), n, SEED + 3), mg_names=list(MG_RANGES))
    print(f"{n} points; f0 {f0.min():.3f}-{f0.max():.3f}, fnu {fnu.min():.4f}-{fnu.max():.4f}, "
          f"finite={np.all(np.isfinite(pk))}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(*(int(a) for a in sys.argv[1:]))
