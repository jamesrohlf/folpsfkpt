"""Record desilike FOLPS / fkpt tracer outputs, to compare before and after the step-7 changes.

Usage: python desilike_check/outputs.py OUT.npz [CASE ...]
Runs on CPU; EH template so no Boltzmann code is needed.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__import__("desilike").__file__), "theories", "tests"))
from test_full_shape import _compile, _direct_template   # noqa: E402

K2 = np.linspace(0.02, 0.3, 30)
K3 = np.column_stack([np.linspace(0.02, 0.15, 8)] * 2)


def cases():
    from desilike.theories.galaxy_clustering import (FKPTJAXTracerSpectrum2Poles, FKPTJAXTracerSpectrum3Poles,
                                                     FOLPSTracerSpectrum2Poles, FOLPSTracerSpectrum3Poles)
    return {
        "folps_p2": lambda: FOLPSTracerSpectrum2Poles(k=K2, template=_direct_template()),
        "folps_p2_noafull": lambda: FOLPSTracerSpectrum2Poles(k=K2, template=_direct_template(), A_full=False),
        "folps_p2_tns": lambda: FOLPSTracerSpectrum2Poles(k=K2, template=_direct_template(), remove_DeltaP=True),
        "folps_p3": lambda: FOLPSTracerSpectrum3Poles(k=K3, template=_direct_template()),
        "fkpt_p2": lambda: FKPTJAXTracerSpectrum2Poles(k=K2, template=_direct_template()),
        "fkpt_p3": lambda: FKPTJAXTracerSpectrum3Poles(k=K3, template=_direct_template()),
        # smearing goes through desilike's bispectrum patch; fkpt uses BispectrumCalculator_fk
        "folps_p3_smear": lambda: FOLPSTracerSpectrum3Poles(k=K3, template=_direct_template(), redshift_smearing=_smear),
        # explicit pt: without it FKPTJAXTracerSpectrum3Poles hands redshift_smearing to the pt, which drops it
        "fkpt_p3_smear": lambda: FKPTJAXTracerSpectrum3Poles(k=K3, pt=_fkpt_pt(), redshift_smearing=_smear),
    }


def _fkpt_pt():
    from desilike.theories.galaxy_clustering.full_shape import FKPTJAXPTSpectrum2Poles
    return FKPTJAXPTSpectrum2Poles(template=_direct_template())


def _smear(kmu):
    import jax.numpy as jnp
    return jnp.exp(-jnp.abs(kmu) * 3.)


if __name__ == "__main__":
    out, names = sys.argv[1], sys.argv[2:]
    todo = cases()
    results = {}
    for name in names or list(todo):
        try:
            results[name] = np.asarray(_compile(todo[name]())())
            print(f"{name}: shape {results[name].shape} finite={np.all(np.isfinite(results[name]))}")
        except Exception as exc:   # record failures instead of stopping
            print(f"{name}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
    np.savez(out, **results)
