"""Compare data/upstream_<case>.npz against data/new_<case>.npz point by point.

Per output (poles / tables / bispectrum) and point: max |new - upstream| / max |upstream|,
the scale taken per row (one multipole, or one table) so columns crossing zero do not blow up.
Writes data/summary.npz and prints a table; exit status 1 if any case fails RTOL.

Usage: python validation/compare.py [RTOL]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usecases import CASES, DATA   # noqa: E402


FLOOR = 1e-6   # rows below FLOOR x the point's largest value are analytically zero (e.g. dP2/d shot noise)


def rel_diff(new, up):
    """(npoint,) worst relative difference; each row (all axes but the last) gets its own scale.

    Row scales are floored at FLOOR x the largest |value| of that point's output, so rows that are
    zero up to rounding (~1e-15 of the largest value) are not reported as O(1) disagreements.
    """
    n, u = new.reshape(len(new), -1, new.shape[-1]), up.reshape(len(up), -1, up.shape[-1])
    scale = np.max(np.abs(u), axis=-1, keepdims=True)
    scale = np.maximum(scale, FLOOR * np.max(scale, axis=1, keepdims=True))
    return np.max(np.abs(n - u) / np.where(scale > 0, scale, 1.), axis=(1, 2))


def main(rtol=1e-7):
    rtol, summary, ok = float(rtol), {}, True
    print(f"{'case':22s} {'points':>6s} {'failed up/new':>13s} {'output':>10s} {'median':>9s} {'p99':>9s} "
          f"{'max':>9s} {'worst pt':>8s} {'time up/new [s]':>16s}")
    for case in CASES:
        try:
            up, new = np.load(DATA / f"upstream_{case}.npz"), np.load(DATA / f"new_{case}.npz")
        except FileNotFoundError:
            print(f"{case:22s} missing")
            ok = False
            continue
        if len(next(up[o] for o in up.files if o not in ("failed", "seconds"))) != \
                len(next(new[o] for o in new.files if o not in ("failed", "seconds"))):
            print(f"{case:22s} point counts differ (a run is still going)")
            ok = False
            continue
        fail = f"{len(up['failed'])}/{len(new['failed'])}"
        outputs = [o for o in up.files if o not in ("failed", "seconds")]
        for o in outputs:
            d = rel_diff(new[o], up[o])
            npt = len(d)
            both_nan = np.isnan(d)
            dd = np.where(both_nan, np.inf, d)
            summary[f"{case}/{o}"] = d
            print(f"{case:22s} {npt:6d} {fail:>13s} {o:>10s} {np.median(dd):9.1e} {np.percentile(dd, 99):9.1e} "
                  f"{np.max(dd):9.1e} {int(np.argmax(dd)):8d} {float(up['seconds']):7.0f}/{float(new['seconds']):<7.0f}")
            ok &= bool(np.max(dd) <= rtol) and fail == "0/0"
    np.savez(DATA / "summary.npz", **summary)
    print(f"\n{'PASS' if ok else 'FAIL'} at rtol {rtol:g}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
