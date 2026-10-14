"""LoopTable: the named contract between a loop engine and postprocessing.

FOLPS and fkptjax both exchange loop terms as positional tuples whose layout
depends on A_full (36/38 entries vs 30/32). This module gives every slot a name
and converts losslessly to and from that layout, so engines and postprocessing
can be swapped without index arithmetic.

Bias-loop names are explicit about which nuisances multiply them, because the
two codes disagree: FOLPS calls the bs2^2 term ``Pb2s2`` (slot 10), while
fkptjax uses ``Pb2s2`` for the b2*bs2 term (slot 9). Here they are
``P_b2bs2`` and ``P_bs2bs2``.
"""
from dataclasses import dataclass, fields
from typing import Any, NamedTuple, Optional

import jax

# Positional order of the k-dependent columns after k (FOLPS combine_loop_terms).
CORE_TERMS = (
    "pk_lin", "f_over_f0",
    "P_dd", "P_dt", "P_tt",
    "P_b1b2", "P_b1bs2", "P_b2b2", "P_b2bs2", "P_bs2bs2",
    "sigma3_pk", "P_b2t", "P_bs2t",
    "I1udd_1", "I2uud_1", "I2uud_2", "I3uuu_2", "I3uuu_3",
    "I2uudd_1D", "I2uudd_2D", "I3uuud_2D", "I3uuud_3D",
    "I4uuuu_2D", "I4uuuu_3D", "I4uuuu_4D",
    "I3uuud_1B", "I4uuuu_1B",
)
# Extra columns present only with A_full=True (b2 and bs2 parts of the A function).
A_FULL_TERMS = (
    "I1udd_1_b2", "I2uud_1_b2", "I2uud_2_b2",
    "I1udd_1_bs2", "I2uud_1_bs2", "I2uud_2_bs2",
)


class LoopTerms(NamedTuple):
    """Loop terms on the output k-grid, for one of the wiggle / no-wiggle spectra."""
    pk_lin: Any
    f_over_f0: Any
    P_dd: Any
    P_dt: Any
    P_tt: Any
    P_b1b2: Any
    P_b1bs2: Any
    P_b2b2: Any
    P_b2bs2: Any
    P_bs2bs2: Any
    sigma3_pk: Any
    P_b2t: Any
    P_bs2t: Any
    I1udd_1: Any
    I2uud_1: Any
    I2uud_2: Any
    I3uuu_2: Any
    I3uuu_3: Any
    I2uudd_1D: Any
    I2uudd_2D: Any
    I3uuud_2D: Any
    I3uuud_3D: Any
    I4uuuu_2D: Any
    I4uuuu_3D: Any
    I4uuuu_4D: Any
    I3uuud_1B: Any
    I4uuuu_1B: Any
    sigma2w: Any
    # A_full-only terms; None when the engine does not compute them.
    I1udd_1_b2: Optional[Any] = None
    I2uud_1_b2: Optional[Any] = None
    I2uud_2_b2: Optional[Any] = None
    I1udd_1_bs2: Optional[Any] = None
    I2uud_1_bs2: Optional[Any] = None
    I2uud_2_bs2: Optional[Any] = None


@dataclass(frozen=True)
class LoopTable:
    """Engine output: loop terms for the wiggle and no-wiggle spectra plus scalars.

    ``A_full`` and ``use_TNS_model`` are static metadata (not traced by jax), so
    postprocessing can branch on them instead of on FOLPS module globals.
    """
    k: Any
    wiggle: LoopTerms
    nowiggle: LoopTerms
    sigma2_nw: Any
    delta_sigma2_nw: Any
    f0: Any
    kernel_constants: Optional[Any] = None   # (A, Ap, CFD3, CFD3p) from beyond-EdS engines
    A_full: bool = False
    use_TNS_model: bool = False

    @classmethod
    def from_folps_tuples(cls, table, table_now, *, use_TNS_model=False, kernel_constants=None):
        """Build from the positional (table, table_now) pair used by FOLPS and fkptjax."""
        ncore = 1 + len(CORE_TERMS)
        if len(table) == ncore + 2:
            A_full = False
        elif len(table) == ncore + len(A_FULL_TERMS) + 2:
            A_full = True
        else:
            raise ValueError(f"unexpected FOLPS table length {len(table)}")
        if len(table_now) != len(table) + 2:
            raise ValueError(f"table_now has {len(table_now)} entries, expected {len(table) + 2}")
        names = CORE_TERMS + (A_FULL_TERMS if A_full else ())

        def terms(t):
            cols = dict(zip(names, t[1:1 + len(names)]))
            return LoopTerms(sigma2w=t[1 + len(names)], **cols)

        return cls(k=table[0], wiggle=terms(table), nowiggle=terms(table_now),
                   sigma2_nw=table_now[-3], delta_sigma2_nw=table_now[-2], f0=table[-1],
                   kernel_constants=kernel_constants, A_full=A_full, use_TNS_model=use_TNS_model)

    def to_folps_tuples(self):
        """Inverse of :meth:`from_folps_tuples`."""
        names = CORE_TERMS + (A_FULL_TERMS if self.A_full else ())
        cols_w = tuple(getattr(self.wiggle, n) for n in names)
        cols_nw = tuple(getattr(self.nowiggle, n) for n in names)
        table = (self.k, *cols_w, self.wiggle.sigma2w, self.f0)
        table_now = (self.k, *cols_nw, self.nowiggle.sigma2w, self.sigma2_nw, self.delta_sigma2_nw, self.f0)
        return table, table_now


jax.tree_util.register_dataclass(
    LoopTable,
    data_fields=[f.name for f in fields(LoopTable) if f.name not in ("A_full", "use_TNS_model")],
    meta_fields=["A_full", "use_TNS_model"],
)
