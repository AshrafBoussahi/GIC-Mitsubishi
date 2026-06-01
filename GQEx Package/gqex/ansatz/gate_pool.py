"""
Gate pool definitions.

Two pool flavours:
  • StandardPool   — original GQE pool (X, Ry±, Rz, CNOT ring/skip)
  • NumberPreservingPool — Givens rotations + fSWAP; all ops preserve
                          total electron count, eliminating ~half of
                          wasted shots (priority-1 improvement).
  • FermionicPool  — single/double excitation operators for ADAPT-GQE

Both inherit from PoolBase which provides the shared pool_size / offsets
interface consumed by the kernel factories and the GQE engine.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Base ─────────────────────────────────────────────────────────────────────

class PoolBase:
    """Minimal interface that every pool must implement."""

    @property
    def pool_size(self) -> int:
        raise NotImplementedError

    def family_offsets(self) -> Dict[str, int]:
        """Map family name → first pool-index (1-based, after identity=0)."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def op_label(self, op_idx: int) -> str:
        """Human-readable label for a pool index (for debugging)."""
        return str(op_idx)


# ─── Standard pool ────────────────────────────────────────────────────────────

@dataclass
class StandardPool(PoolBase):
    """Original GQE discrete gate pool.

    Encoding (in order, each family occupies n contiguous slots):
        0                      identity
        [1..n]                 X  on each qubit
        next n                 Ry(+θ)
        next n                 Ry(-θ)         (if include_ry_neg)
        next n                 Rz(+θ)         (if include_rz)
        next n                 CNOT ring      (if include_cnot)
        next n                 CNOT skip-1    (if include_cnot_lr)
    """
    n_qubits: int
    angle: float = math.pi / 4
    include_x:        bool = True
    include_ry_pos:   bool = True
    include_ry_neg:   bool = True
    include_rz:       bool = False
    include_cnot:     bool = True
    include_cnot_lr:  bool = False

    @property
    def _families(self) -> List[Tuple[str, bool]]:
        return [
            ('x',       self.include_x),
            ('ry_pos',  self.include_ry_pos),
            ('ry_neg',  self.include_ry_neg),
            ('rz',      self.include_rz),
            ('cnot',    self.include_cnot),
            ('cnot_lr', self.include_cnot_lr),
        ]

    @property
    def pool_size(self) -> int:
        n = self.n_qubits
        return 1 + n * sum(1 for _, on in self._families if on)

    def family_offsets(self) -> Dict[str, int]:
        offsets, cur = {}, 1
        for name, on in self._families:
            if on:
                offsets[name] = cur
                cur += self.n_qubits
        return offsets

    def describe(self) -> str:
        offs = self.family_offsets()
        n = self.n_qubits
        s  = f"StandardPool(n={n}, pool_size={self.pool_size}, θ={self.angle:.4f})\n"
        s += f"   op 0          : identity\n"
        for name, off in offs.items():
            s += f"   op {off}..{off+n-1:3d}  : {name}\n"
        return s

    def op_label(self, op_idx: int) -> str:
        if op_idx == 0:
            return 'I'
        offs = self.family_offsets()
        n    = self.n_qubits
        for name, off in offs.items():
            if off <= op_idx and op_idx < off + n:
                return f"{name}[{op_idx - off}]"
        return f"?[{op_idx}]"


# ─── Number-Preserving pool ───────────────────────────────────────────────────

@dataclass
class NumberPreservingPool(PoolBase):
    """Number-preserving gate pool using Givens rotations.

    All operations preserve total electron count (priority-1 improvement).
    This eliminates roughly half of wasted shots that have wrong particle
    number and fundamentally alters the coupon-collector scaling.

    Families (in pool-index order):
        0                       identity
        [1..n_pairs]            Givens+(i, i+1, +θ)  adjacent pairs
        next n_pairs            Givens-(i, i+1, -θ)  adjacent pairs
        next n_pairs_lr         Givens+(i, i+2, +θ)  skip-1  (if lr)
        next n_pairs_lr         Givens-(i, i+2, -θ)  skip-1  (if lr)
        next n-1                fSWAP(i, i+1)         adjacent (if fswap)

    Each Givens(p, q, θ) acts as:
        |00⟩ → |00⟩
        |01⟩ → cos θ|01⟩ + sin θ|10⟩
        |10⟩ → -sin θ|01⟩ + cos θ|10⟩
        |11⟩ → |11⟩
    preserving particle number everywhere.
    """
    n_qubits: int
    angle: float = math.pi / 4
    include_givens_adj:  bool = True   # adjacent-qubit Givens
    include_givens_lr:   bool = False  # skip-1 Givens (longer range)
    include_fswap:       bool = False  # fermionic SWAP (angle=π/2 Givens)
    include_double_exc:  bool = False  # 4-qubit double excitations

    @property
    def _n_adj_pairs(self) -> int:
        return self.n_qubits - 1

    @property
    def _n_lr_pairs(self) -> int:
        return self.n_qubits - 2 if self.n_qubits >= 3 else 0

    @property
    def _n_double_pairs(self) -> int:
        # (0,1,2,3), (2,3,4,5), ... non-overlapping 4-qubit windows
        return self.n_qubits // 4

    @property
    def pool_size(self) -> int:
        n  = 1   # identity
        if self.include_givens_adj:
            n += 2 * self._n_adj_pairs
        if self.include_givens_lr:
            n += 2 * self._n_lr_pairs
        if self.include_fswap:
            n += self._n_adj_pairs
        if self.include_double_exc:
            n += 2 * self._n_double_pairs
        return n

    def family_offsets(self) -> Dict[str, int]:
        offsets: Dict[str, int] = {}
        cur = 1
        if self.include_givens_adj:
            offsets['givens_adj_pos'] = cur; cur += self._n_adj_pairs
            offsets['givens_adj_neg'] = cur; cur += self._n_adj_pairs
        if self.include_givens_lr:
            offsets['givens_lr_pos']  = cur; cur += self._n_lr_pairs
            offsets['givens_lr_neg']  = cur; cur += self._n_lr_pairs
        if self.include_fswap:
            offsets['fswap']          = cur; cur += self._n_adj_pairs
        if self.include_double_exc:
            offsets['dbl_exc_pos']    = cur; cur += self._n_double_pairs
            offsets['dbl_exc_neg']    = cur; cur += self._n_double_pairs
        return offsets

    def describe(self) -> str:
        n    = self.n_qubits
        offs = self.family_offsets()
        s    = (f"NumberPreservingPool(n={n}, pool_size={self.pool_size}, "
                f"θ={self.angle:.4f})\n")
        s   += f"   op 0              : identity\n"
        for name, off in offs.items():
            size = {
                'givens_adj_pos': self._n_adj_pairs,
                'givens_adj_neg': self._n_adj_pairs,
                'givens_lr_pos':  self._n_lr_pairs,
                'givens_lr_neg':  self._n_lr_pairs,
                'fswap':          self._n_adj_pairs,
                'dbl_exc_pos':    self._n_double_pairs,
                'dbl_exc_neg':    self._n_double_pairs,
            }.get(name, 0)
            s += f"   op {off}..{off+size-1:3d}        : {name}\n"
        return s

    def op_label(self, op_idx: int) -> str:
        if op_idx == 0:
            return 'I'
        offs  = self.family_offsets()
        sizes = {
            'givens_adj_pos': self._n_adj_pairs,
            'givens_adj_neg': self._n_adj_pairs,
            'givens_lr_pos':  self._n_lr_pairs,
            'givens_lr_neg':  self._n_lr_pairs,
            'fswap':          self._n_adj_pairs,
            'dbl_exc_pos':    self._n_double_pairs,
            'dbl_exc_neg':    self._n_double_pairs,
        }
        for name, off in offs.items():
            size = sizes.get(name, 0)
            if off <= op_idx and op_idx < off + size:
                return f"{name}[{op_idx - off}]"
        return f"?[{op_idx}]"


# ─── Fermionic excitation pool (ADAPT-GQE) ────────────────────────────────────

@dataclass
class FermionicPool(PoolBase):
    """Pool of fermionic excitation operators for ADAPT-GQE.

    Operators:
      • Single excitations:  a†_p a_q  (one α→α or β→β hop)
      • Double excitations:  a†_p a†_r a_q a_s  (all combinations)

    In the JW interleaved-spin convention used throughout gqex:
      α qubit p  ↔  qubit 2p
      β qubit p  ↔  qubit 2p+1

    After JW mapping each excitation becomes a small Pauli string cluster
    that is stored here for gradient-guided selection in ADAPTEngine.
    """
    n_qubits: int
    include_singles: bool = True
    include_doubles: bool = True

    # Computed at build time
    _single_ops: List = field(default_factory=list, init=False, repr=False)
    _double_ops: List = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._build_ops()

    def _build_ops(self):
        """Build the JW-mapped Pauli clusters for all excitations."""
        from openfermion import FermionOperator, hermitian_conjugated
        from openfermion.transforms import jordan_wigner
        n_orb = self.n_qubits // 2
        self._single_ops = []
        self._double_ops = []

        if self.include_singles:
            for p in range(n_orb):
                for q in range(p):
                    for spin in (0, 1):  # 0=α, 1=β
                        qp = 2 * p + spin
                        qq = 2 * q + spin
                        op = (FermionOperator(f'{qp}^ {qq}') -
                              FermionOperator(f'{qq}^ {qp}'))
                        jw_op = jordan_wigner(op)
                        jw_op.compress(1e-12)
                        if jw_op.terms:
                            self._single_ops.append(jw_op)

        if self.include_doubles:
            for p in range(n_orb):
                for q in range(p):
                    for r in range(n_orb):
                        for s in range(r):
                            for sa, sb in [(0, 1), (0, 0), (1, 1)]:
                                qp = 2 * p + sa; qq = 2 * q + sa
                                qr = 2 * r + sb; qs = 2 * s + sb
                                if len({qp, qq, qr, qs}) < 4:
                                    continue
                                op = (FermionOperator(f'{qp}^ {qr}^ {qq} {qs}') -
                                      FermionOperator(f'{qq}^ {qs}^ {qp} {qr}'))
                                jw_op = jordan_wigner(op)
                                jw_op.compress(1e-12)
                                if jw_op.terms:
                                    self._double_ops.append(jw_op)

    @property
    def operators(self):
        """List of all pool operators as QubitOperators."""
        return self._single_ops + self._double_ops

    @property
    def pool_size(self) -> int:
        return len(self.operators)

    def family_offsets(self) -> Dict[str, int]:
        offsets = {}
        if self.include_singles:
            offsets['singles'] = 0
        if self.include_doubles:
            offsets['doubles'] = len(self._single_ops)
        return offsets

    def describe(self) -> str:
        return (f"FermionicPool(n={self.n_qubits}, "
                f"singles={len(self._single_ops)}, "
                f"doubles={len(self._double_ops)}, "
                f"total={self.pool_size})")


# ─── Factory function ────────────────────────────────────────────────────────

def build_pool(pool_type: str, n_qubits: int, **kwargs) -> PoolBase:
    """Convenience factory.

    pool_type : 'standard' | 'np' | 'fermionic'
    """
    if pool_type == 'standard':
        return StandardPool(n_qubits=n_qubits, **kwargs)
    elif pool_type in ('np', 'number_preserving'):
        return NumberPreservingPool(n_qubits=n_qubits, **kwargs)
    elif pool_type == 'fermionic':
        return FermionicPool(n_qubits=n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown pool_type '{pool_type}'.  "
                         f"Choose from 'standard', 'np', 'fermionic'.")
