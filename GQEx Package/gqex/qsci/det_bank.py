"""
Persistent Determinant Bank — priority-2 improvement.

The DeterminantBank accumulates bitstring determinants across ALL RL
iterations instead of discarding them after each QSCI step.  This
converts wasted shots into an ever-growing SQD sample pool.

Key idea: coupon-collector's problem says that after K independent
circuits × S shots each, the expected number of unique determinants
discovered scales as O(K·S·(1 - e^{-K·S/D})) where D is the total
FCI space size.  Persistence turns K=1 into K=n_iters.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


class DeterminantBank:
    """Thread-safe accumulating bank of computational basis states.

    Attributes
    ----------
    max_size    : maximum number of distinct determinants to retain
    n_alpha     : expected number of alpha electrons (for filtering)
    n_beta      : expected number of beta electrons
    n_qubits    : qubit count (for parity checks)
    """

    def __init__(
        self,
        max_size:  int = 50_000,
        n_alpha:   int = 0,
        n_beta:    int = 0,
        n_qubits:  int = 0,
        apply_symmetry: bool = True,
    ):
        self.max_size       = max_size
        self.n_alpha        = n_alpha
        self.n_beta         = n_beta
        self.n_qubits       = n_qubits
        self.apply_symmetry = apply_symmetry
        self._counts: Counter = Counter()
        self._lock = threading.Lock()
        self._total_shots   = 0
        self._iter_shots: List[int] = []

    # ── Ingestion ─────────────────────────────────────────────────────────

    def update(self, bitstrings: Iterable[str], weight: float = 1.0):
        """Add bitstrings (with counts if Counter, else each once) to the bank.

        Filters on particle-number symmetry before inserting.
        """
        valid = self._filter(bitstrings)
        if self.apply_symmetry:
            valid = self._symmetry_complete(valid)
        with self._lock:
            for b in valid:
                self._counts[b] += weight
            self._total_shots += len(valid)
            self._iter_shots.append(len(valid))
            self._trim()

    def update_from_counter(self, counter: Dict[str, int]):
        """Update from a {bitstring: count} mapping."""
        valid_map = {b: c for b, c in counter.items()
                     if self._valid_det(b)}
        valid_list = []
        for b, c in valid_map.items():
            valid_list.extend([b] * c)
        if self.apply_symmetry:
            valid_list = self._symmetry_complete(valid_list)
        with self._lock:
            self._counts.update(valid_list)
            self._total_shots += sum(valid_map.values())
            self._trim()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def get_determinants(self, top_k: Optional[int] = None) -> List[str]:
        """Return determinants sorted by accumulated weight."""
        with self._lock:
            if top_k:
                return [d for d, _ in self._counts.most_common(top_k)]
            return list(self._counts.keys())

    def get_weighted(self, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return (determinant, weight) pairs sorted by weight."""
        with self._lock:
            items = self._counts.most_common(top_k)
        return [(b, float(c)) for b, c in items]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._counts)

    @property
    def total_shots(self) -> int:
        return self._total_shots

    def clear(self):
        with self._lock:
            self._counts.clear()
            self._total_shots = 0
            self._iter_shots  = []

    def merge(self, other: 'DeterminantBank'):
        """In-place merge of another bank's counts."""
        with other._lock:
            other_counts = dict(other._counts)
        with self._lock:
            self._counts.update(other_counts)
            self._trim()

    def stats(self) -> Dict:
        return {
            'size':         self.size,
            'total_shots':  self._total_shots,
            'n_iters':      len(self._iter_shots),
            'shots_per_iter': (sum(self._iter_shots) / max(len(self._iter_shots), 1)),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _valid_det(self, b: str) -> bool:
        if len(b) != self.n_qubits:
            return False
        if self.n_alpha == 0 and self.n_beta == 0:
            return True  # no filter
        n_a = sum(int(b[i]) for i in range(0, self.n_qubits, 2))
        n_b = sum(int(b[i]) for i in range(1, self.n_qubits, 2))
        return n_a == self.n_alpha and n_b == self.n_beta

    def _filter(self, bitstrings: Iterable[str]) -> List[str]:
        return [b for b in bitstrings if self._valid_det(b)]

    def _trim(self):
        """Keep only max_size most common determinants."""
        if len(self._counts) > self.max_size:
            self._counts = Counter(
                dict(self._counts.most_common(self.max_size))
            )

    def _symmetry_complete(self, dets: List[str]) -> List[str]:
        """Add all spin-orbital configurations with the same orbital occupancy.

        In the interleaved α,β JW encoding qubit 2p = α_p, 2p+1 = β_p.
        For each open-shell orbital, generate all spin assignments that
        preserve n_alpha and n_beta.
        """
        from itertools import combinations
        n_orb = self.n_qubits // 2
        out: Set[str] = set()
        decode = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
        for det in dets:
            if len(det) != self.n_qubits:
                continue
            occ = [decode[(int(det[2*p]), int(det[2*p+1]))]
                   for p in range(n_orb)]
            open_shell = [p for p in range(n_orb) if occ[p] in (1, 2)]
            m = len(open_shell)
            if m == 0:
                out.add(det)
                continue
            n_a_open = sum(1 for p in open_shell if occ[p] == 1)
            for alpha_pos in combinations(range(m), n_a_open):
                new_occ = occ.copy()
                for idx in range(m):
                    new_occ[open_shell[idx]] = 1 if idx in alpha_pos else 2
                chars = ['0'] * (2 * n_orb)
                for p in range(n_orb):
                    if new_occ[p] == 1:
                        chars[2*p] = '1'
                    elif new_occ[p] == 2:
                        chars[2*p+1] = '1'
                    elif new_occ[p] == 3:
                        chars[2*p] = chars[2*p+1] = '1'
                out.add(''.join(chars))
        return list(out)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"DeterminantBank(size={self.size}/{self.max_size}, "
                f"shots={self.total_shots})")
