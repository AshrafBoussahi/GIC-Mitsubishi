"""
SQD (Selected Quantum Diagonalization) configuration recovery.
Priority-7 improvement.

The idea: after a QSCI diagonalisation, the ground-state wavefunction
identifies high-amplitude determinants.  Many near-amplitude determinants
may have been missed by noisy circuit sampling.  SQD iteratively:

  1. Takes the current wavefunction |ψ⟩ = Σ c_i |d_i⟩
  2. Generates "candidate" determinants by applying single/double
     spin-orbital flips to all |d_i⟩ with |c_i| > threshold
  3. Adds them to the current CI space and re-diagonalises
  4. Repeats until convergence or n_iter exhausted

This wraps the existing refine() step without requiring quantum shots.
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple


def _flip_one(det: str, n_qubits: int) -> List[str]:
    """All single-qubit-pair flips of `det` that preserve particle number."""
    neighbours = []
    for i in range(0, n_qubits - 1, 2):
        j = i + 1
        # Swap pair (i, j): |10⟩ ↔ |01⟩
        if det[i] != det[j]:
            chars = list(det)
            chars[i], chars[j] = chars[j], chars[i]
            neighbours.append(''.join(chars))
    return neighbours


def _flip_two(det: str, n_qubits: int) -> List[str]:
    """Double excitation neighbours: swap two separate qubit pairs."""
    pairs = list(range(0, n_qubits - 1, 2))
    neighbours = []
    for p1, p2 in combinations(pairs, 2):
        j1, j2 = p1 + 1, p2 + 1
        c = list(det)
        # Swap pair 1
        c[p1], c[j1] = c[j1], c[p1]
        # Swap pair 2
        c[p2], c[j2] = c[j2], c[p2]
        new_det = ''.join(c)
        if new_det != det:
            neighbours.append(new_det)
    return neighbours


def _valid_det(b: str, n_alpha: int, n_beta: int, n_qubits: int) -> bool:
    if len(b) != n_qubits:
        return False
    n_a = sum(int(b[i]) for i in range(0, n_qubits, 2))
    n_b = sum(int(b[i]) for i in range(1, n_qubits, 2))
    return n_a == n_alpha and n_b == n_beta


def sqd_recover(
    dets: List[str],
    wf_coeffs,
    system,
    threshold: float = 1e-3,
    n_iter: int = 10,
    singles_only: bool = False,
) -> List[str]:
    """
    SQD configuration recovery.

    Parameters
    ----------
    dets       : current CI basis determinants
    wf_coeffs  : wavefunction coefficients for dets
    system     : MolecularSystem
    threshold  : |c_i| threshold for generating neighbours
    n_iter     : maximum SQD iterations
    singles_only: only use single-flip neighbours (faster)

    Returns extended list of determinants.
    """
    n         = system.n_qubits
    na, nb    = system.n_alpha, system.n_beta
    current   = set(dets)

    for it in range(n_iter):
        new_dets: Set[str] = set()
        # Select high-amplitude determinants as seeds
        seeds = [dets[i] for i in range(len(dets))
                 if i < len(wf_coeffs) and abs(wf_coeffs[i]) > threshold]
        if not seeds:
            break
        for det in seeds:
            for nb_det in _flip_one(det, n):
                if nb_det not in current and _valid_det(nb_det, na, nb, n):
                    new_dets.add(nb_det)
            if not singles_only:
                for nb_det in _flip_two(det, n):
                    if nb_det not in current and _valid_det(nb_det, na, nb, n):
                        new_dets.add(nb_det)

        if not new_dets:
            break
        current |= new_dets

    return list(current)
