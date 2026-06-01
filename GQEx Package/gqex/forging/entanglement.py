"""
Entanglement forging — priority-5 qubit-count halving for closed-shell systems.

Algorithm (Eddins et al. 2022):
  For a closed-shell state |ψ⟩ in a 2n-qubit register (n α + n β):

    ⟨ψ|H|ψ⟩ = Σ_{σ∈{0,1}^n} w_σ ⟨ψ_σ|H_n|ψ_σ⟩

  where:
    • |ψ_σ⟩  is an n-qubit state (half the original)
    • H_n     is the n-qubit "forged" Hamiltonian (one spin block)
    • w_σ     are classical combination weights from Schmidt decomposition

  Implementation:
    1. Decompose the 2n-qubit Hamiltonian into α-only, β-only,
       and cross-spin (α⊗β) pieces.
    2. Run n-qubit VQE / GQE circuits for each Schmidt branch.
    3. Classically combine expectation values.

  This halves the required qubit count at the cost of running ~2^n
  classical combinations — practical for n ≤ 6 (manageable 64 terms).

NOTE: Full entanglement forging requires decomposing the fermionic
Hamiltonian into spin blocks.  This module provides the framework and
a simplified implementation for closed-shell molecules where α ≡ β.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openfermion import QubitOperator


@dataclass
class ForgingConfig:
    n_schmidt:   int   = 4      # number of Schmidt vectors to keep
    use_symmetry: bool = True   # exploit α=β symmetry (closed-shell only)
    max_iter:    int   = 100    # VQE iters per Schmidt branch
    verbose:     bool  = True


class EntanglementForging:
    """
    Entanglement forging engine for closed-shell molecules.

    Reduces a 2n-qubit problem to n-qubit problems by exploiting the
    known structure of closed-shell RHF states.

    Parameters
    ----------
    system : MolecularSystem (must have qubit_op populated)
    cfg    : ForgingConfig
    """

    def __init__(self, system, cfg: ForgingConfig):
        self.system = system
        self.cfg    = cfg
        if system.n_alpha != system.n_beta:
            raise ValueError(
                "Entanglement forging requires a closed-shell system "
                f"(n_alpha={system.n_alpha} ≠ n_beta={system.n_beta})"
            )
        self.n_half   = system.n_qubits // 2    # n qubits per spin block
        self.n_alpha  = system.n_alpha
        self._forged_ops = None                   # populated by decompose()

    # ── Hamiltonian decomposition ────────────────────────────────────────

    def decompose(self) -> Dict:
        """Decompose the full 2n-qubit Hamiltonian into spin blocks.

        Returns dict with:
          'alpha_op'  : QubitOperator on n qubits (α-spin block)
          'beta_op'   : QubitOperator on n qubits (β-spin block)
          'cross_ops' : list of (coeff, α_op, β_op) tuples
        """
        n     = self.system.n_qubits
        n_h   = self.n_half
        qubit_op = self.system.qubit_op

        alpha_op  = QubitOperator()
        beta_op   = QubitOperator()
        cross_ops = []

        for term, coeff in qubit_op.terms.items():
            if abs(coeff) < 1e-14:
                continue
            # Classify each Pauli by spin block (even=α, odd=β in JW interleaved)
            alpha_part = []
            beta_part  = []
            for qi, p in term:
                if qi % 2 == 0:
                    alpha_part.append((qi // 2, p))
                else:
                    beta_part.append((qi // 2, p))

            a_op = QubitOperator(alpha_part if alpha_part else '', 1.0)
            b_op = QubitOperator(beta_part  if beta_part  else '', 1.0)

            if not beta_part:
                alpha_op += coeff * a_op
            elif not alpha_part:
                beta_op  += coeff * b_op
            else:
                cross_ops.append((complex(coeff), a_op, b_op))

        self._forged_ops = {
            'alpha_op':  alpha_op,
            'beta_op':   beta_op,
            'cross_ops': cross_ops,
        }
        if self.cfg.verbose:
            print(f"  Forging decomposition: "
                  f"{len(alpha_op.terms)} α-terms, "
                  f"{len(beta_op.terms)} β-terms, "
                  f"{len(cross_ops)} cross-terms")
        return self._forged_ops

    # ── Schmidt vector preparation ────────────────────────────────────────

    def hf_schmidt_vectors(self) -> np.ndarray:
        """Compute Schmidt vectors for the HF reference state.

        For a closed-shell HF state |ψ_HF⟩ = |α_HF⟩ ⊗ |β_HF⟩ there is
        exactly one Schmidt term with weight 1.  This provides the starting
        point for variational optimisation.
        """
        n_h    = self.n_half
        na     = self.n_alpha
        # HF α-sector as integer (occupied orbitals 0..na-1)
        hf_int = sum(1 << p for p in range(na))
        v      = np.zeros(2**n_h)
        v[hf_int] = 1.0
        return v.reshape(1, -1)   # (1, 2^n_h)

    # ── Energy computation ────────────────────────────────────────────────

    def compute_forged_energy(
        self,
        alpha_wf: np.ndarray,
        beta_wf:  Optional[np.ndarray] = None,
        weights:  Optional[np.ndarray] = None,
    ) -> float:
        """Compute the forged energy from one-half wavefunctions.

        Parameters
        ----------
        alpha_wf : (n_schmidt, 2^n_half) array of α-sector wavefunctions
        beta_wf  : (n_schmidt, 2^n_half) β-sector (= alpha_wf if closed-shell)
        weights  : (n_schmidt,) Schmidt weights (uniform if None)
        """
        if self._forged_ops is None:
            self.decompose()

        if beta_wf is None:
            beta_wf = alpha_wf  # closed-shell: α ≡ β
        n_s = alpha_wf.shape[0]
        if weights is None:
            weights = np.ones(n_s) / n_s

        # Build matrices for α and β blocks
        from openfermion.linalg import get_sparse_operator
        n_h = self.n_half

        H_alpha = get_sparse_operator(
            self._forged_ops['alpha_op'], n_qubits=n_h
        ).toarray()
        H_beta  = get_sparse_operator(
            self._forged_ops['beta_op'],  n_qubits=n_h
        ).toarray()

        energy = 0.0
        for k in range(n_s):
            av = alpha_wf[k]
            bv = beta_wf[k]
            w  = weights[k]
            # α contribution
            energy += w * float(np.real(av.conj() @ H_alpha @ av))
            # β contribution
            energy += w * float(np.real(bv.conj() @ H_beta  @ bv))
            # Cross contributions
            for coeff, a_op, b_op in self._forged_ops['cross_ops']:
                Ha_ = get_sparse_operator(a_op, n_qubits=n_h).toarray()
                Hb_ = get_sparse_operator(b_op, n_qubits=n_h).toarray()
                ea  = float(np.real(av.conj() @ Ha_ @ av))
                eb  = float(np.real(bv.conj() @ Hb_ @ bv))
                energy += w * float(np.real(coeff)) * ea * eb

        return energy

    # ── Simplified closed-shell solver ────────────────────────────────────

    def solve_closed_shell_exact(self) -> Tuple[float, np.ndarray]:
        """Exact diagonalisation of the forged n-qubit Hamiltonian.

        For validation; replace with GQE on n qubits for scalable version.
        """
        if self._forged_ops is None:
            self.decompose()
        from openfermion.linalg import get_sparse_operator
        n_h = self.n_half

        # For closed-shell: ⟨H⟩ = 2⟨H_α⟩ + Σ_cross
        H_mat = get_sparse_operator(
            self._forged_ops['alpha_op'], n_qubits=n_h
        ).toarray()
        H_beta = get_sparse_operator(
            self._forged_ops['beta_op'], n_qubits=n_h
        ).toarray()

        # Full energy: both spin blocks act on the same spatial orbital WF
        # (closed-shell approximation)
        H_eff = H_mat + H_beta
        for coeff, a_op, b_op in self._forged_ops['cross_ops']:
            Ha_ = get_sparse_operator(a_op, n_qubits=n_h).toarray()
            Hb_ = get_sparse_operator(b_op, n_qubits=n_h).toarray()
            H_eff += float(np.real(coeff)) * (Ha_ @ Hb_)

        eigs, vecs = np.linalg.eigh(H_eff)
        if self.cfg.verbose:
            print(f"  Forged exact energy (n={n_h} qubits): {eigs[0]:+.6f} Ha")
        return float(eigs[0]), vecs[:, 0]

    def n_qubits_forged(self) -> int:
        """Qubit count after forging."""
        return self.n_half
