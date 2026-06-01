"""
MolecularSystem dataclass and Hamiltonian builder.

Handles active-space selection, JW mapping, CASCI reference, and
optional qubit tapering.  All downstream modules receive a
MolecularSystem that is fully populated by build_molecular_hamiltonian().
"""

import math
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from openfermion import QubitOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from gqex.utils.conversions import of_to_cudaq


# ─── Geometry helpers ────────────────────────────────────────────────────────

def parse_xyz(xyz_block: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Parse a multi-line XYZ geometry string into openfermion format."""
    geom = []
    for line in xyz_block.strip().split('\n'):
        parts = line.split()
        if len(parts) == 4:
            geom.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return geom


# ─── Dataclass ───────────────────────────────────────────────────────────────

@dataclass
class MolecularSystem:
    """All metadata for an electronic-structure problem.

    After calling build_molecular_hamiltonian() all fields below the
    dashed line are populated.
    """
    name: str
    xyz: str
    basis: str = '6-31g'
    charge: int = 0
    spin: int = 0
    n_active_electrons: int = 0
    n_active_orbitals: int = 0
    compute_casci: bool = False

    # ── filled by build() ──────────────────────────────────────────────────
    hamiltonian: Any = None           # cudaq SpinOperator
    qubit_op: Any = None              # openfermion QubitOperator (full JW)
    tapered_qubit_op: Any = None      # after tapering (may equal qubit_op)
    n_qubits: int = 0                 # qubits after tapering
    n_qubits_full: int = 0            # qubits before tapering
    n_alpha: int = 0
    n_beta: int = 0
    hf_energy: float = 0.0
    casci_energy: float = float('nan')
    n_pauli_terms: int = 0
    tapering_sector: Optional[List[int]] = None   # eigenvalue sector used
    frozen_core_energy: float = 0.0

    @property
    def n_orbitals(self) -> int:
        return self.n_qubits // 2

    @property
    def hf_x_qubits(self) -> List[int]:
        """JW interleaved α,β HF determinant flip-list for the active space."""
        return ([2 * p     for p in range(self.n_alpha)] +
                [2 * p + 1 for p in range(self.n_beta)])

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta

    def correlation_energy(self) -> float:
        if math.isnan(self.casci_energy):
            return float('nan')
        return self.casci_energy - self.hf_energy

    def __repr__(self) -> str:
        return (f"MolecularSystem({self.name}, basis={self.basis}, "
                f"{self.n_qubits}q, HF={self.hf_energy:+.6f})")


# ─── Builder ─────────────────────────────────────────────────────────────────

def build_molecular_hamiltonian(
    sys_: MolecularSystem,
    verbose: bool = True,
    taper: bool = False,
    taper_method: str = 'z2',         # 'z2' | 'scbk' | 'none'
    pauli_tol: float = 1e-12,
) -> MolecularSystem:
    """Populate *sys_* in-place and return it.

    Parameters
    ----------
    sys_      : MolecularSystem (partially initialised)
    verbose   : print progress to stdout
    taper     : whether to apply qubit tapering
    taper_method : 'z2' = standard Z2 symmetry tapering,
                   'scbk' = symmetry-conserving BK (-2 qubits),
                   'none' = skip tapering
    pauli_tol : threshold for dropping small Pauli terms
    """
    geom = parse_xyz(sys_.xyz)
    multiplicity = sys_.spin + 1
    sys_.n_alpha = (sys_.n_active_electrons + sys_.spin) // 2
    sys_.n_beta  = sys_.n_active_electrons - sys_.n_alpha

    if verbose:
        print(f"\n  Building {sys_.name}:  {len(geom)} atoms,  basis={sys_.basis}")

    mol = MolecularData(
        geometry=geom,
        basis=sys_.basis,
        charge=sys_.charge,
        multiplicity=multiplicity,
        description=sys_.name,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False)
    sys_.hf_energy = float(mol.hf_energy)

    # Active space (frozen core determined automatically)
    n_frozen_e   = mol.n_electrons - sys_.n_active_electrons
    n_frozen_orb = n_frozen_e // 2
    occupied = list(range(n_frozen_orb))
    active   = list(range(n_frozen_orb, n_frozen_orb + sys_.n_active_orbitals))

    if verbose:
        print(f"  Total:    {mol.n_electrons}e / {mol.n_orbitals}o")
        print(f"  Frozen:   {n_frozen_orb} orbitals  ({n_frozen_e}e)")
        print(f"  Active:   {sys_.n_active_electrons}e / "
              f"{sys_.n_active_orbitals}o  →  {2*sys_.n_active_orbitals} qubits")

    mol_ham    = mol.get_molecular_hamiltonian(occupied_indices=occupied, active_indices=active)
    fermion_op = get_fermion_operator(mol_ham)
    qubit_op   = jordan_wigner(fermion_op)
    qubit_op.compress(abs_tol=pauli_tol)

    sys_.qubit_op       = qubit_op
    sys_.n_qubits_full  = 2 * sys_.n_active_orbitals
    sys_.n_qubits       = sys_.n_qubits_full
    sys_.n_pauli_terms  = len(qubit_op.terms)

    # Optional tapering
    if taper and taper_method != 'none':
        if verbose:
            print(f"  Applying qubit tapering ({taper_method}) ...", end='', flush=True)
        try:
            from gqex.hamiltonian.tapering import taper_hamiltonian
            tap_result = taper_hamiltonian(sys_, method=taper_method, verbose=False)
            sys_.tapered_qubit_op = tap_result.tapered_op
            sys_.n_qubits         = tap_result.n_qubits_tapered
            sys_.tapering_sector  = tap_result.sector
            if verbose:
                saved = sys_.n_qubits_full - sys_.n_qubits
                print(f" done  (saved {saved} qubit{'s' if saved != 1 else ''},"
                      f" now {sys_.n_qubits} qubits)")
            active_op = sys_.tapered_qubit_op
        except Exception as e:
            if verbose:
                print(f" failed ({e}), continuing without tapering")
            active_op = qubit_op
    else:
        active_op = qubit_op

    sys_.hamiltonian = of_to_cudaq(active_op)

    # CASCI reference
    if sys_.compute_casci:
        if verbose:
            print(f"  Computing CASCI reference ...", end='', flush=True)
        try:
            from pyscf import gto, scf, mcscf
            pyscfmol = gto.Mole()
            pyscfmol.atom = sys_.xyz
            pyscfmol.basis = sys_.basis
            pyscfmol.charge = sys_.charge
            pyscfmol.spin = sys_.spin
            pyscfmol.verbose = 0
            pyscfmol.build()
            mf = scf.RHF(pyscfmol)
            mf.kernel()
            mc = mcscf.CASCI(mf, sys_.n_active_orbitals, sys_.n_active_electrons)
            sys_.casci_energy = float(mc.kernel()[0])
            if verbose:
                print(f" done")
        except Exception as e:
            if verbose:
                print(f" failed ({e})")

    if verbose:
        print(f"  HF energy:    {sys_.hf_energy:+.6f} Ha")
        if not math.isnan(sys_.casci_energy):
            print(f"  CASCI energy: {sys_.casci_energy:+.6f} Ha  "
                  f"(corr = {1000*sys_.correlation_energy():+.1f} mHa)")
        print(f"  Pauli terms:  {sys_.n_pauli_terms}  (active qubits: {sys_.n_qubits})")

    return sys_
