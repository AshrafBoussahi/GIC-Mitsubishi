"""
gqex — Generative Quantum Eigensolver eXtended
================================================
Scalable GQE-QSCI pipeline targeting up to ~40 qubits.

Key improvements over baseline GQE:
  1. Number-preserving gate pool (Givens rotations)  → eliminates ~50% wasted shots
  2. Persistent determinant bank                      → converts all shots to SQD input
  3. Qubit tapering (Z2 symmetries)                   → saves 2–4+ qubits
  4. Hamiltonian subspace optimisation                → faster H_sub construction
  5. Entanglement forging (closed-shell)              → halves qubit count
  6. ADAPT-style fermionic pool                       → gradient-guided pool selection
  7. SQD configuration recovery                       → enriches CI subspace
  8. Scaled RetNet with recurrent inference            → O(T) per-step cost

Quick start
-----------
    from gqex import GQExPipeline, MolecularSystem, quick_config

    h2 = MolecularSystem('H2', 'H 0 0 0\\nH 0 0 0.74',
                         basis='sto-3g', n_active_electrons=2,
                         n_active_orbitals=2, compute_casci=True)
    results = GQExPipeline(h2, quick_config('small')).run()
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"

__all__ = [
    "MolecularSystem",
    "build_molecular_hamiltonian",
    "PipelineConfig",
    "quick_config",
    "GQExPipeline",
]

# Lazy imports — only resolved when actually accessed, so that submodules
# that don't need numpy/cudaq can be imported independently.
def __getattr__(name):
    if name == "MolecularSystem":
        from gqex.hamiltonian.molecular_system import MolecularSystem
        return MolecularSystem
    if name == "build_molecular_hamiltonian":
        from gqex.hamiltonian.molecular_system import build_molecular_hamiltonian
        return build_molecular_hamiltonian
    if name == "PipelineConfig":
        from gqex.pipeline.config import PipelineConfig
        return PipelineConfig
    if name == "quick_config":
        from gqex.pipeline.config import quick_config
        return quick_config
    if name == "GQExPipeline":
        from gqex.pipeline.orchestrator import GQExPipeline
        return GQExPipeline
    raise AttributeError(f"module 'gqex' has no attribute {name!r}")
