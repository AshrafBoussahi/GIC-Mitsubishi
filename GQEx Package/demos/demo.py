"""
GQEx demo — BeH2 / 6-31G / 24 qubits (frozen Be 1s)

Reproduces the original GQE.py demo but uses the full gqex package
with all priority improvements enabled.

Run:
    python demo.py
    python demo.py --size large   # ramp up for 40-qubit target
    python demo.py --mol h2o      # water molecule
"""

import argparse
import torch
import numpy as np

torch.manual_seed(7)
np.random.seed(7)

from gqex import MolecularSystem, GQExPipeline, quick_config
from gqex.pipeline.config import (
    PipelineConfig, PoolConfig, TaperingConfig, BankConfig,
    FinetuneConfig, RefinementConfig,
)
from gqex.gqe.engine import GQEConfig
from gqex.qsci.evaluator import QSCIConfig
from gqex.generator.retnet import RetNetConfig


# ─── Molecule library ────────────────────────────────────────────────────────

MOLECULES = {
    # 24-qubit BeH2 in 6-31g (matches the legacy GQE.py demo)
    'beh2': MolecularSystem(
        name               = 'BeH2',
        xyz                = "Be 0.0 0.0  0.0\nH  0.0 0.0  1.33\nH  0.0 0.0 -1.33",
        basis              = '6-31g',
        n_active_electrons = 4,
        n_active_orbitals  = 12,
        compute_casci      = True,
    ),
    # Compact 14-qubit BeH2 for fast iteration / sanity checks
    'beh2_small': MolecularSystem(
        name               = 'BeH2_small',
        xyz                = "Be 0.0 0.0  0.0\nH  0.0 0.0  1.33\nH  0.0 0.0 -1.33",
        basis              = 'sto-3g',
        n_active_electrons = 4,
        n_active_orbitals  = 6,
        compute_casci      = True,
    ),
    'h2': MolecularSystem(
        name               = 'H2',
        xyz                = "H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
        basis              = 'sto-3g',
        n_active_electrons = 2,
        n_active_orbitals  = 2,
        compute_casci      = True,
    ),
    'h2o': MolecularSystem(
        name               = 'H2O',
        xyz                = "O 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0",
        basis              = '6-31g',
        n_active_electrons = 8,
        n_active_orbitals  = 7,
        compute_casci      = True,
    ),
    'lih': MolecularSystem(
        name               = 'LiH',
        xyz                = "Li 0.0 0.0 0.0\nH  0.0 0.0 1.5474",
        basis              = '6-31g',
        n_active_electrons = 4,
        n_active_orbitals  = 5,
        compute_casci      = True,
    ),
    # ~40-qubit target
    'n2_large': MolecularSystem(
        name               = 'N2_large',
        xyz                = "N 0.0 0.0 0.0\nN 0.0 0.0 1.098",
        basis              = 'cc-pvdz',
        n_active_electrons = 10,
        n_active_orbitals  = 20,   # 40 qubits
        compute_casci      = False,
    ),
}


# ─── Config builders ─────────────────────────────────────────────────────────

def make_config(size: str, mol_key: str) -> PipelineConfig:
    """Build a PipelineConfig appropriate for the molecule and run size."""
    base = quick_config(size)
    # Always use NP pool
    base.pool.pool_type = 'np'
    # Turn on bank for medium/large
    if size in ('medium', 'large'):
        base.bank.enabled = True
    return base


# ─── Demo entry point ─────────────────────────────────────────────────────────

def run_demo(mol_key: str = 'n2_large', size: str = 'large'):
    print("=" * 64)
    print(f"  GQEx demo  —  mol={mol_key.upper()}  size={size}")
    print("=" * 64)

    mol = MOLECULES[mol_key]
    cfg = make_config(size, mol_key)

    pipeline = GQExPipeline(mol, cfg)
    results  = pipeline.run()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQEx demo")
    parser.add_argument('--mol',  default='n2_large',
                        choices=list(MOLECULES.keys()),
                        help='molecule to run')
    parser.add_argument('--size', default='large',
                        choices=['small', 'medium', 'large'],
                        help='pipeline size preset')
    args = parser.parse_args()
    run_demo(args.mol, args.size)
