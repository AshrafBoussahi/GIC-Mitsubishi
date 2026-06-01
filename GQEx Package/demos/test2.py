#!/usr/bin/env python3
"""
Production-grade GQEx configuration for 32-qubit active spaces.
Based on Kemmoku et al. (2026), arXiv:2604.09756, Section III.E.

Corrected version — uses ONLY valid GQEConfig fields.
"""

from gqex.pipeline.config import (
    PipelineConfig, PoolConfig, TaperingConfig, BankConfig,
    FinetuneConfig, RefinementConfig
)
from gqex.gqe.engine import GQEConfig
from gqex.qsci.evaluator import QSCIConfig
from gqex.generator.retnet import RetNetConfig
from gqex.adapt.bootstrap import ADAPTConfig


import faulthandler
faulthandler.enable()

import torch
torch.set_num_threads(4)

def make_32qubit_config(
    n_shots: int = 10_000,       # per-circuit shot budget (paper: 1e5)
    n_samples: int = 10,          # circuits per RL iteration (paper: M=10)
    n_iters: int = 100,           # RL iterations (paper: 100)
    d_max: int = 10_000,         # determinant budget
    use_sqd: bool = True,         # SQD for robustness at scale
    sqd_n_iter: int = 5,          # self-consistent recovery iterations
    refine_enabled: bool = True,  # global refinement across circuits
    n_extra_circuits: int = 15,   # extra circuits for refinement pool
    finetune_enabled: bool = False,   # L-BFGS-B angle optimization
    pool_type: str = 'np',        # number-preserving Givens
    taper_enabled: bool = False,   # Z2 tapering reduces effective qubits
    device: str = 'auto',
    verbose: bool = True,
) -> PipelineConfig:
    """
    Build a PipelineConfig tuned for 28–32 qubit active spaces.
    Uses ONLY valid fields for your GQEx package's dataclasses.
    """

    cfg = PipelineConfig()

    # ── 1. RETNET GENERATOR ─────────────────────────────────────────────
    cfg.gen = RetNetConfig(
        pool_size=0,              # auto-set from pool
        hidden_dim=64,           # large capacity for 32 qubits
        n_layers=4,               # deep for long sequences (L~800)
        n_heads=4,                # multi-head retention
        max_len=128,             # must exceed circuit depth + buffer
        gated_mlp=True,
    )

    # ── 2. GQE RL ENGINE ──────────────────────────────────────────────────
    # Paper: M=10, 100 iters, lr=5e-6, AdamW, weight_decay=0.01
    # NOTE: grpo_n_updates and grpo_epsilon are INTERNAL to GQEEngine,
    #       not exposed in GQEConfig.  They are hardcoded or set elsewhere.
    cfg.gqe = GQEConfig(
        depth=0,                  # auto from system
        n_samples=n_samples,      # M = 10 (paper)
        n_iters=n_iters,          # 100 iterations (paper)
        lr=5e-6,                  # paper's learning rate (slower than default)
        reward_mode='qsci',       # QSCI energy as reward
        device=device,
        use_recurrent=True,       # essential for long sequences
        use_bank=True,            # accumulate determinants
        cosine_anneal=True,       # smooth LR decay
        entropy_coef=0.08,        # exploration (tuned down for scale)
        temperature=1.5,
        temp_anneal=True,
        adv_std_floor=5e-3,
        adv_clip=5.0,             # paper's GRPO epsilon = 0.2 maps here
        use_rank_adv=True,
        diversity_coef=0.03,      # encourage circuit diversity
        entropy_floor=0.4,
        entropy_adapt=0.015,
    )

    # ── 3. QSCI / SQD EVALUATOR ─────────────────────────────────────────
    cfg.qsci = QSCIConfig(
        n_shots=n_shots,          # 100_000 (paper's per-circuit budget)
        d_max=d_max,              # 100_000
        apply_symmetry=True,      # enforce N_el, S_z
        verbose=False,
        use_sqd=use_sqd,
        sqd_n_iter=sqd_n_iter,
    )

    # ── 4. GATE POOL ────────────────────────────────────────────────────
    cfg.pool = PoolConfig(
        pool_type=pool_type,
        angle=0.785398,
        include_givens_adj=True,
        include_givens_lr=True,
        include_double_exc=True,
        include_fswap=False,
    )

    # ── 5. DETERMINANT BANK ─────────────────────────────────────────────
    cfg.bank = BankConfig(
        enabled=True,
        max_size=100_000,         # large for 32 qubits
        apply_symmetry=True,
        flush_on_start=False,
    )

    # ── 6. GLOBAL REFINEMENT ────────────────────────────────────────────
    cfg.refine = RefinementConfig(
        enabled=refine_enabled,
        n_extra_circuits=n_extra_circuits,
        only_in_qsci_mode=True,
    )

    # ── 7. L-BFGS-B FINETUNING ──────────────────────────────────────────
    cfg.finetune = FinetuneConfig(
        enabled=finetune_enabled,
        max_iter=1000,
        ftol=1e-10,
        gtol=1e-8,
        method='L-BFGS-B',
    )

    # ── 8. QUBIT TAPERING ───────────────────────────────────────────────
    cfg.taper = TaperingConfig(
        enabled=taper_enabled,
        method='z2',
        pauli_tol=1e-12,
    )

    # ── 9. ADAPT BOOTSTRAP ──────────────────────────────────────────────
    cfg.adapt = ADAPTConfig(
        enabled=False,
        max_gates=50,
        n_candidates_per_iter=40,
        energy_tol=5e-4,
        plateau_patience=4,
    )
    cfg.adapt_prior_strength = 1.5
    cfg.adapt_supervised_epochs = 10

    # ── 10. GLOBAL SWITCHES ─────────────────────────────────────────────
    cfg.compute_casci = False
    cfg.verbose = verbose

    cfg.gqe.depth = 64   # or even 48 — explicit override

    return cfg


# ═══════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from gqex import MolecularSystem, GQExPipeline

    # N2 in 6-31G, (10e, 16o) → 32 qubits, R_NN = 1.8 Å
    n2_system = MolecularSystem(
        name='N2_32qubit',
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.8",
        basis='6-31g',
        n_active_electrons=10,
        n_active_orbitals=16,
        compute_casci=True,
    )

    cfg = make_32qubit_config()
    pipeline = GQExPipeline(n2_system, cfg)
    results = pipeline.run()