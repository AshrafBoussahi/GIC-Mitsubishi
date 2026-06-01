""" #!/usr/bin/env python3
"""
""" Test GQEx pipeline on H2O (dissertation benchmark molecule).
Compare your result to FCI reference: -75.01259 Ha. """
"""

from gqex import MolecularSystem, GQExPipeline, quick_config

# ── Exact H2O from dissertation ─────────────────────────────────────────
system = MolecularSystem(
    name='H2O',
    xyz=(
        "O 0.0000  0.0000  0.1125\n"
        "H 0.0000  0.7938 -0.4500\n"
        "H 0.0000 -0.7938 -0.4500"
    ),
    basis='sto-3g',
    n_active_electrons=8,
    n_active_orbitals=6,
    compute_casci=True,   # computes exact FCI for error reporting
)

# ── Configuration: use 'medium' for 12 qubits, 'large' for maximum accuracy ─
cfg = quick_config('large')

cfg.taper.enabled = False

# Optional: enrich the pool (dissertation uses chemically-motivated excitations)
cfg.pool.pool_type = 'np'
cfg.pool.include_givens_adj = True
cfg.pool.include_givens_lr = True
cfg.pool.include_double_exc = True

# Optional: enable all stages for maximum accuracy
cfg.bank.enabled = True
cfg.refine.enabled = True
cfg.refine.n_extra_circuits = 10
cfg.finetune.enabled = True

# ── Run ─────────────────────────────────────────────────────────────────
pipeline = GQExPipeline(system, cfg)
results = pipeline.run()

# ── Report vs. dissertation reference ───────────────────────────────────
fci_ref = -75.01259
hf_ref = -74.96190

print("\n" + "="*60)
print("  COMPARISON TO DISSERTATION BASELINE")
print("="*60)
print(f"  HF reference (dissertation) : {hf_ref:+.6f} Ha")
print(f"  FCI reference (dissertation): {fci_ref:+.6f} Ha")
print(f"  Your GQEx QSCI energy       : {results['qsci_energy']:+.6f} Ha")
print(f"  Your GQEx Bank energy       : {results['bank_energy']:+.6f} Ha")
print(f"  Your GQEx Global energy     : {results['global_energy']:+.6f} Ha")
print(f"  Your GQEx Finetune energy   : {results['finetune_energy']:+.6f} Ha")
print(f"  Your best circuit ops         : {len(results['best_ops'])} gates")
print("="*60) """















#!/usr/bin/env python3
"""
Test GQEx pipeline on H2O (dissertation benchmark molecule).
Compare your result to FCI reference: -75.01259 Ha.
"""

from gqex import MolecularSystem, GQExPipeline, quick_config

# ── Exact H2O from dissertation ─────────────────────────────────────────
system = MolecularSystem(
    name               = 'BeH2',
    xyz                = "Be 0.0 0.0  0.0\nH  0.0 0.0  1.33\nH  0.0 0.0 -1.33",
    basis              = '6-31g',
    n_active_electrons = 4,
    n_active_orbitals  = 12,
    compute_casci      = True,
)

# ── Configuration: use 'medium' for 12 qubits, 'large' for maximum accuracy ─
cfg = quick_config('large')

cfg.taper.enabled = False

# Optional: enrich the pool (dissertation uses chemically-motivated excitations)
cfg.pool.pool_type = 'np'
cfg.pool.include_givens_adj = True
cfg.pool.include_givens_lr = True
cfg.pool.include_double_exc = True

# Optional: enable all stages for maximum accuracy
cfg.bank.enabled = True
cfg.refine.enabled = True
cfg.refine.n_extra_circuits = 10
cfg.finetune.enabled = True

# ── Run ─────────────────────────────────────────────────────────────────
pipeline = GQExPipeline(system, cfg)
results = pipeline.run()

# ── Report vs. dissertation reference ───────────────────────────────────
fci_ref = -75.01259
hf_ref = -74.96190

print("\n" + "="*60)
print("  COMPARISON TO DISSERTATION BASELINE")
print("="*60)
print(f"  HF reference (dissertation) : {hf_ref:+.6f} Ha")
print(f"  FCI reference (dissertation): {fci_ref:+.6f} Ha")
print(f"  Your GQEx QSCI energy       : {results['qsci_energy']:+.6f} Ha")
print(f"  Your GQEx Bank energy       : {results['bank_energy']:+.6f} Ha")
print(f"  Your GQEx Global energy     : {results['global_energy']:+.6f} Ha")
print(f"  Your GQEx Finetune energy   : {results['finetune_energy']:+.6f} Ha")
print(f"  Your best circuit ops         : {len(results['best_ops'])} gates")
print("="*60)