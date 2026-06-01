<div align="center">

# 🔬 GIC-Mitsubishi Challenge — Team eQoSystem

### Phase 2 Submission · Global Industry Challenge 2026
### *Quantum Materials Discovery Challenge: Scaling Generative Quantum Eigensolver (GQE)*

---

[![Challenge](https://img.shields.io/badge/Challenge-GIC%202026%20Phase%202-blueviolet?style=for-the-badge)](https://github.com/AshrafBoussahi/GIC-Mitsubishi)
[![Provider](https://img.shields.io/badge/Provider-Mitsubishi%20Chemical%20%2B%20AIST-red?style=for-the-badge)](https://github.com/AshrafBoussahi/GIC-Mitsubishi)
[![Focus](https://img.shields.io/badge/Focus-Advanced%20Materials-green?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/Python-3.9%20–%203.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](GQEx%20Package/LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-gqex%200.1.1-orange?style=for-the-badge&logo=pypi)](https://pypi.org/project/gqex/)

</div>

---

## 📋 Table of Contents

1. [Challenge Overview](#-challenge-overview)
2. [Team](#-team)
3. [Our Approach: The GQEx Framework](#-our-approach-the-gqex-framework)
4. [Repository Structure](#-repository-structure)
5. [The `gqex` Python Package](#-the-gqex-python-package)
   - [Architecture](#package-architecture)
   - [Installation](#installation)
   - [Quick Start](#quick-start)
   - [Full Usage Guide](#full-usage-guide)
   - [Configuration Reference](#configuration-reference)
   - [Benchmark Results](#benchmark-results)
6. [Technical Proposal](#-technical-proposal)
7. [Citation & References](#-citation--references)
8. [License](#-license)

---

## 🏆 Challenge Overview

This repository is the **Phase 2 submission** of team **eQoSystem** for the [Global Industry Challenge (GIC) 2026](https://aqora.io/), co-organized by:

- **Mitsubishi Chemical Group** — a global leader in advanced materials and chemical innovation, driving integration of AI and quantum computing into materials discovery.
- **National Institute of Advanced Industrial Science and Technology (AIST)** — Japan's largest public research organization, leading in computational quantum science and high-performance simulation.

### The Problem

Ground-state energy estimation is a foundational challenge in quantum chemistry and materials science. Classical methods hit exponential scaling walls. Quantum approaches like the **Variational Quantum Eigensolver (VQE)** struggle with optimization complexity and circuit scalability.

The **Generative Quantum Eigensolver (GQE)** introduces a promising paradigm where a generative model *produces* quantum circuits that approximate ground states — but scaling it beyond ~12 qubits has remained a critical open challenge.

### The Goal

> Develop a scalable implementation of GQE targeting **~40 qubits**, achieving chemical accuracy (~1.6 mHa) on real molecular systems.

### Evaluation Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | **Scalability** (primary) | ⭐⭐⭐⭐⭐ |
| 2 | Accuracy & Scientific Validity | ⭐⭐⭐⭐ |
| 3 | Algorithmic Innovation | ⭐⭐⭐⭐ |
| 4 | Computational Efficiency | ⭐⭐⭐ |
| 5 | Hybrid System Design | ⭐⭐⭐ |
| 6 | Benchmarking & Validation | ⭐⭐⭐ |
| 7 | Clarity & Reproducibility | ⭐⭐ |

---

## 👥 Team

| Name | Email | Aqora | Role |
|------|-------|-------|------|
| **Achraf Boussahi** | a.boussahi@esi-sba.dz | @AchrafBoussahi | Team Lead & Quantum Expert |
| **Abir Chekroun** | a.chekroun@esi-sba.dz | @Abeer | Quantum Expert / Content & Commercial Lead |
| **Zakaria Lourghi** | z.lourghi@esi-sba.dz | @ZakariaLer | Software Engineer / AI & Data Scientist |

> **Affiliation:** École Supérieure en Informatique de Sidi Bel Abbès (ESI-SBA), Algeria

---

## 🧠 Our Approach: The GQEx Framework

Our solution — **`gqex` (Generative Quantum Eigensolver eXtended)** — re-engineers the baseline GQE for scalability through **8 synergistic innovations**:

### The 8 Scalability Levers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GQEx Scalability Stack                             │
├────┬────────────────────────────────────────────────────────────────────────┤
│ 1  │ Number-Preserving Givens-Rotation Gate Pool  → eliminates ~50% wasted  │
│    │                                                 shots via symmetry      │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 2  │ Persistent Determinant Bank                  → cross-iteration shot     │
│    │                                                 reuse for richer QSCI  │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 3  │ Z2 Qubit Tapering                            → saves 2–4+ qubits via   │
│    │                                                 symmetry reduction      │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 4  │ Hamiltonian Subspace Optimisation            → faster H_sub             │
│    │                                                 diagonalisation         │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 5  │ Entanglement Forging (closed-shell)          → effectively halves qubit │
│    │                                                 count                   │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 6  │ ADAPT-Style Fermionic Pool                   → gradient-guided pool     │
│    │                                                 selection for bootstrap │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 7  │ SQD Configuration Recovery                   → enriches CI subspace     │
│    │                                                 with high-quality dets  │
├────┼────────────────────────────────────────────────────────────────────────┤
│ 8  │ RetNet Generator with O(T) Recurrent          → scalable token-by-token │
│    │  Inference                                      circuit generation       │
└────┴────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Overview

The full GQEx pipeline is a **hybrid quantum-classical** system:

```
Classical Preprocessing
        │
        ▼
  ADAPT-VQE Bootstrap ──────────────────────────────────────┐
  (greedy gate selection, seeds the search)                  │
        │                                                    │
        ▼                                                    │
  RetNet Generator                                           │
  (generates circuits token-by-token using O(T) retention)  │
        │                                                    │
        ▼                                                    │
  GQE REINFORCE Training ◄──────────────────────────────────┤
  (policy gradient with EMA baseline, rank advantages,       │
   diversity bonuses, entropy regularisation)                │
        │                                                    │
        ▼                                                    │
  Persistent Determinant Bank ─────────────────────────────►│
  (accumulates bitstrings across all RL iterations)          │
        │                                                    │
        ▼                                                    │
  QSCI Evaluation ◄────────────────────────────────────────►│
  (Quantum-Selected Configuration Interaction)               │
        │                                                    │
        ▼                                                    │
  Global Refinement                                          │
  (generalised eigenvalue over diverse circuits)             │
        │                                                    │
        ▼                                                    │
  L-BFGS-B Fine-tuning                                       │
  (continuous-angle optimisation of best circuit)            │
        │                                                    │
        ▼                                                    │
  Final Bank QSCI ◄────────────────────────────────────────►│
  (diagonalisation over full persistent bank)                │
        │                                                    │
        ▼                                                    │
   Ground-State Energy Estimate
```

### Key Result — H₂O / STO-3G / 12 qubits

| Method | Energy (Ha) | \|Δ\| vs FCI |
|--------|-------------|--------------|
| Hartree-Fock | −74.961902 | 50.69 mHa |
| ADAPT bootstrap | −74.961902 | 50.69 mHa |
| RL best (raw) | −75.012451 | 0.14 mHa |
| QSCI on elite circuit | −75.012294 | 0.30 mHa |
| **Bank QSCI (`gqex`)** | **−75.012596** | **0.00 mHa ✅** |
| CASCI / FCI reference | −75.012596 | — |

> ✅ **100% of the correlation energy (50.7 mHa) recovered**, well below the 1.6 mHa chemical-accuracy threshold.

---

## 📁 Repository Structure

```
GIC-Mitsubishi/
│
├── 📄 eQoSystem__Phase2_Version1.pdf           ← Our official Phase 2 submission
│
├── 📄 Mitsubishi Chemical and AIST_Phase 2     ← Official challenge description
│      Challenge Description.pdf
│
└── 📦 GQEx Package/                            ← Full Python package implementation
    │
    ├── pyproject.toml                          ← Build & dependency config
    ├── MANIFEST.in
    ├── LICENSE                                 ← MIT License
    │
    ├── gqex/                                   ← Main library source
    │   ├── __init__.py                         ← Public API & lazy imports
    │   │
    │   ├── hamiltonian/                        ← Molecular Hamiltonian construction
    │   │   ├── molecular_system.py             ← MolecularSystem class, JW mapping
    │   │   └── tapering.py                     ← Z2 qubit tapering
    │   │
    │   ├── ansatz/                             ← Gate pools & CUDA-Q kernels
    │   │   ├── gate_pool.py                    ← Standard / NumberPreserving / Fermionic pools
    │   │   └── kernel_factory.py               ← CUDA-Q kernel builder
    │   │
    │   ├── generator/                          ← Circuit generator (RetNet)
    │   │   ├── retnet.py                       ← Multi-head Retentive Network
    │   │   └── sampling.py                     ← Batched circuit sampling utilities
    │   │
    │   ├── qsci/                               ← Quantum-Selected CI
    │   │   ├── evaluator.py                    ← QSCI energy evaluator
    │   │   ├── det_bank.py                     ← Persistent DeterminantBank
    │   │   └── sqd.py                          ← Sample-based quantum diagonalisation
    │   │
    │   ├── gqe/                                ← GQE training engine
    │   │   ├── engine.py                       ← REINFORCE with EMA baseline
    │   │   └── finetune.py                     ← L-BFGS-B continuous fine-tuning
    │   │
    │   ├── adapt/                              ← ADAPT-VQE bootstrap
    │   │   └── bootstrap.py                    ← Greedy gate selection
    │   │
    │   ├── forging/                            ← Entanglement forging
    │   │   └── entanglement.py                 ← Closed-shell qubit halving
    │   │
    │   ├── pipeline/                           ← End-to-end orchestration
    │   │   ├── config.py                       ← PipelineConfig + quick_config presets
    │   │   └── orchestrator.py                 ← GQExPipeline class
    │   │
    │   └── utils/                              ← Shared utilities
    │       └── conversions.py                  ← OpenFermion ↔ CUDA-Q conversions
    │
    ├── demos/                                  ← Runnable demo scripts
    │   ├── demo.py                             ← Main demo (molecule library)
    │   ├── setup.py
    │   ├── test.py
    │   └── test2.py
    │
    └── dist/                                   ← Built distributions
        ├── gqex-0.1.1-py3-none-any.whl
        └── gqex-0.1.1.tar.gz
```

---

## 📦 The `gqex` Python Package

### Package Architecture

Each sub-module addresses a distinct scalability bottleneck of the baseline GQE:

| Module | Bottleneck Addressed |
|--------|---------------------|
| `hamiltonian/` | Slow Hamiltonian construction; qubit overhead |
| `ansatz/` | Wasted shots from symmetry-violating gates |
| `generator/` | Quadratic inference cost in sequence models |
| `qsci/` | Throwing away shots between RL iterations |
| `gqe/` | Policy collapse; poor exploration |
| `adapt/` | Cold-start problem (bad initial circuits) |
| `forging/` | Large qubit counts in closed-shell systems |
| `pipeline/` | Complexity of assembling all modules together |

---

### Installation

#### Basic install (CPU)

```bash
pip install gqex
```

#### With NVIDIA CUDA-Q backend (recommended for performance)

```bash
pip install "gqex[cuda]"
```

> **Note:** CUDA-Q requires NVIDIA hardware and CUDA 12.x. See the [official CUDA-Q guide](https://nvidia.github.io/cuda-quantum/) for prerequisites.

#### Development install (from source)

```bash
git clone https://github.com/AshrafBoussahi/gqex.git
cd gqex
pip install -e ".[dev,cuda]"
```

#### Optional extras

| Extra | Command | Purpose |
|-------|---------|---------|
| `cuda` | `pip install "gqex[cuda]"` | NVIDIA CUDA-Q quantum backend |
| `dev` | `pip install "gqex[dev]"` | Testing, linting (pytest, black, ruff) |
| `viz` | `pip install "gqex[viz]"` | Matplotlib + Jupyter visualization |
| `all` | `pip install "gqex[all]"` | Everything above |

#### Requirements

- Python 3.9 – 3.12
- NumPy ≥ 1.24, SciPy ≥ 1.10
- PyTorch ≥ 2.0
- PySCF ≥ 2.3, OpenFermion ≥ 1.5, OpenFermion-PySCF ≥ 0.5
- *(optional)* CUDA-Q ≥ 0.7

---

### Quick Start

#### Three-line demo

```python
from gqex import GQExPipeline, MolecularSystem, quick_config

h2o = MolecularSystem(
    name               = "H2O",
    xyz                = "O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
    basis              = "sto-3g",
    n_active_electrons = 8,
    n_active_orbitals  = 6,
    compute_casci      = True,
)
results = GQExPipeline(h2o, quick_config("small")).run()
print(f"Final energy: {results['bank_energy']:+.6f} Ha")
```

#### Command-line demo

```bash
gqex-demo --mol h2o      --size small
gqex-demo --mol beh2     --size medium
gqex-demo --mol n2_large --size large
```

Or run directly:

```bash
python demo.py --mol h2o --size small
```

---

### Full Usage Guide

#### 1. Defining a Molecule

```python
from gqex import MolecularSystem

system = MolecularSystem(
    name               = "BeH2",
    xyz                = "Be 0 0 0\nH 0 0 1.33\nH 0 0 -1.33",
    basis              = "6-31g",
    charge             = 0,
    spin               = 0,
    n_active_electrons = 4,
    n_active_orbitals  = 12,
    compute_casci      = True,   # produce a CASCI reference for error reporting
)
```

Build the Hamiltonian explicitly (with tapering):

```python
from gqex.hamiltonian.molecular_system import build_molecular_hamiltonian

build_molecular_hamiltonian(
    system,
    taper        = True,      # enable Z2 qubit tapering
    taper_method = "z2",
)
print(f"Qubits after tapering: {system.n_qubits}")
print(f"Pauli terms:           {system.n_pauli_terms}")
```

#### 2. Choosing a Configuration

Use one of three presets — or build your own:

```python
from gqex import quick_config

cfg = quick_config("small")    # ≤ 12 qubits  — runs in minutes
cfg = quick_config("medium")   # 16–24 qubits — runs in tens of minutes
cfg = quick_config("large")    # 28–40 qubits — runs in hours
```

Override any field:

```python
cfg.gqe.n_iters       = 200
cfg.gqe.n_samples     = 32
cfg.gqe.device        = "cuda"
cfg.taper.enabled     = True
cfg.bank.max_size     = 200_000
```

#### 3. Running the Full Pipeline

```python
from gqex import GQExPipeline

pipeline = GQExPipeline(system, cfg)
results  = pipeline.run()
```

**Pipeline stages:**

| Stage | What it does |
|-------|-------------|
| 0 | ADAPT-VQE bootstrap — greedy gate selection |
| A | RetNet + REINFORCE policy-gradient training |
| B | QSCI evaluation of the elite circuit |
| C | Persistent bank QSCI diagonalisation |
| D | Generalised-eigenvalue global refinement |
| E | L-BFGS-B continuous-angle fine-tuning |
| F | Summary printout |

**Return value:**

```python
results = {
    "adapt_energy":    ...,   # ADAPT bootstrap result
    "rl_best_energy":  ...,   # best discrete circuit from RL
    "qsci_energy":     ...,   # QSCI on the elite circuit
    "bank_energy":     ...,   # QSCI over the full persistent bank  ← headline metric
    "global_energy":   ...,   # generalised-eigenvalue refinement
    "finetune_energy": ...,   # L-BFGS-B continuous fine-tuning
    "best_ops":        [...], # discrete gate-index sequence
    "history":         {...}, # per-iteration training history
    "n_qubits":        ...,
    "pool_size":       ...,
}
```

#### 4. Using Individual Sub-modules

Every component is independently importable:

**Build a number-preserving gate pool:**

```python
from gqex.ansatz import build_pool, build_kernel

pool   = build_pool("np", n_qubits=12,
                   include_givens_adj=True,
                   include_givens_lr=True,
                   include_double_exc=True)
kernel = build_kernel(pool, hf_x_qubits=[0, 1, 2, 3])
print(pool.describe())
```

**Run ADAPT-VQE bootstrap standalone:**

```python
from gqex.adapt import adapt_bootstrap, ADAPTConfig

adapt_ops, adapt_e = adapt_bootstrap(
    pool   = pool,
    system = system,
    kernel = kernel,
    cfg    = ADAPTConfig(max_gates=30, energy_tol=1e-4),
)
```

**Train the RetNet generator:**

```python
from gqex.generator import RetNetGenerator, RetNetConfig

gen = RetNetGenerator(RetNetConfig(
    pool_size  = pool.pool_size,
    hidden_dim = 128,
    n_layers   = 4,
    n_heads    = 4,
    max_len    = 256,
    gated_mlp  = True,
))
```

**Use the persistent determinant bank:**

```python
from gqex.qsci import DeterminantBank

bank = DeterminantBank(max_size=50_000, n_alpha=2, n_beta=2, n_qubits=12)
bank.update(["010101", "101010"])   # contribute bitstring samples
print(bank.stats())                  # {'size': ..., 'total_shots': ...}
```

**Run QSCI directly:**

```python
from gqex.qsci import QSCIEvaluator, QSCIConfig

qsci = QSCIEvaluator(system, QSCIConfig(n_shots=20_000, d_max=500), bank=bank)
energy, wavefunction, n_dets = qsci.evaluate(kernel, adapt_ops)
```

**Fine-tune continuous angles:**

```python
from gqex.gqe import finetune_circuit

theta_opt, e_ft = finetune_circuit(
    best_ops = results["best_ops"],
    pool     = pool,
    system   = system,
    max_iter = 400,
)
```

---

### Configuration Reference

The master configuration is `PipelineConfig`. Key knobs:

| Field | Default | Meaning |
|-------|---------|---------|
| `gqe.depth` | `0` (auto) | Circuit depth (0 → 4 × n_qubits) |
| `gqe.n_samples` | `16` | Circuits sampled per RL iteration |
| `gqe.n_iters` | `120` | Total RL iterations |
| `gqe.lr` | `3e-3` | Adam learning rate |
| `gqe.reward_mode` | `"qsci"` | `"expectation"` (fast) or `"qsci"` (paper mode) |
| `gqe.device` | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `gqe.use_recurrent` | `True` | Use RetNet O(T) recurrent sampling |
| `gqe.use_rank_adv` | `True` | Rank-based advantages (robust to scale) |
| `gqe.diversity_coef` | `0.05` | Bonus for unique circuits in a batch |
| `gqe.entropy_coef` | `0.10` | Entropy regulariser (anti-collapse) |
| `gqe.temperature` | `1.5` | Sampling temperature (annealed to 1.0) |
| `pool.pool_type` | `"np"` | `"standard"`, `"np"`, or `"fermionic"` |
| `taper.enabled` | `False` | Run Z2 qubit tapering |
| `bank.enabled` | `True` | Use persistent determinant bank |
| `bank.max_size` | `100_000` | Cap on bank size |
| `adapt.enabled` | `True` | Run ADAPT-VQE bootstrap |
| `adapt.max_gates` | `30` | Maximum gates the bootstrap appends |
| `finetune.enabled` | `True` | Run L-BFGS-B continuous fine-tuning |
| `gen.hidden_dim` | `128` | RetNet hidden width |
| `gen.n_layers` | `4` | RetNet depth |
| `gen.n_heads` | `4` | RetNet retention heads |
| `qsci.n_shots` | `30_000` | Shots per QSCI evaluation |
| `qsci.d_max` | `800` | Max determinants kept |

---

### Benchmark Results

#### Pre-built Molecule Library

| Key | System | Basis | Qubits | Notes |
|-----|--------|-------|--------|-------|
| `h2` | H₂ | STO-3G | 4 | Toy validation |
| `h2o` | H₂O | 6-31G | 14 | Standard medium benchmark |
| `lih` | LiH | 6-31G | 10 | Frozen-core small system |
| `beh2_small` | BeH₂ | STO-3G | 14 | Fast iteration |
| `beh2` | BeH₂ | 6-31G | 24 | Original GQE paper benchmark |
| `n2_large` | N₂ | cc-pVDZ | 40 | ~40-qubit scaling target |

#### Energy Results

| System | Qubits | Status | Best \|Δ\| from CASCI |
|--------|--------|--------|----------------------|
| H₂O / STO-3G | 12 | ✅ Validated | **0.00 mHa** |
| BeH₂ / 6-31G | 24 | RL converges | 28 mHa (CPU, partial) |
| N₂ / 6-31G | 32 → 28\* | Builds & runs | Bigger-budget target |

*\*after Z2 tapering*

---

## 📄 Technical Proposal

Our full technical write-up is available in the repository:

📎 **[`eQoSystem__Phase2_Version1.pdf`](./eQoSystem__Phase2_Version1.pdf)**

The proposal covers:

- **Problem & Decomposition** — splitting into a discrete investment master and a continuous operations subproblem
- **Architecture** — full hybrid quantum-classical pipeline with 5 detailed modules
- **Quantum Master (cGQE-PCE)** — Conditional GQE with Pauli Correlation Encoding (PCE) for qubit compression
- **Classical Subproblem** — Security-constrained Optimal Power Flow (DC-SCOPF / SOCP) for network physics
- **DPO Training** — Direct Preference Optimization for quantum circuit training without hardware gradients
- **Platform Justification** — qBraid Lab, NVIDIA CUDA-Q, IBM Heron r1 QPU
- **Validation & Current Status** — unit-tested modules, verified IEEE 14-bus baseline

The official challenge description is also included:

📎 **[`Mitsubishi Chemical and AIST_Phase 2 Challenge Description.pdf`](./Mitsubishi%20Chemical%20and%20AIST_Phase%202%20Challenge%20Description.pdf)**

---

## 🗺️ Roadmap

- [ ] **GPU sampling parity** — push `n_samples ≥ 64` on CUDA
- [ ] **PPO-style trust-region updates** for more stable policy training
- [ ] **PySCF `kernel_fixed_space`** integration for fast H_sub diagonalisation
- [ ] **Full Z2 + non-Abelian tapering** (currently Z2-only)
- [ ] **Entanglement-forging pipeline path** (module exists, not yet wired)
- [ ] **Imitation pretraining** on UCCSD orderings before RL
- [ ] **Multi-fidelity sampling** (cheap classical eval + expensive quantum eval)
- [ ] **Continuous-fermion ADAPT** with parameter-shift gradients

---

## 📚 Citation & References

If you use `gqex` in academic work, please cite:

```bibtex
@software{gqex2026,
    author  = {Boussahi, Ashraf},
    title   = {{gqex}: A scalable RetNet-driven Generative Quantum Eigensolver
               package for ~40-qubit quantum chemistry},
    year    = {2026},
    url     = {https://github.com/AshrafBoussahi/gqex},
    version = {0.1.1},
}
```

**Underlying methods:**

| Method | Reference |
|--------|-----------|
| Generative Quantum Eigensolver (GQE) | Nakaji et al., arXiv:2401.09253 (2024) |
| Conditional GQE (cGQE) | Panaganti et al., arXiv:2411.03555 (2024) |
| Retentive Network (RetNet) | Sun et al., arXiv:2307.08621 (2023) |
| Quantum-Selected CI (QSCI) | Kemmoku et al., arXiv:2403.13800 (2024) |
| ADAPT-VQE | Grimsley et al., Nat. Commun. 10, 3007 (2019) |
| Z2 Qubit Tapering | Bravyi et al., arXiv:1701.08213 (2017) |
| Direct Preference Optimization (DPO) | Rafailov et al., NeurIPS 2023 |
| Pauli Correlation Encoding (PCE) | Sciorilli et al., arXiv:2501.06241 (2025) |
| IEEE Reliability Test System | Barrows et al., IEEE Trans. Power Syst. 35(1), 2020 |

---

## 🤝 Contributing

```bash
git clone https://github.com/AshrafBoussahi/gqex.git
cd gqex
pip install -e ".[dev,cuda]"
pytest               # run the test suite
black gqex/ && isort gqex/ && ruff check gqex/
```

Open a Pull Request describing your change. Tests on the H₂O / STO-3G benchmark are appreciated for any algorithmic contribution.

---

## 📜 License

`gqex` is distributed under the **MIT License** — see [`GQEx Package/LICENSE`](GQEx%20Package/LICENSE) for the full text.

Copyright © 2026 Ashraf Boussahi.

---

<div align="center">

**Built with ⚛️ quantum computing and ❤️ by team eQoSystem**

*ESI-SBA, Algeria · GIC 2026 · Phase 2 Submission*

</div>
