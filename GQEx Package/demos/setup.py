from setuptools import setup, find_packages

setup(
    name="gqex",
    version="0.1.0",
    description="Generative Quantum Eigensolver eXtended — scalable GQE-QSCI to ~40 qubits",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "scipy>=1.10",
        "openfermion>=1.5",
        "openfermionpyscf>=0.5",
        "pyscf>=2.3",
    ],
    extras_require={
        "cuda": ["cudaq"],
        "dev":  ["pytest", "black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
