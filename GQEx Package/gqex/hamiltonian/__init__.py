def __getattr__(name):
    if name in ("MolecularSystem", "build_molecular_hamiltonian"):
        from gqex.hamiltonian.molecular_system import MolecularSystem, build_molecular_hamiltonian
        return locals()[name]
    if name in ("taper_hamiltonian", "TaperingResult"):
        from gqex.hamiltonian.tapering import taper_hamiltonian, TaperingResult
        return locals()[name]
    raise AttributeError(name)
