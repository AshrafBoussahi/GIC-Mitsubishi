def __getattr__(name):
    if name in ("of_to_cudaq", "cudaq_to_of", "qubit_op_to_matrix",
                "bitstring_to_index", "index_to_bitstring", "apply_pauli_to_state"):
        import importlib
        mod = importlib.import_module("gqex.utils.conversions")
        return getattr(mod, name)
    raise AttributeError(name)
