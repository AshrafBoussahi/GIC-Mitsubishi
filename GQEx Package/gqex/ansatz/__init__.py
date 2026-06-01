def __getattr__(name):
    if name in ("StandardPool", "NumberPreservingPool", "FermionicPool", "build_pool"):
        import importlib
        mod = importlib.import_module("gqex.ansatz.gate_pool")
        return getattr(mod, name)
    if name in ("make_kernel", "make_np_kernel", "make_param_kernel", "build_kernel"):
        import importlib
        mod = importlib.import_module("gqex.ansatz.kernel_factory")
        return getattr(mod, name)
    raise AttributeError(name)
