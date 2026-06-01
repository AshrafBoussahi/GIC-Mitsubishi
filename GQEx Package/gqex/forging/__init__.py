def __getattr__(name):
    if name in ("EntanglementForging", "ForgingConfig"):
        import importlib
        mod = importlib.import_module("gqex.forging.entanglement")
        return getattr(mod, name)
    raise AttributeError(name)
