def __getattr__(name):
    if name in ("adapt_bootstrap", "adapt_with_qsci", "ADAPTConfig"):
        import importlib
        mod = importlib.import_module("gqex.adapt.bootstrap")
        return getattr(mod, name)
    raise AttributeError(name)
