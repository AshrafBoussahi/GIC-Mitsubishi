def __getattr__(name):
    if name in ("RetNetGenerator", "RetNetConfig"):
        import importlib
        mod = importlib.import_module("gqex.generator.retnet")
        return getattr(mod, name)
    if name in ("sample_circuit", "sample_batch", "log_prob", "log_prob_batch", "beam_search"):
        import importlib
        mod = importlib.import_module("gqex.generator.sampling")
        return getattr(mod, name)
    raise AttributeError(name)
