def __getattr__(name):
    if name in ("GQEEngine", "GQEConfig"):
        import importlib
        mod = importlib.import_module("gqex.gqe.engine")
        return getattr(mod, name)
    if name == "finetune_circuit":
        from gqex.gqe.finetune import finetune_circuit
        return finetune_circuit
    raise AttributeError(name)
