def __getattr__(name):
    if name in ("PipelineConfig", "quick_config",
                "PoolConfig", "TaperingConfig", "BankConfig",
                "FinetuneConfig", "RefinementConfig", "ForgingPipelineConfig"):
        import importlib
        mod = importlib.import_module("gqex.pipeline.config")
        return getattr(mod, name)
    if name == "GQExPipeline":
        from gqex.pipeline.orchestrator import GQExPipeline
        return GQExPipeline
    raise AttributeError(name)
