from gqex.qsci.det_bank import DeterminantBank

def __getattr__(name):
    if name in ("QSCIEvaluator", "QSCIConfig"):
        from gqex.qsci.evaluator import QSCIEvaluator, QSCIConfig
        return locals()[name]
    if name == "sqd_recover":
        from gqex.qsci.sqd import sqd_recover
        return sqd_recover
    raise AttributeError(name)
