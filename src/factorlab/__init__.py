"""轻量级多因子检验与 Brinson / Barra 归因工作台。"""

__version__ = "0.1.0"
__all__ = ["FACTORS", "run_analysis", "__version__"]


def __getattr__(name: str):
    if name == "FACTORS":
        from factorlab.schema import FACTORS

        return FACTORS
    if name == "run_analysis":
        from factorlab.pipeline import run_analysis

        return run_analysis
    raise AttributeError(name)
