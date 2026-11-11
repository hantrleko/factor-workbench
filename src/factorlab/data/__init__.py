"""行情、财务与合成样本。"""

__all__ = [
    "load_fundamentals",
    "load_prices",
    "load_weights",
    "make_synthetic_universe",
]


def __getattr__(name: str):
    if name in {"load_fundamentals", "load_prices", "load_weights"}:
        from factorlab.data import loader

        return getattr(loader, name)
    if name == "make_synthetic_universe":
        from factorlab.data.synthetic import make_synthetic_universe

        return make_synthetic_universe
    raise AttributeError(name)
