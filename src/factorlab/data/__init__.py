"""行情、财务与合成样本。"""

from factorlab.data.loader import load_fundamentals, load_prices, load_weights
from factorlab.data.synthetic import make_synthetic_universe

__all__ = [
    "load_fundamentals",
    "load_prices",
    "load_weights",
    "make_synthetic_universe",
]
