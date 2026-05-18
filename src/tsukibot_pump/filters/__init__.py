"""Filter stack — six selection edges that collectively form the laptop's
defense against the 87% memecoin failure rate on Solana.

Each filter takes a `TokenState` and returns a `FilterOutcome`. Filters are
deliberately decoupled: they can be unit-tested independently and disabled
individually via config.
"""

from .bundle_detector import BundleDetector
from .convergence import ConvergenceDetector
from .cto_detector import CtoDetector
from .curve_predictor import CurvePredictor
from .dev_blacklist import DevBlacklist
from .first_kol_touch import FirstKolTouch

__all__ = [
    "BundleDetector",
    "ConvergenceDetector",
    "CtoDetector",
    "CurvePredictor",
    "DevBlacklist",
    "FirstKolTouch",
]
