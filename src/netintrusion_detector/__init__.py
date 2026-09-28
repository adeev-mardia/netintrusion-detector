"""netintrusion-detector: ML-based network intrusion / anomaly detection on NSL-KDD.

This package trains and evaluates scikit-learn classifiers on the NSL-KDD
network intrusion dataset -- the standard, publicly-available, pre-extracted
flow-feature benchmark used throughout the intrusion-detection ML literature.
No live packet capture is required or performed; everything operates on
already-extracted tabular flow features (durations, byte counts, connection
counts, rate statistics, etc.), which is the standard offline evaluation
methodology for this kind of research.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netintrusion-detector")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
