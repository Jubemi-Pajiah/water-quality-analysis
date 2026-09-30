"""Water quality indices and multivariate analysis of heavy metal data."""

from .indices import wpi, wqi, wqi_class
from .io import drop_constant, load_samples, load_standards

__all__ = ["drop_constant", "load_samples", "load_standards", "wpi", "wqi", "wqi_class"]
__version__ = "1.0.0"
