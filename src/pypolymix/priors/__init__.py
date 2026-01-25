"""Prior factories that can be attached to ``ParameterGroup`` objects."""

from .base import Prior
from .common import GaussianPrior, IIDGaussianPrior, LaplacePrior

__all__ = ["GaussianPrior", "IIDGaussianPrior", "LaplacePrior", "Prior"]
