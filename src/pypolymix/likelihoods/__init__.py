"""Observation-noise models for the ELBO data term."""

from .base import Likelihood
from .gaussian import GaussianLikelihood

__all__ = ["GaussianLikelihood", "Likelihood"]
