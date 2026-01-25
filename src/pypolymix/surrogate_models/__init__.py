"""Collection of deterministic surrogate architectures shipped with pypolymix."""

from .base import SurrogateModel
from .mixture import GatingNetwork, MixtureOfExperts
from .neural_network import NeuralNetwork
from .polynomial_chaos import PolynomialChaosExpansion

__all__ = [
    "GatingNetwork",
    "MixtureOfExperts",
    "NeuralNetwork",
    "PolynomialChaosExpansion",
    "SurrogateModel",
]
