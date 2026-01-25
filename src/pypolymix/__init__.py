"""Public package surface for ``pypolymix``.

The top-level package re-exports the building blocks most users interact with
when defining stochastic surrogate models:

* :mod:`pypolymix.parameter_groups` - parameter priors/variational families
* :mod:`pypolymix.surrogate_models` - deterministic surrogate architectures
* :class:`pypolymix.stochastic_model.StochasticModel` - combines both pieces

This allows short imports such as ``from pypolymix import StochasticModel`` in
examples and the generated API docs.
"""

from . import parameter_groups, surrogate_models
from .stochastic_model import StochasticModel

__all__ = ["parameter_groups", "surrogate_models", "StochasticModel"]
