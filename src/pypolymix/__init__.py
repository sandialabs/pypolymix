"""Public package surface for ``pypolymix``.

The top-level package re-exports the building blocks most users interact with
when defining stochastic surrogate models:

* :mod:`pypolymix.parameter_groups` - parameter priors/variational families
* :mod:`pypolymix.surrogate_models` - deterministic surrogate architectures
* :mod:`pypolymix.likelihoods` - observation-noise models for the ELBO data term
* :func:`pypolymix.elbo.elbo_loss` - combines a likelihood NLL and a KL term into an ELBO loss
* :class:`pypolymix.stochastic_model.StochasticModel` - combines both pieces

This allows short imports such as ``from pypolymix import StochasticModel`` in
examples and the generated API docs.
"""

from . import likelihoods, parameter_groups, surrogate_models
from .elbo import elbo_loss
from .stochastic_model import StochasticModel

__all__ = [
    "elbo_loss",
    "likelihoods",
    "parameter_groups",
    "surrogate_models",
    "StochasticModel",
]
