from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Likelihood(nn.Module, ABC):
    """Abstract base class for observation-noise models used in the ELBO data term.

    Subclasses own the learnable observation-noise parameters (e.g. a per-output
    scale) and expose a single ``nll`` reduction convention that ``elbo_loss``
    assumes: a *mean-per-datapoint* negative log-likelihood, regardless of the
    minibatch size used to compute it. See :func:`pypolymix.elbo.elbo_loss` for
    how this combines with a KL term.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood, summed over the output dimension and
        averaged over the batch dimension (and any leading sample dimension in
        ``pred``, if present: ``pred`` may be ``(S, batch, num_outputs)`` from
        ``StochasticModel.forward`` with ``num_samples=S``, ``target``
        ``(batch, num_outputs)`` or ``(S, batch, num_outputs)``). Returns a
        scalar.

        This is a MEAN-per-datapoint quantity, matching the convention needed
        by :func:`pypolymix.elbo.elbo_loss`: combine as
        ``elbo_loss(nll, kl, n_train=<full dataset row count>)`` regardless of
        the actual minibatch size used to compute ``nll`` -- there is no
        full-batch special case under this convention.
        """
        raise NotImplementedError()

    @abstractmethod
    def no_decay_parameters(self) -> list:
        """Parameters (e.g. ``log_sigma``) that must be excluded
        from weight decay in the optimizer -- see each subclass docstring for
        why."""
        raise NotImplementedError()
