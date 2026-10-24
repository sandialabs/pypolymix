import math

import torch
import torch.nn as nn

from .base import Likelihood

LOG_2PI = math.log(2.0 * math.pi)


class GaussianLikelihood(Likelihood):
    """Learnable per-output Gaussian observation noise.

    NLL = 0.5 * sum_d [ (y_d - yhat_d)^2 / sigma_d^2 + log(2*pi*sigma_d^2) ],
    summed over the d=num_outputs dimension, MEAN over the batch (and sample)
    dimension(s) -- matching the log-likelihood term of the ELBO in
    ``docs/theo.md`` and exactly matching a downstream production codebase's
    already-correct implementation (see this project's handoff notes, PM-07,
    for the line-by-line correspondence).

    The normalizer term (``log(2*pi*sigma_d^2)``) is what makes sigma
    identifiable: drop it and ``d(loss)/d(log_sigma) < 0`` everywhere, so sigma
    runs away to a degenerate extreme under gradient descent regardless of fit
    quality. This exact failure mode (with a learnable precision ``tau`` rather
    than ``sigma``, same root cause) was previously found shipping in this
    library's own tutorial notebooks (``nn.ipynb``, ``moe.ipynb``) -- see
    PM-07's handoff notes for the derivation.

    ``log_sigma`` MUST be excluded from weight decay in the optimizer (see
    :meth:`no_decay_parameters`) -- decoupled weight decay pulls ``log_sigma``
    toward 0 (``sigma`` toward 1) independent of the data, silently asserting a
    specific noise scale instead of letting the likelihood identify it.

    Further caveat (not fixed by this class): even with the normalizer
    present, jointly optimizing the mean and a learned variance via gradient
    descent can converge to poor-but-stable estimates because large sigma
    down-weights a datapoint's contribution to the mean's gradient
    (``1/sigma^2`` scaling) -- see Seitzer, Tavakoli, Antic, Martius (2022),
    "On the Pitfalls of Heteroscedastic Uncertainty Estimation with
    Probabilistic Neural Networks," arXiv:2203.09168 (ICLR 2022), who propose
    beta-NLL as a mitigation not implemented here.

    Example:
        ```python
        >>> import torch
        >>> from pypolymix.likelihoods import GaussianLikelihood
        >>> likelihood = GaussianLikelihood(num_outputs=2)
        >>> pred = torch.zeros(4, 2)
        >>> target = torch.zeros(4, 2)
        >>> likelihood.nll(pred, target).shape
        torch.Size([])
        ```
    """

    def __init__(self, num_outputs: int, init_sigma: float = 1.0):
        """Create a learnable per-output log-sigma parameter.

        Args:
            num_outputs: Number of output dimensions (``d`` in the docstring above).
            init_sigma: Initial observation-noise standard deviation, shared across
                outputs at construction time.
        """
        super().__init__()
        if not init_sigma > 0.0:
            raise ValueError(f"init_sigma must be positive, got {init_sigma}")
        self.num_outputs = int(num_outputs)
        self.log_sigma = nn.Parameter(torch.full((self.num_outputs,), math.log(init_sigma)))

    def nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL, summed over outputs and averaged over batch/sample dims.

        Matches a downstream production codebase's ``lightning.py`` ``_objective``
        Gaussian branch exactly:
        ``nll = 0.5 * (residual**2 * exp(-2*log_sigma)).sum(-1).mean()``
        ``nll = nll + 0.5 * (LOG_2PI + 2*log_sigma).sum()``
        """
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(0).expand_as(pred)
        residual = pred - target
        data_term = 0.5 * (residual**2 * torch.exp(-2.0 * self.log_sigma)).sum(-1).mean()
        normalizer = 0.5 * (LOG_2PI + 2.0 * self.log_sigma).sum()
        return data_term + normalizer

    def no_decay_parameters(self) -> list:
        """``log_sigma`` must never be weight-decayed -- see class docstring."""
        return [self.log_sigma]
