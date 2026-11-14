import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT

from .._math import inverse_softplus
from .base import Likelihood


class StudentTLikelihood(Likelihood):
    """Learnable per-output Student-t observation noise with a learned,
    floored degrees-of-freedom: ``df = df_floor + softplus(raw_df)`` (``df_floor``
    defaults to 2.0, keeping the variance finite -- Student-t variance is
    undefined for ``df <= 2``).

    Use when residuals are leptokurtic (heavy-tailed) relative to Gaussian --
    a Gaussian MLE matches the variance, which a heavy tail dominates, so it
    over-covers the bulk of the residual distribution. ``df`` is often
    strongly identified by the data even when the per-point log-likelihood
    surface is nearly flat in ``df`` (profile-likelihood argument): compare
    ``2*N*d(log-lik)`` to a chi-squared(1) reference if you need to check
    significance for your own data before trusting a near-floor ``df``
    estimate.

    Like :class:`pypolymix.likelihoods.gaussian.GaussianLikelihood`, ``raw_df``
    (and the per-output scale parameter) MUST be excluded from weight decay --
    see :meth:`no_decay_parameters`.

    Example:
        ```python
        >>> import torch
        >>> from pypolymix.likelihoods import StudentTLikelihood
        >>> likelihood = StudentTLikelihood(num_outputs=2)
        >>> pred = torch.zeros(4, 2)
        >>> target = torch.zeros(4, 2)
        >>> likelihood.nll(pred, target).shape
        torch.Size([])
        ```
    """

    def __init__(
        self,
        num_outputs: int,
        init_scale: float = 1.0,
        init_df: float = 8.0,
        df_floor: float = 2.0,
    ):
        """Create learnable per-output log-scale and raw-df parameters.

        Args:
            num_outputs: Number of output dimensions.
            init_scale: Initial Student-t scale, shared across outputs at construction time.
            init_df: Initial degrees of freedom; must be strictly greater than ``df_floor``.
            df_floor: Minimum degrees of freedom, keeping the variance finite
                (Student-t variance is undefined for ``df <= 2``).
        """
        super().__init__()
        if not init_scale > 0.0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if not init_df > df_floor:
            raise ValueError(f"init_df must be > df_floor ({df_floor}), got {init_df}")
        self.num_outputs = int(num_outputs)
        self.df_floor = float(df_floor)
        self.log_scale = nn.Parameter(torch.full((self.num_outputs,), math.log(init_scale)))
        # softplus^{-1}(init_df - df_floor): start df at init_df, let it descend.
        raw0 = inverse_softplus(torch.tensor(init_df - df_floor, dtype=torch.float64)).item()
        self.raw_df = nn.Parameter(torch.full((self.num_outputs,), raw0))

    @property
    def df(self) -> torch.Tensor:
        """Degrees of freedom: ``df_floor + softplus(raw_df)``."""
        return self.df_floor + F.softplus(self.raw_df)

    def nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Student-t NLL, summed over outputs and averaged over batch/sample dims.

        Matches a downstream production codebase's ``lightning.py`` ``_objective``
        Student-t branch exactly:
        ``nll = -StudentT(df, 0, scale).log_prob(residual).sum(-1).mean()``,
        with ``StudentT.log_prob`` already carrying its own df-dependent
        normalizer (the ``lgamma`` terms), so ``df`` is identifiable directly
        from this term.
        """
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(0).expand_as(pred)
        residual = pred - target
        scale = torch.exp(self.log_scale)
        df = self.df
        return -StudentT(df, torch.zeros_like(scale), scale).log_prob(residual).sum(-1).mean()

    def no_decay_parameters(self) -> list:
        """``log_scale`` and ``raw_df`` must never be weight-decayed -- see class docstring."""
        return [self.log_scale, self.raw_df]
