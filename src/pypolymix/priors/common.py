"""Common prior implementations used by parameter groups."""

from typing import Optional, Union

import torch
import torch.distributions as td

from .base import Prior

TensorOrFloat = Union[float, torch.Tensor]


def _as_tensor(x: TensorOrFloat, device, dtype, shape):
    """Broadcast ``x`` to ``shape`` while respecting ``device``/``dtype``."""
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x, device=device, dtype=dtype)
    return t.expand(shape).to(device=device, dtype=dtype)


class IIDGaussianPrior(Prior):
    """Independent Gaussian prior with per-parameter mean and standard deviation.

    Example:
        >>> prior = IIDGaussianPrior(mean=0.0, std=0.5)
        >>> prior.distribution(torch.Size([4]), None, None).sample().shape
        torch.Size([4])
    """

    def __init__(self, mean: TensorOrFloat = 0.0, std: TensorOrFloat = 1.0):
        """Persist broadcastable buffers for the requested mean and std."""
        super().__init__()
        self.register_buffer("_mean_buf", torch.as_tensor(mean), persistent=False)
        self.register_buffer("_std_buf", torch.as_tensor(std), persistent=False)

    def distribution(self, event_shape, device, dtype) -> td.Distribution:
        """Return ``Independent(N(mean, std), 1)`` with broadcasted parameters."""
        mean = _as_tensor(self._mean_buf, device, dtype, event_shape)
        std = _as_tensor(self._std_buf, device, dtype, event_shape)
        base = td.Normal(mean, std)
        return td.Independent(base, 1)


class GaussianPrior(Prior):
    """Full-covariance Gaussian prior :math:`\\mathcal{N}(\\mu, \\Sigma)`.

    Users must provide either ``covariance_matrix`` or ``scale_tril`` when
    instantiating the prior; the other argument should be ``None``.

    Example:
        ```python
        >>> mean = torch.zeros(2)
        >>> cov = torch.eye(2)
        >>> prior = GaussianPrior(mean, covariance_matrix=cov)
        >>> isinstance(prior.distribution(torch.Size([2]), None, None), td.MultivariateNormal)
        True
        ```
    """

    def __init__(
        self,
        mean: torch.Tensor,
        covariance_matrix: Optional[torch.Tensor] = None,
        scale_tril: Optional[torch.Tensor] = None,
    ):
        """Store either ``covariance_matrix`` or ``scale_tril`` for later use."""
        super().__init__()
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Provide exactly one of covariance_matrix or scale_tril.")

        self.register_buffer("_mean", mean.clone(), persistent=False)
        if scale_tril is not None:
            self.register_buffer("_scale_tril", scale_tril.clone(), persistent=False)
            self._cov = None
        else:
            self.register_buffer("_cov", covariance_matrix.clone(), persistent=False)
            self._scale_tril = None

    def distribution(
        self,
        event_shape: torch.Size,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> td.Distribution:
        """Validate tensor shapes and build a multivariate normal distribution."""
        d = event_shape[0]
        mean = self._mean.to(device=device, dtype=dtype)
        if mean.shape != (d,):
            raise ValueError(f"Prior mean shape {mean.shape} != {(d,)}")

        if self._scale_tril is not None:
            L = self._scale_tril.to(device=device, dtype=dtype)
            if L.shape != (d, d):
                raise ValueError(f"scale_tril shape {L.shape} != {(d, d)}")
            return td.MultivariateNormal(loc=mean, scale_tril=L)

        cov = self._cov.to(device=device, dtype=dtype)
        if cov.shape != (d, d):
            raise ValueError(f"covariance_matrix shape {cov.shape} != {(d, d)}")
        return td.MultivariateNormal(loc=mean, covariance_matrix=cov)


class UniformPrior(Prior):
    """Independent uniform prior :math:`U(\\text{low}, \\text{high})` per parameter.

    ``docs/theo.md`` documents that, for a uniform prior,
    :math:`\\text{KL}(q \\,\\|\\, p)` reduces to
    :math:`-\\tfrac{1}{2}\\log|\\Sigma_q| + \\text{const}`. That reduction is only
    exact when all of :math:`q`'s probability mass lies within ``[low, high]``;
    it is an approximation (ignoring the vanishingly small tail mass) whenever
    :math:`q` has unbounded support, as an ``IIDGaussianGroup``/``GaussianGroup``
    does.

    **Caveat (verified against this project's torch version):**
    ``torch.distributions`` *does* register an analytic
    ``kl_divergence(Normal, Uniform)`` (via ``_kl_normal_infinity``), and it
    unconditionally returns ``+inf`` -- regardless of ``low``/``high`` or how
    well ``q``'s mean is centered -- because a Gaussian has nonzero density
    everywhere outside any bounded interval, so the true KL divergence from an
    unbounded distribution to a compactly-supported one genuinely is infinite.
    Consequently ``ParameterGroup.distribution_loss()`` takes its analytic
    branch (no ``NotImplementedError`` is raised) and returns a constant
    ``+inf`` tensor with no gradient, for *any* Gaussian variational family
    paired with this prior; it never reaches the Monte Carlo fallback for that
    combination. Making ``distribution_loss()`` degrade gracefully for this
    pairing (e.g. detecting non-finite analytic KL and falling back to the MC
    estimator) is tracked separately as KL-estimator machinery work and is
    intentionally not changed here.

    ``Uniform`` also has compact support in its own right: if a variational
    family *with bounded support* places mass or samples outside
    ``[low, high]``, ``log_prob`` returns ``-inf``, which can produce ``NaN``
    gradients in a Monte Carlo KL estimate. Choose ``low``/``high`` generously
    wide relative to the expected posterior.

    Example:
        >>> prior = UniformPrior(low=-1.0, high=1.0)
        >>> prior.distribution(torch.Size([4]), None, None).sample().shape
        torch.Size([4])
    """

    def __init__(self, low: TensorOrFloat = -10.0, high: TensorOrFloat = 10.0):
        """Persist broadcastable buffers for the requested low/high bounds."""
        super().__init__()
        self.register_buffer("_low", torch.as_tensor(low), persistent=False)
        self.register_buffer("_high", torch.as_tensor(high), persistent=False)

    def distribution(self, event_shape, device, dtype) -> td.Distribution:
        """Return ``Independent(Uniform(low, high), 1)`` with broadcasted parameters."""
        low = _as_tensor(self._low, device, dtype, event_shape)
        high = _as_tensor(self._high, device, dtype, event_shape)
        return td.Independent(td.Uniform(low, high), 1)


class LaplacePrior(Prior):
    """IID Laplace prior that encourages sparsity.

    Example:
        ```python
        >>> prior = LaplacePrior(loc=0.0, scale=1e-1)
        >>> prior.distribution(torch.Size([3]), None, None).sample().shape
        torch.Size([3])
        ```
    """

    def __init__(self, loc=0.0, scale=1.0):
        """Persist broadcastable buffers for the Laplace hyper-parameters."""
        super().__init__()
        self.register_buffer("_loc", torch.as_tensor(loc), persistent=False)
        self.register_buffer("_scale", torch.as_tensor(scale), persistent=False)

    def distribution(self, event_shape, device, dtype):
        """Return ``Independent(Laplace(loc, scale), 1)`` with broadcasted params."""
        loc = _as_tensor(self._loc, device, dtype, event_shape)
        scale = _as_tensor(self._scale, device, dtype, event_shape)
        return td.Independent(td.Laplace(loc, scale), 1)
