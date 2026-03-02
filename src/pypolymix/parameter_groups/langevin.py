"""Implicit Langevin parameter group driven by a learned score surrogate."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributions as td
import torch.nn as nn

from ..priors.base import Prior
from ..surrogate_models.base import SurrogateModel
from .base import ParameterGroup


class LangevinGroup(ParameterGroup):
    """Parameter group based on unadjusted Langevin dynamics.

    The group learns a score model ``s(theta)`` and generates samples by iterating:

    ``theta_{k+1} = theta_k + step_size * s(theta_k) + sqrt(2 * step_size) * xi_k``

    where ``xi_k ~ Normal(0, I)``.

    Example:
        ```python
        >>> from pypolymix.surrogate_models import NeuralNetwork
        >>> score_model = NeuralNetwork(num_inputs=6, num_outputs=6, width=16, depth=2)
        >>> group = LangevinGroup("coeffs", num_params=6, score_model=score_model)
        >>> samples = group.sample_parameters(num_samples=8)
        >>> samples.shape
        torch.Size([8, 6])
        ```
    """

    def __init__(
        self,
        name: str,
        num_params: int,
        score_model: SurrogateModel,
        prior: Optional[Prior] = None,
        *,
        num_particles: int = 32,
        num_diffusion_steps: int = 50,
        step_size: float = 1e-3,
        init_std: float = 1.0,
    ):
        super().__init__(name, num_params, prior=prior)

        if num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be > 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")
        if init_std <= 0.0:
            raise ValueError("init_std must be > 0")

        if not hasattr(score_model, "num_inputs") or not hasattr(score_model, "num_outputs"):
            raise ValueError(
                "score_model must define 'num_inputs' and 'num_outputs' attributes to validate dimensions."
            )
        if score_model.num_inputs != num_params:
            raise ValueError(
                f"score_model.num_inputs ({score_model.num_inputs}) must equal num_params ({num_params})."
            )
        if score_model.num_outputs != num_params:
            raise ValueError(
                f"score_model.num_outputs ({score_model.num_outputs}) must equal num_params ({num_params})."
            )

        self.score_model = score_model
        self.num_particles = int(num_particles)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.step_size = float(step_size)
        self.init_std = float(init_std)

        # Deterministic parameter vector for the score surrogate.
        self.score_params = nn.Parameter(torch.randn(score_model.num_params()) * 0.01)
        self._last_samples: Optional[torch.Tensor] = None

    def _score(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned score model on a batch of particles."""
        model_params = self.score_params.unsqueeze(0)  # (1, n_score_params)
        score = self.score_model(theta, model_params).squeeze(0)  # (n_particles, num_params)
        return score

    def sample_parameters(self, num_samples: int | None = None) -> torch.Tensor:
        """Draw parameter samples by running Langevin dynamics."""
        if num_samples is None:
            num_samples = self.num_particles
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        device = self.score_params.device
        dtype = self.score_params.dtype
        theta = torch.randn(num_samples, self.num_params, device=device, dtype=dtype) * self.init_std
        noise_scale = torch.sqrt(torch.tensor(2.0 * self.step_size, device=device, dtype=dtype))

        for _ in range(self.num_diffusion_steps):
            drift = self.step_size * self._score(theta)
            diffusion = noise_scale * torch.randn_like(theta)
            theta = theta + drift + diffusion

        self._last_samples = theta
        return theta

    def variational_distribution(self) -> td.Distribution:
        """Langevin sampling defines an implicit posterior, not an analytic distribution."""
        raise RuntimeError(
            "LangevinGroup defines an implicit sampler and does not expose an analytic variational distribution."
        )

    def distribution_loss(self) -> torch.Tensor:
        """Monte Carlo estimate of ``-E_q[log p(theta)]`` under recent particles."""
        samples = self._last_samples
        if samples is None:
            samples = self.sample_parameters(self.num_particles)

        p = self.prior.distribution(
            event_shape=torch.Size([self.num_params]),
            device=samples.device,
            dtype=samples.dtype,
        )
        return -p.log_prob(samples).mean()
