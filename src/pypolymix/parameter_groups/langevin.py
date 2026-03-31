"""Implicit Langevin parameter group driven by a learned score or energy surrogate."""

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

    The group learns either a score model ``s(theta)`` or an energy model
    ``E(theta)`` and generates samples by iterating:

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
        score_model: Optional[SurrogateModel] = None,
        energy_model: Optional[SurrogateModel] = None,
        prior: Optional[Prior] = None,
        *,
        num_particles: int = 32,
        num_diffusion_steps: int = 50,
        step_size: float = 1e-3,
        init_std: float = 1.0,
        theta_clip: float | None = None,
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
        if theta_clip is not None and theta_clip <= 0.0:
            raise ValueError("theta_clip must be > 0 when provided")

        if (score_model is None) == (energy_model is None):
            raise ValueError("Provide exactly one of score_model or energy_model.")

        self.score_model = score_model
        self.energy_model = energy_model
        self.num_particles = int(num_particles)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.step_size = float(step_size)
        self.init_std = float(init_std)
        self.theta_clip = None if theta_clip is None else float(theta_clip)

        surrogate = score_model if score_model is not None else energy_model
        self._mode = "score" if score_model is not None else "energy"
        self._validate_surrogate(surrogate)

        # Deterministic parameter vector for the score/energy surrogate.
        self.surrogate_params = nn.Parameter(torch.randn(surrogate.num_params()) * 0.01)
        self._last_samples: Optional[torch.Tensor] = None

    def _validate_surrogate(self, surrogate_model: SurrogateModel) -> None:
        """Validate the score or energy surrogate dimensions."""
        if not hasattr(surrogate_model, "num_inputs") or not hasattr(surrogate_model, "num_outputs"):
            raise ValueError(
                "Langevin surrogate must define 'num_inputs' and 'num_outputs' attributes to validate dimensions."
            )
        if surrogate_model.num_inputs != self.num_params:
            raise ValueError(
                f"Langevin surrogate num_inputs ({surrogate_model.num_inputs}) must equal num_params ({self.num_params})."
            )

        expected_outputs = self.num_params if self._mode == "score" else 1
        if surrogate_model.num_outputs != expected_outputs:
            raise ValueError(
                f"{self._mode}_model.num_outputs ({surrogate_model.num_outputs}) must equal {expected_outputs}."
            )

    def _energy(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned scalar energy/log-density surrogate."""
        if self.energy_model is None:
            raise RuntimeError("energy_model is not configured for this LangevinGroup.")

        model_params = self.surrogate_params.unsqueeze(0)
        energy = self.energy_model(theta, model_params).squeeze(0).squeeze(-1)
        return energy

    def _score(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned score field on a batch of particles."""
        if self.score_model is not None:
            model_params = self.surrogate_params.unsqueeze(0)  # (1, n_surrogate_params)
            return self.score_model(theta, model_params).squeeze(0)  # (n_particles, num_params)

        create_graph = torch.is_grad_enabled()
        theta_for_grad = theta if theta.requires_grad else theta.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self._energy(theta_for_grad)
            score = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=theta_for_grad,
                create_graph=create_graph,
            )[0]
        return score

    def _clip_theta_norm(self, theta: torch.Tensor) -> torch.Tensor:
        """Clip particle norms row-wise before score evaluation."""
        if self.theta_clip is None:
            return theta

        norms = theta.norm(dim=1, keepdim=True)
        safe_norms = norms.clamp_min(torch.finfo(theta.dtype).eps)
        scales = torch.clamp(self.theta_clip / safe_norms, max=1.0)
        return theta * scales

    def sample_parameters(self, num_samples: int | None = None) -> torch.Tensor:
        """Draw parameter samples by running Langevin dynamics."""
        if num_samples is None:
            num_samples = self.num_particles
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        device = self.surrogate_params.device
        dtype = self.surrogate_params.dtype
        theta = torch.randn(num_samples, self.num_params, device=device, dtype=dtype) * self.init_std
        noise_scale = torch.sqrt(torch.tensor(2.0 * self.step_size, device=device, dtype=dtype))

        for _ in range(self.num_diffusion_steps):
            theta = self._clip_theta_norm(theta)
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
