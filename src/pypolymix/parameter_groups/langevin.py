"""Implicit Langevin parameter group driven by a learned score surrogate."""

from __future__ import annotations

from typing import Optional
import warnings

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
        self.theta_clip = None if theta_clip is None else float(theta_clip)

        # Deterministic parameter vector for the score surrogate.
        self.score_params = nn.Parameter(torch.randn(score_model.num_params()) * 0.01)
        self._last_samples: Optional[torch.Tensor] = None

    def _fresh_particles(self, num_particles: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draw fresh particles from the Langevin initializer."""
        return torch.randn(num_particles, self.num_params, device=device, dtype=dtype) * self.init_std

    def _replace_invalid_rows(
        self,
        tensor: torch.Tensor,
        valid_rows: torch.Tensor,
        *,
        stage: str,
    ) -> torch.Tensor:
        """Replace non-finite particle rows with fresh initial draws."""
        if valid_rows.all():
            return tensor

        repaired = tensor.clone()
        num_bad = int((~valid_rows).sum().item())
        repaired[~valid_rows] = self._fresh_particles(num_bad, tensor.device, tensor.dtype)
        warnings.warn(
            f"LangevinGroup detected {num_bad} non-finite particle trajectories during {stage}; "
            "restarting those rows from the initializer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return repaired

    def _score(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned score model on a batch of particles."""
        model_params = self.score_params.unsqueeze(0)  # (1, n_score_params)
        score = self.score_model(theta, model_params).squeeze(0)  # (n_particles, num_params)
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

        device = self.score_params.device
        dtype = self.score_params.dtype
        theta = self._fresh_particles(num_samples, device, dtype)
        noise_scale = torch.sqrt(torch.tensor(2.0 * self.step_size, device=device, dtype=dtype))

        for step in range(self.num_diffusion_steps):
            theta = self._clip_theta_norm(theta)
            score = self._score(theta)
            valid_score = torch.isfinite(score).all(dim=1)
            if not valid_score.all():
                bad = int((~valid_score).sum().item())
                print(
                    f"[LangevinGroup] non-finite score at step {step}: "
                    f"{bad}/{num_samples} invalid rows, "
                    f"max|theta|={torch.nan_to_num(theta).abs().max().item():.3e}, "
                    f"max|score|={torch.nan_to_num(score).abs().max().item():.3e}"
                )
                theta = self._replace_invalid_rows(theta, valid_score, stage="score evaluation")
                score = self._score(theta)
            drift = self.step_size * score
            valid_drift = torch.isfinite(drift).all(dim=1)
            if not valid_drift.all():
                bad = int((~valid_drift).sum().item())
                print(
                    f"[LangevinGroup] non-finite drift at step {step}: "
                    f"{bad}/{num_samples} invalid rows, "
                    f"step_size={self.step_size:.3e}, "
                    f"max|score|={torch.nan_to_num(score).abs().max().item():.3e}, "
                    f"max|drift|={torch.nan_to_num(drift).abs().max().item():.3e}"
                )
            diffusion = noise_scale * torch.randn_like(theta)
            theta = theta + drift + diffusion
            valid_theta = torch.isfinite(theta).all(dim=1)
            if not valid_theta.all():
                bad = int((~valid_theta).sum().item())
                print(
                    f"[LangevinGroup] non-finite particles after update at step {step}: "
                    f"{bad}/{num_samples} invalid rows, "
                    f"max|drift|={torch.nan_to_num(drift).abs().max().item():.3e}, "
                    f"max|diffusion|={torch.nan_to_num(diffusion).abs().max().item():.3e}, "
                    f"max|theta|={torch.nan_to_num(theta).abs().max().item():.3e}"
                )
                theta = self._replace_invalid_rows(theta, valid_theta, stage="Langevin update")

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
