"""Exact-score (preconditioned) Stochastic Gradient Langevin Dynamics group."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributions as td

from ..priors.base import Prior
from .base import ParameterGroup

LogLikelihoodFn = Callable[[torch.Tensor], torch.Tensor]


class SGLDGroup(ParameterGroup):
    """Parameter group that samples the posterior directly via (optionally
    preconditioned) Stochastic Gradient Langevin Dynamics, using the *exact*
    autograd score of a user-supplied log-likelihood -- no trained/amortized
    score network, no backprop-through-the-chain training loop, and no
    ``kl_estimator`` (there is nothing being variationally fit; the dynamics
    themselves target the true posterior).

    Unadjusted Langevin update (Welling & Teh, *"Bayesian Learning via
    Stochastic Gradient Langevin Dynamics,"* ICML 2011):

    ``theta_{k+1} = theta_k + step_size * score(theta_k) + sqrt(2 * step_size) * xi_k``

    where ``score(theta) = grad_theta [log_likelihood_fn(theta) + log_prior(theta)]``,
    computed fresh via autograd at every step (no network to train).

    With ``preconditioned=True`` (the default), each particle maintains its
    **own** RMSProp-style running second-moment estimate of its score and
    rescales its effective step size per-parameter (preconditioned SGLD; Li,
    Chen, Carlson & Carin, *"Preconditioned Stochastic Gradient Langevin
    Dynamics for Deep Neural Networks,"* AAAI 2016):

    ``V <- alpha * V + (1 - alpha) * score**2``
    ``G = 1 / (sqrt(V) + eps)``
    ``theta_{k+1} = theta_k + 0.5 * step_size * G * score + sqrt(step_size * G) * xi_k``

    dropping the curvature-correction term Li et al. show is negligible when
    the preconditioner varies slowly relative to the step size. Preconditioner
    state is kept **per particle** (never pooled/shared across particles), so
    particles remain fully statistically independent -- this matters: a
    shared/trained statistic coupling particles together is exactly the
    mechanism found (see ``experiments/langevin_diagnosis/moe/FINDINGS_moe.md``,
    Experiments 11-13) to cause mode collapse in a *trained*, shared-network
    sampler (``LangevinGroup``) on multimodal posteriors; this class avoids
    that mechanism by construction, not by tuning.

    Use ``preconditioned=False`` for a plain fixed-step ULA sampler -- the
    right choice whenever the exact score is already well-conditioned (e.g. a
    closed-form/linear-Gaussian posterior), where preconditioning adds
    complexity without the stabilization it is designed for; see the
    Experiments above for a case (a 1D non-Gaussian posterior) where plain
    ULA strictly outperformed a particle-repulsion scheme, and a case (a
    79-parameter neural-network posterior) where plain ULA was numerically
    unusable without preconditioning.

    **Important usage caveat, unlike every other ``ParameterGroup`` in this
    library:** ``sample_parameters()`` reinitializes ``num_samples`` particles
    from ``N(0, init_std^2)`` and runs the *full* ``num_diffusion_steps``
    chain from scratch on *every* call (matching ``LangevinGroup``'s own
    fresh-every-call convention). There is no persistent chain state across
    calls. This is appropriate for the intended usage pattern -- call
    ``sample_parameters()`` once (or a few times for evaluation) to obtain a
    representative particle set from a posterior with no other trainable
    unknowns -- but makes this class a poor fit for a joint training loop that
    calls ``sample_parameters()`` every epoch alongside *other*, genuinely
    trainable parameter groups: at a realistic ``num_diffusion_steps`` (often
    several thousand, see the Experiments above), repeating the full chain
    every epoch is likely to dominate the loop's total cost. No persistent
    "continue the chain" API is provided (yet).

    ``distribution_loss()`` always returns ``0`` (a zero scalar, not merely a
    small number): this group is not fitting a variational family against a
    KL term, so there is no quantity for it to contribute to a joint ELBO's
    loss. ``variational_distribution()`` raises, matching ``LangevinGroup``'s
    implicit sampler -- there is no analytic ``q``.

    Example:
        ```python
        >>> import torch
        >>> from pypolymix.parameter_groups import SGLDGroup
        >>> x = torch.linspace(-1, 1, 20).unsqueeze(-1)
        >>> y = 2.0 * x + 0.1 * torch.randn_like(x)
        >>> def log_likelihood_fn(theta):  # theta: (S, 2) -> (S,)
        ...     pred = theta[:, 0:1] + theta[:, 1:2] * x.squeeze(-1)[None, :]
        ...     return -0.5 / 0.1**2 * (pred - y.squeeze(-1)[None, :]).pow(2).sum(dim=1)
        >>> group = SGLDGroup("ab", num_params=2, log_likelihood_fn=log_likelihood_fn,
        ...                    num_diffusion_steps=200, step_size=1e-2)
        >>> samples = group.sample_parameters(num_samples=16)
        >>> samples.shape
        torch.Size([16, 2])
        ```
    """

    def __init__(
        self,
        name: str,
        num_params: int,
        log_likelihood_fn: LogLikelihoodFn,
        prior: Optional[Prior] = None,
        *,
        num_particles: int = 128,
        num_diffusion_steps: int = 2000,
        step_size: float = 1e-2,
        init_std: float = 1.0,
        preconditioned: bool = True,
        preconditioner_alpha: float = 0.9,
        preconditioner_eps: float = 1e-8,
    ):
        """Create the sampler.

        Args:
            log_likelihood_fn: Callable mapping a batch of particles
                ``theta`` with shape ``(S, num_params)`` to a per-particle
                summed log-likelihood tensor of shape ``(S,)``, differentiable
                w.r.t. ``theta`` via autograd (e.g. closing over a surrogate
                model and training data). This is the one piece of
                problem-specific information this class needs; everything
                else (the prior's log-density) is handled generically via
                ``prior``.
            num_particles: Default particle count used when
                ``sample_parameters`` is called with no explicit
                ``num_samples``.
            num_diffusion_steps: Number of ULA/pSGLD steps run per call to
                ``sample_parameters``. Unlike ``LangevinGroup``, there is no
                trained score network to amortize this cost -- each step
                calls ``log_likelihood_fn`` and its prior's ``log_prob``, so
                total per-call cost scales linearly with this value.
            step_size: Nominal step size ``h``. With ``preconditioned=True``
                this is rescaled per-parameter by the adaptive preconditioner
                ``G``, not used directly as the raw step.
            init_std: Standard deviation of the ``N(0, init_std^2)``
                initialization drawn fresh on every ``sample_parameters`` call.
            preconditioned: If ``True`` (default), use per-particle
                RMSProp-style preconditioning (pSGLD). If ``False``, plain
                fixed-step ULA.
            preconditioner_alpha: EMA decay for the per-particle running
                second-moment estimate ``V``. Ignored if ``preconditioned=False``.
            preconditioner_eps: Numerical floor added to ``sqrt(V)`` before
                inverting, to avoid division by zero. Ignored if
                ``preconditioned=False``.
        """
        super().__init__(name, num_params, prior=prior, kl_estimator=None)

        if not callable(log_likelihood_fn):
            raise ValueError("log_likelihood_fn must be callable.")
        if num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be > 0")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0")
        if init_std <= 0.0:
            raise ValueError("init_std must be > 0")
        if preconditioned and not (0.0 < preconditioner_alpha < 1.0):
            raise ValueError("preconditioner_alpha must be in (0, 1)")
        if preconditioned and preconditioner_eps <= 0.0:
            raise ValueError("preconditioner_eps must be > 0")

        self.log_likelihood_fn = log_likelihood_fn
        self.num_particles = int(num_particles)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.step_size = float(step_size)
        self.init_std = float(init_std)
        self.preconditioned = bool(preconditioned)
        self.preconditioner_alpha = float(preconditioner_alpha)
        self.preconditioner_eps = float(preconditioner_eps)

        # No learnable nn.Parameter: nothing here is fit by an outer
        # optimizer. This buffer exists purely so the module has *some*
        # registered tensor to anchor device/dtype to (e.g. after
        # `group.to("cuda")`), mirroring the role `surrogate_params` plays
        # for LangevinGroup's device/dtype but carrying no trainable content.
        self.register_buffer("_device_anchor", torch.zeros(()))
        self._last_samples: Optional[torch.Tensor] = None

    def _log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        """``log_likelihood_fn(theta) + log_prior(theta)``, shape ``(S,)``."""
        log_lik = self.log_likelihood_fn(theta)
        dist = self.prior.distribution(torch.Size([self.num_params]), theta.device, theta.dtype)
        log_prior = dist.log_prob(theta)
        return log_lik + log_prior

    def _score(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact ``grad_theta log p(theta)`` via autograd. No ``create_graph``:
        this group never needs to backprop through its own sampling process
        (there is nothing it trains), unlike ``LangevinGroup._score``."""
        theta_for_grad = theta if theta.requires_grad else theta.detach().requires_grad_(True)
        with torch.enable_grad():
            log_post = self._log_posterior(theta_for_grad)
            (score,) = torch.autograd.grad(outputs=log_post.sum(), inputs=theta_for_grad)
        return score

    def _init_particles(self, num_samples: int) -> torch.Tensor:
        device = self._device_anchor.device
        dtype = self._device_anchor.dtype
        return torch.randn(num_samples, self.num_params, device=device, dtype=dtype) * self.init_std

    def _step(self, theta: torch.Tensor, V: Optional[torch.Tensor]):
        """One (optionally preconditioned) ULA update, shared by
        :meth:`sample_parameters` and :meth:`sample_trajectory`. Returns
        ``(new_theta, new_V)`` -- ``V`` is the per-particle running
        second-moment estimate (``None`` and ignored when
        ``preconditioned=False``)."""
        score = self._score(theta)
        noise = torch.randn_like(theta)
        if self.preconditioned:
            sq = score**2
            V = sq if V is None else self.preconditioner_alpha * V + (1 - self.preconditioner_alpha) * sq
            G = 1.0 / (torch.sqrt(V) + self.preconditioner_eps)
            theta = theta + 0.5 * self.step_size * G * score + torch.sqrt(self.step_size * G) * noise
        else:
            noise_scale = (2.0 * self.step_size) ** 0.5
            theta = theta + self.step_size * score + noise_scale * noise
        return theta, V

    def sample_parameters(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """Draw parameter samples by running (preconditioned) ULA from a fresh
        random initialization. See the class docstring's usage caveat: this
        reruns the full chain every call, with no persistent state."""
        if num_samples is None:
            num_samples = self.num_particles
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        theta = self._init_particles(num_samples)
        V = None

        with torch.no_grad():
            for _ in range(self.num_diffusion_steps):
                theta, V = self._step(theta, V)

        self._last_samples = theta
        return theta

    def sample_trajectory(self, num_samples: Optional[int] = None, record_every: int = 1) -> torch.Tensor:
        """Run the same (preconditioned) ULA dynamics as
        :meth:`sample_parameters`, but record and return every
        ``record_every``-th particle configuration along the way.

        Intended for pedagogical/diagnostic use (e.g. plotting particle
        trajectories against diffusion step as a pseudotime axis, to see
        whether/when a particle cloud splits across modes), not for training.

        Returns:
            Tensor of shape ``(num_recorded, num_samples, num_params)``,
            always including step 0 (the initial draw) and the final step
            (``num_diffusion_steps``), regardless of ``record_every``.
        """
        if num_samples is None:
            num_samples = self.num_particles
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if record_every <= 0:
            raise ValueError("record_every must be > 0")

        theta = self._init_particles(num_samples)
        V = None

        with torch.no_grad():
            trajectory = [theta.clone()]
            for step in range(1, self.num_diffusion_steps + 1):
                theta, V = self._step(theta, V)
                if step % record_every == 0 or step == self.num_diffusion_steps:
                    trajectory.append(theta.clone())

        self._last_samples = theta
        return torch.stack(trajectory, dim=0)

    def variational_distribution(self) -> td.Distribution:
        """SGLD defines an implicit sampler, not an analytic distribution."""
        raise RuntimeError(
            "SGLDGroup defines an implicit sampler and does not expose an analytic variational distribution."
        )

    def distribution_loss(self) -> torch.Tensor:
        """Always zero: this group is not a variational family being fit
        against a KL term -- the (preconditioned) ULA dynamics already target
        the exact posterior, with no free parameters of this group's own for
        an outer optimizer to adjust. Returning an always-zero (not merely
        small) scalar makes this explicit rather than silently omitting a
        term another group's ``distribution_loss`` might expect to be present."""
        return torch.zeros((), device=self._device_anchor.device, dtype=self._device_anchor.dtype)
