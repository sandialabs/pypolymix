"""Non-centered ("whitening") reparameterization wrapper for parameter groups."""

from __future__ import annotations

import warnings
from typing import Callable

import torch
import torch.distributions as td

from ..priors.common import IIDGaussianPrior
from .base import ParameterGroup


def _looks_like_standard_normal(prior) -> bool:
    """Best-effort check that ``prior`` already behaves like ``N(0, I)``.

    Instantiates the prior for a 1-element event and compares its ``mean``/
    ``stddev`` against 0/1. Any exception (a prior that cannot be instantiated
    with this trivial shape, or that has no finite mean/variance, e.g.
    ``UniformPrior``) is treated as "does not look standard normal" rather than
    propagated, since this helper is only used to decide whether to emit a
    warning.
    """
    try:
        dist = prior.distribution(event_shape=torch.Size([1]), device=None, dtype=torch.float64)
        mean = dist.mean
        std = dist.stddev
    except Exception:
        return False
    return bool(torch.allclose(mean, torch.zeros_like(mean))) and bool(
        torch.allclose(std, torch.ones_like(std))
    )


class WhitenedGroup(ParameterGroup):
    r"""Non-centered reparameterization wrapper: ``theta = center + scale * z``.

    Wraps any base :class:`ParameterGroup` (which models ``z``) so that its
    prior is forced to ``N(0, I)`` regardless of what prior the base group was
    constructed with, and ``theta``'s effective prior becomes
    ``N(center, diag(scale)^2)`` implicitly. ``center`` and ``scale`` are
    persistent buffers (not learned), so a checkpoint's whitening transform
    round-trips through ``load_state_dict`` without special handling.

    **Why this exists.** Every stochastic parameter family in this package has
    a scale problem: :class:`LowRankGaussianGroup`'s ``min_variance`` floor
    becomes *binding* when the true prior std is itself smaller than the
    floor, and :class:`LangevinGroup`'s scalar ``init_std`` cannot represent a
    prior whose scale varies by orders of magnitude across parameters.
    Reparameterizing ``theta = center + scale * z``, putting ``z``'s prior at
    ``N(0, I)`` always, and letting the base group model ``z`` instead of
    ``theta`` sidesteps both problems: the floor now applies in dimensionless
    ``z``-space, and the prior is scale-free by construction. This is also the
    standard "non-centered parameterization" fix for the Neal/Betancourt
    "funnel" pathology in hierarchical models (Betancourt & Girolami,
    "Hamiltonian Monte Carlo for Hierarchical Models," arXiv:1312.0906).

    **Design decision 1: force vs. validate-and-raise on ``base_group.prior``.**
    ``__init__`` *forces* ``base_group.prior`` to ``IIDGaussianPrior(0, 1)``,
    overwriting whatever the caller originally passed to ``base_group``, and
    emits a ``warnings.warn`` if the previous prior did not already look like
    ``N(0, I)``. Forcing (rather than validating-and-raising) was chosen
    because "``z``'s prior is always ``N(0, I)``" is the entire point of
    whitening -- there is no coherent way to whiten against anything else --
    so a caller-supplied base prior that disagrees is overwhelmingly more
    likely to be a mistake (e.g. a base group reused from elsewhere with its
    original prior still attached) than a deliberate request for different
    behavior. Forcing fails safe: the resulting model is always correct, and
    the warning plus the deliberately *not*-``info``-level severity gives the
    caller a chance to notice and double check.

    **Design decision 2: ``LangevinGroup`` has no analytic ``q(z)``.**
    :class:`LangevinGroup` raises ``RuntimeError`` from
    ``variational_distribution()`` by design -- it is an implicit sampler, not
    an analytic family. ``WhitenedGroup.variational_distribution()`` therefore
    also raises ``RuntimeError`` when wrapping a ``LangevinGroup``: it simply
    delegates to ``base_group.variational_distribution()`` and lets that
    exception propagate, rather than trying to special-case the implicit
    sampler. ``distribution_loss()`` still works for a whitened
    ``LangevinGroup`` because it delegates unconditionally to
    ``base_group.distribution_loss()``, which (as of this writing, before
    PM-04's KL-estimator scaffolding lands) estimates only the cross-entropy
    term ``-E_q[log p(z)]`` for ``LangevinGroup``, not the full KL -- see
    ``src/pypolymix/parameter_groups/langevin.py:148-159``.
    # TODO(PM-04 integration): once PM-04 gives ``LangevinGroup`` a pluggable,
    # entropy-aware KL estimator, re-check that ``WhitenedGroup`` wrapping a
    # ``LangevinGroup`` still composes correctly -- i.e. that
    # ``base_group.distribution_loss()`` keeps meaning "KL(q(z) || N(0, I))"
    # (or whatever PM-04 settles on) rather than something that assumes
    # ``z``'s *own* prior, pre-whitening.

    **Why ``variational_distribution()`` is not simply "the right answer" for
    the KL computation.** For base groups that *do* define an analytic
    ``q(z)`` (everything except ``LangevinGroup``),
    ``variational_distribution()`` below returns the exact pushforward of
    ``q(z)`` through the affine map, via
    ``torch.distributions.TransformedDistribution`` with an
    ``AffineTransform(loc=center, scale=scale, event_dim=1)``. That
    *distribution object* is correct (its ``rsample``/``log_prob`` genuinely
    describe ``q(theta)``, including the Jacobian correction) and is provided
    for callers who want ``q(theta)`` for its own sake (e.g. plotting,
    reporting credible intervals). It is deliberately **not** used by
    ``distribution_loss()``, though. ``theta = center + scale * z`` is a fixed
    (non-learned) bijection, and KL divergence is invariant under bijections,
    so ``KL(q(theta) || p(theta)) == KL(q(z) || N(0, I))`` exactly --
    ``distribution_loss()`` returns ``base_group.distribution_loss()``
    unchanged, which reuses whatever analytic-or-Monte-Carlo KL machinery the
    base group already has for ``z``-space. Recomputing the same quantity by
    building the transformed distribution and calling
    ``kl_divergence``/Monte Carlo on *that* instead would be strictly worse: no
    analytic KL is registered for ``TransformedDistribution`` against
    ``IIDGaussianPrior``'s ``N(center, diag(scale)^2)``, so it would silently
    fall back to a noisier Monte Carlo estimate of a quantity already known
    exactly, and it would raise (rather than work) for ``LangevinGroup``.

    **Resolving the "does `q = p` at init?" discrepancy.** Wrapping a base
    group in ``WhitenedGroup`` guarantees that the *prior* starts at exactly
    ``N(0, I)`` in ``z``-space. It does **not**, by itself, guarantee that the
    base group's *variational distribution* ``q(z)`` starts there too --
    that is a property of the base group's own default initialization, which
    this wrapper has no control over. Concretely, for
    :class:`IIDGaussianGroup`: its default ``mean`` is ``torch.randn(n)`` (not
    zero), *and* its default ``log_std`` is a constant ``-0.5``, i.e.
    ``std = exp(-0.5) \\approx 0.6065`` (not 1) -- see
    ``src/pypolymix/parameter_groups/gaussian.py``. Zero-initializing only
    ``mean`` is therefore **not sufficient** for ``q(z) = N(0, I)`` at init:
    with the default ``log_std``, ``KL(N(0, 0.6065) || N(0, 1)) \\approx 0.18``
    per parameter, not 0. Achieving exact ``q = p`` at initialization requires
    the caller to zero-initialize *every* parameter that controls ``q(z)``
    (both ``mean`` and ``log_std`` for ``IIDGaussianGroup``) so that the base
    group's own ``q(z)`` starts at exactly ``N(0, I)``; ``WhitenedGroup``
    guarantees the *target* is ``N(0, I)``, not that the base group starts
    there.

    Example:
        ```python
        >>> import torch
        >>> from pypolymix.parameter_groups import IIDGaussianGroup, WhitenedGroup
        >>> _ = torch.manual_seed(0)
        >>> base = IIDGaussianGroup("z", num_params=3)
        >>> center = torch.tensor([1.0, 2.0, 3.0])
        >>> scale = torch.tensor([0.1, 0.2, 0.3])
        >>> group = WhitenedGroup(base, center, scale)
        >>> _ = torch.manual_seed(1234)
        >>> theta = group.sample_parameters(num_samples=4)
        >>> _ = torch.manual_seed(1234)
        >>> z = base.sample_parameters(num_samples=4)
        >>> torch.allclose(theta, center + scale * z)
        True
        ```
    """

    def __init__(self, base_group: ParameterGroup, center: torch.Tensor, scale: torch.Tensor):
        """Wrap ``base_group`` so it models ``z`` in ``theta = center + scale * z``.

        Args:
            base_group: Any ``ParameterGroup`` that will model the whitened
                variable ``z``. Its ``prior`` is overwritten in place (see
                design decision 1 above).
            center: Per-parameter offset, broadcast-compatible with
                ``(num_params,)``.
            scale: Per-parameter strictly-positive scale, broadcast-compatible
                with ``(num_params,)``.

        Raises:
            ValueError: If ``base_group.num_params`` disagrees with
                ``center.numel()``/``scale.numel()``, or if ``scale`` is not
                strictly positive everywhere.
        """
        center = torch.as_tensor(center).float()
        scale = torch.as_tensor(scale).float()

        if center.numel() != base_group.num_params or scale.numel() != base_group.num_params:
            raise ValueError(
                "base_group.num_params "
                f"({base_group.num_params}) must match center.numel() "
                f"({center.numel()}) and scale.numel() ({scale.numel()})."
            )
        center = center.reshape(base_group.num_params)
        scale = scale.reshape(base_group.num_params)
        if not torch.all(scale > 0):
            raise ValueError("scale must be strictly positive everywhere.")

        super().__init__(
            base_group.name, base_group.num_params, prior=IIDGaussianPrior(mean=0.0, std=1.0)
        )

        if not _looks_like_standard_normal(base_group.prior):
            warnings.warn(
                f"WhitenedGroup is overwriting base_group.prior ({base_group.prior!r}) "
                "with IIDGaussianPrior(mean=0.0, std=1.0). Whitening requires z's "
                "prior to be the standard normal by construction; passing a "
                "base_group with a different prior is very likely a mistake.",
                stacklevel=2,
            )
        base_group.prior = IIDGaussianPrior(mean=0.0, std=1.0)

        self.register_buffer("center", center, persistent=True)
        self.register_buffer("scale", scale, persistent=True)
        self.base_group = base_group

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Draw ``theta = center + scale * z`` for ``z ~ base_group``."""
        z = self.base_group.sample_parameters(num_samples)
        return self.center + self.scale * z

    def variational_distribution(self) -> td.Distribution:
        """Return the exact pushforward of ``q(z)`` through the affine map.

        Raises:
            RuntimeError: If ``base_group`` does not define an analytic
                variational distribution (e.g. ``LangevinGroup`` -- see design
                decision 2 above); this propagates unchanged from
                ``base_group.variational_distribution()``.
        """
        base_q = self.base_group.variational_distribution()
        return td.TransformedDistribution(
            base_q, [td.AffineTransform(loc=self.center, scale=self.scale, event_dim=1)]
        )

    def distribution_loss(self) -> torch.Tensor:
        """Return ``KL(q(z) || N(0, I))``, which equals ``KL(q(theta) || p(theta))``.

        See the class docstring for why this delegates to
        ``base_group.distribution_loss()`` unchanged rather than recomputing
        the KL from ``variational_distribution()``.
        """
        return self.base_group.distribution_loss()

    @classmethod
    def from_point_estimate(
        cls,
        base_group_factory: Callable[[int], ParameterGroup],
        center: torch.Tensor,
        relative_scale: float = 0.1,
        *,
        eps: float = 1e-3,
    ) -> "WhitenedGroup":
        """Build a ``WhitenedGroup`` centered on a deterministic point estimate.

        This is exactly the pattern recommended by hand in
        ``docs/examples/constitutive_model/README.md`` ("Going stochastic"):
        ``group.log_std.copy_(torch.log(0.1 * theta0.abs().clamp_min(1e-3)))``.
        Here, ``scale = relative_scale * center.abs().clamp_min(eps)``, i.e.
        each parameter's std is ``relative_scale`` fraction of its point
        estimate's magnitude (floored at ``eps`` so parameters whose point
        estimate is exactly or near zero still get a usable, nonzero scale).

        Args:
            base_group_factory: Callable that takes ``num_params`` (inferred
                from ``center.numel()``) and returns a freshly constructed
                ``ParameterGroup`` to model ``z``, e.g.
                ``lambda n: IIDGaussianGroup("theta", n)``.
            center: Point estimate ``theta0``, e.g. from a prior deterministic
                fit.
            relative_scale: Fraction of ``|theta0|`` used as each parameter's
                std. Defaults to ``0.1``, matching the README precedent.
            eps: Floor on ``center.abs()`` before scaling, so parameters whose
                point estimate is ~0 still get a nonzero scale. Defaults to
                ``1e-3``, matching the README precedent.

        Returns:
            A ``WhitenedGroup`` wrapping the base group produced by
            ``base_group_factory``.
        """
        center = torch.as_tensor(center).float()
        scale = relative_scale * center.abs().clamp_min(eps)
        base_group = base_group_factory(center.numel())
        return cls(base_group, center, scale)
