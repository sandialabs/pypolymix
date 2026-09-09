"""Mixture-of-experts surrogate built from ``pypolymix`` components.

The model is a plain ``pypolymix.StochasticModel``: a
:class:`~pypolymix.surrogate_models.MixtureOfExperts` surrogate whose parameter
vector is supplied by a list of parameter groups. In the deterministic case every
group is a :class:`~pypolymix.parameter_groups.DeterministicGroup`, so the model
learns a single point estimate and ``distribution_loss()`` is never used.

Two details are worth understanding before changing anything here.

**Why one group per layer, rather than one group for the whole mixture.**
``DeterministicGroup`` initializes its parameters as ``torch.randn(n)``, which is
a unit-variance normal regardless of layer shape. A depth-4 network initialized
that way saturates immediately and does not train. Splitting the parameter vector
into one group per weight matrix and one per bias vector lets each group be seeded
with the same distribution ``torch.nn.Linear`` would have used. The split is also
the natural handle for making *part* of the model stochastic later: swap the
groups you care about for ``IIDGaussianGroup`` and leave the rest deterministic.

**Group order is load-bearing.** ``StochasticModel`` concatenates group samples in
list order to form the flat parameter vector, and ``MixtureOfExperts`` slices that
vector as ``[expert 0 | expert 1 | ... | expert N-1 | gate]``. Groups must be
appended in exactly that order. ``StochasticModel`` validates the total count, not
the order, so a permutation fails silently.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

from pypolymix import StochasticModel
from pypolymix.parameter_groups import DeterministicGroup
from pypolymix.surrogate_models import (GatingNetwork, MixtureOfExperts,
                                        NeuralNetwork)


def linear_init(shape: Sequence[int], fan_in: int, generator=None) -> torch.Tensor:
    """Draw a flat parameter vector the way ``torch.nn.Linear`` would.

    ``nn.Linear`` seeds weights with ``kaiming_uniform_(a=sqrt(5))`` and biases
    with ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``. For ``a = sqrt(5)`` the Kaiming
    bound reduces to ``1/sqrt(fan_in)`` as well, so both cases are the same
    uniform distribution and no ``nn.init`` call is needed.

    Args:
        shape: ``(in_features, out_features)`` for a weight, ``(out_features,)``
            for a bias.
        fan_in: Input width of the layer this parameter belongs to.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Flat tensor with ``prod(shape)`` entries.
    """
    bound = 1.0 / math.sqrt(max(int(fan_in), 1))
    values = torch.empty(math.prod(int(s) for s in shape))
    return values.uniform_(-bound, bound, generator=generator)


def network_groups(network, prefix: str, generator=None) -> List[DeterministicGroup]:
    """Build one initialized :class:`DeterministicGroup` per weight and bias.

    Args:
        network: A ``pypolymix`` ``NeuralNetwork`` (or anything exposing the same
            ``param_slices`` of ``(start, end, shape)`` triples, weights first).
        prefix: Name prefix for the generated groups.
        generator: Optional ``torch.Generator``.

    Returns:
        Groups in the order the surrogate expects to slice them.
    """
    groups: List[DeterministicGroup] = []
    fan_in = network.num_inputs
    for index, (_, _, shape) in enumerate(network.param_slices):
        layer = index // 2
        if len(shape) == 2:
            fan_in = int(shape[0])
            name = f"{prefix}_layer{layer}_weight"
        else:
            name = f"{prefix}_layer{layer}_bias"
        group = DeterministicGroup(name, math.prod(int(s) for s in shape))
        with torch.no_grad():
            group.params.copy_(linear_init(shape, fan_in, generator=generator))
        groups.append(group)
    return groups


def build_moe(cfg) -> StochasticModel:
    """Assemble the deterministic mixture-of-experts surrogate from a ``Config``."""
    activation = cfg.activation_fn()
    generator = torch.Generator().manual_seed(cfg.seed)

    experts = [
        NeuralNetwork(
            num_inputs=cfg.num_inputs,
            num_outputs=cfg.num_outputs,
            width=cfg.expert_width,
            depth=cfg.expert_depth,
            activation=activation,
        )
        for _ in range(cfg.num_experts)
    ]
    gate = GatingNetwork(
        num_inputs=cfg.num_inputs,
        num_experts=cfg.num_experts,
        width=cfg.gate_width,
        depth=cfg.gate_depth,
        activation=activation,
    )
    surrogate = MixtureOfExperts(experts, gate)

    groups: List[DeterministicGroup] = []
    for index, expert in enumerate(experts):
        groups.extend(network_groups(expert, f"expert{index}", generator=generator))
    groups.extend(network_groups(gate.network, "gate", generator=generator))

    return StochasticModel(surrogate, groups)


def predict(model: StochasticModel, z: torch.Tensor) -> torch.Tensor:
    """Evaluate the model on transformed inputs, returning ``(batch, num_outputs)``.

    Every group is deterministic, so all draws are identical and one suffices.
    """
    return model(z, num_samples=1).squeeze(0)


def gating_weights(model: StochasticModel, z: torch.Tensor) -> torch.Tensor:
    """Return the mixture weights ``(batch, num_experts)`` for transformed inputs."""
    params = model.sample_parameters(1)
    return model.surrogate_model.get_gating_weights(z, params).squeeze(0)


def parameter_summary(model: StochasticModel) -> str:
    """One-line description of the parameter layout, for the training log."""
    groups = list(model.parameter_groups)
    return (
        f"{model.num_params():,d} parameters in {len(groups)} groups "
        f"({model.surrogate_model.num_experts} experts + gate)"
    )


__all__ = [
    "build_moe",
    "gating_weights",
    "linear_init",
    "network_groups",
    "parameter_summary",
    "predict",
]
