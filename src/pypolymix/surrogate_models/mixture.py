from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SurrogateModel
from .neural_network import NeuralNetwork


class GatingNetwork(SurrogateModel):
    """Gating network that outputs mixture weights via softmax.

    Example:
        ```python
        >>> gating = GatingNetwork(num_inputs=1, num_experts=3, width=8)
        >>> params = torch.randn(2, gating.num_params())
        >>> x = torch.zeros(4, 1)
        >>> gating(x, params).shape
        torch.Size([2, 4, 3])
        ```
    """

    def __init__(self, num_inputs: int, num_experts: int, **kwargs):
        """Use ``NeuralNetwork`` under the hood to produce expert weights."""
        super().__init__()
        self.network = NeuralNetwork(num_inputs, num_outputs=num_experts, **kwargs)
        self.num_inputs = num_inputs
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute mixture weights.

        Args:
            x: (batch_size, num_inputs)
            params: (num_samples, num_params)

        Returns:
            Gating weights: (num_samples, batch_size, num_experts)
        """
        logits = self.network(x, params)  # (num_samples, batch_size, num_experts)
        return F.softmax(logits, dim=-1)

    def num_params(self) -> int:
        return self.network.num_params()


class MixtureOfExperts(SurrogateModel):
    """Mixture of Experts surrogate model compatible with stochastic parameter sampling.

    Example:
        ```python
        >>> from pypolymix.surrogate_models import NeuralNetwork
        >>> experts = [NeuralNetwork(num_inputs=1, num_outputs=1, width=4) for _ in range(2)]
        >>> gating = GatingNetwork(num_inputs=1, num_experts=len(experts))
        >>> moe = MixtureOfExperts(experts, gating)
        >>> params = torch.randn(3, moe.num_params())
        >>> x = torch.randn(5, 1)
        >>> moe(x, params).shape
        torch.Size([3, 5, 1])
        ```
    """

    def __init__(self, experts: List[SurrogateModel], gating_network: SurrogateModel):
        """Initialize the mixture of experts architecture.

        Args:
            experts: List of expert surrogate models.
            gating_network: A surrogate model returning mixture weights.
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network

        # Sanity checks
        num_inputs = {expert.num_inputs for expert in experts}
        if len(num_inputs) > 1:
            raise ValueError("All experts must have the same input dimension.")

        self.num_inputs = next(iter(num_inputs))
        self.num_experts = len(experts)
        self.num_outputs = experts[0].num_outputs

        # Precompute parameter index boundaries
        self.param_slices = []
        start = 0
        for expert in self.experts:
            n = expert.num_params()
            self.param_slices.append((start, start + n))
            start += n

        # Gating network parameters at the end
        gating_start = start
        gating_end = gating_start + gating_network.num_params()
        self.param_slices.append((gating_start, gating_end))

        self._num_params = gating_end

    def num_params(self) -> int:
        """Total number of scalar parameters across all experts and gating network."""
        return self._num_params

    def get_expert_outputs(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate each expert with the parameter slice assigned to it.

        Returns:
            Tensor with shape ``(num_samples, num_experts, batch_size, num_outputs)``.
        """
        # Compute expert outputs (stacked over experts)
        expert_outputs = []
        for e_idx, expert in enumerate(self.experts):
            start, end = self.param_slices[e_idx]
            y = expert(x, params[:, start:end])  # (num_samples, batch_size, num_outputs)
            expert_outputs.append(y)

        return torch.stack(expert_outputs, dim=1)  # (num_samples, num_experts, batch, num_outputs)

    def get_gating_weights(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute the mixture weights produced by the gating network."""
        g_start, g_end = self.param_slices[-1]
        return self.gating_network(x, params[:, g_start:g_end])  # (num_samples, batch, num_experts)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Vectorized forward pass through all experts and the gating network.

        Args:
            x: (batch_size, num_inputs)
            params: (num_samples, total_params)

        Returns:
            Tensor of shape ``(num_samples, batch_size, num_outputs)`` containing
            the expert ensemble prediction for each draw of ``params``.
        """

        # Compute expert outputs
        expert_outputs = self.get_expert_outputs(
            x, params
        )  # (num_samples, num_experts, batch, num_outputs)

        # Compute gating weights
        gating_weights = self.get_gating_weights(x, params)  # (num_samples, batch, num_experts)

        # Weighted combination across experts
        y = torch.einsum("tbe,tebo->tbo", gating_weights, expert_outputs)
        return y
