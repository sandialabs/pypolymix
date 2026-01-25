"""Fully-connected neural-network surrogate model."""

from typing import Callable

import torch
import torch.nn.functional as F

from .base import SurrogateModel


class NeuralNetwork(SurrogateModel):
    """Neural network driven by sampled parameters.

    Example:
        ```python
        >>> surrogate = NeuralNetwork(num_inputs=2, num_outputs=1, width=8, depth=2)
        >>> surrogate.num_params()
        105
        >>> params = torch.randn(3, surrogate.num_params())
        >>> x = torch.randn(5, 2)
        >>> surrogate(x, params).shape
        torch.Size([3, 5, 1])
        ```
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int = 1,
        width: int = 16,
        depth: int = 1,
        activation: Callable = F.relu,
    ):
        """Initialize the neural network architecture.

        Args:
            num_inputs: Dimensionality of ``x``.
            num_outputs: Number of response dimensions.
            width: Hidden-layer width.
            depth: Number of hidden layers.
            activation: Callable applied after each hidden linear block.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.activation = activation

        # Define layer shapes (weights and biases)
        self.layer_shapes = []
        in_features = num_inputs
        for _ in range(depth):
            self.layer_shapes.append((in_features, width))  # weight
            self.layer_shapes.append((width,))  # bias
            in_features = width

        # Output layer
        self.layer_shapes.append((in_features, num_outputs))  # weight
        self.layer_shapes.append((num_outputs,))  # bias

        # Precompute parameter indices for slicing
        self.param_slices = []
        start = 0
        for shape in self.layer_shapes:
            n_params = torch.tensor(shape).prod().item()
            end = start + n_params
            self.param_slices.append((start, end, shape))
            start = end

        self._num_params = start  # total number of scalar parameters

    def num_params(self) -> int:
        """Return the number of scalar parameters implied by the architecture."""
        return self._num_params

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural network for multiple parameter samples in parallel.

        Args:
            x: Tensor of shape (batch_size, num_inputs)
            params: Tensor of shape (num_samples, num_params)

        Returns:
            y: Tensor of shape (num_samples, batch_size, num_outputs)
        """
        num_samples, _ = params.shape
        # batch_size = x.shape[0]

        # Expand inputs for broadcasting: (num_samples, batch_size, num_inputs)
        y = x.unsqueeze(0).expand(num_samples, -1, -1)

        # Loop over layers, but vectorized across samples
        for j in range(0, len(self.param_slices), 2):
            w_slice, b_slice = self.param_slices[j], self.param_slices[j + 1]

            # Extract weights/biases for all samples
            w = params[:, w_slice[0] : w_slice[1]].reshape(
                num_samples, *w_slice[2]
            )  # (num_samples, in_features, out_features)
            b = params[:, b_slice[0] : b_slice[1]].reshape(
                num_samples, *b_slice[2]
            )  # (num_samples, out_features)

            # Batched linear transformation: (num_samples, batch, out_features)
            y = torch.einsum("tbi,tio->tbo", y, w) + b.unsqueeze(1)

            # Apply activation except on last layer
            if j < len(self.param_slices) - 2:
                y = self.activation(y)

        return y
