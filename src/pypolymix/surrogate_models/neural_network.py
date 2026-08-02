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
        batch_norm: bool = False,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ):
        """Initialize the neural network architecture.

        Args:
            num_inputs: Dimensionality of ``x``.
            num_outputs: Number of response dimensions.
            width: Hidden-layer width.
            depth: Number of hidden layers.
            activation: Callable applied after each hidden linear block.
            batch_norm: Whether to apply batch normalization after hidden linear
                blocks and before activation.
            batch_norm_eps: Small positive value added to hidden batch variances.
            batch_norm_momentum: Momentum used to update running batch-normalization
                statistics during training.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum

        if batch_norm:
            self.register_buffer("batch_norm_running_mean", torch.zeros(depth, width))
            self.register_buffer("batch_norm_running_var", torch.ones(depth, width))

        # Define layer shapes. Batch norm applies to hidden layers only.
        self.layer_shapes = []
        in_features = num_inputs
        for _ in range(depth):
            self.layer_shapes.append((in_features, width))  # weight
            self.layer_shapes.append((width,))  # bias
            if batch_norm:
                self.layer_shapes.append((width,))  # batch-norm gamma
                self.layer_shapes.append((width,))  # batch-norm beta
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

    def _batch_norm(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply hidden-layer batch normalization with running statistics."""
        running_mean = self.batch_norm_running_mean[layer_idx].to(
            device=y.device, dtype=y.dtype
        )
        running_var = self.batch_norm_running_var[layer_idx].to(device=y.device, dtype=y.dtype)

        if self.training:
            batch_mean = y.mean(dim=(0, 1), keepdim=True)
            batch_var = y.var(dim=(0, 1), unbiased=False, keepdim=True)

            with torch.no_grad():
                n = y.shape[0] * y.shape[1]
                if n > 1:
                    running_batch_var = y.var(dim=(0, 1), unbiased=True)
                else:
                    running_batch_var = batch_var.reshape(-1)

                mean_update = batch_mean.reshape(-1).detach().to(
                    device=self.batch_norm_running_mean.device,
                    dtype=self.batch_norm_running_mean.dtype,
                )
                var_update = running_batch_var.detach().to(
                    device=self.batch_norm_running_var.device,
                    dtype=self.batch_norm_running_var.dtype,
                )
                self.batch_norm_running_mean[layer_idx].mul_(1 - self.batch_norm_momentum).add_(
                    self.batch_norm_momentum * mean_update
                )
                self.batch_norm_running_var[layer_idx].mul_(1 - self.batch_norm_momentum).add_(
                    self.batch_norm_momentum * var_update
                )

            mean = batch_mean
            var = batch_var
        else:
            mean = running_mean.reshape(1, 1, -1)
            var = running_var.reshape(1, 1, -1)

        return (y - mean) * torch.rsqrt(var + self.batch_norm_eps)

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

        slice_idx = 0
        for layer_idx in range(self.depth + 1):
            w_slice = self.param_slices[slice_idx]
            b_slice = self.param_slices[slice_idx + 1]
            slice_idx += 2

            # Extract weights/biases for all samples
            w = params[:, w_slice[0] : w_slice[1]].reshape(
                num_samples, *w_slice[2]
            )  # (num_samples, in_features, out_features)
            b = params[:, b_slice[0] : b_slice[1]].reshape(
                num_samples, *b_slice[2]
            )  # (num_samples, out_features)

            # Batched linear transformation: (num_samples, batch, out_features)
            y = torch.einsum("tbi,tio->tbo", y, w) + b.unsqueeze(1)

            # Apply batch normalization and activation except on last layer.
            if layer_idx < self.depth:
                if self.batch_norm:
                    gamma_slice = self.param_slices[slice_idx]
                    beta_slice = self.param_slices[slice_idx + 1]
                    slice_idx += 2

                    gamma = params[:, gamma_slice[0] : gamma_slice[1]].reshape(
                        num_samples, *gamma_slice[2]
                    )
                    beta = params[:, beta_slice[0] : beta_slice[1]].reshape(
                        num_samples, *beta_slice[2]
                    )

                    y = self._batch_norm(y, layer_idx)
                    y = y * gamma.unsqueeze(1) + beta.unsqueeze(1)

                y = self.activation(y)

        return y
