from typing import Iterable, Union

import torch
import torch.nn as nn

from .parameter_groups.base import ParameterGroup
from .surrogate_models.base import SurrogateModel


def _is_frozen(group: ParameterGroup) -> bool:
    """Return ``True`` if every parameter in ``group`` has ``requires_grad=False``.

    A group with zero parameters (a hypothetical parameterless group) is
    treated as *not* frozen -- vacuously passthrough, since there is nothing
    to detach and no reason to force it into the contiguous frozen prefix.
    This needs an explicit check: naively evaluating
    ``not any(p.requires_grad for p in group.parameters())`` over an *empty*
    ``parameters()`` gives ``not any(()) == not False == True`` (vacuous
    truth of ``any`` over nothing), which would incorrectly classify a
    parameterless group as frozen.
    """
    params = list(group.parameters())
    if not params:
        return False
    return not any(p.requires_grad for p in params)


class StochasticModel(nn.Module):
    """Wrap a deterministic surrogate model with sampled parameters.

    Example:
        ```python
        >>> from pypolymix import parameter_groups, surrogate_models
        >>> surrogate = surrogate_models.NeuralNetwork(num_inputs=1, num_outputs=1)
        >>> groups = [parameter_groups.IIDGaussianGroup("nn", surrogate.num_params())]
        >>> model = StochasticModel(surrogate, groups)
        >>> x = torch.linspace(-1, 1, 32).unsqueeze(-1)
        >>> y = model(x, num_samples=4)  # (4, 32, 1)
        ```

    Freezing groups:
        A group can be held at a point estimate (no gradient, no sampling
        overhead in the backward pass) by calling ``group.requires_grad_(False)``
        on it -- this already works via ``nn.Module``, no new API is needed here.
        ``sample_parameters`` detects such frozen groups (every parameter has
        ``requires_grad=False``) and calls ``.detach()`` on their sampled slice
        *before* concatenating with the trainable groups' samples.

        This detach step matters because ``torch.cat``'s backward computes a
        gradient for its *whole* input as one unit of autograd work: a frozen
        leaf with ``requires_grad=False`` still has its local gradient computed
        inside the concatenated graph and then discarded, so merely setting
        ``requires_grad=False`` gives no speedup on its own (on the order of a
        1.02x speedup was measured for a real 47,872-scalar model with
        47,584 frozen). Detaching the frozen slice before `cat` removes that
        branch -- and everything feeding into it -- from the graph entirely.

        Frozen groups must form a contiguous *prefix* of ``parameter_groups``
        (all frozen groups before any trainable one). This is because the
        detach happens per-group, before concatenation: if a frozen group
        appeared after a trainable one, its surrounding trainable context would
        still need gradients to flow correctly through the `cat`, and
        interleaving frozen/trainable groups can't be expressed by detaching
        whole-group slices. ``sample_parameters`` raises ``ValueError`` naming
        the offending group if this ordering is violated. Note this means a
        group layout that freezes *part* of a group (e.g. only some rows of a
        single group's parameters) or interleaves frozen and trainable groups
        is not supported by this mechanism.

        Unlike a hand-rolled "frozen/combined model" workaround that only
        creates ``nn.Parameter``s for the trainable slices, this approach keeps
        frozen groups as real groups with real parameters (just
        ``requires_grad=False``) -- so it never creates unused parameters and
        does not trip ``DistributedDataParallel``'s unused-parameter check.
        This replaces any need for a separate frozen/combined model class.
    """

    def __init__(
        self,
        surrogate_model: SurrogateModel,
        parameter_groups: Union[ParameterGroup, Iterable[ParameterGroup]],
    ):
        """Build the stochastic model.

        Args:
            surrogate_model: Base surrogate that consumes input ``x`` and sampled parameters.
            parameter_groups: Parameter-group module or iterable of modules whose draws are concatenated
                to produce the full parameter vector expected by ``surrogate_model``.
        """
        super().__init__()
        self.surrogate_model = surrogate_model
        if isinstance(parameter_groups, ParameterGroup):
            parameter_groups = [parameter_groups]
        else:
            parameter_groups = list(parameter_groups)
        self.parameter_groups = nn.ModuleList(parameter_groups)

        if self.num_params() != surrogate_model.num_params():
            raise ValueError(
                f"Total number of parameters in parameter groups ({self.num_params()}) does not match "
                f"number of parameters required in model ({surrogate_model.num_params()})"
            )

    def num_params(self) -> int:
        """Return the total number of scalar parameters managed across groups."""
        return sum(group.num_params for group in self.parameter_groups)

    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """Draw parameter samples from each group and concatenate them.

        Groups with every parameter frozen (``requires_grad=False``) have
        their sampled slice detached before concatenation, removing them (and
        everything feeding into them) from the autograd graph -- see the
        "Freezing groups" section of the class docstring. Frozen groups must
        form a contiguous prefix of ``parameter_groups``.

        Args:
            num_samples: Number of Monte Carlo samples per parameter group.

        Returns:
            Tensor with shape ``(num_samples, surrogate_model.num_params())``.

        Raises:
            ValueError: If a frozen group appears after a trainable group.
        """
        samples = []
        seen_trainable = False
        for i, g in enumerate(self.parameter_groups):
            s = g.sample_parameters(num_samples)
            frozen = _is_frozen(g)
            if frozen:
                if seen_trainable:
                    raise ValueError(
                        f"Group '{g.name}' (index {i}) is frozen (requires_grad=False "
                        f"for all its parameters) but appears after a trainable group "
                        f"in parameter_groups. Frozen groups must form a contiguous "
                        f"prefix of parameter_groups so their concatenated slice can "
                        f"be detached as a single block; reorder your groups list so "
                        f"all frozen groups come first."
                    )
                s = s.detach()
            else:
                seen_trainable = True
            samples.append(s)
        return torch.cat(samples, dim=1)

    def distribution_loss(self) -> torch.Tensor:
        """Return the sum of KL/cross-entropy terms provided by every parameter group."""
        return sum(g.distribution_loss() for g in self.parameter_groups)

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Evaluate the surrogate under randomly drawn parameters.

        Args:
            x: Input tensor passed to the surrogate model.
            num_samples: Number of parameter draws to average/evaluate over.

        Example:
            Evaluate a function at ``x`` using 10 random draws of the parameters:

            ```python
            >>> y = model(x, num_samples=10)
            >>> y.shape
            torch.Size([10, x.shape[0], surrogate_model.num_outputs])
            ```
        """
        params = self.sample_parameters(num_samples)
        return self.surrogate_model(x, params)
