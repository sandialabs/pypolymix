import pytest
import torch

from pypolymix import StochasticModel
from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.stochastic_model import _is_frozen
from pypolymix.surrogate_models import NeuralNetwork


def _build_model():
    """2 -> 2 -> 1 network (9 params total): a frozen prefix group (3 params)
    followed by a trainable group (6 params)."""
    torch.manual_seed(0)
    surrogate = NeuralNetwork(num_inputs=2, num_outputs=1, width=2, depth=1)
    frozen_group = DeterministicGroup("frozen", num_params=3)
    trainable_group = IIDGaussianGroup("trainable", num_params=6)
    model = StochasticModel(
        surrogate_model=surrogate, parameter_groups=[frozen_group, trainable_group]
    )
    return surrogate, model


def test_is_frozen_true_when_all_params_require_no_grad():
    group = DeterministicGroup("g", num_params=3)
    group.requires_grad_(False)
    assert _is_frozen(group)


def test_is_frozen_false_when_any_param_requires_grad():
    group = DeterministicGroup("g", num_params=3)
    assert not _is_frozen(group)


def test_is_frozen_false_for_parameterless_group():
    """A group with zero parameters is vacuously not frozen (passthrough)."""

    class _ParameterlessGroup(DeterministicGroup):
        def parameters(self, recurse: bool = True):
            return iter(())

    group = _ParameterlessGroup("empty", num_params=0)
    assert not _is_frozen(group)


def test_frozen_group_has_no_grad_and_trainable_group_has_finite_grad():
    surrogate, model = _build_model()
    frozen_group, trainable_group = model.parameter_groups
    frozen_group.requires_grad_(False)

    x = torch.randn(5, 2)
    params = model.sample_parameters(num_samples=4)
    y = surrogate(x, params)
    y.sum().backward()

    for p in frozen_group.parameters():
        # Never part of the graph, so no gradient is ever assigned.
        assert p.grad is None

    for p in trainable_group.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_frozen_group_after_trainable_group_raises_value_error():
    surrogate, model = _build_model()
    frozen_group, trainable_group = model.parameter_groups
    frozen_group.requires_grad_(False)

    # Reorder so the trainable group comes first -- frozen groups must form a
    # contiguous prefix.
    model.parameter_groups = torch.nn.ModuleList([trainable_group, frozen_group])

    with pytest.raises(ValueError, match=f"'{frozen_group.name}'"):
        model.sample_parameters(num_samples=4)


def test_detach_does_not_change_forward_output_or_trainable_gradients():
    """Detaching the frozen prefix must only affect what gets a gradient, not
    any numerical result: forward output and the trainable group's gradients
    must be bit-identical to a model with no freezing applied (same seeds,
    same frozen group's parameters held at the same fixed values)."""
    surrogate, model_frozen = _build_model()
    _, model_plain = _build_model()
    # Make sure both models start from identical parameter values.
    model_plain.load_state_dict(model_frozen.state_dict())

    frozen_group, _ = model_frozen.parameter_groups
    frozen_group.requires_grad_(False)

    x = torch.randn(5, 2)

    torch.manual_seed(42)
    params_frozen = model_frozen.sample_parameters(num_samples=8)
    y_frozen = model_frozen.surrogate_model(x, params_frozen)

    torch.manual_seed(42)
    params_plain = model_plain.sample_parameters(num_samples=8)
    y_plain = model_plain.surrogate_model(x, params_plain)

    assert torch.equal(y_frozen, y_plain)

    y_frozen.sum().backward()
    y_plain.sum().backward()

    _, trainable_frozen = model_frozen.parameter_groups
    _, trainable_plain = model_plain.parameter_groups
    for p_frozen, p_plain in zip(trainable_frozen.parameters(), trainable_plain.parameters()):
        assert torch.equal(p_frozen.grad, p_plain.grad)
