# unit tests for src/pypolymix/surrogate_models/mixture.py
import math
import pytest
import torch
from torch import nn
from pypolymix.surrogate_models.base import SurrogateModel
from pypolymix.surrogate_models.mixture import GatingNetwork, MixtureOfExperts


class ScalarParamExpert(SurrogateModel):
    """Simple expert whose output is one selected scalar parameter.

    For each parameter sample t and input batch item b, returns:

        y[t, b, :] = params[t, param_index]

    This makes it easy to verify that MixtureOfExperts slices parameters
    correctly and combines expert outputs correctly.
    """

    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 1,
        n_params: int = 1,
        param_index: int = 0,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._num_params = n_params
        self.param_index = param_index

    def num_params(self) -> int:
        return self._num_params

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        assert params.shape[1] == self._num_params

        num_samples = params.shape[0]
        batch_size = x.shape[0]

        value = params[:, self.param_index]
        return value[:, None, None].expand(
            num_samples,
            batch_size,
            self.num_outputs,
        )


class SoftmaxParamGating(SurrogateModel):
    """Simple gating model using its parameter slice as logits.

    params has shape:

        (num_samples, num_experts)

    Output has shape:

        (num_samples, batch_size, num_experts)
    """

    def __init__(self, num_inputs: int = 1, num_experts: int = 2):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_experts = num_experts

    def num_params(self) -> int:
        return self.num_experts

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        assert params.shape[1] == self.num_experts

        batch_size = x.shape[0]
        logits = params[:, None, :].expand(-1, batch_size, -1)
        return torch.softmax(logits, dim=-1)


def test_mixture_of_experts_is_surrogate_model_and_module():
    experts = [
        ScalarParamExpert(num_inputs=1),
        ScalarParamExpert(num_inputs=1),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)

    moe = MixtureOfExperts(experts, gating)

    assert isinstance(moe, SurrogateModel)
    assert isinstance(moe, nn.Module)


def test_num_params_and_param_slices_are_correct():
    experts = [
        ScalarParamExpert(num_inputs=1, n_params=2),
        ScalarParamExpert(num_inputs=1, n_params=3),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)

    moe = MixtureOfExperts(experts, gating)

    assert moe.num_params() == 7
    assert moe.param_slices == [
        (0, 2),  # first expert
        (2, 5),  # second expert
        (5, 7),  # gating network
    ]


def test_constructor_rejects_experts_with_mismatched_input_dimensions():
    experts = [
        ScalarParamExpert(num_inputs=1),
        ScalarParamExpert(num_inputs=2),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)

    with pytest.raises(ValueError):
        MixtureOfExperts(experts, gating)


def test_get_expert_outputs_has_expected_shape_and_values():
    experts = [
        ScalarParamExpert(num_inputs=1),
        ScalarParamExpert(num_inputs=1),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)
    moe = MixtureOfExperts(experts, gating)

    x = torch.randn(4, 1)

    # Parameter layout:
    # expert 0: [2.0]
    # expert 1: [10.0]
    # gating:   [0.0, 0.0]
    params = torch.tensor([[2.0, 10.0, 0.0, 0.0]])

    outputs = moe.get_expert_outputs(x, params)

    assert outputs.shape == (1, 2, 4, 1)

    torch.testing.assert_close(
        outputs[0, 0],
        torch.full((4, 1), 2.0),
    )
    torch.testing.assert_close(
        outputs[0, 1],
        torch.full((4, 1), 10.0),
    )


def test_get_gating_weights_has_expected_shape_and_sums_to_one():
    experts = [
        ScalarParamExpert(num_inputs=1),
        ScalarParamExpert(num_inputs=1),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)
    moe = MixtureOfExperts(experts, gating)

    x = torch.randn(3, 1)

    params = torch.tensor(
        [
            [2.0, 10.0, 0.0, 0.0],
            [1.0, 5.0, math.log(3.0), 0.0],
        ]
    )

    weights = moe.get_gating_weights(x, params)

    assert weights.shape == (2, 3, 2)

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(2, 3),
    )

    # First sample has equal logits, so weights are [0.5, 0.5].
    torch.testing.assert_close(
        weights[0],
        torch.full((3, 2), 0.5),
    )

    # Second sample has logits [log(3), 0], so weights are [0.75, 0.25].
    expected_second = torch.tensor([[0.75, 0.25]]).expand(3, 2)
    torch.testing.assert_close(
        weights[1],
        expected_second,
    )


def test_forward_combines_experts_with_gating_weights():
    experts = [
        ScalarParamExpert(num_inputs=1),
        ScalarParamExpert(num_inputs=1),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)
    moe = MixtureOfExperts(experts, gating)

    x = torch.randn(3, 1)

    params = torch.tensor(
        [
            # expert outputs: 2 and 10
            # gating weights: [0.5, 0.5]
            # mixture: 0.5 * 2 + 0.5 * 10 = 6
            [2.0, 10.0, 0.0, 0.0],

            # expert outputs: 1 and 5
            # gating weights: softmax([log(3), 0]) = [0.75, 0.25]
            # mixture: 0.75 * 1 + 0.25 * 5 = 2
            [1.0, 5.0, math.log(3.0), 0.0],
        ]
    )

    y = moe(x, params)

    assert y.shape == (2, 3, 1)

    expected = torch.tensor(
        [
            [[6.0], [6.0], [6.0]],
            [[2.0], [2.0], [2.0]],
        ]
    )

    torch.testing.assert_close(y, expected)


def test_forward_supports_multiple_outputs():
    experts = [
        ScalarParamExpert(num_inputs=1, num_outputs=2),
        ScalarParamExpert(num_inputs=1, num_outputs=2),
    ]
    gating = SoftmaxParamGating(num_inputs=1, num_experts=2)
    moe = MixtureOfExperts(experts, gating)

    x = torch.randn(4, 1)

    params = torch.tensor(
        [
            [2.0, 10.0, 0.0, 0.0],
        ]
    )

    y = moe(x, params)

    assert y.shape == (1, 4, 2)

    expected = torch.full((1, 4, 2), 6.0)
    torch.testing.assert_close(y, expected)


def test_gating_network_outputs_valid_simplex_weights():
    gating = GatingNetwork(num_inputs=2, num_experts=3, width=4)

    x = torch.randn(5, 2)
    params = torch.randn(7, gating.num_params())

    weights = gating(x, params)

    assert weights.shape == (7, 5, 3)

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(7, 5),
    )

    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)
