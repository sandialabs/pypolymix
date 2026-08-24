from pypolymix import StochasticModel
from pypolymix.surrogate_models import NeuralNetwork
from pypolymix.parameter_groups import IIDGaussianGroup
import pytest
import torch

def test_num_params_wraps_correctly():
    '''
    2 -> 2 -> 1 is the network
    This leaves 2*2 + 1*2 = 6 weights
    and 2 + 1 = 3 biases
    for a total of 9 params
    '''
    NN = NeuralNetwork(num_inputs=2, num_outputs=1, width=2, depth=1)
    iid_gaussian = IIDGaussianGroup(name="test_name", num_params=9)
    stochastic_model = StochasticModel(surrogate_model=NN, parameter_groups=iid_gaussian)
    assert stochastic_model.num_params() == 9

def test_contradicting_num_params_ValueError():
    ''' Same network as last time
    but we use the wrong num_params for the parameter group'''
    NN = NeuralNetwork(num_inputs=2, num_outputs=1, width=2, depth=1)
    iid_gaussian = IIDGaussianGroup(name="test_name", num_params=10) # WRONG
    with pytest.raises(ValueError):
        stochastic_model = StochasticModel(surrogate_model=NN, parameter_groups=iid_gaussian)


def test_forward_returns_tensor():
    NN = NeuralNetwork(num_inputs=2, num_outputs=1, width=2, depth=1)
    iid_gaussian = IIDGaussianGroup(name="test_name", num_params=9)
    stochastic_model = StochasticModel(surrogate_model=NN, parameter_groups=iid_gaussian)
    inputs = torch.randn(5,2)
    output = stochastic_model.forward(inputs, num_samples=10)
    assert isinstance(output, torch.Tensor)

def test_forward_shape():
    NN = NeuralNetwork(num_inputs=2, num_outputs=1, width=2, depth=1)
    iid_gaussian = IIDGaussianGroup(name="test_name", num_params=9)
    stochastic_model = StochasticModel(surrogate_model=NN, parameter_groups=iid_gaussian)
    inputs = torch.randn(5,2)
    output = stochastic_model.forward(inputs, num_samples=10)
    assert output.shape == torch.Size([10, 5, 1])
