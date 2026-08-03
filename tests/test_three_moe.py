import matplotlib.pyplot as plt
import torch
from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.surrogate_models import PolynomialChaosExpansion, MixtureOfExperts, GatingNetwork
from pypolymix import StochasticModel
NUM_EXPERTS = 3

def make_three_region_data():

    torch.manual_seed(1)

    num_training_points = 150

    x_min = -1.0
    x_max = 1.0

    X = x_min + (x_max - x_min) * torch.rand(
        num_training_points, 1
    )

    # Three equally wide regions:
    # [-1, -1/3), [-1/3, 1/3), [1/3, 1]
    edges = torch.linspace(x_min, x_max, NUM_EXPERTS + 1)

    # Region index: 0, 1, or 2
    region = torch.bucketize(
        X.squeeze(-1),
        edges[1:-1],
    )

    # Convert the global x coordinate to a local coordinate in [0, 1]
    left_edges = edges[region]
    right_edges = edges[region + 1]

    local_x = (
        X.squeeze(-1) - left_edges
    ) / (
        right_edges - left_edges
    )

    # Vertical baseline of each half-sine
    offsets = torch.tile(torch.tensor([-1, 1]), (-(NUM_EXPERTS // -2) ,))

    # Amplitude of each half-sine
    amplitudes = torch.tile(torch.tensor([-1, 1]), (-(NUM_EXPERTS // -2) ,))

    # Noise standard deviation
    noise_std = 0.05

    Y = (
        offsets[region]
        + amplitudes[region] * torch.sin(torch.pi * local_x)
        + noise_std * torch.randn(num_training_points)
    ).unsqueeze(-1)
    return X, Y, region, edges


X, Y, region, edges = make_three_region_data()
# Plot synthetic data
_, ax = plt.subplots()
ax.scatter(X, Y)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()

def make_three_expert_problem():
    torch.manual_seed(1)

    X, Y, region, edges = make_three_region_data()
    width = 16
    depth = 1

    # Goal: experts learn the three pieces of the function
    # and the gating network chooses which expert to use for each piece
    experts = [PolynomialChaosExpansion(num_inputs=1, num_outputs=1, degree=3) for _ in range(NUM_EXPERTS)]
    gating_network = GatingNetwork(num_inputs=1,num_experts=NUM_EXPERTS, width=width, depth=depth, activation=torch.nn.functional.tanh)

    surrogate_model = MixtureOfExperts(
        experts=experts,
        gating_network=gating_network,
    )

    parameter_groups = []
    # Expert parameters
    # each expert has four parameters: 2 stochastic and 2 deterministic
    parameter_groups = [
        IIDGaussianGroup("expert1_stochastic_coeffs", 2),
        DeterministicGroup("expert1_deterministic_coeffs", 2),
        IIDGaussianGroup("expert2_stochastic_coeffs", 2),
        DeterministicGroup("expert2_deterministic_coeffs", 2),
        IIDGaussianGroup("expert3_stochastic_coeffs", 2),
        DeterministicGroup("expert3_deterministic_coeffs", 2),
    ]

    # GATING NETWORK parameters
    # Gating network input layer
    parameter_groups += [
        DeterministicGroup("gating_input_layer_weights", width),
        DeterministicGroup("gating_input_layer_biases", width),
    ]
    # no hidden -> hidden weights and biases since there's only 1 layer

    # Gating network output layer: map the 16 output values to 3 hidden weights
    parameter_groups += [
        IIDGaussianGroup("gating_output_weights", NUM_EXPERTS * width),
        IIDGaussianGroup("gating_output_biases", NUM_EXPERTS),
    ]

    model = StochasticModel(surrogate_model=surrogate_model, parameter_groups=parameter_groups)

    return surrogate_model, model, X, Y, region, edges

def test_three_region_data_generation():
    X, Y, region, edges = make_three_region_data()

    assert X.shape == (150, 1)
    assert Y.shape == (150, 1)
    assert region.shape == (150,)
    assert edges.shape == (4,)

    assert set(region.tolist()).issubset({0, 1, 2})
