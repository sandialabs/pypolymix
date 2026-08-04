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


def train_model(surrogate_model, model, X, Y,
                num_epochs=10_000, num_samples=100,
                weight_factor=1e-2, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=10 * lr,
        total_steps=num_epochs,
    )

    # print(f"Training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        params = model.sample_parameters(num_samples=num_samples)
        Y_hat = surrogate_model(X, params)

        data_loss = loss_fn(Y_hat, Y.unsqueeze(0).expand_as(Y_hat)) / num_samples
        distribution_loss = model.distribution_loss()
        total_loss = data_loss + weight_factor * distribution_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if (epoch + 1) % 1000 == 0:
            current_lr = scheduler.get_last_lr()[0]
            # print(
            #     f"Epoch {epoch + 1:5d} | "
            #     f"learning rate = {current_lr:.6f} | "
            #     f"data loss = {data_loss.item():.4f} | "
            #     f"distribution loss = {distribution_loss.item():.4f} | "
            #     f"total loss = {total_loss.item():.4f}"
            # )
    return total_loss.item()

def test_three_region_data_generation():
    X, Y, region, edges = make_three_region_data()

    assert X.shape == (150, 1)
    assert Y.shape == (150, 1)
    assert region.shape == (150,)
    assert edges.shape == (4,)

    assert set(region.tolist()).issubset({0, 1, 2})

def test_three_expert_training_returns_float():
    surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
    total_loss = train_model(surrogate_model,model, X, Y, num_epochs=10, num_samples=2)

    assert isinstance(total_loss, float)
    assert torch.isfinite(torch.tensor(total_loss))

def is_better(expected_better, expected_worse):
    '''
    Check that expected_better is not too much more than expected_worse
    Ideally it would be less over multiple samples
    But the tests are already long-running, so for now close is good enough
    '''
    return expected_better < expected_worse + 10

def test_epochs():
    epoch_losses = {}
    for num in [1_000, 3_000, 10_000]:
        surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
        total_loss = train_model(surrogate_model, model, X, Y, num_epochs=num)
        epoch_losses[num] = total_loss
    print(epoch_losses)
    assert is_better(epoch_losses[3_000], epoch_losses[1_000])
    assert is_better(epoch_losses[10_000], epoch_losses[3000])

def test_num_samples():
    loss = {}
    for num in [1, 10, 100]:
        surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
        loss[num] = train_model(surrogate_model, model, X, Y, num_samples=num)
    assert is_better(loss[10], loss[1])
    assert is_better(loss[100], loss[10])
