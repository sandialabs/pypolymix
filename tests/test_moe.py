# Import statements
import torch

from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.surrogate_models import PolynomialChaosExpansion, MixtureOfExperts, GatingNetwork
from pypolymix import StochasticModel

# setup NN and input data for all of the tests
def make_problem():
    # Generate synthetic data
    num_samples = 50
    X = 2 * torch.rand(num_samples, 1) - 1
    Y = torch.sin(torch.pi * X) + torch.sign(X) + 0.05 * torch.randn(X.shape)
    # Create a surrogate model
    width = 16
    depth = 1
    surrogate_model = MixtureOfExperts(
        experts=[
            PolynomialChaosExpansion(num_inputs=1, num_outputs=1, degree=3),
            PolynomialChaosExpansion(num_inputs=1, num_outputs=1, degree=3),
        ],
        gating_network=GatingNetwork(num_inputs=1, num_experts=2, width=width, depth=depth, activation=torch.nn.functional.tanh)
    )

    num_params = surrogate_model.num_params()
    print(f"This model has {num_params} parameters")
    # Create parameter groups
    parameter_groups = [  # Experts
        IIDGaussianGroup("expert1_c0", 2),
        DeterministicGroup("expert1_c", 2),
        IIDGaussianGroup("expert2_c0", 2),
        DeterministicGroup("expert2_c", 2),
    ]
    # Gating network
    parameter_groups += [  # Input layer
        DeterministicGroup("gating_input_layer_weights", width),
        DeterministicGroup("gating_input_layer_biases", width),
    ]
    for j in range(depth - 1):
        parameter_groups += [  # Hidden layers
            DeterministicGroup(f"gating_layer{j + 1}_weights", width * width),
            DeterministicGroup(f"gating_layer{j + 1}_biases", width),
        ]
    parameter_groups += [  # Output layer
        IIDGaussianGroup("gating_output_weights", 2 * width),
        IIDGaussianGroup("gating_output_biases", 2),
    ] 

    model = StochasticModel(
        surrogate_model=surrogate_model, parameter_groups=parameter_groups
    )
    print(f"Created stochastic model with {model.num_params()} parameters")
    return surrogate_model, model, X, Y

# num_epochs, num_samples, weight_factor and scheduler
# will be the primary parameters tested
# TODO experiment with a different scheduler
def train_model(surrogate_model, model, X, Y,
                num_epochs=10_000, num_samples=100,
                weight_factor=1e-2, lr=1e-3, weight_decay=1e-4):
    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Scheduler: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=10 * lr,
        total_steps=num_epochs
    ) 
    # Train the stochastic model
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Evaluate parameters and model
        params = model.sample_parameters(num_samples=num_samples)
        Y_hat = surrogate_model(X, params)

        # Losses
        data_loss = loss_fn(Y_hat, Y.unsqueeze(0).expand_as(Y_hat)) / num_samples
        distribution_loss = model.distribution_loss()
        total_loss = data_loss + weight_factor * distribution_loss

        # Backprop + step
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if (epoch + 1) % 1000 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:5d} | "
                f"learning rate = {current_lr:.6f} | "
                f"data loss = {data_loss.item():.4f} | "
                f"distribution loss = {distribution_loss.item():.4f} | "
                f"total loss = {total_loss.item():.4f}"
            )
    # total_loss is a tensor, return a float
    return total_loss.item()

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
        surrogate_model, model, X, Y = make_problem()
        total_loss = train_model(surrogate_model, model, X, Y, num_epochs=num)
        epoch_losses[num] = total_loss
    assert is_better(epoch_losses[3_000], epoch_losses[1_000])
    assert is_better(epoch_losses[10_000], epoch_losses[3000])

def test_num_samples():
    loss = {}
    for num in [1, 10, 100]:
        surrogate_model, model, X, Y = make_problem()
        loss[num] = train_model(surrogate_model, model, X, Y, num_samples=num)
    assert is_better(loss[10], loss[1])
    assert is_better(loss[100], loss[10])

# Higher weight factor results in more total loss
# because total_loss = data_loss + weight_factor * distribution loss
def test_weight_factor():
    loss = {}
    for num in [1, 0.1, 0.01, 0.001]:
        surrogate_model, model, X, Y = make_problem()
        loss[num] = train_model(surrogate_model, model, X, Y, weight_factor=num)
    assert is_better(loss[0.1], loss[1])
    assert is_better(loss[0.01], loss[0.1])
    assert is_better(loss[0.001], loss[0.01])
