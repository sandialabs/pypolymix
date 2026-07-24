import torch

from pypolymix.parameter_groups import IIDGaussianGroup, DeterministicGroup
from pypolymix.surrogate_models import NeuralNetwork
from pypolymix import StochasticModel
import copy

# setup NN and input data for all of the tests
def make_problem():
    _ = torch.manual_seed(2048)
    num_samples = 30
    X = 2 * torch.rand(num_samples, 1) - 1
    y1 = torch.sin(torch.pi * X) + 0.05 * torch.randn(X.shape)
    y2 = torch.cos(torch.pi * X) + 0.05 * torch.randn(X.shape)
    Y = torch.hstack((y1, y2))

    width = 16
    depth = 1
    surrogate_model = NeuralNetwork(num_inputs=1, num_outputs=2, width=width, depth=depth, activation=torch.nn.functional.tanh)
    num_params = surrogate_model.num_params()
    print(f"This model has {num_params} parameters")

    # TODO the interface for creating parameter_groups requires manual math (width * width)
    # that could potentially be done programmatically
    parameter_groups = [  # Input layer
        DeterministicGroup("input_layer_weights", width),
        DeterministicGroup("input_layer_biases", width),
    ]
    for j in range(depth - 1):
        parameter_groups += [  # Hidden layers
            DeterministicGroup(f"layer{j + 1}_weights", width * width),
            DeterministicGroup(f"layer{j + 1}_biases", width),
        ]
    parameter_groups += [  # Output layer
        IIDGaussianGroup("output_weights", 2 * width),
        IIDGaussianGroup("output_biases", 2),
    ]

    model = StochasticModel(
        surrogate_model=surrogate_model, parameter_groups=parameter_groups
    )
    print(f"Created stochastic model with {model.num_params()} parameters")
    return surrogate_model, model, X, Y

# num_epochs, num_samples, weight_factor and scheduler
# will be the primary parameters tested
def train_model(surrogate_model, model, X, Y,
                num_epochs=10_000, num_samples=100,
                weight_factor=1e-2, lr=1e-3, weight_decay=1e-4):
    # Learnable global precision
    log_tau = torch.nn.Parameter(torch.tensor(0.0))  # τ = exp(log_tau)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_tau], lr=lr, weight_decay=weight_decay
    )

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
        data_loss *= torch.exp(log_tau)
        distribution_loss = model.distribution_loss()
        total_loss = data_loss + weight_factor * distribution_loss

        # Backprop + step
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # # Logging
        # if (epoch + 1) % 100 == 0:
        #     current_lr = scheduler.get_last_lr()[0]
        #     print(
        #         f"Epoch {epoch + 1:5d} | "
        #         f"learning rate = {current_lr:.6f} | "
        #         f"data loss = {data_loss.item():.4f} | "
        #         f"distribution loss = {distribution_loss.item():.4f} | "
        #         f"total loss = {total_loss.item():.4f}"
        #     )
    # total_loss is a tensor, return a float
    return total_loss.item()

def test_epochs():
    epoch_losses = []
    for num in [1_000, 3_000, 10_000]:
        surrogate_model, model, X, Y = make_problem()
        total_loss = train_model(surrogate_model, model, X, Y, num_epochs=num)
        epoch_losses.append(total_loss)
    assert sorted(epoch_losses, reverse=True) == epoch_losses, "Failed: Model does not improve with more epoch losses"

# TODO figure out if more samples isn't necessarily better
# or if this fails due to a bug in code
def test_num_samples():
    sample_loss = []
    for num in [10, 30, 100]:
        surrogate_model, model, X, Y = make_problem()
        total_loss = train_model(surrogate_model, model, X, Y, num_samples=num)
        sample_loss.append(total_loss)
    assert sorted(sample_loss, reverse=True) == sample_loss, "Failed: Model does not improve with more samples"

'''
print()
print("weight_factor\tTotal Loss")
# Lower weight factor results in less total loss
# because it is multiplied by distribution loss
# not sure if this assertion is meaningful
weight_factor_loss = []
for num in [1e-1, 1e-2, 1e-3]:
    total_loss = train_model(copy.deepcopy(model), weight_factor=num)
    weight_factor_loss.append(total_loss)
    print(f"{num}\t\t{total_loss:.2f}")
assert sorted(weight_factor_loss, reverse=True) == weight_factor_loss, "Model does not improve with decreased weight factor"
'''
