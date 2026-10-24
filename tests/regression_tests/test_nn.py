import torch

import pypolymix
from pypolymix import StochasticModel
from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.surrogate_models import NeuralNetwork


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
    surrogate_model = NeuralNetwork(
        num_inputs=1, num_outputs=2, width=width, depth=depth, activation=torch.nn.functional.tanh
    )
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
    # surrogate_model: The neural network function
    # model: the network's stochastic manager of weights and biases
    model = StochasticModel(surrogate_model=surrogate_model, parameter_groups=parameter_groups)
    print(f"Created stochastic model with {model.num_params()} parameters")
    return surrogate_model, model, X, Y


# Note: this can't be factored out from here and test_moe.py
# because this training loop uses pypolymix.elbo_loss with a per-output
# GaussianLikelihood (not the old log_tau precision trick / weight_factor
# anti-pattern -- see PM-07/PM-09 handoff notes).
# num_epochs, num_samples and scheduler are the primary parameters tested.
def train_model(
    surrogate_model, model, X, Y, num_epochs=10_000, num_samples=100, lr=1e-3, weight_decay=1e-4
):
    # Learnable per-output Gaussian observation noise
    likelihood = pypolymix.likelihoods.GaussianLikelihood(num_outputs=Y.shape[-1])

    # Optimizer: AdamW (likelihood's log_sigma excluded from weight decay)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": weight_decay},
            {"params": likelihood.no_decay_parameters(), "weight_decay": 0.0},
        ],
        lr=lr,
    )

    # Scheduler: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=10 * lr, total_steps=num_epochs
    )

    # Train the stochastic model
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Evaluate parameters and model
        Y_hat = model(X, num_samples=num_samples)  # (S, batch, outputs)

        # Losses
        nll = likelihood.nll(Y_hat, Y.unsqueeze(0).expand_as(Y_hat))
        kl = model.distribution_loss()
        total_loss = pypolymix.elbo_loss(nll, kl, n_train=X.shape[0], temperature=1.0)

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
        #         f"nll = {nll.item():.4f} | "
        #         f"kl = {kl.item():.4f} | "
        #         f"total loss = {total_loss.item():.4f} | "
        #         f"log_sigma = {likelihood.log_sigma.detach().tolist()}"
        #     )
    # total_loss is a tensor, return a float
    return total_loss.item()


def is_better(expected_better, expected_worse):
    """
    Check that expected_better is not too much more than expected_worse
    Ideally it would be less over multiple samples
    But the tests are already long-running, so for now close is good enough

    The margin (1.0) is re-derived for the corrected elbo_loss scale: with the
    old log_tau/weight_factor recipe total_loss ran ~1-8 and a margin of 10
    was already generous; with the normalizer fix total_loss instead runs
    ~0.5-3 (it can even go negative, since a Gaussian NLL is a log-density,
    not a non-negative quantity), so a margin on the same order as that scale
    still catches a real regression without being vacuous.
    """
    return expected_better < expected_worse + 1.0


def test_returns_float():
    surrogate_model, model, X, Y = make_problem()
    total_loss = train_model(surrogate_model, model, X, Y, num_epochs=10, num_samples=2)
    assert isinstance(total_loss, float)


def test_epochs():
    loss = {}
    for num in [1_000, 3_000, 10_000]:
        surrogate_model, model, X, Y = make_problem()
        loss[num] = train_model(surrogate_model, model, X, Y, num_epochs=num)
    assert is_better(loss[3_000], loss[1_000])
    assert is_better(loss[10_000], loss[3_000])


def test_num_samples():
    loss = {}
    for num in [1, 10, 100]:
        surrogate_model, model, X, Y = make_problem()
        loss[num] = train_model(surrogate_model, model, X, Y, num_samples=num)
    assert is_better(loss[10], loss[1])
    assert is_better(loss[100], loss[10])


# Note: there used to be a test_weight_factor() here that swept the
# weight_factor anti-pattern's arbitrary KL scale and asserted a smaller
# weight_factor gives smaller total_loss. pypolymix.elbo_loss removes that
# knob entirely (KL is divided by n_train, the only scale consistent with a
# proper ELBO -- see elbo_loss's docstring), so there is no longer a
# weight_factor to sweep.
