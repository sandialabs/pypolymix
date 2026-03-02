# API Reference

`pypolymix` splits stochastic surrogate modeling into three composable layers:

1. **Surrogate models** (`pypolymix.surrogate_models`): deterministic forward
   models that expect an input tensor and a tensor of parameters.
2. **Parameter groups** (`pypolymix.parameter_groups`): variational families with
   associated priors defined over blocks of surrogate parameters.
3. **Stochastic model** (`pypolymix.StochasticModel`): glues a surrogate to one
   or more parameter groups and exposes a familiar PyTorch `nn.Module`.

## Stochastic Model

Wrap any surrogate in the `StochasticModel` framework and provide a list of 
parameter groups whose samples are concatenated before being fed to the surrogate.

```python
import torch
from pypolymix import StochasticModel, parameter_groups, surrogate_models

surrogate = surrogate_models.NeuralNetwork(num_inputs=1, num_outputs=1)
group = parameter_groups.IIDGaussianGroup("nn", surrogate.num_params())
model = StochasticModel(surrogate, [group])

x = torch.linspace(-1, 1, 32).unsqueeze(-1)
y = model(x, num_samples=8)  # (8, 32, 1)
loss = y.mean() + model.distribution_loss()
```

::: pypolymix.stochastic_model.StochasticModel

## Parameter Groups

Parameter groups describe how parameters are sampled and regularised. They can
be mixed (e.g. deterministic biases and stochastic weights) by instantiating
multiple groups and passing them to the same stochastic model.

### DeterministicGroup

Use when you want point estimates for a parameter block while still leveraging
the same interface as the stochastic groups.

::: pypolymix.parameter_groups.deterministic.DeterministicGroup

### IIDGaussianGroup

Independent Normal posterior with per-parameter mean and (log) std that supports
reparameterised sampling for variational inference.

::: pypolymix.parameter_groups.gaussian.IIDGaussianGroup

### GaussianGroup

Full-covariance Gaussian variational family parameterised by a Cholesky factor,
useful when posterior correlations cannot be ignored.

::: pypolymix.parameter_groups.gaussian.GaussianGroup

### LowRankGaussianGroup

Low-rank plus diagonal Gaussian family that captures the largest correlations
with a configurable rank while staying closer to `O(d)` memory.

::: pypolymix.parameter_groups.gaussian.LowRankGaussianGroup

### LangevinGroup

Implicit posterior sampler driven by unadjusted Langevin dynamics:
`theta <- theta + step_size * score(theta) + sqrt(2 * step_size) * noise`.

`LangevinGroup` keeps the same `ParameterGroup` interface and can be mixed with
the other groups inside `StochasticModel`.

The score model is passed in as any `SurrogateModel` satisfying:

- `score_model.num_inputs == num_params`
- `score_model.num_outputs == num_params`

`NeuralNetwork` is the most common choice for this role.

::: pypolymix.parameter_groups.langevin.LangevinGroup

## Priors

All parameter groups accept a `Prior` object that
creates a `torch.distributions.Distribution` on demand. Priors can therefore
share learnable buffers or be reused across groups.

### IIDGaussianPrior

::: pypolymix.priors.common.IIDGaussianPrior

### GaussianPrior

::: pypolymix.priors.common.GaussianPrior

### LaplacePrior

::: pypolymix.priors.common.LaplacePrior

## Surrogate Models

Surrogates implement the deterministic mapping from `(x, params)` to outputs.
They are ordinary PyTorch modules, but operate on batched parameter samples.

### NeuralNetwork

Fully-connected MLP whose weights/biases are supplied dynamically via sampled
parameters.

::: pypolymix.surrogate_models.neural_network.NeuralNetwork

### PolynomialChaosExpansion

Legendre polynomial chaos expansion with configurable dimension, degree, and
number of outputs.

::: pypolymix.surrogate_models.polynomial_chaos.PolynomialChaosExpansion

### Mixture Components

The mixture module contains both the gating network and the full
Mixture-of-Experts surrogate, enabling scalable ensembles driven by sampled
parameters.

#### GatingNetwork

::: pypolymix.surrogate_models.mixture.GatingNetwork

#### MixtureOfExperts

::: pypolymix.surrogate_models.mixture.MixtureOfExperts
