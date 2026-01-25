<p align="center">
  <img src="https://raw.githubusercontent.com/sandialabs/pypolymix/main/docs/img/logo.png" alt="pypolymix logo" width="160">
</p>

| Docs | Code style |
|:----:|:----------:|
| [![Docs](https://img.shields.io/badge/docs-online-blue)](https://sandialabs.github.io/pypolymix) | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |

# Pypolymix

`pypolymix` is a light-weight PyTorch companion for building **stochastic surrogate models**.

## Documentation

Project documentation lives at https://sandialabs.github.io/pypolymix.

## Installation

From PyPI:
```
python -m pip install pypolymix
```

Optional extras:
```
python -m pip install "pypolymix[examples]"
python -m pip install "pypolymix[docs]"
python -m pip install "pypolymix[dev]"
```

For local development:
```
python -m pip install -e ".[dev,docs,examples]"
```

## Quickstart

Fit a 1D polynomial chaos model with a mix of stochastic and deterministic coefficients:

```python
import torch

from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.surrogate_models import PolynomialChaosExpansion
from pypolymix import StochasticModel

torch.manual_seed(0)
x = torch.linspace(-1, 1, 200).unsqueeze(-1)
y = torch.sin(3 * x) + 0.1 * torch.randn_like(x)

surrogate_model = PolynomialChaosExpansion(num_inputs=1, num_outputs=1, degree=5)
num_params = surrogate_model.num_params()

parameter_groups = [
    IIDGaussianGroup("stochastic", 2),
    DeterministicGroup("deterministic", num_params - 2),
]

model = StochasticModel(surrogate_model=surrogate_model, parameter_groups=parameter_groups)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
for _ in range(500):
    optimizer.zero_grad()
    preds = model(x, num_samples=16).mean(dim=0)
    data_loss = torch.mean((preds - y) ** 2)
    loss = data_loss + 1e-3 * model.distribution_loss()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    samples = model(x, num_samples=100)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/sandialabs/pypolymix/main/docs/img/pce.png" alt="Polynomial chaos example" width="720">
</p>

## Development

If you use Poetry:
```
poetry install --with dev,docs,examples
```

Install the `pre-commit` hook:
```
pre-commit install
```

## Examples

Notebooks live in `docs/examples/` and are rendered on the documentation site.
