"""Input and output transforms -- a suggestion, not a universal recipe.

Constitutive rates are hostile to a neural network in their raw units. In the toy
dataset the state variables span eight orders of magnitude, the rates span more
than twenty, and two of the three rates change sign. Fitting a mean-squared error
on those numbers optimizes almost exclusively the largest-magnitude rows.

This module provides a transform pair that worked well on a real creep dataset,
plus a plain standardization pair so that swapping the transform out is a
demonstrated operation rather than an aspiration. **Treat the defaults as a
starting point and check them against your own data** -- see "Choosing a transform"
in the README, and :func:`suggest_power`.

The suggested pair applies, in order:

1. ``log10`` to the strictly positive input columns, leaving the rest alone;
2. median/IQR centering of the inputs (robust to the heavy tails that survive
   step 1);
3. for the outputs that are rates of a positive state, the change of variable
   ``dq/dt -> d log10(q)/dt = (dq/dt) / (q ln 10)``, which removes the state's
   own dynamic range from its rate;
4. a signed power ``sign(v) |v|^p`` with a small ``p``, which compresses the
   remaining decades while preserving sign and the zero;
5. median/IQR centering of the outputs.

Everything is invertible, and ``evaluate.py`` inverts it to report errors in raw
physical units.

**The fitted transform is part of the model.** It is written to
``<run_dir>/transforms.json`` at the start of training and reloaded for
evaluation. Refitting or deleting it silently invalidates every checkpoint in that
run directory: the weights were trained against one normalization and will be
evaluated against another, with no error raised anywhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

LN10 = math.log(10.0)

_REGISTRY: Dict[str, type] = {}


def register(cls):
    """Register a transform class so it can round-trip through ``transforms.json``."""
    _REGISTRY[cls.__name__] = cls
    return cls


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #


class InputTransform(nn.Module):
    """Maps raw inputs ``X`` to network inputs ``Z``.

    Subclasses implement :meth:`fit`, :meth:`forward` and :meth:`config`.
    """

    def fit(self, x: torch.Tensor) -> "InputTransform":
        """Estimate statistics from raw training inputs. Returns ``self``."""
        raise NotImplementedError

    def config(self) -> dict:
        """Return the JSON-serializable state of this transform."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"kind": type(self).__name__, **self.config()}


class OutputTransform(nn.Module):
    """Maps raw rates ``Y`` to network targets ``Z`` and back.

    ``X`` is passed to both directions because a useful change of variable may
    depend on the state (see the ``d log10(q)/dt`` step above).
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "OutputTransform":
        """Estimate statistics from raw training pairs. Returns ``self``."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Map network outputs back to raw rates."""
        raise NotImplementedError

    def config(self) -> dict:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"kind": type(self).__name__, **self.config()}


def _robust_center_scale(values: torch.Tensor, min_scale: float) -> tuple:
    """Return column-wise median and IQR, with the IQR floored at ``min_scale``."""
    center = values.nanquantile(0.5, dim=0)
    spread = values.nanquantile(0.75, dim=0) - values.nanquantile(0.25, dim=0)
    return center, spread.clamp_min(min_scale)


# --------------------------------------------------------------------------- #
# Suggested pair
# --------------------------------------------------------------------------- #


@register
class LogMedianIQR(InputTransform):
    """``log10`` on selected columns, then median/IQR centering.

    Args:
        log_columns: Indices of strictly positive columns to take ``log10`` of.
            Every other column passes through unchanged.
        eps: Floor applied before the logarithm.
        min_scale: Lower bound on the IQR, guarding constant columns.
    """

    def __init__(
        self,
        log_columns: Sequence[int] = (0, 1, 2, 5),
        eps: float = 1e-30,
        min_scale: float = 1e-6,
    ):
        super().__init__()
        self.log_columns = tuple(int(c) for c in log_columns)
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        self.register_buffer("center", None)
        self.register_buffer("scale", None)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.log_columns:
            return x
        out = x.clone()
        idx = list(self.log_columns)
        out[:, idx] = torch.log10(x[:, idx].clamp_min(self.eps))
        return out

    def fit(self, x: torch.Tensor) -> "LogMedianIQR":
        features = self._features(x.double())
        center, scale = _robust_center_scale(features, self.min_scale)
        self.center = center.float()
        self.scale = scale.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._features(x) - self.center) / self.scale

    def config(self) -> dict:
        return {
            "log_columns": list(self.log_columns),
            "eps": self.eps,
            "min_scale": self.min_scale,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


@register
class SignedPowerMedianIQR(OutputTransform):
    """State-relative rates, signed power compression, then median/IQR centering.

    Args:
        state_indices: Column of ``X`` holding the state whose rate is output ``j``.
        log_rate_outputs: Outputs to rewrite as ``d log10(state)/dt``. Only valid
            for strictly positive states. Pass an empty tuple to disable.
        powers: Signed-power exponent per output, each in ``(0, 1]``. Smaller
            values compress harder; ``1.0`` is a no-op.
        eps: Floor applied to states before dividing by them.
        min_scale: Lower bound on the IQR.
    """

    def __init__(
        self,
        state_indices: Sequence[int] = (0, 1, 2),
        log_rate_outputs: Sequence[int] = (1, 2),
        powers: Sequence[float] = (1 / 11, 1 / 7, 1 / 7),
        eps: float = 1e-30,
        min_scale: float = 1e-12,
    ):
        super().__init__()
        self.state_indices = tuple(int(i) for i in state_indices)
        self.log_rate_outputs = tuple(int(i) for i in log_rate_outputs)
        self.eps = float(eps)
        self.min_scale = float(min_scale)
        powers_t = torch.as_tensor(list(powers), dtype=torch.float32)
        if powers_t.numel() != len(self.state_indices):
            raise ValueError("powers must have one entry per output")
        if not bool(((powers_t > 0) & (powers_t <= 1)).all()):
            raise ValueError(f"powers must lie in (0, 1], got {powers_t.tolist()}")
        self.register_buffer("powers", powers_t)
        self.register_buffer("center", None)
        self.register_buffer("scale", None)

    def _state_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-output divisor implementing ``d log10(q)/dt``."""
        divisor = torch.ones((x.shape[0], len(self.state_indices)), dtype=x.dtype, device=x.device)
        for j in self.log_rate_outputs:
            divisor[:, j] = x[:, self.state_indices[j]].clamp_min(self.eps) * LN10
        return divisor

    def _compress(self, values: torch.Tensor) -> torch.Tensor:
        powers = self.powers.to(values.dtype)
        return torch.sign(values) * torch.pow(values.abs(), powers)

    def _expand(self, values: torch.Tensor) -> torch.Tensor:
        powers = self.powers.to(values.dtype)
        return torch.sign(values) * torch.pow(values.abs(), 1.0 / powers)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "SignedPowerMedianIQR":
        compressed = self._compress(y.double() / self._state_scale(x.double()))
        center, scale = _robust_center_scale(compressed, self.min_scale)
        self.center = center.float()
        self.scale = scale.float()
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self._compress(y / self._state_scale(x)) - self.center) / self.scale

    def inverse(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._expand(z * self.scale + self.center) * self._state_scale(x)

    def config(self) -> dict:
        return {
            "state_indices": list(self.state_indices),
            "log_rate_outputs": list(self.log_rate_outputs),
            "powers": self.powers.tolist(),
            "eps": self.eps,
            "min_scale": self.min_scale,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


# --------------------------------------------------------------------------- #
# Plain alternative
# --------------------------------------------------------------------------- #


@register
class StandardizeInput(InputTransform):
    """Column-wise mean/standard-deviation scaling of the raw inputs."""

    def __init__(self, min_scale: float = 1e-12):
        super().__init__()
        self.min_scale = float(min_scale)
        self.register_buffer("center", None)
        self.register_buffer("scale", None)

    def fit(self, x: torch.Tensor) -> "StandardizeInput":
        values = x.double()
        self.center = values.mean(dim=0).float()
        self.scale = values.std(dim=0).clamp_min(self.min_scale).float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.scale

    def config(self) -> dict:
        return {
            "min_scale": self.min_scale,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


@register
class StandardizeOutput(OutputTransform):
    """Column-wise mean/standard-deviation scaling of the raw rates.

    Provided as a baseline. On heavy-tailed rate data this typically trains to a
    much worse rollout than :class:`SignedPowerMedianIQR`, which is the point.
    """

    def __init__(self, min_scale: float = 1e-30):
        super().__init__()
        self.min_scale = float(min_scale)
        self.register_buffer("center", None)
        self.register_buffer("scale", None)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "StandardizeOutput":
        values = y.double()
        self.center = values.mean(dim=0).float()
        self.scale = values.std(dim=0).clamp_min(self.min_scale).float()
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y - self.center) / self.scale

    def inverse(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.center

    def config(self) -> dict:
        return {
            "min_scale": self.min_scale,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


# --------------------------------------------------------------------------- #
# Pair, persistence, diagnostics
# --------------------------------------------------------------------------- #


class TransformPair(nn.Module):
    """Bundle of an :class:`InputTransform` and an :class:`OutputTransform`."""

    def __init__(self, input_transform: InputTransform, output_transform: OutputTransform):
        super().__init__()
        self.input = input_transform
        self.output = output_transform

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "TransformPair":
        self.input.fit(x)
        self.output.fit(x, y)
        return self

    def save(self, path) -> None:
        payload = {"version": 1, "input": self.input.to_dict(), "output": self.output.to_dict()}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @staticmethod
    def load(path) -> "TransformPair":
        payload = json.loads(Path(path).read_text())
        return TransformPair(_rebuild(payload["input"]), _rebuild(payload["output"]))


def _rebuild(spec: dict):
    """Reconstruct a registered transform from its serialized ``config``."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown transform {kind!r}; register it with @transforms.register before loading"
        )
    center = spec.pop("center")
    scale = spec.pop("scale")
    obj = _REGISTRY[kind](**spec)
    obj.center = torch.as_tensor(center, dtype=torch.float32)
    obj.scale = torch.as_tensor(scale, dtype=torch.float32)
    return obj


def suggested_pair(
    state_indices: Sequence[int],
    log_columns: Sequence[int],
    log_rate_outputs: Sequence[int],
    powers: Sequence[float],
) -> TransformPair:
    """Build the suggested transform pair described in the module docstring."""
    return TransformPair(
        LogMedianIQR(log_columns=log_columns),
        SignedPowerMedianIQR(
            state_indices=state_indices,
            log_rate_outputs=log_rate_outputs,
            powers=powers,
        ),
    )


def suggest_power(values, candidates: Sequence[float] = ()) -> float:
    """Suggest a signed-power exponent for one output column.

    Picks the exponent whose transformed distribution has the smallest excess
    kurtosis, i.e. is closest to Gaussian-tailed. This is a crude but useful
    diagnostic: an exponent that is too large leaves a spike at zero with extreme
    tails, and one that is too small over-inflates near-zero noise.

    Args:
        values: 1-D array of raw rates for a single output.
        candidates: Exponents to try. Defaults to ``1/1 ... 1/24``.

    Returns:
        The best exponent found.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data) & (data != 0.0)]
    if data.size == 0:
        return 1.0
    if data.size > 200_000:
        data = np.random.default_rng(0).choice(data, 200_000, replace=False)
    grid = list(candidates) if len(candidates) else [1.0 / k for k in range(1, 25)]

    best_p, best_score = 1.0, np.inf
    for p in grid:
        z = np.sign(data) * np.abs(data) ** p
        std = z.std()
        if std <= 0 or not np.isfinite(std):
            continue
        excess_kurtosis = float((((z - z.mean()) / std) ** 4).mean() - 3.0)
        if abs(excess_kurtosis) < best_score:
            best_p, best_score = float(p), abs(excess_kurtosis)
    return best_p


__all__ = [
    "InputTransform",
    "LogMedianIQR",
    "OutputTransform",
    "SignedPowerMedianIQR",
    "StandardizeInput",
    "StandardizeOutput",
    "TransformPair",
    "register",
    "suggest_power",
    "suggested_pair",
]
