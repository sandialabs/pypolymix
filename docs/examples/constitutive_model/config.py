"""Configuration for the constitutive-model surrogate example.

Every knob lives in a single :class:`Config` dataclass. The argparse front end is
derived from the dataclass fields, so adding a field automatically adds a flag --
there is no second place to keep in sync.

Run ``python train.py --help`` to see the generated interface.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict

import torch
import torch.nn.functional as F

#: Activations selectable by name from the command line.
ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
}


@dataclass
class Config:
    """All settings for data loading, model construction, training and evaluation."""

    # ------------------------------------------------------------------ paths
    data_dir: str = "toy_data"
    """Directory holding the prepared dataset (see ``data.py`` for the contract)."""

    run_dir: str = "runs/demo"
    """Output directory for checkpoints, ``transforms.json``, metrics and figures."""

    # ----------------------------------------------------------- architecture
    num_inputs: int = 6
    """Width of ``x = [state, controls]``. Cross-checked against ``meta.json``."""

    num_outputs: int = 3
    """Width of ``y = dstate/dt``. Cross-checked against ``meta.json``."""

    num_experts: int = 4
    """Number of expert networks in the mixture."""

    expert_width: int = 64
    """Hidden width of every expert."""

    expert_depth: int = 4
    """Number of hidden layers in every expert."""

    gate_width: int = 16
    """Hidden width of the gating network."""

    gate_depth: int = 1
    """Number of hidden layers in the gating network."""

    activation: str = "silu"
    """Hidden activation for experts and gate. One of ``ACTIVATIONS``."""

    # ---------------------------------------------------------- optimization
    lr: float = 1e-3
    """AdamW learning rate."""

    weight_decay: float = 1e-2
    """AdamW weight decay."""

    batch_size: int = 4096
    """Rows per optimizer step."""

    epochs: int = 150
    """Total training epochs. Training resumes from ``last.pt`` if present."""

    grad_clip: float = 1.0
    """Global gradient-norm clip. Set to 0 to disable."""

    coswr_t0: int = 10
    """First cycle length of ``CosineAnnealingWarmRestarts``, in epochs."""

    coswr_tmult: int = 2
    """Cycle-length multiplier. ``epochs = t0 * (tmult**k - 1) / (tmult - 1)``
    completes exactly ``k`` cycles."""

    coswr_eta_min: float = 1e-6
    """Learning-rate floor of the cosine schedule."""

    # ----------------------------------------------------------- transforms
    output_powers: str = "0.0909,0.1429,0.1429"
    """Signed-power exponents, one per output. See ``transforms.py``; these are a
    starting point, not a universal truth."""

    log_rate_outputs: str = "1,2"
    """Outputs to rewrite as ``d log10(state)/dt`` before the signed power.
    Comma-separated indices, or an empty string to disable."""

    log_input_columns: str = "0,1,2,5"
    """Input columns to take ``log10`` of. Must be strictly positive columns."""

    state_indices: str = "0,1,2"
    """Column of ``X`` holding the state whose rate is output ``j``. Must agree
    with ``state_indices`` in the dataset's ``meta.json``."""

    fit_max_rows: int = 1_000_000
    """Rows sampled from the training split when fitting the transforms. Robust
    quantiles converge quickly; there is no need to stream hundreds of millions
    of rows."""

    # -------------------------------------------------------------- runtime
    seed: int = 0
    """Seed for torch/numpy. Controls initialization and shuffling."""

    device: str = "auto"
    """``auto``, ``cpu``, ``cuda``, or an explicit device string."""

    num_workers: int = 0
    """DataLoader worker processes. 0 is usually fastest for memory-mapped arrays."""

    amp: bool = False
    """Enable bfloat16 autocast. Only meaningful on CUDA."""

    # ------------------------------------------------------------------ misc
    def resolved_device(self) -> torch.device:
        """Return the concrete device, resolving ``auto``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def activation_fn(self):
        """Return the callable named by :attr:`activation`."""
        try:
            return ACTIVATIONS[self.activation]
        except KeyError:
            raise ValueError(
                f"unknown activation {self.activation!r}; choose from {sorted(ACTIVATIONS)}"
            ) from None

    def powers(self) -> list:
        """Parse :attr:`output_powers` into a list of floats."""
        values = [float(p) for p in self.output_powers.split(",") if p.strip()]
        if len(values) != self.num_outputs:
            raise ValueError(
                f"output_powers has {len(values)} entries but num_outputs is {self.num_outputs}"
            )
        return values

    def log_rate_indices(self) -> tuple:
        """Parse :attr:`log_rate_outputs` into a tuple of output indices."""
        return tuple(int(i) for i in self.log_rate_outputs.split(",") if i.strip())

    def log_input_indices(self) -> tuple:
        """Parse :attr:`log_input_columns` into a tuple of input column indices."""
        return tuple(int(i) for i in self.log_input_columns.split(",") if i.strip())

    def state_column_indices(self) -> tuple:
        """Parse :attr:`state_indices` into a tuple of input column indices."""
        return tuple(int(i) for i in self.state_indices.split(",") if i.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy, for embedding in a checkpoint."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Rebuild from :meth:`to_dict`, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add one flag per :class:`Config` field, using the field default and docstring."""
    defaults = Config()
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        if f.type is bool or isinstance(current, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=current)
        else:
            parser.add_argument(flag, type=type(current), default=current)
    return parser


def from_args(argv=None, description: str = "") -> Config:
    """Parse ``argv`` into a :class:`Config`."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args(argv)
    return Config(**vars(args))


# ``field`` is re-exported for downstream configs that extend this dataclass.
__all__ = ["ACTIVATIONS", "Config", "add_arguments", "field", "from_args"]
