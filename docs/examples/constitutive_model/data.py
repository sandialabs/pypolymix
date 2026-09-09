"""Dataset contract, memory-mapped loader, and a synthetic dataset generator.

The example learns the right-hand side of a constitutive law

    dq/dt = f(q, u)

where ``q`` is the internal state and ``u`` the externally imposed controls. A
single row of the dataset is one ``(x, y)`` pair with

    x = [q (n_state), u (n_control)]      y = dq/dt (n_state)

**On-disk contract.** ``train.py`` and ``evaluate.py`` read exactly this layout,
and nothing else. To use your own data, write these files:

```text
<data_dir>/
  meta.json                      names, state/control split, row counts
  train/X.npy   (N, 6) float32   val/X.npy   test/X.npy    (test optional)
  train/Y.npy   (N, 3) float32   val/Y.npy   test/Y.npy
  rollout/case_0000.npz          {"t": (T,), "X": (T, 6)}  (optional)
  rollout/case_0001.npz          ...
```

``state_indices`` in ``meta.json`` says which columns of ``X`` hold ``q``; output
``j`` is understood to be ``d X[:, state_indices[j]] / dt``. That correspondence is
the one structural assumption in this example and it is what makes the trajectory
rollout in ``evaluate.py`` possible. Rows may be in any order and cases need not be
contiguous; only the rollout files carry time.

Running this module generates a synthetic dataset so the example is runnable with
no data at hand:

```bash
python data.py --out toy_data
```

The toy system is a three-state creep model (equivalent strain plus mobile and
immobile dislocation densities) driven by temperature, stress and irradiation
flux. It is not calibrated to any material. Its purpose is to reproduce the
features that motivate the rest of the example: state variables spanning many
orders of magnitude, rates spanning even more, sign changes in the density rates,
and thermally activated regime changes that give a mixture of experts something
to partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                              SequentialSampler)

# --------------------------------------------------------------------------- #
# Toy constitutive model
# --------------------------------------------------------------------------- #

INPUT_NAMES = ["evm", "rhom", "rhoi", "temperature", "stress", "flux"]
OUTPUT_NAMES = ["devm_dt", "drhom_dt", "drhoi_dt"]
STATE_INDICES = [0, 1, 2]
CONTROL_INDICES = [3, 4, 5]

R_GAS = 8.314  # J / (mol K)
Q_ACT = 250.0e3  # J / mol, activation energy for glide and recovery
STRESS_REF = 100.0  # MPa
FLUX_REF = 1.0e-7  # dpa / s
N_EXP = 4.0  # stress exponent
A_GLIDE = 5.0e-5  # strain rate prefactor
K_MULT = 5.0e7  # dislocation multiplication at forest obstacles
K_TRAP = 2.0e2  # mobile -> immobile trapping
K_RECM = 5.0e8  # thermal recovery of mobile density
K_RECI = 2.0e7  # thermal recovery of immobile density
K_IRR = 1.0  # irradiation enhancement of immobile recovery
RHO_ANNEALED = 1.0e10  # 1/m^2, density of a fully annealed network
EVM_MAX = 0.15  # stop a case once this strain is reached
T_CAP = 1.0e10  # s, stop a case at this clock regardless
MAX_RHS_EVALS = 200_000  # per-case solver budget; see _integrate_case


def toy_rhs(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Evaluate ``dq/dt`` for the toy creep model.

    Args:
        state: ``(..., 3)`` array of ``[evm, rhom, rhoi]``, all strictly positive.
        controls: ``(..., 3)`` array of ``[temperature, stress, flux]``.

    Returns:
        ``(..., 3)`` array of ``[devm_dt, drhom_dt, drhoi_dt]``.
    """
    _, rhom, rhoi = state[..., 0], state[..., 1], state[..., 2]
    temperature, stress, flux = controls[..., 0], controls[..., 1], controls[..., 2]

    thermal = np.exp(-Q_ACT / (R_GAS * temperature))
    reduced_stress = stress / STRESS_REF

    devm_dt = A_GLIDE * rhom * reduced_stress**N_EXP * thermal
    multiplication = K_MULT * np.sqrt(rhoi) * devm_dt
    trapping = K_TRAP * rhom * devm_dt
    # Recovery relaxes each density toward the annealed network density rather
    # than toward zero. Without that floor a cold, unstressed case decays for
    # ever, the state underflows, and the log-state solver stalls.
    recovery_m = K_RECM * (rhom - RHO_ANNEALED) * thermal
    recovery_i = K_RECI * (rhoi - RHO_ANNEALED) * thermal * (1.0 + K_IRR * flux / FLUX_REF)

    drhom_dt = multiplication - trapping - recovery_m
    drhoi_dt = trapping - recovery_i
    return np.stack([devm_dt, drhom_dt, drhoi_dt], axis=-1)


class _BudgetExceeded(Exception):
    """Raised from the right-hand side when a case exceeds its solver budget."""


def _integrate_case(state0: np.ndarray, controls: np.ndarray, num_points: int, rng) -> tuple:
    """Integrate one case in log-state and sample it on a log-spaced clock.

    Integrating ``d log(q)/dt = f(q, u) / q`` keeps every state strictly positive
    without needing a solver constraint. The same trick is used by the rollout in
    ``evaluate.py``.

    Returns ``(t, X, Y)`` or ``None`` if the case was rejected.
    """
    from scipy.integrate import solve_ivp

    # A stiff draw can make LSODA crawl for minutes. Budgeting right-hand-side
    # evaluations bounds the cost of a single case; the caller simply draws
    # another one.
    budget = [MAX_RHS_EVALS]

    def log_rhs(_t, log_q):
        budget[0] -= 1
        if budget[0] < 0:
            raise _BudgetExceeded
        q = np.exp(log_q)
        return toy_rhs(q, controls) / q

    def strain_limit(_t, log_q):
        return log_q[0] - np.log(EVM_MAX)

    strain_limit.terminal = True
    strain_limit.direction = 1.0

    # First pass at loose tolerance, only to locate the end of the case: either the
    # strain limit is reached or the clock runs out. Dense output is deliberately
    # avoided -- evaluating LSODA interpolants is an order of magnitude more
    # expensive than simply integrating a second time on a known grid.
    try:
        probe = solve_ivp(
            log_rhs,
            (0.0, T_CAP),
            np.log(state0),
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
            events=strain_limit,
        )
        if not probe.success or probe.t[-1] <= 0.0:
            return None

        # Log-spaced clock: creep spans many decades in time and a linear grid
        # would spend every sample in the final decade.
        t_end = float(probe.t[-1])
        t = np.geomspace(t_end * 1e-9, t_end, num_points)
        solution = solve_ivp(
            log_rhs,
            (0.0, t_end),
            np.log(state0),
            method="LSODA",
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
        )
    except _BudgetExceeded:
        return None
    if not solution.success or solution.y.shape[1] != num_points:
        return None

    log_q = solution.y
    if not np.all(np.isfinite(log_q)):
        return None

    state = np.exp(log_q).T  # (num_points, 3)
    if not np.all(state > 0.0):
        return None

    controls_t = np.broadcast_to(controls, (num_points, 3))
    x = np.concatenate([state, controls_t], axis=1)
    y = toy_rhs(state, controls_t)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None

    # A small amount of measurement noise keeps the problem honest; without it a
    # surrogate can chase solver round-off.
    y = y * (1.0 + 1e-3 * rng.standard_normal(y.shape))
    return t, x.astype(np.float32), y.astype(np.float32)


def generate(
    out_dir: Path,
    num_cases: int = 384,
    num_points: int = 512,
    seed: int = 0,
    val_cases: int = 32,
    test_cases: int = 32,
    rollout_cases: int = 32,
) -> Dict[str, int]:
    """Generate a synthetic dataset in the on-disk contract described above."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)

    cases: List[tuple] = []
    attempts = 0
    while len(cases) < num_cases and attempts < 8 * num_cases:
        attempts += 1
        controls = np.array(
            [
                rng.uniform(700.0, 1000.0),  # temperature, K
                10.0 ** rng.uniform(np.log10(20.0), np.log10(300.0)),  # stress, MPa
                10.0 ** rng.uniform(-9.0, -6.0),  # flux, dpa/s
            ]
        )
        state0 = np.array(
            [
                1.0e-5,  # evm
                10.0 ** rng.uniform(11.0, 12.0),  # rhom, 1/m^2
                10.0 ** rng.uniform(12.0, 14.0),  # rhoi, 1/m^2
            ]
        )
        case = _integrate_case(state0, controls, num_points, rng)
        if case is not None:
            cases.append(case)

    if len(cases) < num_cases:
        raise RuntimeError(f"only {len(cases)}/{num_cases} cases integrated successfully")

    order = rng.permutation(num_cases)
    n_train = num_cases - val_cases - test_cases - rollout_cases
    if n_train <= 0:
        raise ValueError("num_cases is too small for the requested split")
    bounds = np.cumsum([n_train, val_cases, test_cases, rollout_cases])
    split_ids = {
        "train": order[: bounds[0]],
        "val": order[bounds[0] : bounds[1]],
        "test": order[bounds[1] : bounds[2]],
        "rollout": order[bounds[2] :],
    }

    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        x = np.concatenate([cases[i][1] for i in split_ids[split]], axis=0)
        y = np.concatenate([cases[i][2] for i in split_ids[split]], axis=0)
        shuffle = rng.permutation(len(x))
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", np.ascontiguousarray(x[shuffle]))
        np.save(split_dir / "Y.npy", np.ascontiguousarray(y[shuffle]))
        counts[split] = int(len(x))

    rollout_dir = out_dir / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(split_ids["rollout"]):
        t, x, _ = cases[i]
        np.savez_compressed(rollout_dir / f"case_{k:04d}.npz", t=t, X=x)

    meta = {
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "state_indices": STATE_INDICES,
        "control_indices": CONTROL_INDICES,
        "splits": counts,
        "num_rollout_cases": int(len(split_ids["rollout"])),
        "source": "synthetic toy creep model from data.py",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return counts


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_meta(data_dir) -> dict:
    """Read and validate ``meta.json``."""
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate the toy dataset with `python data.py --out {data_dir}` "
            "or write your own following the contract in the data.py docstring."
        )
    meta = json.loads(path.read_text())
    for key in ("input_names", "output_names", "state_indices"):
        if key not in meta:
            raise ValueError(f"{path} is missing required key {key!r}")
    if len(meta["state_indices"]) != len(meta["output_names"]):
        raise ValueError(
            "meta.json: state_indices and output_names must have the same length; "
            "output j is interpreted as the rate of state_indices[j]"
        )
    return meta


class DerivativeDataset(torch.utils.data.Dataset):
    """Memory-mapped ``(X, Y)`` split.

    ``__getitem__`` accepts a sequence of indices and returns a whole batch, which
    is roughly two orders of magnitude faster than fetching rows one at a time and
    collating. Use :func:`make_loader` to drive it correctly.
    """

    def __init__(self, data_dir, split: str):
        split_dir = Path(data_dir) / split
        self.x = np.load(split_dir / "X.npy", mmap_mode="r")
        self.y = np.load(split_dir / "Y.npy", mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError(f"{split}: X has {len(self.x)} rows but Y has {len(self.y)}")
        self.split = split

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, index):
        idx = np.asarray(index)
        return {
            "X": torch.from_numpy(np.ascontiguousarray(self.x[idx], dtype=np.float32)),
            "Y": torch.from_numpy(np.ascontiguousarray(self.y[idx], dtype=np.float32)),
        }

    def full(self) -> tuple:
        """Return the whole split as two in-memory float32 tensors."""
        return (
            torch.from_numpy(np.array(self.x, dtype=np.float32)),
            torch.from_numpy(np.array(self.y, dtype=np.float32)),
        )


def make_loader(
    dataset: DerivativeDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator=None,
) -> DataLoader:
    """Wrap a :class:`DerivativeDataset` in a batch-at-a-time ``DataLoader``."""
    base = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    # batch_size=None disables per-item fetching, so the sampler's index list is
    # handed to DerivativeDataset.__getitem__ in one go.
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_rollout_cases(data_dir, limit: int = 0) -> List[dict]:
    """Load rollout trajectories, sorted by file name."""
    paths: Sequence[Path] = sorted((Path(data_dir) / "rollout").glob("case_*.npz"))
    if limit:
        paths = paths[:limit]
    cases = []
    for path in paths:
        with np.load(path) as handle:
            cases.append(
                {
                    "name": path.stem,
                    "t": np.asarray(handle["t"], dtype=np.float64),
                    "X": np.asarray(handle["X"], dtype=np.float64),
                }
            )
    return cases


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    """Generate the synthetic dataset and print a short summary of its ranges."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="toy_data", help="output directory")
    parser.add_argument("--num-cases", type=int, default=384)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    counts = generate(
        Path(args.out),
        num_cases=args.num_cases,
        num_points=args.num_points,
        seed=args.seed,
    )
    print(f"wrote {args.out}")
    for split, n in counts.items():
        print(f"  {split:<5} {n:>9,d} rows")

    x = np.load(Path(args.out) / "train" / "X.npy", mmap_mode="r")
    y = np.load(Path(args.out) / "train" / "Y.npy", mmap_mode="r")
    print("\ntrain ranges (min / max):")
    for j, name in enumerate(INPUT_NAMES):
        print(f"  x[{j}] {name:<12} {x[:, j].min():>12.4g} {x[:, j].max():>12.4g}")
    for j, name in enumerate(OUTPUT_NAMES):
        column = np.asarray(y[:, j])
        negative = 100.0 * float((column < 0).mean())
        print(
            f"  y[{j}] {name:<12} {column.min():>12.4g} {column.max():>12.4g}"
            f"   ({negative:.0f}% negative)"
        )

    # Advisory only: see transforms.suggest_power.
    from transforms import suggest_power

    powers = [suggest_power(np.asarray(y[:, j])) for j in range(y.shape[1])]
    print("\nsuggested signed-power exponents for --output-powers:")
    print("  " + ",".join(f"{p:.4f}" for p in powers))


if __name__ == "__main__":
    main()
