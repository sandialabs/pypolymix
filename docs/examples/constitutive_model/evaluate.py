"""Evaluate a trained constitutive surrogate.

```bash
python evaluate.py --run-dir runs/demo
```

Two very different questions are asked, and they do not have the same answer.

**One-step accuracy** compares ``f(x)`` against the true ``dq/dt`` on held-out
rows. It is cheap, it is what the training loss optimizes, and it is *not*
sufficient. A surrogate can have excellent one-step error and still be useless.

**Rollout accuracy** integrates ``dq/dt = f(q, u)`` forward from the initial state
of a held-out trajectory and compares against the true trajectory. Errors compound
and the model is asked about states it partly generated itself, so this is the
test that reflects how a constitutive model is actually used. Integration is done
in log-state, ``d log q / dt = f(q, u) / q``, which keeps positive states positive
and is far better conditioned across the many decades these variables cover.

Artifacts written to ``<run_dir>/``:

| file                 | contents                                              |
| -------------------- | ----------------------------------------------------- |
| `one_step.csv`       | per-output RMSE / NRMSE, raw and transformed units    |
| `rollout.csv`        | per-case, per-output relative trajectory error        |
| `figures/parity.png` | predicted vs true rate, per output                    |
| `figures/gating.png` | mixture weights across the input space                |
| `figures/rollout.png`| integrated vs true trajectories                       |
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from config import Config
from data import DerivativeDataset, load_rollout_cases
from model import build_moe, gating_weights, predict
from transforms import TransformPair


class Surrogate:
    """A trained model plus its transforms, exposing the raw-units right-hand side."""

    def __init__(self, run_dir: Path, checkpoint: str = "best", device: str = "auto"):
        path = run_dir / f"{checkpoint}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; train first with train.py")
        state = torch.load(path, map_location="cpu", weights_only=False)

        self.cfg = Config.from_dict(state["config"])
        self.meta = state.get("meta", {})
        self.epoch = int(state.get("epoch", -1))
        self.best_val = float(state.get("best_val", float("nan")))
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        self.model = build_moe(self.cfg)
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()
        self.transforms = TransformPair.load(run_dir / "transforms.json").to(self.device)

    @torch.no_grad()
    def rhs(self, x_raw: np.ndarray) -> np.ndarray:
        """Return ``dq/dt`` in raw physical units for raw inputs ``(batch, n_in)``."""
        x = torch.as_tensor(np.atleast_2d(x_raw), dtype=torch.float32, device=self.device)
        z_pred = predict(self.model, self.transforms.input(x))
        return self.transforms.output.inverse(x, z_pred).cpu().numpy()

    @torch.no_grad()
    def gates(self, x_raw: np.ndarray) -> np.ndarray:
        """Return mixture weights ``(batch, num_experts)`` for raw inputs."""
        x = torch.as_tensor(np.atleast_2d(x_raw), dtype=torch.float32, device=self.device)
        return gating_weights(self.model, self.transforms.input(x)).cpu().numpy()


# --------------------------------------------------------------------------- #
# One-step
# --------------------------------------------------------------------------- #


def one_step_metrics(surrogate: Surrogate, data_dir: str, split: str, batch_size: int) -> list:
    """Per-output errors on a held-out split, in raw and transformed units."""
    dataset = DerivativeDataset(data_dir, split)
    names = surrogate.meta.get("output_names") or [
        f"y{j}" for j in range(surrogate.cfg.num_outputs)
    ]

    n_out = surrogate.cfg.num_outputs
    sse_raw = np.zeros(n_out)
    sst_raw = np.zeros(n_out)
    sse_z = np.zeros(n_out)
    sst_z = np.zeros(n_out)
    count = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = np.arange(start, min(start + batch_size, len(dataset)))
            batch = dataset[rows]
            x = batch["X"].to(surrogate.device)
            y = batch["Y"].to(surrogate.device)

            z_target = surrogate.transforms.output(x, y)
            z_pred = predict(surrogate.model, surrogate.transforms.input(x))
            y_pred = surrogate.transforms.output.inverse(x, z_pred)

            sse_raw += ((y_pred - y) ** 2).sum(0).double().cpu().numpy()
            sst_raw += (y**2).sum(0).double().cpu().numpy()
            sse_z += ((z_pred - z_target) ** 2).sum(0).double().cpu().numpy()
            sst_z += (z_target**2).sum(0).double().cpu().numpy()
            count += len(rows)

    rows_out = []
    for j in range(n_out):
        rows_out.append(
            {
                "output": names[j],
                "n_rows": count,
                "raw_rmse": float(np.sqrt(sse_raw[j] / count)),
                "raw_nrmse": float(np.sqrt(sse_raw[j] / max(sst_raw[j], 1e-300))),
                "transformed_rmse": float(np.sqrt(sse_z[j] / count)),
                "transformed_nrmse": float(np.sqrt(sse_z[j] / max(sst_z[j], 1e-300))),
            }
        )
    return rows_out


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #


def rollout_case(surrogate: Surrogate, case: dict, state_indices, rtol=1e-6, atol=1e-9):
    """Integrate the surrogate over one held-out trajectory.

    Returns the predicted state array ``(T, n_state)`` or ``None`` if the solver
    failed. Controls are held at their recorded values; only the state is
    integrated.
    """
    from scipy.integrate import solve_ivp

    t = case["t"]
    x_true = case["X"]
    state_indices = list(state_indices)
    control_indices = [c for c in range(x_true.shape[1]) if c not in state_indices]
    controls = x_true[0, control_indices]

    template = np.zeros((1, x_true.shape[1]))
    template[0, control_indices] = controls

    def log_rhs(_t, log_q):
        q = np.exp(np.clip(log_q, -700.0, 700.0))
        template[0, state_indices] = q
        return surrogate.rhs(template)[0] / q

    try:
        solution = solve_ivp(
            log_rhs,
            (t[0], t[-1]),
            np.log(x_true[0, state_indices]),
            method="LSODA",
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )
    except Exception:  # noqa: BLE001 - a diverging surrogate can raise from LSODA
        return None
    if not solution.success or solution.y.shape[1] != len(t):
        return None
    return np.exp(solution.y).T


def rollout_metrics(surrogate: Surrogate, cases: list, state_indices) -> tuple:
    """Integrate every case and score it in log10-state, which is the honest metric
    for variables spanning many decades."""
    names = surrogate.meta.get("input_names") or [f"x{j}" for j in range(surrogate.cfg.num_inputs)]
    rows, trajectories = [], []
    for case in cases:
        predicted = rollout_case(surrogate, case, state_indices)
        truth = case["X"][:, list(state_indices)]
        record = {"case": case["name"], "n_steps": len(case["t"])}
        if predicted is None:
            record["status"] = "failed"
            for j, s in enumerate(state_indices):
                record[f"log10_rmse_{names[s]}"] = float("nan")
        else:
            record["status"] = "ok"
            for j, s in enumerate(state_indices):
                error = np.log10(np.maximum(predicted[:, j], 1e-300)) - np.log10(
                    np.maximum(truth[:, j], 1e-300)
                )
                record[f"log10_rmse_{names[s]}"] = float(np.sqrt(np.mean(error**2)))
        rows.append(record)
        trajectories.append(predicted)
    return rows, trajectories


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _sample_rows(data_dir, split, seed, max_points):
    dataset = DerivativeDataset(data_dir, split)
    count = min(max_points, len(dataset))
    rows = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    return dataset[rows]["X"].numpy(), dataset[rows]["Y"].numpy()


def figure_parity(surrogate, data_dir, split, out_path, max_points=20000) -> None:
    """Predicted against true rate, on symmetric-log axes because rates change sign."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    x, truth = _sample_rows(data_dir, split, 0, max_points)
    pred = surrogate.rhs(x)
    names = surrogate.meta.get("output_names") or [f"y{j}" for j in range(truth.shape[1])]

    n = truth.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0))
    for j, ax in enumerate(np.atleast_1d(axes)):
        magnitude = np.abs(truth[:, j])
        linthresh = max(float(np.percentile(magnitude[magnitude > 0], 25)), 1e-300)
        ax.scatter(truth[:, j], pred[:, j], s=1, alpha=0.15, color="C0", linewidth=0)
        lo = min(truth[:, j].min(), pred[:, j].min())
        hi = max(truth[:, j].max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.5, zorder=99)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("symlog", linthresh=linthresh)
        # symlog puts a tick on every decade of both branches; thin them out or the
        # labels collide around the linear region.
        for axis in (ax.xaxis, ax.yaxis):
            locator = ticker.SymmetricalLogLocator(base=10.0, linthresh=linthresh)
            locator.set_params(numticks=4)
            axis.set_major_locator(locator)
            axis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_xlabel(f"true {names[j]}")
        ax.set_ylabel(f"predicted {names[j]}")
    fig.suptitle(f"one-step parity ({split} split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def figure_gating(surrogate, data_dir, split, out_path, max_points=20000) -> None:
    """Show how the gate partitions the input space -- the payoff of the mixture.

    Plotted against a *state* variable rather than a control: controls are constant
    along a trajectory, so a control axis would only show one point per case.
    """
    import matplotlib.pyplot as plt

    x, _ = _sample_rows(data_dir, split, 1, max_points)
    gates = surrogate.gates(x)
    names = surrogate.meta.get("input_names") or [f"x{j}" for j in range(x.shape[1])]
    state_indices = surrogate.cfg.state_column_indices()
    controls = [c for c in range(x.shape[1]) if c not in state_indices]

    a = state_indices[0]
    b = controls[0] if controls else state_indices[-1]
    dominant = gates.argmax(axis=1)
    num_experts = gates.shape[1]
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for e in range(num_experts):
        mask = dominant == e
        if mask.any():
            axes[0].scatter(
                x[mask, a], x[mask, b], s=2, color=colors(e), linewidth=0, label=f"expert {e}"
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(names[a])
    axes[0].set_ylabel(names[b])
    axes[0].set_title("dominant expert")
    axes[0].legend(markerscale=6, fontsize=8, loc="best")

    for e in range(num_experts):
        axes[1].scatter(x[:, a], gates[:, e], s=2, alpha=0.25, color=colors(e), linewidth=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(names[a])
    axes[1].set_ylabel("gating weight")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("mixture weights")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def figure_rollout(surrogate, cases, trajectories, state_indices, out_path, max_cases=6) -> None:
    import matplotlib.pyplot as plt

    names = surrogate.meta.get("input_names") or [f"x{j}" for j in range(surrogate.cfg.num_inputs)]
    shown = [i for i, tr in enumerate(trajectories) if tr is not None][:max_cases]
    if not shown:
        return
    n_state = len(state_indices)

    fig, axes = plt.subplots(
        n_state, len(shown), figsize=(2.6 * len(shown), 2.4 * n_state), squeeze=False, sharex="col"
    )
    for col, i in enumerate(shown):
        t = cases[i]["t"]
        truth = cases[i]["X"][:, list(state_indices)]
        for row in range(n_state):
            ax = axes[row][col]
            ax.plot(t, truth[:, row], color="black", linewidth=1.5, label="truth")
            ax.plot(
                t, trajectories[i][:, row], color="red", linewidth=1.5, ls="--", label="surrogate"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(names[state_indices[row]])
            if row == n_state - 1:
                ax.set_xlabel("time [s]")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
        axes[0][col].set_title(cases[i]["name"], fontsize=8)
    fig.suptitle("trajectory rollout: integrated surrogate vs truth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="runs/demo")
    parser.add_argument("--data-dir", default="", help="defaults to the value used for training")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--split", default="test", help="split used for one-step metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--rollout-cases", type=int, default=16)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    surrogate = Surrogate(run_dir, args.checkpoint, args.device)
    data_dir = args.data_dir or surrogate.cfg.data_dir
    state_indices = surrogate.cfg.state_column_indices()

    print(f"checkpoint: {run_dir / (args.checkpoint + '.pt')} (epoch {surrogate.epoch})")
    print(f"data:       {data_dir}")

    split = args.split
    if not (Path(data_dir) / split).exists():
        split = "val"
    metrics = one_step_metrics(surrogate, data_dir, split, args.batch_size)
    write_csv(run_dir / "one_step.csv", metrics)
    print(f"\none-step errors on the {split} split ({metrics[0]['n_rows']:,d} rows):")
    print(f"  {'output':<12} {'raw NRMSE':>12} {'transformed NRMSE':>19}")
    for row in metrics:
        print(f"  {row['output']:<12} {row['raw_nrmse']:>12.4g} {row['transformed_nrmse']:>19.4g}")

    cases = load_rollout_cases(data_dir, limit=args.rollout_cases)
    trajectories = []
    if cases:
        rows, trajectories = rollout_metrics(surrogate, cases, state_indices)
        write_csv(run_dir / "rollout.csv", rows)
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"\nrollout over {len(rows)} cases ({len(ok)} integrated successfully):")
        keys = [k for k in rows[0] if k.startswith("log10_rmse_")]
        for key in keys:
            values = np.array([r[key] for r in ok], dtype=float)
            label = key.removeprefix("log10_rmse_")
            print(f"  {label:<12} median log10 RMSE {np.median(values):.4f}")
    else:
        print("\nno rollout cases found; skipping trajectory evaluation")

    if not args.no_figures:
        figures = run_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        figure_parity(surrogate, data_dir, split, figures / "parity.png")
        figure_gating(surrogate, data_dir, split, figures / "gating.png")
        if cases:
            figure_rollout(surrogate, cases, trajectories, state_indices, figures / "rollout.png")
        print(f"\nwrote figures to {figures}")


if __name__ == "__main__":
    main()
