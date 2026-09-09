"""Train the deterministic mixture-of-experts constitutive surrogate.

```bash
python data.py --out toy_data          # once, if you have no data of your own
python train.py --data-dir toy_data --run-dir runs/demo
```

The loss is a plain mean-squared error in *transformed* space,

    L = mean( ( f(T_x(X)) - T_y(X, Y) )^2 )

which is the entire objective for the deterministic case. There is no
regularization term: with ``DeterministicGroup`` the parameter groups contribute
nothing to the loss, and weight decay is handled by AdamW.

Outputs written to ``--run-dir``:

| file              | contents                                                  |
| ----------------- | --------------------------------------------------------- |
| `transforms.json` | the fitted transforms -- part of the model, see transforms.py |
| `last.pt`         | most recent epoch; training resumes from this file        |
| `best.pt`         | lowest validation loss seen                               |
| `metrics.csv`     | one row per epoch                                         |

Re-running the same command resumes from ``last.pt``. To start over, delete the
run directory.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import config as config_module
import numpy as np
import torch
import torch.nn.functional as F
from data import DerivativeDataset, load_meta, make_loader
from model import build_moe, parameter_summary, predict
from transforms import TransformPair, suggested_pair


def build_transforms(cfg, dataset: DerivativeDataset, run_dir: Path) -> TransformPair:
    """Load ``transforms.json`` if it exists, otherwise fit it and write it out.

    Refitting an existing file is deliberately avoided: the checkpoints in this
    directory were trained against the stored normalization.
    """
    path = run_dir / "transforms.json"
    if path.exists():
        print(f"transforms: loaded {path}")
        return TransformPair.load(path)

    n = len(dataset)
    if n > cfg.fit_max_rows:
        rows = np.sort(np.random.default_rng(cfg.seed).choice(n, cfg.fit_max_rows, replace=False))
        sample = dataset[rows]
        x, y = sample["X"], sample["Y"]
    else:
        x, y = dataset.full()

    pair = suggested_pair(
        state_indices=cfg.state_column_indices(),
        log_columns=cfg.log_input_indices(),
        log_rate_outputs=cfg.log_rate_indices(),
        powers=cfg.powers(),
    )
    pair.fit(x, y)
    pair.save(path)
    print(f"transforms: fitted on {len(x):,d} rows -> {path}")
    return pair


def loss_on_batch(model, transforms: TransformPair, batch, device) -> torch.Tensor:
    """Mean-squared error between prediction and target, both in transformed space."""
    x_raw = batch["X"].to(device, non_blocking=True)
    y_raw = batch["Y"].to(device, non_blocking=True)
    z_in = transforms.input(x_raw)
    z_target = transforms.output(x_raw, y_raw)
    return F.mse_loss(predict(model, z_in), z_target)


@torch.no_grad()
def evaluate_split(model, transforms, loader, device) -> float:
    """Row-weighted mean loss over a whole split."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        rows = batch["X"].shape[0]
        total += float(loss_on_batch(model, transforms, batch, device)) * rows
        count += rows
    return total / max(count, 1)


def main(argv=None) -> None:
    cfg = config_module.from_args(argv, description=__doc__.split("\n")[0])
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.resolved_device()

    # ---------------------------------------------------------------- data
    meta = load_meta(cfg.data_dir)
    if len(meta["input_names"]) != cfg.num_inputs:
        raise ValueError(
            f"meta.json declares {len(meta['input_names'])} inputs "
            f"but --num-inputs is {cfg.num_inputs}"
        )
    if len(meta["output_names"]) != cfg.num_outputs:
        raise ValueError(
            f"meta.json declares {len(meta['output_names'])} outputs "
            f"but --num-outputs is {cfg.num_outputs}"
        )
    if tuple(meta["state_indices"]) != cfg.state_column_indices():
        raise ValueError(
            f"meta.json declares state_indices={meta['state_indices']} "
            f"but --state-indices is {list(cfg.state_column_indices())}"
        )

    train_set = DerivativeDataset(cfg.data_dir, "train")
    val_set = DerivativeDataset(cfg.data_dir, "val")
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = make_loader(
        train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=generator
    )
    val_loader = make_loader(val_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"device: {device}")
    print(f"data:   {len(train_set):,d} train rows, {len(val_set):,d} val rows")

    # ------------------------------------------------ transforms and model
    transforms = build_transforms(cfg, train_set, run_dir).to(device)
    model = build_moe(cfg).to(device)
    print(f"model:  {parameter_summary(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.coswr_t0, T_mult=cfg.coswr_tmult, eta_min=cfg.coswr_eta_min
    )

    # ------------------------------------------------------------- resume
    start_epoch, best_val = 0, float("inf")
    last_path = run_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        print(f"resumed from {last_path} at epoch {start_epoch} (best val {best_val:.6f})")
        if start_epoch >= cfg.epochs:
            print("nothing to do; increase --epochs or delete the run directory")
            return

    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists()
    metrics_file = metrics_path.open("a", newline="")
    writer = csv.writer(metrics_file)
    if write_header:
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    # ------------------------------------------------------------- train
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
    )
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tic = time.time()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                loss = loss_on_batch(model, transforms, batch, device)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            rows = batch["X"].shape[0]
            total += float(loss.detach()) * rows
            count += rows
        scheduler.step()

        train_loss = total / max(count, 1)
        val_loss = evaluate_split(model, transforms, val_loader, device)
        elapsed = time.time() - tic
        lr = optimizer.param_groups[0]["lr"]
        writer.writerow([epoch, train_loss, val_loss, lr, elapsed])
        metrics_file.flush()

        checkpoint = {
            "config": cfg.to_dict(),
            "meta": meta,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
        }
        torch.save(checkpoint, last_path)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, run_dir / "best.pt")
            marker = "  *"
        if epoch % 5 == 0 or epoch == cfg.epochs - 1 or marker:
            print(
                f"epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}  "
                f"lr {lr:.2e}  {elapsed:5.1f}s{marker}"
            )

    metrics_file.close()
    print(f"\nbest validation loss {best_val:.6f}")
    print(f"wrote {run_dir / 'best.pt'}")
    print(f"next: python evaluate.py --run-dir {run_dir}")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
