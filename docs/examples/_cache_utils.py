"""Lightweight on-disk caching for expensive notebook cells.

Used by the ``toy_*`` example notebooks (``toy_sine_regression_families.ipynb``,
``toy_moe_multimodal.ipynb``) to avoid re-running multi-hour training loops /
sampling chains every time the notebook is re-executed (e.g. after a
markdown-only edit or a plot-styling tweak). This is notebook-authoring
infrastructure, not part of the ``pypolymix`` package itself.

**Only cache plain data** (tensors, and dicts/lists of tensors, numpy arrays,
and python scalars) -- never live ``nn.Module``/``ParameterGroup`` objects
directly; they may hold notebook-local Python closures, which standard
pickling (what ``torch.save`` uses under the hood) cannot serialize. The
robust pattern for *trained* families (the Gaussian families, and
``LangevinGroup``) is to cache ``group.state_dict()`` (plain tensors only)
plus the training ``history``; reconstruct a fresh ``group``/``model`` via
the same factory function used originally, then
``group.load_state_dict(...)`` to restore the fit instantly, skipping the
training loop.

Usage:
    ```python
    cache = CacheDir(".cache/toy_sine_regression_families", force=FORCE_RECOMPUTE)

    def _compute():
        group, model, history = train_a("iid")
        return {"state_dict": group.state_dict(), "history": history}

    result = cache.get("parta_iid_v1", _compute)
    ```

**Cache invalidation is manual, by design**: bump the version suffix in a
cache key (e.g. ``"parta_iid_v1"`` -> ``"parta_iid_v2"``) whenever you change
code in a way that would change that key's result. There is no automatic
hashing of source code or config -- for a paper-facing notebook, an explicit,
visible version bump is more trustworthy than a cache that silently decides
on its own whether to trust itself.
"""

from __future__ import annotations

import pathlib
import time
from typing import Callable, Optional, TypeVar

import torch

T = TypeVar("T")


class CacheDir:
    """A directory of ``torch.save``'d cache entries, one file per key."""

    def __init__(self, directory, force: bool = False, verbose: bool = True):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.verbose = verbose

    def _path(self, key: str) -> pathlib.Path:
        safe_key = key.replace("/", "_")
        return self.directory / f"{safe_key}.pt"

    def get(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Return the cached value for ``key`` if present (and ``force`` is
        not set), otherwise call ``compute_fn()``, cache the result, and
        return it. ``compute_fn`` must return plain data only -- see the
        module docstring."""
        path = self._path(key)
        if not self.force and path.exists():
            if self.verbose:
                print(f"[cache hit]  {key}  <-  {path.name}")
            return torch.load(path, weights_only=False)
        t0 = time.time()
        result = compute_fn()
        torch.save(result, path)
        if self.verbose:
            elapsed = time.time() - t0
            print(f"[cache miss] {key}  ->  {path.name}  (computed in {elapsed:.1f}s)")
        return result

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def clear(self, key: Optional[str] = None) -> None:
        """Delete one cached entry by key, or (if ``key`` is ``None``) every
        entry in this cache directory. Use to force recomputation of
        specific results without blowing away the whole cache."""
        if key is None:
            for p in self.directory.glob("*.pt"):
                p.unlink()
        else:
            self._path(key).unlink(missing_ok=True)
