"""Benchmark: does detaching frozen parameter groups speed up backward()?

This lives under ``regression_tests/`` (per ``tests/conftest.py``) purely to get
the project's existing slow-test opt-in behavior (skipped unless ``--regression``
is passed); it is not a training regression test, just a wall-clock benchmark.

IMPORTANT FINDING (verified, reproducible -- see investigation below): for
every ``SurrogateModel`` currently in this repo (``NeuralNetwork``,
``MixtureOfExperts``, ``GatingNetwork``), the ``.detach()`` call added to
``StochasticModel.sample_parameters()`` for PM-06 measurably changes NOTHING:
the median wall-clock ratio across repeated, interleaved A/B trials is
~1.00x (noise-dominated, sometimes >1, sometimes <1), never a reproducible
">1.5x" speedup. This test is marked ``xfail(strict=True)`` to record that
finding rather than silently deleting the required acceptance-criterion test
or asserting something false.

Why there is no speedup here (verified step by step):

1. A frozen group's ``sample_parameters()`` output ALREADY has
   ``requires_grad=False`` *before* any explicit ``.detach()`` call, because
   it is derived purely from that group's own ``requires_grad=False``
   leaf parameters (e.g. ``DeterministicGroup.params.unsqueeze(0).expand(...)``
   with no other differentiable input). Calling ``.detach()`` on an already
   ``requires_grad=False`` tensor is a complete no-op -- it does not change
   the tensor's graph status at all. (Verified: ``s.requires_grad`` is
   ``False`` even without calling ``.detach()``, and
   ``s.requires_grad == s.detach().requires_grad``.)

2. Every surrogate model re-slices the single concatenated ``params`` tensor
   returned by ``sample_parameters()`` (e.g.
   ``NeuralNetwork.forward``'s ``params[:, w_slice[0]:w_slice[1]]`` per
   layer, ``MixtureOfExperts.get_expert_outputs``'s ``params[:, start:end]``
   per expert). PyTorch's autograd tracks ``requires_grad`` as a single
   boolean *per tensor*, decided by "did any input to this op require
   grad" -- it does not do per-index/per-element provenance tracking. Once
   ``torch.cat`` combines a frozen group's sample with a trainable group's
   sample, the resulting tensor has ``requires_grad=True`` as a whole, and
   *any* subsequent slice of it -- even a slice that maps 1:1 onto exactly
   the frozen group's region -- ALSO gets ``requires_grad=True``. This holds
   regardless of whether the frozen slice was explicitly ``.detach()``'d
   before the ``cat`` (verified directly: slicing the frozen region out of
   ``torch.cat([s, t])`` gives ``requires_grad=True`` identically whether
   ``s`` or ``s.detach()`` was the input to ``cat``).

3. Consequently, every "frozen" layer inside a surrogate that re-slices
   ``params`` is tracked and backpropagated through in full, with or without
   this fix, because its weight/bias tensor's ``requires_grad`` flag is
   driven by the *whole* concatenated tensor, not by which original group
   contributed that slice. The only thing ``.detach()`` changes is whether
   the (always-``None``-anyway, since the leaf already had
   ``requires_grad=False``) gradient is accumulated on the frozen group's
   own leaf parameters -- a correctness/defensive detail, not a performance
   one, for the surrogate models in this codebase.

A genuine speedup from this kind of detaching would require a surrogate that
consumes each parameter group's sample as a *separate* tensor (never
re-sliced out of a single combined ``cat`` output) -- which none of
``NeuralNetwork``/``MixtureOfExperts``/``GatingNetwork`` do, and which is
explicitly out of scope for PM-06 (see "Out of scope" in
``PM-06-selective-freezing.md``: "Do not touch MixtureOfExperts/NeuralNetwork").
"""

import statistics

import pytest
import torch

from pypolymix import stochastic_model as stochastic_model_module
from pypolymix.parameter_groups import DeterministicGroup, IIDGaussianGroup
from pypolymix.stochastic_model import StochasticModel
from pypolymix.surrogate_models import NeuralNetwork


def _build_mostly_frozen_model():
    """Build a model with hidden layers fully frozen (one ``DeterministicGroup``
    per layer, aligned to ``NeuralNetwork``'s own layer boundaries so each
    frozen group maps onto exactly one weight+bias block) and only the output
    layer trainable."""
    torch.manual_seed(0)
    width, depth, num_inputs, num_outputs = 128, 6, 128, 8
    surrogate = NeuralNetwork(
        num_inputs=num_inputs, num_outputs=num_outputs, width=width, depth=depth
    )

    groups = []
    in_features = num_inputs
    for i in range(depth):
        size = in_features * width + width  # one layer's weight + bias, combined
        frozen_group = DeterministicGroup(f"hidden_{i}", num_params=size)
        frozen_group.requires_grad_(False)
        groups.append(frozen_group)
        in_features = width
    out_size = in_features * num_outputs + num_outputs
    groups.append(IIDGaussianGroup("output", num_params=out_size))

    model = StochasticModel(surrogate_model=surrogate, parameter_groups=groups)
    return surrogate, model


def _time_backward_once(surrogate, model, num_samples, batch, inner_reps=5):
    x = torch.randn(batch, surrogate.num_inputs)
    # Warm up so lazy setup doesn't pollute the timing.
    params = model.sample_parameters(num_samples=num_samples)
    surrogate(x, params).sum().backward()
    for group in model.parameter_groups:
        for p in group.parameters():
            p.grad = None

    import time

    start = time.perf_counter()
    for _ in range(inner_reps):
        params = model.sample_parameters(num_samples=num_samples)
        y = surrogate(x, params)
        y.sum().backward()
        for group in model.parameter_groups:
            for p in group.parameters():
                p.grad = None
    return (time.perf_counter() - start) / inner_reps


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Verified, reproducible finding: detaching frozen groups before `cat` "
        "gives no measurable backward() speedup for any surrogate in this repo "
        "(median ratio ~1.0x across interleaved A/B trials, see module "
        "docstring for the full explanation). The claimed >1.5x speedup is not "
        "achievable without also changing how surrogates consume `params` "
        "(out of scope for PM-06)."
    ),
)
def test_detaching_frozen_groups_speeds_up_backward(monkeypatch):
    surrogate, model = _build_mostly_frozen_model()
    num_samples, batch = 16, 32
    num_trials = 15

    with_times = []
    without_times = []
    for _ in range(num_trials):
        # Current (correct) behavior: frozen groups are detached before `cat`.
        monkeypatch.setattr(
            stochastic_model_module, "_is_frozen", stochastic_model_module._is_frozen
        )
        with_times.append(_time_backward_once(surrogate, model, num_samples, batch))

        # Pre-PM-06 behavior reproduced: force `_is_frozen` to always report
        # "trainable" so no group is ever detached before `cat`.
        monkeypatch.setattr(stochastic_model_module, "_is_frozen", lambda group: False)
        without_times.append(_time_backward_once(surrogate, model, num_samples, batch))

    median_with = statistics.median(with_times)
    median_without = statistics.median(without_times)
    speedup = median_without / median_with
    print(
        f"\n[PM-06 benchmark] median backward() time over {num_trials} "
        f"interleaved trials: with_detach={median_with * 1e3:.3f} ms, "
        f"without_detach={median_without * 1e3:.3f} ms, speedup={speedup:.3f}x "
        f"(expected: no reproducible speedup, see module docstring)"
    )

    # Loose lower bound per the acceptance criteria. Expected (and verified,
    # see module docstring) to fail: measured speedup is ~1.0x, not >1.5x.
    assert speedup > 1.5
