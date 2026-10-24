"""Direct numerical parity against a downstream production codebase's
``_objective`` formula.

``GaussianLikelihood.nll`` and ``StudentTLikelihood.nll`` must match
``creep_surrogate/train/lightning.py``'s ``_objective`` method (lines
~205-212 at the time of writing) bit-for-bit (to float32 precision), since
PM-07's reduction convention was deliberately designed to be a drop-in
replacement for that already-correct implementation (see PM-07's handoff
notes and CS-01's later migration). The reference formulas below are a
verbatim transcription of that method -- do not "simplify" them, the point of
this test is to catch any accidental reformulation.
"""

import math

import torch
import torch.nn.functional as F
from torch.distributions import StudentT

from pypolymix.likelihoods import GaussianLikelihood, StudentTLikelihood

LOG_2PI = math.log(2.0 * math.pi)  # lightning.py:19
DF_FLOOR = 2.0  # lightning.py:26


def _lightning_gaussian_nll(residual: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Verbatim transcription of lightning.py:211-212 (Gaussian branch)."""
    nll = 0.5 * (residual**2 * torch.exp(-2.0 * log_sigma)).sum(-1).mean()
    nll = nll + 0.5 * (LOG_2PI + 2.0 * log_sigma).sum()
    return nll


def _lightning_student_t_nll(
    residual: torch.Tensor, log_sigma: torch.Tensor, raw_df: torch.Tensor
) -> torch.Tensor:
    """Verbatim transcription of lightning.py:207-209 (Student-t branch),
    using lightning.py:29-35's ``student_t_df_from_raw``."""
    scale = torch.exp(log_sigma)
    df = DF_FLOOR + F.softplus(raw_df)  # student_t_df_from_raw
    return -StudentT(df, torch.zeros_like(scale), scale).log_prob(residual).sum(-1).mean()


def test_gaussian_likelihood_nll_matches_lightning_objective_formula():
    torch.manual_seed(0)
    num_outputs = 3
    num_samples, batch = 5, 7
    log_sigma = torch.tensor([-0.5, 0.0, 0.3], dtype=torch.float32)
    pred = torch.randn(num_samples, batch, num_outputs, dtype=torch.float32)
    target = torch.randn(batch, num_outputs, dtype=torch.float32)
    residual = pred - target.unsqueeze(0).expand_as(pred)

    expected = _lightning_gaussian_nll(residual, log_sigma)

    likelihood = GaussianLikelihood(num_outputs=num_outputs)
    with torch.no_grad():
        likelihood.log_sigma.copy_(log_sigma)
    actual = likelihood.nll(pred, target)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_student_t_likelihood_nll_matches_lightning_objective_formula():
    torch.manual_seed(2)
    num_outputs = 3
    num_samples, batch = 5, 7
    log_sigma = torch.tensor([-0.2, 0.1, 0.4], dtype=torch.float32)
    raw_df = torch.tensor([0.3, -0.6, 1.5], dtype=torch.float32)
    pred = torch.randn(num_samples, batch, num_outputs, dtype=torch.float32)
    target = torch.randn(batch, num_outputs, dtype=torch.float32)
    residual = pred - target.unsqueeze(0).expand_as(pred)

    expected = _lightning_student_t_nll(residual, log_sigma, raw_df)

    likelihood = StudentTLikelihood(num_outputs=num_outputs, df_floor=DF_FLOOR)
    with torch.no_grad():
        likelihood.log_scale.copy_(log_sigma)
        likelihood.raw_df.copy_(raw_df)
    actual = likelihood.nll(pred, target)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
