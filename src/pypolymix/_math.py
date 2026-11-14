"""Small numerical helpers shared across the package (not part of the public API)."""

from typing import Union

import torch


def inverse_softplus(y: Union[float, torch.Tensor]) -> torch.Tensor:
    """Numerically stable inverse of ``softplus(x) = log(1 + exp(x))`` for ``y > 0``.

    The textbook form ``log(expm1(y))`` overflows once ``expm1(y)`` exceeds the
    dtype's range (``y > ~88.7`` in float32, ``y > ~709.8`` in float64), returning
    ``inf`` and silently corrupting any parameter initialized from it. The
    algebraically identical ``y + log(-expm1(-y))`` stays finite for every
    ``y > 0`` and keeps full precision for small ``y``.

    Example:
        ```python
        >>> import torch
        >>> import torch.nn.functional as F
        >>> y = torch.tensor([1e-6, 1.0, 1000.0])
        >>> torch.allclose(F.softplus(inverse_softplus(y)), y)
        True
        ```
    """
    y = torch.as_tensor(y, dtype=y.dtype if isinstance(y, torch.Tensor) else torch.get_default_dtype())
    return y + torch.log(-torch.expm1(-y))
