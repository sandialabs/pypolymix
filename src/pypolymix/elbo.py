import torch


def elbo_loss(
    nll: torch.Tensor, kl: torch.Tensor, n_train: int, temperature: float = 1.0
) -> torch.Tensor:
    """Combine a mean-per-datapoint data NLL and a KL term into an ELBO loss.

    ``loss = nll + temperature * kl / n_train``

    ``n_train`` is the FULL training-set size (not the minibatch size) and is
    REQUIRED -- there is no default and no "full-batch means skip this"
    shortcut. This is what makes a per-datapoint-mean data term (as produced
    by ``Likelihood.nll()``, see its docstring) combine correctly with a KL
    that is computed once over the whole parameter space, for ANY batch size,
    including a batch size equal to ``n_train`` (full-batch training still
    needs this division; see Blundell, Cornebise, Kavukcuoglu, Wierstra
    (2015), "Weight Uncertainty in Neural Networks," arXiv:1505.05424 (ICML
    2015), for the general minibatch-KL-scaling derivation this is a special
    case of).

    ``temperature`` defaults to 1.0, the only value consistent with a proper
    ELBO / variational free energy. A different temperature is a deliberate
    "cold/warm posterior" choice (see Wenzel et al. (2020), "How Good is the
    Bayes Posterior in Deep Neural Networks Really?," arXiv:2002.02405) --
    document and justify any non-default value explicitly wherever you set
    it; do not use this as a free hyperparameter to tune away a bad fit (that
    is the "weight_factor" anti-pattern this helper replaces; see this
    project's handoff notes PM-09..PM-12 for the production bug it caused).

    Args:
        nll: Mean-per-datapoint negative log-likelihood (e.g. from
            ``Likelihood.nll()``), summed over output dimensions and averaged
            over the batch (and sample) dimension(s). Scalar.
        kl: KL divergence summed once over the whole parameter space (e.g.
            ``StochasticModel.distribution_loss()``). Scalar.
        n_train: Full training-set row count. Must be a positive int.
        temperature: Scales the KL term; see docstring above. Defaults to 1.0.

    Returns:
        Scalar ELBO loss, to be minimized.

    Example:
        ```python
        >>> import torch
        >>> from pypolymix.elbo import elbo_loss
        >>> elbo_loss(torch.tensor(1.0), torch.tensor(2.0), n_train=4)
        tensor(1.5000)
        ```
    """
    if n_train is None or n_train <= 0:
        raise ValueError(
            f"n_train must be a positive int (the full training-set size), got {n_train!r}"
        )
    return nll + temperature * kl / float(n_train)
