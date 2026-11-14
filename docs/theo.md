# Variational Inference

Notations

* Observed data: \(\mathcal{D} =\{(x_i,y_i)\vert i=0,1,\ldots d-1\}\)
* Model parameters: \(\mathbf{c}=\{c_0,c_1,\ldots,c_{n-1}\}\)
* Likelihood & prior densities: \(p(\mathcal{D}\vert\boldsymbol{c})\) & \(p(\boldsymbol{c})\)
* Posterior & approximate posterior densities: \(p(\boldsymbol{c}\vert\mathcal{D})\) & \(q_{\boldsymbol{\theta}}(\boldsymbol{c})\)

In a Bayesian framework, the posterior distribution for the parameters \(\boldsymbol{c}\) of a model \(M\) can be estimated as 
$$
p(\boldsymbol{c}\vert\mathcal{D}) \propto p(\mathcal{D}\vert\boldsymbol{c}) p(\boldsymbol{c})
$$
There are no closed-form solutions \(p(\boldsymbol{c}\vert\mathcal{D})\) for generic models. Sampling the posterior distribution \(p(\boldsymbol{c}\vert\mathcal{D})\) is typically hampered by high-dimensionality of \(\boldsymbol{c}\) and the high computational cost to estimate the likelihood \(p(\mathcal{D}\vert\boldsymbol{c})\) for a large number of model evaluations for example in Markov chain Monte Carlo schemes.

To work around this challenge, instead of sampling or evaluating \(p(\boldsymbol{c}\vert\mathcal{D})\),  we will *approximate* it with a simpler distribution \(q_{\boldsymbol{\theta}}(\boldsymbol{c})\):
$$
p(\boldsymbol{c}\vert\mathcal{D}) \approx q_{\boldsymbol{\theta}}(\boldsymbol{c})
$$
where \(\boldsymbol{\theta}\) are *variational parameters* to be optimized by minimizing the distance between the "true" and approximate distributions.  The algorithm works as follows:

* Choose a tractable family of distributions \(\mathcal{Q} = \{q_\boldsymbol{\theta} : \boldsymbol{\theta} \in \Theta\}\)
* Use KL divergence as the measure of closeness, i.e.

$$
\text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c}) \parallel p(\boldsymbol{c}\vert\mathcal{D})) = \int_{\boldsymbol{c}} q_{\boldsymbol{\theta}}(\boldsymbol{c}) \log\frac{q_{\boldsymbol{\theta}}(\boldsymbol{c})}{p(\boldsymbol{c}\vert\mathcal{D}))}\,d\boldsymbol{c}
$$
to find \(\boldsymbol{\theta}^* \in \Theta\) such that \(q_{\boldsymbol{\theta}^*}(\boldsymbol{c})\) is closest to \(p(\boldsymbol{c}\vert\mathcal{D})\) among all possible choices for \(\boldsymbol{\theta}\).

The optimization algorithm uses the following **Evidence Lower Bound** (ELBO) expression for the loss function

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{q}[\log p(\mathcal{D}\vert\boldsymbol{c})] - \text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c})\parallel p(\boldsymbol{c})) + \text{const.}
$$

The current version of the library implements variational inference algorithms that assume Gaussian approximate posteriors, i.e. for \(\mathcal{Q}\), as well as a Gaussian likelihood. For the prior distribution \(p(\boldsymbol{c})\), both uniform and Gaussian distributions are implemented.

For the case where both the approximate posterior and the prior are multivariate Gaussian distributions

$$
q_{\boldsymbol{\theta}}(\boldsymbol{c})=\mathcal{N}(\boldsymbol{c} \mid \boldsymbol{\mu}_\boldsymbol{\theta},\boldsymbol{\Sigma}_{\boldsymbol{\theta}}),\,\,
p(\boldsymbol{c}) = \mathcal{N}(\boldsymbol{c} \mid \boldsymbol{\mu}_{p}, \boldsymbol{\Sigma}_{p}),
$$

the KL divergence can be computed analytically as

$$
\text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c}) \parallel p(\boldsymbol{c})) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}_{\boldsymbol{\theta}}) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_{\boldsymbol{\theta}})^\top\boldsymbol{\Sigma}_p^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_{\boldsymbol{\theta}}) - n + \log\frac{|\boldsymbol{\Sigma}_p|}{|\boldsymbol{\Sigma}_{\boldsymbol{\theta}}|}\right]
$$

If the prior is uniform, the above expression reduces to

$$
\text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c}) \parallel p(\boldsymbol{c}))=- \frac{1}{2}\log|\boldsymbol{\Sigma}_{\boldsymbol{\theta}}|+\text{const.}
$$

This reduction is the classical *improper, unbounded* flat-prior limit: it drops
the prior-dependent term as an additive "const" without evaluating it, which is
only valid because that term does not depend on \(\boldsymbol{\theta}\) and is
never optimized -- it is a standard informal device for an idealized prior with
no finite normalization, not a statement about any particular *bounded*
distribution. `pypolymix.priors.UniformPrior` is a literal bounded
\(\mathcal{U}(\text{low}, \text{high})\) (`torch.distributions.Uniform`), a
different mathematical object from this improper limit: the true analytic KL
from an unbounded \(q\) (e.g. any Gaussian-family group) to a bounded
\(\mathcal{U}(\text{low}, \text{high})\) is genuinely \(+\infty\), because a
Gaussian has nonzero density everywhere outside \([\text{low}, \text{high}]\).
`torch.distributions` registers exactly this analytic result and
`ParameterGroup.distribution_loss()` returns it unmodified (non-differentiable)
for this pairing today -- see `UniformPrior`'s docstring and
`tests/priors/test_Uniform.py` for the verified behavior, and the
"KL/entropy estimators for implicit samplers" section below for the general
difficulty of mismatched-support prior/posterior pairs.

The first term on the right-hand side of the ELBO function

$$
\mathbb{E}_{q}[\log p(\mathcal{D}\vert\boldsymbol{c})]\approx 
\frac{1}{N_s}\sum_{k=1}^{N_s}\log p(\mathcal{D}\vert\boldsymbol{c}_k)
$$

with samples \(\boldsymbol{c}_k\) drawn from \(\mathcal{N}(\mu_\boldsymbol{\theta},\Sigma_\boldsymbol{\theta})\). Using the reparametrization trick, 

$$
\boldsymbol{c} = \mu_\boldsymbol{\theta} + L\boldsymbol{\epsilon}
$$

where \(\boldsymbol{\epsilon}\propto\mathcal{N}(\boldsymbol{0},\mathbb{I})\) and \(L\) is the lower-triangular matrix resulted from the Cholesky decomposition of \(\Sigma_\boldsymbol{\theta}\), i.e. \(\Sigma_\boldsymbol{\theta}=L L^T\). The expected log-likelihood expression above becomes

$$
\mathbb{E}_{q}[\log p(\mathcal{D}\vert\boldsymbol{c})]=\mathbb{E}_{\boldsymbol{\epsilon}}[\log p(\mathcal{D}\vert\mu_\boldsymbol{\theta} + L\boldsymbol{\epsilon})]\approx 
\frac{1}{N_s}\sum_{k=1}^{N_s}\log p(\mathcal{D}\vert\mu_\boldsymbol{\theta} + L\boldsymbol{\epsilon}_k).
$$

Gradients with respect to \(\boldsymbol{\theta}\) components \(\mu_{\boldsymbol{\theta}}\) and \(\Sigma_\boldsymbol{\theta}\), are estimated through automatic differentiation tools if the model that depends of parameters \(\boldsymbol{c}\) is differentiable.

## KL scaling is not a free hyperparameter

The ELBO's KL coefficient is exactly 1:
$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{q}[\log p(\mathcal{D}\vert\boldsymbol{c})] - \text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c})\parallel p(\boldsymbol{c})),
$$
full stop. There is no free "KL weight" to tune in a properly specified variational objective.

**Worked cautionary example.** Suppose the data term is a Gaussian negative log-likelihood with (known, fixed) noise scale \(\sigma_{\text{true}}\),
$$
\text{NLL}(\sigma) = \frac{S(\boldsymbol{c})}{2\sigma^2} + \frac{N}{2}\log(2\pi\sigma^2), \qquad S(\boldsymbol{c}) = \sum_{i=1}^{N}(y_i - \hat{y}_i(\boldsymbol{c}))^2,
$$
and that, instead of the proper ELBO, someone minimizes
$$
\mathcal{L}'(\boldsymbol{\theta}) = \text{NLL}(\sigma_{\text{true}}) + \alpha\,\text{KL}(q\parallel p), \qquad 0 < \alpha < 1,
$$
rescaling the KL term by an arbitrary constant \(\alpha\). Multiplying \(\mathcal{L}'\) by \(1/\alpha\) does not change its minimizer (scaling a loss by a positive constant never changes where it is minimized), so minimizing \(\mathcal{L}'\) is identical to minimizing
$$
\frac{1}{\alpha}\mathcal{L}'(\boldsymbol{\theta}) = \frac{1}{\alpha}\text{NLL}(\sigma_{\text{true}}) + \text{KL}(q\parallel p),
$$
which has the required coefficient of 1 on the KL term. Matching the \(S(\boldsymbol{c})\)-dependent part of \(\frac{1}{\alpha}\text{NLL}(\sigma_{\text{true}}) = \frac{S(\boldsymbol{c})}{2\alpha\sigma_{\text{true}}^2} + \text{const}\) against \(\text{NLL}(\sigma_{\text{eff}}) = \frac{S(\boldsymbol{c})}{2\sigma_{\text{eff}}^2} + \text{const}\) gives
$$
\sigma_{\text{eff}}^2 = \alpha\,\sigma_{\text{true}}^2.
$$
So down-weighting the KL term by \(\alpha\) is mathematically equivalent to running the *correctly weighted* ELBO (coefficient 1 on KL) under a likelihood that assumes noise scale \(\sigma_{\text{eff}} = \sqrt{\alpha}\,\sigma_{\text{true}}\) instead of the true \(\sigma_{\text{true}}\). Concretely, \(\alpha = 0.001\) gives \(\sigma_{\text{true}}/\sigma_{\text{eff}} = 1/\sqrt{\alpha} \approx 32\): the assumed noise understates the true noise by roughly \(32\times\), so the data term dominates the (now too-weak) prior regularization and the resulting posterior is about \(32\times\) too confident.

## Combining a per-datapoint-mean NLL with the KL: `elbo_loss`

`pypolymix.likelihoods.Likelihood.nll()` returns a negative log-likelihood that is summed over the output dimension and averaged (MEAN) over the batch (and sample) dimension(s) -- a per-datapoint-mean quantity, regardless of the minibatch size used to compute it.

Combining this mean-per-datapoint data term with a KL computed once over the whole parameter space requires
$$
\text{loss} = \overline{\text{NLL}} + \frac{\text{KL}(q\parallel p)}{n_{\text{train}}},
$$
with \(n_{\text{train}}\) the FULL training-set row count, for **any** batch size -- including a batch equal to the full dataset. There is no full-batch special case under this convention: a mean-per-datapoint data term always needs to be divided down to match a single, whole-dataset KL, however large the batch is. (This differs from the alternative "sum-per-minibatch" convention, which does have a full-batch special case; the two conventions must not be mixed.) Blundell, Cornebise, Kavukcuoglu & Wierstra (2015), "Weight Uncertainty in Neural Networks," [arXiv:1505.05424](https://arxiv.org/abs/1505.05424), give the general minibatch-weighting scheme with per-minibatch weights \(\pi_i\) (useful when minibatches are not equal-sized); `elbo_loss`'s uniform division by \(n_{\text{train}}\) is the simple, always-correct special case of that scheme for the mean-NLL convention used here.

`pypolymix.elbo_loss` implements exactly this combination:

```python
import torch
from pypolymix.elbo import elbo_loss
from pypolymix.likelihoods import GaussianLikelihood

likelihood = GaussianLikelihood(num_outputs=2)
pred = torch.zeros(4, 2)
target = torch.zeros(4, 2)
nll_mean = likelihood.nll(pred, target)  # mean-per-datapoint, any batch size

kl = torch.tensor(2.0)       # KL(q || p), summed once over all parameters
n_train = 1000               # full training-set row count

loss = elbo_loss(nll_mean, kl, n_train=n_train)
```

## Learnable likelihood scale: normalizer + no-decay

A learnable observation-noise scale (e.g. `GaussianLikelihood`'s \(\sigma\)) is only identifiable from the data if two requirements hold:

1. **The normalizer term is present.** For the Gaussian case, dropping \(\log(2\pi\sigma_d^2)\) and keeping only the quadratic data term leaves \(\partial(\text{loss})/\partial(\log\sigma) < 0\) everywhere, so \(\sigma\) is driven to a degenerate extreme by gradient descent regardless of fit quality -- the normalizer is what penalizes making \(\sigma\) arbitrarily large (or small) to shrink the data term for free.
2. **The scale parameter is excluded from weight decay.** Decoupled weight decay pulls a parameter toward 0 independent of the data; applied to `log_sigma` this silently asserts a specific noise scale instead of letting the likelihood identify it from the residuals.

`GaussianLikelihood` already includes the normalizer term in `nll()` and exposes `no_decay_parameters()` so the optimizer can be built with two parameter groups:

```python
import torch
import torch.nn as nn
from pypolymix.likelihoods import GaussianLikelihood

model = nn.Linear(3, 2)
likelihood = GaussianLikelihood(num_outputs=2)

optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "weight_decay": 1e-4},
    {"params": likelihood.no_decay_parameters(), "weight_decay": 0.0},
], lr=1e-3)
```

Seitzer, Tavakoli, Antic & Martius (2022), "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks," [arXiv:2203.09168](https://arxiv.org/abs/2203.09168), identify a further, not-fully-solved caveat even once both requirements above are satisfied: a correctly-normalized heteroscedastic NLL can still converge to poor-but-stable mean estimates, because a large \(\sigma\) down-weights a datapoint's contribution to the mean's gradient (the data term scales as \(1/\sigma^2\)). They propose \(\beta\)-NLL as a mitigation; `pypolymix` does not implement it.

## Non-centered (whitened) parameterization

Hierarchical and strongly-scaled models are prone to a "funnel" pathology: when the prior scale varies by orders of magnitude across parameters, a fixed variational family's initialization and/or any scale floor it imposes (to prevent, e.g., a diagonal covariance from collapsing to zero) interacts badly with parameters whose true prior scale is much smaller than that floor, trapping optimization in a region where it cannot resolve the funnel's narrow neck. Betancourt & Girolami (2015), "Hamiltonian Monte Carlo for Hierarchical Models," [arXiv:1312.0906](https://arxiv.org/abs/1312.0906), analyze this pathology and its standard fix: a non-centered ("whitened") reparameterization.

`pypolymix.parameter_groups.WhitenedGroup` implements this fix by wrapping a base parameter group so that it models a standardized variable \(z\) instead of \(\boldsymbol{\theta}\) directly:
$$
\boldsymbol{\theta} = \boldsymbol{c} + \boldsymbol{s}\odot\boldsymbol{z}, \qquad p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0}, \mathbb{I}),
$$
with fixed (non-learned) offset \(\boldsymbol{c}\) and per-parameter scale \(\boldsymbol{s}\). Because this affine map is a fixed bijection, KL divergence is invariant under it: \(\text{KL}(q(\boldsymbol{\theta})\parallel p(\boldsymbol{\theta})) = \text{KL}(q(\boldsymbol{z})\parallel \mathcal{N}(\boldsymbol{0},\mathbb{I}))\) exactly, so `WhitenedGroup.distribution_loss()` simply reuses whatever KL machinery the wrapped base group already has for \(z\)-space, now always comparing against a scale-free \(N(0,I)\) prior:

```python
import torch
from pypolymix.parameter_groups import IIDGaussianGroup, WhitenedGroup

base = IIDGaussianGroup("z", num_params=3)
center = torch.tensor([1.0, 2.0, 3.0])
scale = torch.tensor([0.1, 0.2, 0.3])
group = WhitenedGroup(base, center, scale)

theta = group.sample_parameters(num_samples=4)  # theta = center + scale * z
kl = group.distribution_loss()                  # KL(q(z) || N(0, I))
```

## KL/entropy estimators for implicit samplers

`pypolymix.parameter_groups.LangevinGroup` (and any other implicit sampler without an analytic \(q\)) cannot use the ordinary analytic-or-reparameterized-Monte-Carlo KL path above, because that path requires evaluating \(\log q(\boldsymbol{\theta})\), which an implicit sampler never defines. `pypolymix.parameter_groups.kl_estimators.CrossEntropyOnly`, `LangevinGroup`'s historical default, instead computes only the cross-entropy term \(-\mathbb{E}_q[\log p(\boldsymbol{\theta})]\) from particles alone; because this omits the entropy \(-H[q]\) entirely, minimizing it alone drives \(q\) toward a point mass at the prior's mode ("particle collapse"), not toward the true posterior. `DiagonalGaussianEntropyKL` and `KNNEntropyKL` add a plug-in entropy estimate back in -- a cheap per-dimension diagonal-Gaussian estimate and a nonparametric \(k\)-nearest-neighbor (Kozachenko-Leonenko) estimate, respectively -- at the cost of an added Gaussianity assumption (the former) or \(O(S^2)\) pairwise distances and higher small-sample variance (the latter). See the [KL/Entropy Estimator Comparison](examples/kl_estimators_comparison.ipynb) notebook for an empirical illustration on a toy posterior. `pypolymix` does not yet prescribe a default recommendation among these estimators beyond `LangevinGroup`'s current `CrossEntropyOnly` default -- this is an open question under active investigation, not a settled recommendation.

## A note on temperature / cold posteriors

`pypolymix.elbo_loss` accepts a `temperature` argument that scales the KL term, defaulting to 1.0 -- the only value consistent with a proper ELBO. A *deliberate*, documented `temperature != 1.0` is a different thing entirely from the undocumented ad hoc KL rescaling described above: Wenzel, Roth, Veeling, Świątkowski, Tran, Mandt, Snoek, Salimans, Jenatton & Nowozin (2020), "How Good is the Bayes Posterior in Deep Neural Networks Really?," [arXiv:2002.02405](https://arxiv.org/abs/2002.02405), find that "cold posteriors" (`temperature < 1`) often improve predictive performance in deep Bayesian neural networks, for reasons that remain debated in the literature. Using a non-default temperature is a researched, citable modeling choice that must be reported as such wherever it is set; using it as a free knob to make an otherwise-unstable training run "look stable" is the undocumented-rescaling bug from the first section above, not a cold-posterior choice.