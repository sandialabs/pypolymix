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

The optimization algorithm uses the following **E**vidence **L**ower **BO**und (ELBO) expression for the loss function

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{q}[\log p(\mathcal{D}\vert\boldsymbol{c})] - \text{KL}(q_{\boldsymbol{\theta}}(\boldsymbol{c})\parallel p(\boldsymbol{c})) + \text{const.}
$$

The current version of the library implements variational inference algorithms that assume Gaussian approximate posteriors, i.e. for \(\mathcal{Q}\), as well as a Gausian likelihood. For the prior distribution \(p(\boldsymbol{c})\), both uniform and Gaussian distributions are implemented.

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