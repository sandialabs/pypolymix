# The training code is the same no matter how many experts
import torch

import pypolymix


def train_moe_model(
    surrogate_model, model, X, Y, num_epochs=10_000, num_samples=100, lr=1e-3, weight_decay=1e-4
):
    # Learnable per-output Gaussian observation noise
    likelihood = pypolymix.likelihoods.GaussianLikelihood(num_outputs=Y.shape[-1])

    # Optimizer: AdamW (likelihood's log_sigma excluded from weight decay)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": weight_decay},
            {"params": likelihood.no_decay_parameters(), "weight_decay": 0.0},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=10 * lr,
        total_steps=num_epochs,
    )

    # print(f"Training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        params = model.sample_parameters(num_samples=num_samples)
        Y_hat = surrogate_model(X, params)

        nll = likelihood.nll(Y_hat, Y.unsqueeze(0).expand_as(Y_hat))
        kl = model.distribution_loss()
        total_loss = pypolymix.elbo_loss(nll, kl, n_train=X.shape[0], temperature=1.0)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        # if (epoch + 1) % 1000 == 0:
        #     current_lr = scheduler.get_last_lr()[0]
        # print(
        #     f"Epoch {epoch + 1:5d} | "
        #     f"learning rate = {current_lr:.6f} | "
        #     f"nll = {nll.item():.4f} | "
        #     f"kl = {kl.item():.4f} | "
        #     f"total loss = {total_loss.item():.4f} | "
        #     f"log_sigma = {likelihood.log_sigma.detach().tolist()}"
        # )
    return total_loss.item()
