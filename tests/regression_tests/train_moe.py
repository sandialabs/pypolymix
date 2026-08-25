# The training code is the same no matter how many experts
import torch

def train_moe_model(surrogate_model, model, X, Y,
                num_epochs=10_000, num_samples=100,
                weight_factor=1e-2, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.MSELoss(reduction="sum")

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

        data_loss = loss_fn(Y_hat, Y.unsqueeze(0).expand_as(Y_hat)) / num_samples
        distribution_loss = model.distribution_loss()
        total_loss = data_loss + weight_factor * distribution_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        # if (epoch + 1) % 1000 == 0:
        #     current_lr = scheduler.get_last_lr()[0]
            # print(
            #     f"Epoch {epoch + 1:5d} | "
            #     f"learning rate = {current_lr:.6f} | "
            #     f"data loss = {data_loss.item():.4f} | "
            #     f"distribution loss = {distribution_loss.item():.4f} | "
            #     f"total loss = {total_loss.item():.4f}"
            # )
    return total_loss.item()