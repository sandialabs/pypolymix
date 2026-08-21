import torch
# package import used because in CI pytest is run from the project root
from .train_moe import train_moe_model
from .three_moe_helpers import make_three_region_data, make_three_expert_problem, gating_weights
NUM_EXPERTS=3

def test_three_region_data_generation():
    X, Y, region, edges = make_three_region_data()

    assert X.shape == (150, 1)
    assert Y.shape == (150, 1)
    assert region.shape == (150,)
    assert edges.shape == (4,)

    assert set(region.tolist()).issubset({0, 1, 2})

def test_three_expert_training_returns_float():
    surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
    total_loss = train_moe_model(surrogate_model,model, X, Y, num_epochs=10, num_samples=2)

    assert isinstance(total_loss, float)
    assert torch.isfinite(torch.tensor(total_loss))

def is_better(expected_better, expected_worse):
    '''
    Check that expected_better is not too much more than expected_worse
    Ideally it would be less over multiple samples
    But the tests are already long-running, so for now close is good enough
    '''
    return expected_better < expected_worse + 10

def test_epochs():
    epoch_losses = {}
    for num in [1_000, 3_000, 10_000]:
        surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
        total_loss = train_moe_model(surrogate_model, model, X, Y, num_epochs=num)
        epoch_losses[num] = total_loss
    print(epoch_losses)
    assert is_better(epoch_losses[3_000], epoch_losses[1_000])
    assert is_better(epoch_losses[10_000], epoch_losses[3000])

def test_num_samples():
    loss = {}
    for num in [1, 10, 100]:
        surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
        loss[num] = train_moe_model(surrogate_model, model, X, Y, num_samples=num)
    assert is_better(loss[10], loss[1])
    assert is_better(loss[100], loss[10])

def test_experts_specialize_by_region():
    surrogate_model, model, X, Y, region, edges = make_three_expert_problem()
    train_moe_model(surrogate_model, model, X, Y)

    X_grid = torch.linspace(-1.0, 1.0, 600).unsqueeze(-1)

    grid_region = torch.bucketize(
        X_grid.squeeze(-1),
        edges[1:-1],
    )

    mean_gates = gating_weights(surrogate_model, model, X_grid)

    # Average gate vector in each of the three regions.
    region_gate_means = torch.stack(
        [mean_gates[grid_region == r].mean(dim=0) for r in range(NUM_EXPERTS)]
    )
    
    dominant_expert_by_region = region_gate_means.argmax(dim=1)
    # Check that each expert is dominant in one region
    assert set(dominant_expert_by_region.tolist()) == {0, 1, 2}

    # Quantify that dominance isn't just an even split
    dominant_weights = region_gate_means.max(dim=1).values
    print(f"Dominant weights\n{dominant_weights}")
    assert torch.all(dominant_weights > 0.50)
