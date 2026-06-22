from dict_to_tensors import dict_to_tensors
import numpy as np
import torch

if __name__ == "__main__":
    obj = {
        "data": {
            "x1": np.array([1, 2, 3]),
            "x2": np.array([1, 2, 3]),
            "U": {
                "y1": np.array([4, 5, 6]),
            }
        }
    }
    X, Y = dict_to_tensors(obj)
    print(X)
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    
    # X and Y are 2-D tensors
    assert X.ndim == 2
    assert Y.ndim == 2

    # X and Y are the correct values
    assert torch.allclose(X, torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float32))
    assert torch.allclose(Y, torch.tensor([[4], [5], [6]], dtype=torch.float32))
    
    print("All Tests Pass")