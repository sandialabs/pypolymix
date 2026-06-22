import torch
import numpy as np

def dict_to_tensors(obj):
    '''
    Input schema
    {
        'data': {
            'var1': np.array([values...]),  # Input variable 1
            'var2': np.array([values...]),  # Input variable 2
            ...
            'U': {
                'output1': np.array([values...]),  # Output variable 1
                'output2': np.array([values...]),  # Output variable 2
                ...
            }
        }
    }

    Note that all input and output keys map to numpy arrays of the same length
    
    Returns 2-D float tensor of input vars, 2-D float tensor of output vars
    '''
    output_dict = obj["data"]["U"]
    input_dict = {k: v for k, v in obj["data"].items() if k != "U"}

    # conversion to numpy.ndarray done because of a warning that
    # creating a tensor from a list of numpy.ndarrays is extremely slow
    X_np = np.column_stack(list(input_dict.values())).astype(np.float32)
    Y_np = np.column_stack(list(output_dict.values())).astype(np.float32)
    return torch.from_numpy(X_np), torch.from_numpy(Y_np)
