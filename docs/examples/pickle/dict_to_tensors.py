# use: uv run python print_pickle.py path/to/file
# PATH is the path to the .pickle file
import torch

def dict_to_tensors(obj):
    '''
    Input schema
    {
        'data': {
            'var1': [values...],  # Input variable 1
            'var2': [values...],  # Input variable 2
            ...
            'U': {
                'output1': [values...],  # Output variable 1
                'output2': [values...],  # Output variable 2
                ...
            }
        }
    }

    Note that all input and output keys map to numpy arrays of the same length
    
    Returns 2-D float tensor of input vars, 2-D float tensor of output vars
    '''
    output_dict = obj["data"]["U"]
    input_dict = {k: v for k, v in obj["data"].items() if k != "U"}
    
    X_cols = [[torch.tensor(arr, dtype=torch.float32)] for arr in input_dict.values()]
    Y_cols = [[torch.tensor(arr, dtype=torch.float32)] for arr in output_dict.values()]
    X = torch.hstack(X_cols)
    Y = torch.hstack(Y_cols)
    return X,Y
