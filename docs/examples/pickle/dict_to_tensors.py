# use: uv run python print_pickle.py path/to/file
# PATH is the path to the .pickle file
import torch

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
    X = torch.tensor(list(input_dict.values()), dtype=torch.float32).T
    Y = torch.tensor(list(output_dict.values()), dtype=torch.float32).T
    return X,Y
