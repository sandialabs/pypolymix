# data_transformation.py

Transforms a CSV file with inputs, predicted outputs, and reference outputs to three .pickle files that can be used for data visualization in a GUI.

Quickstart: 

```uv run python data_transformation.py --help```

**Note that this is a writeup for the requirements and decisions involved in developing `data_transformation.py`.**

`relative_error_transformation.py` has a similar idea and the same output format, except the output columns in the csv contain relative error instead of reference and predicted values.

## Given: test_predictions.csv

A CSV file with 1.2M rows and 12 columns.

6 Inputs:
- input_evm
- input_rhom
- input_rhoi
- input_temperature
- input_vmJ2
- input_flux

3 Predicted Outputs:
- predicted_devm_dt
- predicted_drhom_dt
- predicted_drhoi_dt

3 Reference Outputs:
- reference_devm_dt
- reference_drhom_dt,
- reference_drhoi_dt

Note that some of the entry values are things like 6.86e-13 insetad of being written out as decimals.

## Wanted: Visualize the differences between predicted and reference outputs in the GUI

Steps:
1. Load input and reference output data into the GUI
    1. Transform the input and reference data from `test_predictions.csv` to the data structure that the GUI expects
    2. Store the data structure in a .pickle file on disk

2. Load the training results into the GUI
    1. Transform all data from `test_predictions.csv` to the data structure that the GUI expects
    2. Store the data structure in a .pickle file on disk

3. Configure the GUI so that the visualizations are meaningful

## GUI Input Data Structure
The object in the pickle file is stored as a dictionary.

The dictionary has one key: `'data'`
`obj['data']` has entries for each input. The output is stored under `obj['data']['U']`

All input/output variables should have a numpy array as their value. All numpy arrays should be the same length.

example format:
- inputs: `obj['data']['x1']` and `obj['data']['x2']`
- outputs: `obj['data']['U']['out1']` and `obj['data']['U']['out2']`

## GUI Training Results Data Structure

For visualizing training results, the GUI requires two files: `train_output.pickle` and `SM.pickle`. We store these files in a specified directory, or ./training_results_<input1>_<input2> if none is specified.

#### `train_output.pickle`

Contains the predictions and the actual input and output values.

Stored as a dictionary with the following keys
- "U" -> a dictionary (str -> np.array) of the actual output values
- "Usm" -> a dictionary (str -> np.array) of the **s**urrogate **m**odel's predicted output values
- "unmapped" -> a dictionary (str -> np.array) of the input values

Note that the GUI assumes U and Usm are in a scaled output space. Our data has them in physical output space. To work around this, we use the physical values and provide an identity transform in SM["output_maps"].

#### `SM.pickle`

Stores the trained **S**urrogate **M**odel in the format required by the GUI. Since the input data is provided, but not the surrogate model, we'll make a minimal SM.pickle file.
