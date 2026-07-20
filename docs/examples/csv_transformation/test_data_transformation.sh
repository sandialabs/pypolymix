#!/bin/zsh

[[ -f "test_predictions.csv" ]] || echo "test_predictions.csv missing";

# test with no directory specified
uv run python data_transformation.py test_predictions.csv input_evm input_rhom >/dev/null
[[ -d "training_results_input_evm_input_rhom" ]] || echo "Failed to create default directory"

[[ -f "training_results_input_evm_input_rhom/data.pickle" ]] || echo "Failed to create data.pickle"

for file in "data.pickle" "train_output.pickle" "sm.pickle"; do
    [[ -f "training_results_input_evm_input_rhom/$file" ]] || echo "${file} was not created"
done

rm -r training_results_input_evm_input_rhom