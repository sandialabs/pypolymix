#!/bin/zsh
CSV_INPUT="test_predictions.csv"

[[ -f "$CSV_INPUT" ]] || echo "${CSV_INPUT} missing";

# test with no directory specified
uv run python data_transformation.py test_predictions.csv input_evm input_rhom >/dev/null
[[ -d "training_results_input_evm_input_rhom" ]] || echo "Failed to create default directory"

for file in "data.pickle" "train_output.pickle" "sm.pickle"; do
    [[ -f "training_results_input_evm_input_rhom/$file" ]] || echo "${file} was not created"
done

rm -r training_results_input_evm_input_rhom

# test with directory specified
uv run python data_transformation.py test_predictions.csv input_evm input_rhom specified/output >/dev/null
[[ -d "specified/output" ]] || echo "Failed to create specified directory"

for file in "data.pickle" "train_output.pickle" "sm.pickle"; do
    [[ -f "specified/output/$file" ]] || echo "${file} was not created"
done

rm -r specified
