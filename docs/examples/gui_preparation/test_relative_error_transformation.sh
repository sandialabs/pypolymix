#!/bin/zsh
CSV_INPUT="one_step_control_log_error_map_runs.csv"

[[ -f "$CSV_INPUT" ]] || echo "${CSV_INPUT} missing";

# test with no directory specified
uv run python relative_error_transformation.py $CSV_INPUT temperature vmJ2 >/dev/null

DEFAULT_DIR="reference_results_temperature_vmJ2"
[[ -d "$DEFAULT_DIR" ]] || echo "Failed to create default directory"

for file in "data.pickle" "train_output.pickle" "sm.pickle"; do
    [[ -f "$DEFAULT_DIR/$file" ]] || echo "${file} was not created"
done

rm -r "$DEFAULT_DIR"

# test with directory specified
uv run python relative_error_transformation.py $CSV_INPUT temperature vmJ2 specified/output >/dev/null
[[ -d "specified/output" ]] || echo "Failed to create specified directory"

for file in "data.pickle" "train_output.pickle" "sm.pickle"; do
    [[ -f "specified/output/$file" ]] || echo "${file} was not created"
done

rm -r specified
