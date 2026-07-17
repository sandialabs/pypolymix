import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def _auto_scale_factor(values):
    """
    Return a scale factor that makes the variable range 1

    x_scaled = x_raw * scale_factor
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not finite.any():
        return 1.0

    v = values[finite]
    span = float(np.nanmax(v) - np.nanmin(v))

    if span == 0.0 or not np.isfinite(span):
        return 1.0

    return 1.0 / span


def make_gui_error_visualization_files(
    df,
    input_x="input_temperature",
    input_y="input_vmJ2",
    output_dir=None,
    n_samples=50_000,
    x_scale="auto",
    y_scale="auto",
    random_seed=12345,
):
    """
    Generate GUI-compatible visualization files for one two-input parameterization.

    Files written:
        data.pickle
        train_output.pickle
        SM.pickle

    This is intended for visualization of prediction error, not for creating a
    real trained surrogate model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original input, reference, and predicted columns.

    input_x : str
        First input variable to visualize.

    input_y : str
        Second input variable to visualize.

    output_dir : str or None
        Folder where output files are written. If None, a folder name is generated.

    n_samples : int or None
        Optional downsampling count. Use None to keep all rows.

    x_scale, y_scale : "auto", "none", or float
        Scaling applied to the two selected input variables.

        If "auto":
            variable is scaled by 1 / range(variable).

        If "none":
            variable is not scaled.

        If float:
            variable is multiplied by that factor.

    random_seed : int
        Random seed for reproducible downsampling.

    Returns
    -------
    dict
        Summary of generated files and GUI variable names.
    """

    input_names = [column for column in df.columns if column.startswith("input_")]
    reference_names = [column for column in df.columns if column.startswith("reference_")]
    predicted_names = [column for column in df.columns if column.startswith("predicted_")]
    if input_x not in input_names:
        raise ValueError(f"input_x={input_x!r} is not one of {input_names}")
    if input_y not in input_names:
        raise ValueError(f"input_y={input_y!r} is not one of {input_names}")
    if input_x == input_y:
        raise ValueError("input_x and input_y must be different.")
    
    # reference_devm_dt -> output_devm_dt
    output_pairs = {}

    for ref_col in reference_names:
        suffix = ref_col.replace("reference_", "")
        pred_col = "predicted_" + suffix
        gui_out_name = "output_" + suffix

        if pred_col not in predicted_names:
            raise KeyError(f"Could not find predicted column for {ref_col}: expected {pred_col}")

        output_pairs[gui_out_name] = (ref_col, pred_col)


    # ------------------------------------------------------------------
    # Output folder.
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = f"training_results_{input_x}_{input_y}"

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Downsample rows.
    # ------------------------------------------------------------------
    n_total = len(df)
    rng = np.random.default_rng(random_seed)

    if n_samples is not None and n_total > n_samples:
        idx = rng.choice(n_total, size=n_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_total)

    # ------------------------------------------------------------------
    # Extract raw selected inputs.
    # ------------------------------------------------------------------
    x_raw = df[input_x].to_numpy(dtype=float)[idx]
    y_raw = df[input_y].to_numpy(dtype=float)[idx]

    # ------------------------------------------------------------------
    # Resolve scaling.
    # ------------------------------------------------------------------
    def resolve_scale(scale_spec, values):
        if isinstance(scale_spec, str):
            if scale_spec.lower() == "auto":
                return _auto_scale_factor(values)
            if scale_spec.lower() == "none":
                return 1.0
            raise ValueError("Scale spec must be 'auto', 'none', or a numeric factor.")
        return float(scale_spec)

    x_factor = resolve_scale(x_scale, x_raw)
    y_factor = resolve_scale(y_scale, y_raw)

    x = x_raw * x_factor
    y = y_raw * y_factor

    # GUI names for the two plotted inputs.
    x_gui_name = f"plot_{input_x}"
    y_gui_name = f"plot_{input_y}"

    # ------------------------------------------------------------------
    # Drop rows that are non-finite in selected inputs or any output.
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(x) & np.isfinite(y)

    for _, (ref_col, pred_col) in output_pairs.items():
        ref_values = df[ref_col].to_numpy(dtype=float)[idx]
        pred_values = df[pred_col].to_numpy(dtype=float)[idx]

        finite_mask &= np.isfinite(ref_values)
        finite_mask &= np.isfinite(pred_values)

    x = x[finite_mask]
    y = y[finite_mask]
    idx = idx[finite_mask]

    if len(x) < 3:
        raise ValueError("Fewer than 3 finite rows remain; cannot make a useful 2D visualization.")

    if float(np.min(x)) == float(np.max(x)):
        raise ValueError(f"Scaled x input {x_gui_name!r} is constant.")

    if float(np.min(y)) == float(np.max(y)):
        raise ValueError(f"Scaled y input {y_gui_name!r} is constant.")

    # ------------------------------------------------------------------
    # Build data.pickle.
    #
    # This is loaded using "Load Data (.pickle file)".
    # It contains only the two visualization inputs and all reference outputs.
    # ------------------------------------------------------------------
    data_obj = {
        "data": {
            x_gui_name: x,
            y_gui_name: y,
            "U": {},
        }
    }

    for gui_out_name, (ref_col, pred_col) in output_pairs.items():
        data_obj["data"]["U"][gui_out_name] = df[ref_col].to_numpy(dtype=float)[idx]

    data_path = os.path.join(output_dir, "data.pickle")

    with open(data_path, "wb") as f:
        pickle.dump(data_obj, f)

    # ------------------------------------------------------------------
    # Build train_output.pickle.
    #
    # This is loaded using "Load Training".
    # It contains the same two visualization inputs and all reference/predicted
    # outputs with matching keys.
    # ------------------------------------------------------------------
    train_output_obj = {
        "U": {},
        "Usm": {},
        "unmapped": {
            x_gui_name: x,
            y_gui_name: y,
        },
    }

    for gui_out_name, (ref_col, pred_col) in output_pairs.items():
        train_output_obj["U"][gui_out_name] = df[ref_col].to_numpy(dtype=float)[idx]
        train_output_obj["Usm"][gui_out_name] = df[pred_col].to_numpy(dtype=float)[idx]

    train_output_path = os.path.join(output_dir, "train_output.pickle")

    with open(train_output_path, "wb") as f:
        pickle.dump(train_output_obj, f)

    # ------------------------------------------------------------------
    # Build minimal SM.pickle.
    #
    # This is a placeholder model object. It is not a real trained surrogate.
    # It exists so the GUI's Training Result plotting code can run.
    # ------------------------------------------------------------------
    identity = FunctionTransformer(validate=False)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    nodes = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=float,
    )

    conn = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=int,
    )

    output_maps = {}
    nodal_values = {}

    for gui_out_name, (ref_col, pred_col) in output_pairs.items():
        output_maps[gui_out_name] = identity

        pred_values = df[pred_col].to_numpy(dtype=float)[idx]
        pred_values_finite = pred_values[np.isfinite(pred_values)]

        if len(pred_values_finite) == 0:
            representative_value = 0.0
        else:
            representative_value = float(np.nanmedian(pred_values_finite))

        nodal_values[gui_out_name] = np.full(nodes.shape[0], representative_value, dtype=float)

    SM = {
        "mesh": {
            "nodes": nodes,
            "conn": conn,

            # Placeholder only. Not physically meaningful.
            "M": np.eye(nodes.shape[0]),
        },
        "input_maps": {
            x_gui_name: identity,
            y_gui_name: identity,
        },
        "output_maps": output_maps,
        "nodal_values": nodal_values,
    }

    sm_path = os.path.join(output_dir, "SM.pickle")

    with open(sm_path, "wb") as f:
        pickle.dump(SM, f)

    # ------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------
    print("Wrote GUI visualization files:")
    print(f"  {data_path}")
    print(f"  {train_output_path}")
    print(f"  {sm_path}")
    print()
    print("GUI input variables:")
    print(f"  {x_gui_name}")
    print(f"  {y_gui_name}")
    print()
    print("Input scaling:")
    print(f"  {x_gui_name} = {input_x} * {x_factor}")
    print(f"  {y_gui_name} = {input_y} * {y_factor}")
    print()
    print("GUI output variables:")
    for out_name in output_pairs:
        print(f"  {out_name}")
    print()
    print(f"Rows written: {len(x)}")



def main():
    parser = argparse.ArgumentParser(
        description="""Generate GUI-compatible visualization files from prediction and reference CSV data.
Outputs 3 files in a new folder called training_results_<input1>_<input2>.
- data.pickle: can be loaded into the GUI with "load data"
- train_output.pickle: the training results. Can be loaded into the GUI with "load training"
- SM.pickle: A supplementary file that the GUI requires to be in the same directory as train_output.pickle""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "csv_file",
        help="CSV file containing input, reference, and predicted columns.",
    )

    parser.add_argument(
        "input_variable_1",
        help="First input variable to visualize.",
    )

    parser.add_argument(
        "input_variable_2",
        help="Second input variable to visualize.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    
    make_gui_error_visualization_files(
        df,
        input_x=args.input_variable_1,
        input_y=args.input_variable_2,
        n_samples=50_000,
        x_scale="auto",
        y_scale="auto",
    )

if __name__ == "__main__":
    main()
