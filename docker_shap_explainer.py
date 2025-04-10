import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import shap


def predict_with_docker(X, input_dir, output_dir, temp_input, temp_output):
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    temp_input_path = os.path.join(input_dir, os.path.basename(temp_input))
    X.to_csv(temp_input_path, index=False)

    docker_input_path = f"/app/{temp_input}"
    docker_output_path = f"/app/{temp_output}"

    cmd = [
        'docker', 'run', '--rm',
        '-v', f"{input_dir_abs}:/app/in",
        '-v', f"{output_dir_abs}:/app/out",
        'forth_copd_args',
        '--input_data', docker_input_path,
        '--output', docker_output_path
    ]

    print(f"Running Docker command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Docker command failed: {result.stderr}")
        return None

    output_path = os.path.join(output_dir, os.path.basename(temp_output))
    if os.path.exists(output_path):
        predictions = pd.read_csv(output_path)
        return predictions
    else:
        print(f"Output file not found: {output_path}")
        return None


def docker_prediction_function(x):
    df = pd.DataFrame(x, columns=input_columns)
    preds = predict_with_docker(df, args.input_dir, args.output_dir, args.temp_input, args.temp_output)

    if preds is None:
        return np.zeros((x.shape[0], len(output_columns)))

    if args.add_noise:
        noise = np.random.normal(0, 0.01, size=preds.values.shape)
        return preds.values + noise
    else:
        return preds.values


def explain_with_shap(X, output_columns):
    print("Creating SHAP explainer...")
    shap_values_list = []

    for i, output_col in enumerate(output_columns):
        print(f"Processing output {output_col} ({i + 1}/{len(output_columns)})...")

        def single_output_func(x, col_idx=i):
            full_output = docker_prediction_function(x)
            return full_output[:, col_idx:col_idx + 1]

        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(
            single_output_func,
            background,
            link="identity",
            l1_reg="num_features(10)"
        )

        current_shap_values = explainer.shap_values(X, nsamples=200)

        if len(current_shap_values.shape) == 1:
            current_shap_values = current_shap_values.reshape(-1, 1)

        print(f"Shape of shap_values for {output_col}: {current_shap_values.shape}")
        print(
            f"Min/Max SHAP values for {output_col}: {np.min(current_shap_values):.6f}, {np.max(current_shap_values):.6f}")

        shap_values_list.append(current_shap_values)

    return shap_values_list, None


def generate_shap_json(X, shap_values, output_columns):
    shap_data = {
        "overall_importance": {},
        "sample_explanations": []
    }

    X_processed = X.copy()
    for col in X_processed.columns:
        X_processed[col] = X_processed[col].apply(lambda x: None if pd.isna(x) else x)

    for i, output_name in enumerate(output_columns):
        if i >= len(shap_values):
            print(f"No SHAP values for output {output_name}, skipping in JSON")
            continue

        current_shap_values = shap_values[i]

        if current_shap_values.shape[1] == len(X.columns):
            feature_importance = np.abs(current_shap_values).mean(axis=0)
            feature_names = X.columns
        else:
            min_cols = min(current_shap_values.shape[1], len(X.columns))
            feature_importance = np.abs(current_shap_values[:, :min_cols]).mean(axis=0)
            feature_names = X.columns[:min_cols]

        importance_pairs = list(zip(feature_names, feature_importance))

        def safe_abs_value(x):
            # Safely handle potentially multi-dimensional values
            value = x[1]
            if hasattr(value, 'shape') and value.shape:
                value = value.item() if value.size == 1 else np.mean(value)
            if np.isnan(value):
                return 0
            return float(np.abs(value))

        importance_pairs.sort(key=safe_abs_value, reverse=True)

        shap_data["overall_importance"][output_name] = {
            feature: 0.0 if (hasattr(importance, 'shape') and importance.shape and np.isnan(
                importance).any()) or np.isnan(importance)
            else float(importance.item() if hasattr(importance,
                                                    'shape') and importance.shape and importance.size == 1 else importance)
            for feature, importance in importance_pairs
        }

    num_samples = min(10, X.shape[0])
    for s in range(num_samples):
        sample_explanation = {
            "sample_id": s,
            "input_values": X_processed.iloc[s].to_dict(),
            "explanations": {}
        }

        for i, output_name in enumerate(output_columns):
            if i >= len(shap_values):
                continue

            current_shap_values = shap_values[i]

            if len(current_shap_values.shape) == 1:
                sample_shap = current_shap_values[s:s + 1]
            else:
                sample_shap = current_shap_values[s]

            if len(sample_shap) == len(X.columns):
                feature_names = X.columns
                feature_values = sample_shap
            else:
                min_cols = min(len(sample_shap), len(X.columns))
                feature_names = X.columns[:min_cols]
                feature_values = sample_shap[:min_cols]

            feature_contributions = list(zip(feature_names, feature_values))

            def safe_abs_value_for_sample(x):
                # Safely handle potentially multi-dimensional values
                value = x[1]
                if hasattr(value, 'shape') and value.shape:
                    value = value.item() if value.size == 1 else np.mean(value)
                if np.isnan(value):
                    return 0
                return float(np.abs(value))

            feature_contributions.sort(key=safe_abs_value_for_sample, reverse=True)

            sample_explanation["explanations"][output_name] = {
                feature: 0.0 if (hasattr(value, 'shape') and value.shape and np.isnan(value).any()) or np.isnan(value)
                else float(value.item() if hasattr(value, 'shape') and value.shape and value.size == 1 else value)
                for feature, value in feature_contributions
            }

        shap_data["sample_explanations"].append(sample_explanation)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                # For numpy float scalars
                return float(obj)
            if isinstance(obj, np.ndarray):
                # Handle array of size 1 as a scalar
                if obj.size == 1:
                    return obj.item()
                return obj.tolist()
            if pd.isna(obj):
                return None
            return super(NpEncoder, self).default(obj)

    with open('shap_values.json', 'w', encoding='utf-8') as f:
        json.dump(shap_data, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print("SHAP values saved to shap_values.json")
    return shap_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHAP explainer for Docker-based COPD model')
    parser.add_argument('--input_dir', type=str, default="copd_in",
                        help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, default="copd_out",
                        help='Directory for output files')
    parser.add_argument('--temp_input', type=str, default="in/temp_input.csv",
                        help='Temporary input file path (relative to Docker container)')
    parser.add_argument('--temp_output', type=str, default="out/temp_output.csv",
                        help='Temporary output file path (relative to Docker container)')
    parser.add_argument('--sample_size', type=int, default=2,
                        help='Number of samples to use for SHAP analysis')
    parser.add_argument('--input_file', type=str, default="x_sample_docker.csv",
                        help='Input CSV filename')
    parser.add_argument('--output_file', type=str, default="output.csv",
                        help='Output CSV filename')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add small random noise to predictions to help SHAP differentiate')

    args = parser.parse_args()
    np.random.seed(42)

    input_path = os.path.join(args.input_dir, args.input_file)
    output_path = os.path.join(args.output_dir, args.output_file)

    X = pd.read_csv(input_path)
    y = pd.read_csv(output_path)

    input_columns = X.columns
    output_columns = y.columns

    print(f"Loaded input data with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Loaded output data with {y.shape[1]} target variables")

    X_sample = X.head(args.sample_size)
    shap_values, explainer = explain_with_shap(X_sample, output_columns)
    shap_data = generate_shap_json(X_sample, shap_values, output_columns)

    print("Explanation complete!")
