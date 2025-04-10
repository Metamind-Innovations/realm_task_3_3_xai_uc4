import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def predict_with_docker(X, input_dir, output_dir, temp_input, temp_output):
    # Get absolute paths for input and output
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)

    # Save input to a temporary CSV in the input directory
    temp_input_path = os.path.join(input_dir, os.path.basename(temp_input))
    X.to_csv(temp_input_path, index=False)

    # Prepare docker input and output paths
    docker_input_path = f"/app/{temp_input}"
    docker_output_path = f"/app/{temp_output}"

    # Run docker container with the specified paths
    cmd = [
        'docker', 'run',
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

    # Load and return predictions
    output_path = os.path.join(output_dir, os.path.basename(temp_output))
    if os.path.exists(output_path):
        predictions = pd.read_csv(output_path)
        return predictions
    else:
        print(f"Output file not found: {output_path}")
        return None


def docker_prediction_function(x):
    # Convert the numpy array to a DataFrame with the correct column names
    df = pd.DataFrame(x, columns=input_columns)

    # Run prediction
    preds = predict_with_docker(df, args.input_dir, args.output_dir, args.temp_input, args.temp_output)

    if preds is None:
        # Return zeros if prediction fails
        return np.zeros((x.shape[0], 4))

    # Convert predictions to numpy array
    return preds.values


def explain_with_shap(X, output_columns):
    print("Creating SHAP explainer...")

    # For efficiency, use a subset of the data as background
    background = shap.sample(X, min(100, len(X)))

    # Create an explainer that calls the Docker container
    explainer = shap.KernelExplainer(docker_prediction_function, background)

    print("Calculating SHAP values...")
    # Calculate SHAP values (this will call the Docker container multiple times)
    shap_values = explainer.shap_values(X)

    # Create visualizations
    os.makedirs('shap_plots', exist_ok=True)

    # Generate and save plots for each output
    for i, output_name in enumerate(output_columns):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X, show=False)
        plt.title(f'SHAP Summary Plot for {output_name}')
        plt.tight_layout()
        plt.savefig(f'shap_plots/shap_summary_{output_name}.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X, plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance for {output_name}')
        plt.tight_layout()
        plt.savefig(f'shap_plots/shap_importance_{output_name}.png')
        plt.close()

    return shap_values, explainer


def generate_shap_json(X, shap_values, output_columns):
    # Create a JSON structure for SHAP values
    shap_data = {
        "overall_importance": {},
        "sample_explanations": []
    }

    # Calculate overall feature importance for each output
    for i, output_name in enumerate(output_columns):
        # Average absolute SHAP values across samples
        feature_importance = np.abs(shap_values[i]).mean(axis=0)

        # Sort features by importance
        importance_pairs = list(zip(X.columns, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Store in JSON
        shap_data["overall_importance"][output_name] = {
            feature: float(importance)
            for feature, importance in importance_pairs
        }

    # Add sample-level explanations
    num_samples = min(10, X.shape[0])  # Limit to 10 samples for brevity
    for i in range(num_samples):
        sample_explanation = {
            "sample_id": i,
            "input_values": X.iloc[i].to_dict(),
            "explanations": {}
        }

        for j, output_name in enumerate(output_columns):
            # Get SHAP values for this sample and output
            sample_shap = shap_values[j][i]

            # Combine with feature names and sort by absolute value
            feature_contributions = list(zip(X.columns, sample_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Store in JSON
            sample_explanation["explanations"][output_name] = {
                feature: float(value)
                for feature, value in feature_contributions
            }

        shap_data["sample_explanations"].append(sample_explanation)

    # Save to JSON file
    with open('shap_values.json', 'w', encoding='utf-8') as f:
        json.dump(shap_data, f, indent=2, ensure_ascii=False)

    print("SHAP values saved to shap_values.json")
    return shap_data


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SHAP explainer for Docker-based COPD model')
    parser.add_argument('--input_dir', type=str, default="C:/Users/gigak/Desktop/REALM_UC4/copd_in",
                        help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, default="C:/Users/gigak/Desktop/REALM_UC4/copd_out",
                        help='Directory for output files')
    parser.add_argument('--temp_input', type=str, default="in/temp_input.csv",
                        help='Temporary input file path (relative to Docker container)')
    parser.add_argument('--temp_output', type=str, default="out/temp_output.csv",
                        help='Temporary output file path (relative to Docker container)')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of samples to use for SHAP analysis')
    parser.add_argument('--input_file', type=str, default="x_sample_docker.csv",
                        help='Input CSV filename')
    parser.add_argument('--output_file', type=str, default="output.csv",
                        help='Output CSV filename')

    args = parser.parse_args()

    # Load your original data
    input_path = os.path.join(args.input_dir, args.input_file)
    output_path = os.path.join(args.output_dir, args.output_file)

    X = pd.read_csv(input_path)
    y = pd.read_csv(output_path)

    # Store column names globally for the prediction function
    input_columns = X.columns
    output_columns = y.columns

    print(f"Loaded input data with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Loaded output data with {y.shape[1]} target variables")

    # Use sample size from arguments
    X_sample = X.head(args.sample_size)

    # Run SHAP analysis
    shap_values, explainer = explain_with_shap(X_sample, output_columns)

    # Generate JSON with SHAP values
    shap_data = generate_shap_json(X_sample, shap_values, output_columns)

    print("Explanation complete!")
