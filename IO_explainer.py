import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Filter warnings to keep output clean
warnings.filterwarnings('ignore')


def load_data(input_path, output_path):
    """Load and preprocess input and output data"""
    # Load the data
    input_data = pd.read_csv(input_path)
    output_data = pd.read_csv(output_path)

    # Replace empty strings with NaN and handle numeric conversion
    input_data = input_data.replace('', np.nan)
    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            try:
                input_data[col] = pd.to_numeric(input_data[col])
            except:
                pass

    # Fill NaN values with the median for numeric columns
    for col in input_data.columns:
        if pd.api.types.is_numeric_dtype(input_data[col]):
            median_val = input_data[col].median(skipna=True)
            if not pd.isna(median_val):
                input_data[col] = input_data[col].fillna(median_val)
            else:
                input_data[col] = input_data[col].fillna(0)

    return input_data, output_data


def calculate_feature_importance(input_data, output_data):
    """Calculate correlation-based feature importance"""
    # Only use numeric columns from input data
    numeric_input = input_data.select_dtypes(include=[np.number])

    # Dictionary to store correlation values
    correlation_results = {}

    # Calculate correlations between each input feature and each output
    for output_col in output_data.columns:
        if output_data[output_col].nunique() > 1:  # Skip if only one unique value
            correlation_results[output_col] = []

            # For each feature, calculate correlation with the output
            for feature in numeric_input.columns:
                # Try different correlation methods
                try:
                    # Point biserial correlation for binary outcomes
                    if output_data[output_col].nunique() == 2:
                        corr, p_value = pointbiserialr(numeric_input[feature], output_data[output_col])
                    else:
                        # Spearman correlation (more robust than Pearson)
                        corr, p_value = spearmanr(numeric_input[feature], output_data[output_col], nan_policy='omit')

                    correlation_results[output_col].append({
                        'feature': feature,
                        'correlation': float(corr) if not np.isnan(corr) else 0,
                        'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                        'significant': bool(p_value < 0.05 if not np.isnan(p_value) else False)
                    })
                except Exception as e:
                    # Add a placeholder for failed correlations
                    correlation_results[output_col].append({
                        'feature': feature,
                        'correlation': 0,
                        'p_value': 1.0,
                        'significant': False,
                        'error': str(e)
                    })

            # Sort by absolute correlation value
            correlation_results[output_col] = sorted(
                correlation_results[output_col],
                key=lambda x: abs(x['correlation']),
                reverse=True
            )

    return correlation_results


def analyze_feature_distributions(input_data, output_data):
    """Analyze how feature distributions differ across prediction outcomes"""
    distribution_insights = {}

    # Only use numeric columns from input data
    numeric_input = input_data.select_dtypes(include=[np.number])

    # For each output column
    for output_col in output_data.columns:
        output_values = output_data[output_col].unique()

        # Skip if only one unique prediction value
        if len(output_values) <= 1:
            continue

        distribution_insights[output_col] = {}

        # For each feature, calculate statistics grouped by the output value
        for feature in numeric_input.columns:
            # Calculate summary statistics for each group
            group_stats = {}
            for value in output_values:
                feature_values = numeric_input.loc[output_data[output_col] == value, feature]

                # Handle NaN values for JSON serialization
                mean = float(feature_values.mean()) if not np.isnan(feature_values.mean()) else None
                median = float(feature_values.median()) if not np.isnan(feature_values.median()) else None
                std = float(feature_values.std()) if not np.isnan(feature_values.std()) else None
                min_val = float(feature_values.min()) if not np.isnan(feature_values.min()) else None
                max_val = float(feature_values.max()) if not np.isnan(feature_values.max()) else None

                # Convert numpy types to Python native types for JSON serialization
                group_stats[int(value)] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'count': int(feature_values.count())
                }

            distribution_insights[output_col][feature] = group_stats

    return distribution_insights


def calculate_feature_correlations(input_data):
    """Calculate correlations between all input features"""
    # Only use numeric columns from input data
    numeric_input = input_data.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_input.corr().fillna(0)

    # Convert correlation matrix to dictionary format
    feature_correlations = {}
    for feature1 in corr_matrix.columns:
        feature_correlations[feature1] = {}
        for feature2 in corr_matrix.columns:
            value = corr_matrix.loc[feature1, feature2]
            # Handle NaN or infinite values
            if np.isnan(value) or np.isinf(value):
                feature_correlations[feature1][feature2] = None
            else:
                feature_correlations[feature1][feature2] = float(value)

    return feature_correlations


def calculate_pca_components(input_data):
    """Calculate PCA components for dimensionality reduction visualization"""
    # Only use numeric columns from input data
    numeric_input = input_data.select_dtypes(include=[np.number])

    try:
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_input)

        # Apply PCA
        pca = PCA(n_components=min(2, numeric_input.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        # Create result dictionary
        pca_results = {
            'variance_explained': [float(v) for v in pca.explained_variance_ratio_],
            'components': [{'PC1': float(row[0]), 'PC2': float(row[1]) if X_pca.shape[1] > 1 else 0} for row in X_pca],
            'feature_weights': {}
        }

        # Add feature weights for each principal component
        for i, feature in enumerate(numeric_input.columns):
            pca_results['feature_weights'][feature] = {
                'PC1': float(pca.components_[0, i]),
                'PC2': float(pca.components_[1, i]) if pca.components_.shape[0] > 1 else 0
            }

        return pca_results
    except Exception as e:
        return {'error': str(e)}


def calculate_output_statistics(output_data):
    """Calculate statistics about the output predictions"""
    output_stats = {}

    for col in output_data.columns:
        # Get value counts
        value_counts = output_data[col].value_counts().to_dict()
        # Convert keys to strings for JSON compatibility
        value_counts = {str(k): int(v) for k, v in value_counts.items()}

        output_stats[col] = {
            'distribution': value_counts,
            'unique_values': int(output_data[col].nunique()),
            'most_common': int(output_data[col].mode()[0])
        }

    return output_stats


def generate_metrics_json(input_data, output_data):
    """Generate a comprehensive JSON with all metrics"""
    # Calculate all the metrics
    feature_importance = calculate_feature_importance(input_data, output_data)
    feature_distributions = analyze_feature_distributions(input_data, output_data)
    feature_correlations = calculate_feature_correlations(input_data)
    pca_components = calculate_pca_components(input_data)
    output_statistics = calculate_output_statistics(output_data)

    # Feature descriptions based on documentation
    feature_descriptions = {
        'age': 'Patient age in years',
        'Sex': '0: Female, 1: Male',
        'Pneumonia': '0: not known, 1: active, 2: active bilateral',
        'PH': 'Measurement from blood test',
        'DiaPr': 'Diastolic pressure (mmHg)',
        'Respiratory rate': 'Respiratory rate (f/min)',
        'SPO2': 'Percentage of arterial oxygen saturation',
        'GCS': 'Glasgow Coma Scale (3 to 15)',
        'SysPr': 'Systolic pressure (mmHg)',
        'Pulse rate': 'Pulse rate in bpm',
        'SM PY': 'Accumulated smoking in pack years',
        'smoker': '0: non smoker, 1: ex smoker, 2: active smoker, 3: social smoker',
        'ex sm years': 'Time passed since active smoking has stopped (years)',
        'hospitalizations': 'Count of registered admissions in the clinic',
        '(MT)': 'Malignant tumor: 0: not known, 1: active, 2: past'
    }

    # Output descriptions
    output_descriptions = {
        'VenDep': 'Ventilator Dependence (0: False, 1: True)',
        'ARF': 'Acute Respiratory Failure Type (0: No ARF, 1: Low O2, 2: High CO2)',
        'ARF_combo': 'ARF Occurrence (0: No ARF, 1: ARF Type 1 or 2)',
        'Mortality': 'Mortality Prediction (0: Live, 1: Deceased)'
    }

    # Helper function to handle NaN, Infinity values for JSON serialization
    def safe_float(value):
        if value is None:
            return None
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    # Create the metrics JSON
    metrics = {
        'model_summary': {
            'input_shape': {
                'rows': int(input_data.shape[0]),
                'columns': int(input_data.shape[1])
            },
            'output_shape': {
                'rows': int(output_data.shape[0]),
                'columns': int(output_data.shape[1])
            },
            'input_features': {
                column: {
                    'description': feature_descriptions.get(column, 'No description available'),
                    'dtype': str(input_data[column].dtype),
                    'min': safe_float(input_data[column].min()) if pd.api.types.is_numeric_dtype(
                        input_data[column]) else None,
                    'max': safe_float(input_data[column].max()) if pd.api.types.is_numeric_dtype(
                        input_data[column]) else None,
                    'mean': safe_float(input_data[column].mean()) if pd.api.types.is_numeric_dtype(
                        input_data[column]) else None,
                    'median': safe_float(input_data[column].median()) if pd.api.types.is_numeric_dtype(
                        input_data[column]) else None,
                    'std': safe_float(input_data[column].std()) if pd.api.types.is_numeric_dtype(
                        input_data[column]) else None,
                    'unique_values': int(input_data[column].nunique())
                } for column in input_data.columns
            },
            'output_features': {
                column: {
                    'description': output_descriptions.get(column, column),
                    'unique_values': int(output_data[column].nunique())
                } for column in output_data.columns
            },
            'output_statistics': output_statistics
        },
        'feature_importance': feature_importance,
        'feature_distributions': feature_distributions,
        'feature_correlations': feature_correlations,
        'pca_analysis': pca_components
    }

    # Add overall top features section for easy reference
    top_features = {}
    for output_col, features in feature_importance.items():
        top_features[output_col] = [f['feature'] for f in features[:5]]  # Top 5 features

    metrics['top_features'] = top_features

    return metrics


def explain_model(input_path, output_path):
    """Main function to explain the COPD model"""
    print(f"Loading data from {input_path} and {output_path}...")
    input_data, output_data = load_data(input_path, output_path)

    print("Generating metrics JSON...")
    metrics = generate_metrics_json(input_data, output_data)

    # Save to JSON file
    output_file = 'copd_model_metrics.json'

    # Custom JSON encoder to handle NaN, Infinity values
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NpEncoder, self).default(obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print(f"Done! Metrics saved to {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output predictions CSV')

    args = parser.parse_args()

    explain_model(args.input, args.output)
