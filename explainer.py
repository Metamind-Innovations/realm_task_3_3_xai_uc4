import argparse
from typing import Dict, Literal, List, Any, Tuple, Union
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
import dice_ml
from ASCOPD_model import DockerModelWrapper
from utils import is_between, load_csv, store_json, concat_dfs


RANDOM_STATE = 2025
NUMBER_REPEATS = 5
CATEGORICAL_COLUMNS = ["Sex"]
NUMERICAL_COLUMNS = [
    "age",
    "Hemoglobin",
    "Platelet Count",
    "Urea Nitrogen",
    "Creatinine",
    "Sodium",
    "Potassium",
    "ALT",
    "Glucose",
    "PH",
    "pO2",
    "pCO2",
    "Bicarbonate",
    "Absolute Neutrophil Count",
    "Absolute Monocyte Count",
    "Eosinophils",
    "Absolute Lymphocyte Count",
]


def pass_checks(
    tabular_data: pd.DataFrame,
    actual_target: pd.DataFrame,
    target_col: str,
) -> bool:
    """Validate that input DataFrames and required columns pass basic checks.

    Args:
        tabular_data (pd.DataFrame): The main tabular dataset.
        actual_target (pd.DataFrame): DataFrame containing the actual target values.
        target_col (str): The name of the target column that must exist in `actual_target`.

    Returns:
        bool: True if all checks pass, otherwise False.

    Checks performed:
        1. None of the inputs are None.
        2. None of the DataFrames are empty.
        3. DataFrames have the same length.
        4. `target_col` exists in `actual_target`.
    """

    if (tabular_data is None) or (actual_target is None):
        return False

    if tabular_data.empty or actual_target.empty:
        return False

    if not len(tabular_data) == len(actual_target):
        return False

    if target_col not in actual_target.columns:
        return False

    return True


def select_method(sens: float) -> Literal["feature_permutation", "counterfactuals"]:
    """
    Select an explainability method based on sensitivity level.
    Values < 0.5 -> "feature_permutation".
    Values >= 0.5 -> "counterfactuals".

    Args:
        sens (float): Sensitivity parameter in the range [0, 1].

    Returns:
        Literal["feature_permutation", "counterfactuals"]:
            The name of the selected explainability method.
    """

    return "feature_permutation" if sens < 0.5 else "counterfactuals"


def data_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a DataFrame.
    This function fills missing values in numerical columns with the median,
    and in categorical columns with the mode (most frequent value).

    Args:
        data (pd.DataFrame): Input DataFrame containing numerical and categorical columns.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """

    return data.apply(
        lambda col: (
            col.fillna(col.median())
            if col.name in NUMERICAL_COLUMNS
            else col.fillna(col.mode().iloc[0])
        ),
        axis=0,
    )


def feature_permutation(
    X: pd.DataFrame, y: pd.DataFrame, model: DockerModelWrapper
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute feature importance using permutation importance with a Dockerized model.
    This function wraps `sklearn.inspection.permutation_importance` to estimate how much
    each feature contributes to model performance. Missing values in X are
    imputed (median for numerical, mode for categorical). It evaluates feature importance
    with F1 scoring, and aggregates results across multiple repeats.

    Args:
        X (pd.DataFrame): Input features used for prediction.
        y (pd.DataFrame): True target values corresponding to `X`.
        model (DockerModelWrapper): Dockerized model wrapper.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            - A list of dictionaries with feature names and their mean permutation importance.
              Example: `[{"Feature": "age", "Permutation_Importance": 0.12}, ...]`
            - A dictionary containing the full permutation importance results from sklearn,
              including per-repeat scores (`importances`), mean (`importances_mean`),
              and standard deviation (`importances_std`).
    """

    X_imputed = data_imputation(X)

    importance_results = permutation_importance(
        estimator=model,
        X=X_imputed,
        y=y,
        scoring="f1",
        n_repeats=NUMBER_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    summary_results = list(
        map(
            lambda col, imp: {"Feature": col, "Permutation_Importance": imp},
            X.columns,
            importance_results.importances_mean,
        )
    )

    importance_results = dict(importance_results)
    importance_results["features"] = importance_results.columns.to_list()

    return summary_results, importance_results


def counterfactuals(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_col: str,
    model: DockerModelWrapper,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
    """
    Generate counterfactual explanations and compute global and local feature importance
    using DiCE for a given dataset and model.

    The function:
    1. Merges features (X) and target (y) into a single dataframe.
    2. Creates a DiCE data interface with numerical and categorical features.
    3. Wraps the model in a DiCE model interface.
    4. Imputes missing values in X (median for numerical, mode for categorical).
    5. Computes global feature importance from generated counterfactuals.
    6. Returns global and local importance results.

    Args:
        X (pd.DataFrame): Input features dataframe.
        y (pd.DataFrame): Target labels dataframe.
        target_col (str): Name of the target column in y.
        model (DockerModelWrapper): Pretrained Dockerized model wrapper.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
            - summary_results: List of dicts with global feature importance:
                [{"Feature": feature_name, "Counterfactuals_Importance": importance_value}, ...]
            - importance_results: Local feature importance per query instance.
    """

    data_merged = concat_dfs([X, y])

    data_interface = dice_ml.Data(
        dataframe=data_merged,
        continuous_features=NUMERICAL_COLUMNS,
        categorical_features=CATEGORICAL_COLUMNS,
        outcome_name=target_col,
    )

    model = dice_ml.Model(model=model, backend="PYT")

    explainer = dice_ml.Dice(
        model_interface=model,
        data_interface=data_interface,
        method="random",
    )

    X_imputed = data_imputation(X)

    global_importance_score_obj = explainer.global_feature_importance(
        X_imputed,
        total_CFs=10,
        posthoc_sparsity_param=None,
        desired_class="opposite",
    )

    summary_results = list(
        map(
            lambda x: {"Feature": x[0], "Counterfactuals_Importance": x[1]},
            global_importance_score_obj.summary_importance.items(),
        )
    )

    importance_results = global_importance_score_obj.local_importance

    return summary_results, importance_results


def run_explainability_analysis(
    tabular_data: Union[str, Path],
    actual_target: Union[str, Path],
    target_col: str,
    output_dir: Union[str, Path],
    sensitivity: float,
) -> List[Dict[str, Any]]:
    """
    Runs an explainability analysis on tabular data using either permutation importance
    or counterfactuals, depending on the specified sensitivity.
    The function:
        1. Loads tabular feature data and actual target labels.
        2. Checks data consistency.
        3. Selects the explainability method based on sensitivity.
           - Low sensitivity (<0.5): Feature permutation
           - High sensitivity (>=0.5): Counterfactuals
        4. Configures the Dockerized model.
        5. Computes feature importances.
        6. Stores detailed results (if applicable) and summary results as JSON.

    Args:
        tabular_data (Union[str, Path]): Path to CSV file containing input features.
        actual_target (Union[str, Path]): Path to CSV file containing true target labels.
        target_col (str): Name of the target column for the model.
        output_dir (Path): Directory where results will be stored.
        sensitivity (float): Sensitivity parameter (0 to 1) controlling the method selection.
    """

    # Init results
    results = [{}]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tabular_data = load_csv(tabular_data)
    actual_target = load_csv(actual_target)

    if not pass_checks(tabular_data, actual_target, target_col):
        raise ValueError(f"Inconsistent data provided. Check again the data provided.")

    method = select_method(sensitivity)
    print(f"Using method: {method} based on sensitivity: {sensitivity}")

    # Perform analysis
    if method == "feature_permutation":
        model = DockerModelWrapper(target=target_col)
        results, detailed_results = feature_permutation(
            X=tabular_data, y=actual_target, model=model
        )

    elif method == "counterfactuals":
        model = DockerModelWrapper(
            target=target_col,
            features=tabular_data.columns.to_list(),
            internal_data_type="tensor",
        )
        results, detailed_results = counterfactuals(
            X=tabular_data, y=actual_target, target_col=target_col, model=model
        )

    # Store results to output dir
    store_json(
        data=detailed_results,
        path=output_dir.joinpath(f"{method}_analysis_detailed_results.json"),
    )
    store_json(data=results, path=output_dir.joinpath(f"{method}_analysis.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and explain feature importance"
    )

    parser.add_argument(
        "--tabular_data", required=True, help="Path to tabular data CSV file"
    )
    parser.add_argument(
        "--actual_target",
        required=True,
        help="Path to groundtruth target data CSV file",
    )
    parser.add_argument(
        "--target_col",
        required=True,
        choices=["VenDep", "ARF", "Mortality"],
        help="Name of target column name in actual and predicted target CSV file",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output dir for results JSON files",
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="Sensitivity value (0-1): <0.5 uses permutation feature importance, >=0.5 uses counterfactuals",
    )
    args = parser.parse_args()

    if not is_between(x=args.sensitivity):
        raise ValueError("Sensitivity must be between 0 and 1")

    run_explainability_analysis(
        tabular_data=args.tabular_data,
        actual_target=args.actual_target,
        target_col=args.target_col,
        output_dir=Path(args.output),
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
