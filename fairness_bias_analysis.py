import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Literal, Dict, List, Tuple, Any, Union
from pathlib import Path
from utils import load_csv, store_json


ANALYSIS_MAP = {
    "equalized_odds": "error_rates_by_group",
    "demographic_parity": "prediction_rates_by_group",
}


def pass_checks(
    tabular_data: pd.DataFrame,
    actual_target: pd.DataFrame,
    pred_target: pd.DataFrame,
    target_col: str,
    demographic_cols: Dict[str, str],
) -> bool:
    """Validate that input DataFrames and required columns pass basic checks.

    Args:
        tabular_data (pd.DataFrame): The main tabular dataset with demographic columns.
        actual_target (pd.DataFrame): DataFrame containing the actual target values.
        pred_target (pd.DataFrame): DataFrame containing the predicted target values.
        target_col (str): The name of the target column that must exist in both
            `actual_target` and `pred_target`.
        demographic_cols (Dict[str, str]): A mapping of demographic categories
            to corresponding column names in `tabular_data`.

    Returns:
        bool: True if all checks pass, otherwise False.

    Checks performed:
        1. None of the inputs are None.
        2. None of the DataFrames are empty.
        3. All DataFrames have the same length.
        4. `target_col` exists in both `actual_target` and `pred_target`.
        5. At least one demographic column exists in `tabular_data`.
    """

    if (tabular_data is None) or (actual_target is None) or (pred_target is None):
        return False

    if tabular_data.empty or actual_target.empty or pred_target.empty:
        return False

    if not (len(tabular_data) == len(actual_target) == len(pred_target)):
        return False

    if (target_col not in actual_target.columns) or (
        target_col not in pred_target.columns
    ):
        return False

    if not any(col in tabular_data.columns for col in demographic_cols.values()):
        return False

    return True


def prepare_data_for_analysis(
    tabular_data: pd.DataFrame,
    actual_target: pd.DataFrame,
    pred_target: pd.DataFrame,
    target_col: str,
    demographic_cols: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare demographic, actual target, and predicted target DataFrames for analysis.

    Args:
        tabular_data (pd.DataFrame): The main dataset containing demographic columns.
        actual_target (pd.DataFrame): DataFrame containing the actual target values.
        pred_target (pd.DataFrame): DataFrame containing the predicted target values.
        target_col (str): The column name for the target variable in both
            `actual_target` and `pred_target`.
        demographic_cols (Dict[str, str]): Mapping of demographic keys to column names
            in `tabular_data`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - dem_df: DataFrame containing the available demographic columns.
            - actual_df: DataFrame with a single column `actual`.
            - pred_df: DataFrame with a single column `pred`.
    """

    dem_df = tabular_data[
        [col for col in demographic_cols.values() if col in tabular_data.columns]
    ]

    actual_df = actual_target[[target_col]].rename(columns={target_col: "actual"})
    pred_df = pred_target[[target_col]].rename(columns={target_col: "pred"})

    return dem_df, actual_df, pred_df


def age_to_cat(age_col: pd.Series) -> pd.Series:
    """Convert a numeric age column into categorical age groups.

    Age bins: <40, 40-54, 55-69, 70+

    Args:
        age_col (pd.Series): Numeric age values.

    Returns:
        pd.Series: A categorical series with age groups as labels.
    """

    age_bins = [0, 40, 55, 70, 120]
    age_labels = ["<40", "40-54", "55-69", "70+"]

    return pd.cut(
        age_col, bins=age_bins, labels=age_labels, include_lowest=True, right=True
    )


def concat_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames along columns (axis=1).

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames to concatenate.

    Returns:
        pd.DataFrame: A single DataFrame with columns from all input DataFrames.
    """

    return pd.concat(df_list, axis=1)


def compute_fpr(true: pd.Series, pred: pd.Series) -> float:
    """Compute False Positive Rate (FPR) for binary classification.

    Args:
        true (pd.Series): Ground truth binary labels (0/1).
        pred (pd.Series): Predicted binary labels (0/1).

    Returns:
        float: False Positive Rate = FP / (FP + TN)
    """

    cm = confusion_matrix(true, pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return fpr


def calculate_metric(
    data: pd.DataFrame,
    demographic_cols: Dict[str, str],
    analysis: Literal["equalized_odds", "demographic_parity"],
) -> Dict[str, Any]:
    """Calculate fairness metrics (Equalized Odds or Demographic Parity) per demographic group.

    Args:
        data (pd.DataFrame): DataFrame containing "actual" and "pred" and demographic columns.
        demographic_cols (Dict[str, str]): Mapping of demographic keys to column names in `data`.
        analysis (Literal["equalized_odds", "demographic_parity"]): Type of analysis to perform.

    Returns:
        Dict[str, Any]: Nested dictionary with metrics per demographic column and group.
    """

    metrics = {}

    for col in demographic_cols.values():
        if col not in data.columns:
            continue

        metrics[col] = {ANALYSIS_MAP.get(analysis): {}}

        unique_vals = data[col].unique().tolist()

        for val in unique_vals:
            if pd.isna(val):
                continue

            spec_cat_data = data[data[col] == val].copy()

            # Check which analysis is requested
            if analysis == "equalized_odds":
                fpr = compute_fpr(
                    true=spec_cat_data["actual"], pred=spec_cat_data["pred"]
                )
                analysis_results = {
                    "false_positive_rate": fpr,
                }
            elif analysis == "demographic_parity":
                analysis_results = spec_cat_data["pred"].mean()

            metrics[col][ANALYSIS_MAP.get(analysis)][f"{val}"] = analysis_results

    return metrics


def fairness_bias_analysis(
    tabular_data: Union[str, Path],
    actual_target: Union[str, Path],
    pred_target: Union[str, Path],
    target_col: str,
    output_path: Union[str, Path],
    demographic_cols: Dict[str, str] = {"age": "age", "gender": "Sex"},
) -> Dict[str, Any]:
    """Run fairness and bias analysis using Equalized Odds and Demographic Parity.

    This function loads tabular data, actual target labels, and predicted target labels,
    validates them, prepares them for analysis, and computes fairness metrics across
    demographic subgroups. The results are then stored as a JSON file.

    Args:
        tabular_data (Union[str, Path]): Path to the tabular dataset CSV.
        actual_target (Union[str, Path]): Path to CSV containing actual target labels.
        pred_target (Union[str, Path]): Path to CSV containing predicted target labels.
        target_col (str): Column name of the target variable in both actual an
        output_path (Union[str, Path]): Path where the results JSON will be saved.
        demographic_cols (Dict[str, str], optional): Mapping of demographic keys to their corresponding
                        column names in the tabular data. Defaults to {"age": "age", "gender": "Sex"}.
    """

    # Init results
    results = {"equalized_odds_metrics": {}, "demographic_parity_metrics": {}}

    # Load data
    tabular_data = load_csv(tabular_data)
    actual_target = load_csv(actual_target)
    pred_target = load_csv(pred_target)

    if not pass_checks(
        tabular_data, actual_target, pred_target, target_col, demographic_cols
    ):
        raise ValueError(f"Inconsistent data provided. Check again the data provided.")

    tabular_data, actual_target, pred_target = prepare_data_for_analysis(
        tabular_data=tabular_data,
        actual_target=actual_target,
        pred_target=pred_target,
        target_col=target_col,
        demographic_cols=demographic_cols,
    )

    # Convert age columns to categorical
    if demographic_cols.get("age") in tabular_data.columns:
        tabular_data[demographic_cols.get("age")] = age_to_cat(
            tabular_data[demographic_cols.get("age")]
        )

    # Concatenated df with demographic columns and 'actual', 'pred' columns
    concat_data = concat_dfs([tabular_data, actual_target, pred_target])

    # Equalized odds
    results["equalized_odds_metrics"] = calculate_metric(
        data=concat_data, demographic_cols=demographic_cols, analysis="equalized_odds"
    )

    # Demographic parity
    results["demographic_parity_metrics"] = calculate_metric(
        data=concat_data,
        demographic_cols=demographic_cols,
        analysis="demographic_parity",
    )

    # Store results
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store_json(data=results, path=output_path)
    print(f"Fairness - Bias analysis completed. Results saved to {output_path}")


def main():
    """CLI entry point for fairness and bias analysis of ASCOPD model predictions.

    This function parses command-line arguments for tabular data, actual and predicted
    targets, demographic column names, and output path. It then runs the fairness and
    bias analysis pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Analyze fairness and bias in ASCOPD model predictions"
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
        "--pred_target", required=True, help="Path to predicted target data CSV file"
    )
    parser.add_argument(
        "--target_col",
        required=True,
        choices=["VenDep", "ARF", "Mortality"],
        help="Name of target column name in actual and predicted target CSV file",
    )
    parser.add_argument(
        "--age_col",
        required=False,
        default="age",
        help="Name of age column in the tabular data",
    )
    parser.add_argument(
        "--gender_col",
        required=False,
        default="Sex",
        help="Name of gender column in the tabular data",
    )
    parser.add_argument(
        "--output",
        default="output/fairness_analysis.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    fairness_bias_analysis(
        tabular_data=Path(args.tabular_data),
        actual_target=Path(args.actual_target),
        pred_target=Path(args.pred_target),
        target_col=args.target_col,
        demographic_cols={"age": args.age_col, "gender": args.gender_col},
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
