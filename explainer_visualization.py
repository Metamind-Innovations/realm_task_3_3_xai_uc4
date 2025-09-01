from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from utils import load_json


def is_feature_permulation_results(data: list) -> bool:
    """Check if the input list corresponds to feature permutation results.

    Args:
        data (list): Input list of dicts, where the elements are expected to contain
                     the string 'Permutation_Importance' if it is permutation results.

    Returns:
        bool: True if 'Permutation_Importance' is found in the first dict of the list, False otherwise.
    """

    return (
        True
        if ("Permutation_Importance" in data[0] and "Feature" in data[0])
        else False
    )


def is_counterfactuals(data: list) -> bool:
    """Check if the input list corresponds to counterfactuals results.

    Args:
        data (list): Input list of dicts, where the elements are expected to contain
                     the string 'Counterfactuals_Importance' if it is counterfactuals results.

    Returns:
        bool: True if 'Counterfactuals_Importance' is found in the first dict of the list, False otherwise.
    """

    return (
        True
        if ("Counterfactuals_Importance" in data[0] and "Feature" in data[0])
        else False
    )


def plot_importances(
    features: Union[list, pd.Series],
    importances: Union[list, pd.Series],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (6, 4),
    output_path: str = None,
    show: bool = False,
) -> None:
    """Plot a horizontal bar chart of feature importances with annotations.
    Features and importances can be provided as lists or pandas Series.
    The function adds a small space after the longest bar and annotates each bar
    with its importance value.

    Args:
        features (list or pd.Series): Names of the features.
        importances (list or pd.Series): Importance values corresponding to the features.
        title (str, optional): Plot title. Defaults to "".
        xlabel (str, optional): Label for the x-axis. Defaults to "".
        ylabel (str, optional): Label for the y-axis. Defaults to "".
        figsize (tuple[int, int], optional): Figure size in inches. Defaults to (6, 4).
        output_path (str, optional): File path to save the plot. If None, does not save.
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
    """

    plt.figure(figsize=figsize)

    bars = plt.barh(features, importances, color="steelblue")
    plt.gca().invert_yaxis()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xmax = max(importances) * 1.12
    plt.xlim(0, xmax)

    for bar, val in zip(bars, importances):
        plt.text(
            val + (xmax * 0.01),  # slight offset to the right of the bar
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_explanations(analysis_results: Path, output_dir: Path) -> None:
    """Visualize permutation feature importance or counterfactual explanations and save plots.
    This function loads analysis results from a JSON file, determines whether the results
    are from permutation feature importance or counterfactual analysis, sorts the features
    by importance, and creates a horizontal bar plot with annotated importance values.

    Args:
        analysis_results (str or Path): Path to the JSON file containing analysis results.
        output_dir (str or Path): Directory where the generated plots will be saved.

    Raises:
        ValueError: If the input data does not contain expected keys for permutation feature
                    importance or counterfactual analysis.
    """

    analysis_results = load_json(analysis_results)

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_feature_permulation_results(analysis_results):
        method = "Permutation Feature"
        sort_column = "Permutation_Importance"
    elif is_counterfactuals(analysis_results):
        method = "Counterfactuals"
        sort_column = "Counterfactuals_Importance"
    else:
        raise ValueError(
            "Invalid data passed. Expected keys were not found in analysis results."
        )

    analysis_results = pd.DataFrame(analysis_results)
    analysis_results.sort_values(by=sort_column, ascending=False, inplace=True)

    features = analysis_results["Feature"].to_list()
    importances = analysis_results[sort_column].to_list()

    plot_importances(
        features=features,
        importances=importances,
        title=f"{method} Importance Analysis",
        xlabel="Importance",
        ylabel="Feature",
        output_path=output_dir.joinpath(f"{method}_Importance.png"),
    )

    print(f"Plots stored in {output_dir}")


def main() -> None:
    """Entry point for visualizing explainability analysis results.
    Parses command-line arguments to specify the analysis results JSON file
    and the output directory, then calls `visualize_explanations` to generate
    and save the feature importance plots.
    """

    parser = argparse.ArgumentParser(
        description="Visualize explainability analysis results for ASCOPD model predictions"
    )
    parser.add_argument(
        "--analysis_results",
        help="Analysis JSON file path",
    )
    parser.add_argument(
        "--output", default="output", help="Output dir for visualizations"
    )

    args = parser.parse_args()

    visualize_explanations(
        analysis_results=Path(args.analysis_results), output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
