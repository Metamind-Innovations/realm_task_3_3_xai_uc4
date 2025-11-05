from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from utils import load_json


def is_feature_permulation_results(data: list) -> bool:
    """Check if the input list corresponds to feature permutation results.

    :param data: Input list of dicts, where the elements are expected to contain
                 the string 'Permutation_Importance' if it is permutation results.
    :type data: list
    :returns: True if 'Permutation_Importance' is found in the first dict of the list, False otherwise.
    :rtype: bool
    """

    return (
        True
        if ("Permutation_Importance" in data[0] and "Feature" in data[0])
        else False
    )


def is_counterfactuals(data: list) -> bool:
    """Check if the input list corresponds to counterfactuals results.

    :param data: Input list of dicts, where the elements are expected to contain
                 the string 'Counterfactuals_Importance' if it is counterfactuals results.
    :type data: list
    :returns: True if 'Counterfactuals_Importance' is found in the first dict of the list, False otherwise.
    :rtype: bool
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
        figsize: tuple[int, int] = (8, 4),
        output_path: str = None,
        show: bool = False,
        explanation_text: str = None,
) -> None:
    """Plot a horizontal bar chart of feature importances with annotations.
    Features and importances can be provided as lists or pandas Series.
    The function adds a small space after the longest bar and annotates each bar
    with its importance value.

    :param features: Names of the features.
    :type features: list or pd.Series
    :param importances: Importance values corresponding to the features.
    :type importances: list or pd.Series
    :param title: Plot title. Defaults to "".
    :type title: str, optional
    :param xlabel: Label for the x-axis. Defaults to "".
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis. Defaults to "".
    :type ylabel: str, optional
    :param figsize: Figure size in inches. Defaults to (6, 4).
    :type figsize: tuple[int, int], optional
    :param output_path: File path to save the plot. If None, does not save.
    :type output_path: str, optional
    :param show: Whether to display the plot interactively. Defaults to False.
    :type show: bool, optional
    :param explanation_text: Explanatory text to display below the plot. Defaults to None.
    :type explanation_text: str, optional
    """

    plt.figure(figsize=figsize)

    bars = plt.barh(features, importances, color="steelblue")
    plt.gca().invert_yaxis()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xmin = min(importances)
    xmax = max(importances)
    x_range = xmax - xmin

    xlim_min = xmin - (x_range * 0.12) if xmin < 0 else 0
    xlim_max = xmax + (x_range * 0.12) if xmax > 0 else 0

    plt.xlim(xlim_min, xlim_max)

    for bar, val in zip(bars, importances):
        if val >= 0:
            text_x = val + (x_range * 0.01)
            text_ha = "left"
        else:
            text_x = val - (x_range * 0.01)
            text_ha = "right"

        plt.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha=text_ha,
        )

    if explanation_text:
        plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=10,
                    style='italic', color='#555555', wrap=True)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
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

    :param analysis_results: Path to the JSON file containing analysis results.
    :type analysis_results: str or Path
    :param output_dir: Directory where the generated plots will be saved.
    :type output_dir: str or Path
    :raises ValueError: If the input data does not contain expected keys for permutation feature
                        importance or counterfactual analysis.
    """

    raw_data = load_json(analysis_results)

    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = None
    method_from_metadata = None

    if isinstance(raw_data, dict) and "results" in raw_data:
        metadata = raw_data.get("metadata", {})
        sensitivity = metadata.get("sensitivity")
        method_from_metadata = metadata.get("method")
        analysis_results = raw_data["results"]
    else:
        analysis_results = raw_data

    if is_feature_permulation_results(analysis_results):
        method = "feature_permutation" if method_from_metadata is None else method_from_metadata
        sort_column = "Permutation_Importance"
        explanation_text = ("Positive values indicate features that improve model performance; "
                            "Negative values indicate features that degrade model performance when used by the model.")
    elif is_counterfactuals(analysis_results):
        method = "counterfactuals" if method_from_metadata is None else method_from_metadata
        sort_column = "Counterfactuals_Importance"
        explanation_text = (
            "Shows which characteristics would need to change to alter the ventilator dependence prediction; "
            "higher values indicate features that are key drivers of the model's decisions")
    else:
        raise ValueError(
            "Invalid data passed. Expected keys were not found in analysis results."
        )

    analysis_results = pd.DataFrame(analysis_results)
    analysis_results.sort_values(by=sort_column, ascending=False, inplace=True)

    features = analysis_results["Feature"].to_list()
    importances = analysis_results[sort_column].to_list()

    if sensitivity is not None and method_from_metadata is not None:
        formatted_method = method_from_metadata.replace('_', ' ').title()
        title = f'Sensitivity [0,1]: {sensitivity}, Methodology: {formatted_method}'
    else:
        formatted_method = method.replace('_', ' ').title()
        title = f"{formatted_method} Importance Analysis"

    plot_importances(
        features=features,
        importances=importances,
        title=title,
        xlabel="Importance",
        ylabel="Feature",
        output_path=output_dir.joinpath(f"{method.replace('_', ' ').title()} Importance.png"),
        explanation_text=explanation_text,
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
