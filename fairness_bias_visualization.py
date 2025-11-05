import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_json
from typing import Dict

AGE_COLORS = {
    '<40': '#66C2A5',
    '40-54': '#FC8D62',
    '55-69': '#8DA0CB',
    '70+': '#E78AC3'
}

AGE_GROUPS = ['<40', '40-54', '55-69', '70+']

SEX_COLORS = {
    '0.0': '#FF69B4',
    '1.0': '#4169E1'
}


def format_method_name(method_name: str) -> str:
    """Format method name by capitalizing and removing underscores.

    :param method_name: Method name with underscores (e.g., 'equalized_odds').
    :type method_name: str
    :returns: Formatted method name (e.g., 'Equalized Odds').
    :rtype: str
    """
    return method_name.replace('_', ' ').title()


def create_consolidated_visualization(data: Dict, demographic_key: str, output_dir: Path) -> None:
    """Create a 1x2 consolidated visualization for fairness and bias metrics.

    :param data: Dictionary containing fairness analysis results.
    :type data: Dict
    :param demographic_key: The demographic attribute to visualize ('age' or 'Sex').
    :type demographic_key: str
    :param output_dir: Directory path to save the visualization.
    :type output_dir: Path
    """
    fpr_data = {}
    prediction_data = {}

    fairness_method = data.get('metadata', {}).get('fairness_method', 'equalized_odds')
    bias_method = data.get('metadata', {}).get('bias_method', 'demographic_parity')

    fairness_method_display = format_method_name(fairness_method)
    bias_method_display = format_method_name(bias_method)

    if 'equalized_odds_metrics' in data and demographic_key in data['equalized_odds_metrics']:
        error_rates = data['equalized_odds_metrics'][demographic_key].get('error_rates_by_group', {})
        for group, rates in error_rates.items():
            if 'false_positive_rate' in rates:
                fpr_data[group] = rates['false_positive_rate']

    if 'demographic_parity_metrics' in data and demographic_key in data['demographic_parity_metrics']:
        pred_rates = data['demographic_parity_metrics'][demographic_key].get('prediction_rates_by_group', {})
        for group, rate in pred_rates.items():
            prediction_data[group] = rate

    if not fpr_data and not prediction_data:
        print(f"No data available for demographic: {demographic_key}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Fairness and Bias Analysis by {demographic_key}', fontsize=16, fontweight='bold')

    color_map = AGE_COLORS if demographic_key == 'age' else SEX_COLORS

    if fpr_data:
        groups = sorted(fpr_data.keys(), key=lambda x: (
            AGE_GROUPS.index(x) if x in AGE_GROUPS else 999) if demographic_key == 'age' else x)
        values = [fpr_data[g] for g in groups]
        colors = [color_map.get(g, '#4ECDC4') for g in groups]

        bars = axes[0].bar(groups, values, color=colors)
        axes[0].set_title(f'Fairness Calculation using {fairness_method_display}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(demographic_key)
        axes[0].set_ylabel('False Positive Rate')
        axes[0].set_ylim(0, max(values) * 1.15 if values else 1)

        for bar, val in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + (max(values) * 0.02 if values else 0.02),
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        fig.text(0.25, 0.02, 'Shows false positive rates across demographic groups; lower is better',
                 ha='center', fontsize=10, style='italic', color='#555555')
    else:
        axes[0].text(0.5, 0.5, 'No FPR data available', ha='center', va='center')
        axes[0].axis('off')

    if prediction_data:
        groups = sorted(prediction_data.keys(), key=lambda x: (
            AGE_GROUPS.index(x) if x in AGE_GROUPS else 999) if demographic_key == 'age' else x)
        values = [prediction_data[g] for g in groups]
        colors = [color_map.get(g, '#4ECDC4') for g in groups]

        bars = axes[1].bar(groups, values, color=colors)
        axes[1].set_title(f'Bias Calculation using {bias_method_display}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(demographic_key)
        axes[1].set_ylabel('Prediction Rate')
        axes[1].set_ylim(0, max(values) * 1.15 if values else 1)

        for bar, val in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                val + (max(values) * 0.02 if values else 0.02),
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        fig.text(0.75, 0.02,
                 'How often the model predicts positive outcomes; similar values across groups indicate less bias',
                 ha='center', fontsize=10, style='italic', color='#555555')
    else:
        axes[1].text(0.5, 0.5, 'No prediction data available', ha='center', va='center')
        axes[1].axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    filename = f"fairness_bias_{demographic_key.lower()}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def visualize_fairness_bias_analysis(analysis_results: Path, output_dir: Path) -> None:
    """Create consolidated visualizations for all fairness/bias metrics in the JSON results.

    :param analysis_results: Path to the JSON file with results.
    :type analysis_results: Path
    :param output_dir: Directory to save the plots.
    :type output_dir: Path
    """
    data = load_json(analysis_results)

    output_dir.mkdir(parents=True, exist_ok=True)

    if 'equalized_odds_metrics' in data:
        for demographic_key in data['equalized_odds_metrics'].keys():
            create_consolidated_visualization(data, demographic_key, output_dir)

    print(f"All visualizations saved to {output_dir}")


def main() -> None:
    """Entry point for visualizing fairness and bias analysis results.
    Parses command-line arguments for the JSON file containing metrics
    and the output directory to save the visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Visualize fairness and bias analysis results for ASCOPD model predictions"
    )
    parser.add_argument(
        "--analysis_results",
        default="fairness_analysis.json",
        help="Analysis JSON file path",
    )
    parser.add_argument(
        "--output", default="output", help="Output dir for visualizations"
    )

    args = parser.parse_args()

    visualize_fairness_bias_analysis(
        analysis_results=Path(args.analysis_results), output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
