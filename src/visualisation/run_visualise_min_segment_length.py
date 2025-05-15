from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.gridspec import GridSpec

from src.data_generation.generate_synthetic_segmented_dataset import CorrType
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, DataCompleteness, \
    SYNTHETIC_DATA_DIR, get_data_dir, get_clustering_quality_multiple_data_variants_result_folder, ResultsType, \
    get_image_results_path, base_dataset_result_folder_for_type
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import fontsize, reset_matplotlib, Backends


def plot_combined_mae_stats(all_mae_results, lengths, labels, correlations, threshold=0.1, backend=Backends.none.value):
    """
    Plot mean MAE with median and IQR for different segment lengths for all correlation types in one figure.

    :param all_mae_results: Dictionary with correlation types as keys, each containing mae_results
    :param lengths: list of segment lengths
    :param labels: list of labels to show for ticks
    :param correlations: list of correlation types
    :param threshold: value for red threshold line
    :param backend: see Backends
    """
    reset_matplotlib(backend)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)

    # Create axes using the GridSpec
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2])]

    # Share y-axis among all subplots
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])

    # Plot each correlation type in its subplot
    for i, cor in enumerate(correlations):
        ax = axes[i]
        mae_results = all_mae_results[cor]

        # Set subplot title
        ax.set_title(cor.capitalize(), fontsize=fontsize, fontweight='bold')

        # Plot mean MAE
        ax.plot(lengths, mae_results['mean'], 'o-', color='blue', label='Mean MAE')

        # Plot median MAE
        ax.plot(lengths, mae_results['50%'], 's--', color='green', label='Median MAE')

        # Create an array for x-axis values
        x = np.array(lengths)

        # Plot IQR as a shaded region
        ax.fill_between(x, mae_results['25%'], mae_results['75%'], color='gray', alpha=0.3, label='IQR (25%-75%)')

        # Set labels and title
        ax.set_xlabel('Observations', fontsize=fontsize)
        ax.set_ylim(0.0, 0.25)  # ensure consistent y axis

        # Only add y-axis label to the first subplot
        if i == 0:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.set_ylabel('MAE', fontsize=fontsize)
        else:
            # Keep y-axis ticks but remove labels for other subplots
            ax.tick_params(axis='y', which='both', labelleft=False)

        # Set x-axis to log scale since segment lengths span a wide range
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Add horizontal threshold line at y=threshold
        ax.axhline(y=threshold, color='red', linestyle='-', linewidth=1.5, label='Threshold')

        # Add a legend only to the first subplot to save space
        if i == 0:
            ax.legend(fontsize=fontsize)

        # avoid showing all labels
        ax.set_xticks(lengths) # all lengths have a tick
        tick_labels = []
        for tick in lengths:
            if tick in labels:
                tick_labels.append(str(tick))
            else:
                tick_labels.append('')  # Empty string for no label
        ax.set_xticklabels(tick_labels)

    plt.tight_layout()

    return fig


def mae_to_df(mae_results, segment_lengths):
    # Create a dictionary for the DataFrame
    df_dict = {'segment_length': segment_lengths}

    # Add each statistic to the dictionary
    for stat in ['mean', '50%', '25%', '75%']:
        if stat in mae_results:
            df_dict[stat] = mae_results[stat]

    # Create DataFrame
    return pd.DataFrame(df_dict)


if __name__ == "__main__":
    # visualise min segment length for different correlations
    root_result_dir = ROOT_RESULTS_DIR
    ds = GENERATED_DATASETS_FILE_PATH
    data_dir = SYNTHETIC_DATA_DIR
    overall_ds_name = "n30"
    correlations = [CorrType.spearman, CorrType.pearson, CorrType.kendall]

    normal = SyntheticDataType.non_normal_correlated

    folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
    file_name = str(path.join(folder, "spearman_kendall_pearson_minimal_segment_lengths.png"))

    # load data
    complete = DataCompleteness.complete
    nn_100 = DescribeSubjectsForDataVariant(wandb_run_file=ds, overall_ds_name=overall_ds_name,
                                            data_type=normal, data_dir=get_data_dir(data_dir, complete),
                                            load_data=True, additional_corr=[CorrType.pearson, CorrType.kendall])

    # lengths to plot
    lengths = [10, 15, 20, 30, 60, 80, 100, 200, 400, 600, 800]
    labels = [10, 20, 30, 60, 100, 200, 400, 800] # fewer to not overlap

    all_mae_results = {}
    # Calculate MAE results for each correlation type
    for cor in correlations:
        all_mae_results[cor] = nn_100.mean_mae_for_segment_lengths(lengths, cor_type=cor)

    # Create combined figure
    fig = plot_combined_mae_stats(all_mae_results, lengths, labels, correlations, threshold=0.1,
                                  backend=Backends.visible_tests.value)

    folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
    file_name = str(path.join(folder, "spearman_kendall_pearson_minimal_segment_lengths.png"))
    fig.savefig(file_name, dpi=300, bbox_inches='tight')

    # Save the combined figure
    folder_name = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.dataset_description,
        overall_dataset_name=overall_ds_name,
        results_dir=data_dir,
        distance_measure="")

    for cor in correlations:
        df = mae_to_df(all_mae_results[cor], lengths)
        df.to_csv(str(path.join(folder_name, cor + "_minimal_segment_lengths_mae_stats.csv")))
