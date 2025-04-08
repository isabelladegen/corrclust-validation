from itertools import chain
from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, CorrType
from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, DataCompleteness, \
    SYNTHETIC_DATA_DIR, get_data_dir, get_clustering_quality_multiple_data_variants_result_folder, ResultsType, \
    get_image_results_path
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import fontsize, reset_matplotlib, Backends


def plot_mae_with_statistics(mae_results: {}, lengths: [int], threshold: float = 0.1,
                             backend: str = Backends.none.value):
    """
    Plot mean MAE with median and IQR for different segment lengths.
    :param mae_results :Dictionary with keys 'mean', '50%', '25%', '75%', each containing a list of values
        corresponding to the segment lengths.
    :param lengths : list of segment lengths
    :param threshold : value for red threshold line
    :param backend : see Backends
    """
    reset_matplotlib(backend)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean MAE
    ax.plot(lengths, mae_results['mean'], 'o-', color='blue', label='Mean MAE')

    # Plot median MAE
    ax.plot(lengths, mae_results['50%'], 's--', color='green', label='Median MAE')

    # Create an array for x-axis values
    x = np.array(lengths)

    # Plot IQR as a shaded region
    ax.fill_between(x, mae_results['25%'], mae_results['75%'], color='gray', alpha=0.3, label='IQR (25%-75%)')

    # Set labels and title
    ax.set_xlabel('Segment Length', fontsize=fontsize)
    ax.set_ylabel('MAE', fontsize=fontsize)

    # Set x-axis to log scale since segment lengths span a wide range
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add horizontal threshold line at y=0.05
    ax.axhline(y=0.1, color='red', linestyle='-', linewidth=1.5, label='Threshold')

    # Add a legend
    ax.legend(fontsize=fontsize)

    # Improve x-axis tick labels
    ax.set_xticks(lengths)
    ax.set_xticklabels(lengths)

    plt.tight_layout()
    plt.show()
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

    # load data
    complete = DataCompleteness.complete
    nn_100 = DescribeSubjectsForDataVariant(wandb_run_file=ds, overall_ds_name=overall_ds_name,
                                            data_type=normal, data_dir=get_data_dir(data_dir, complete),
                                            load_data=True, additional_corr=[CorrType.pearson, CorrType.kendall])

    # lengths to plot
    lengths = [10, 15, 20, 30, 60, 80, 100, 200, 400, 600, 800]
    for cor in correlations:
        mae_results = nn_100.mean_mae_for_segment_lengths(lengths)

        # plot results
        fig = plot_mae_with_statistics(mae_results, lengths, Backends.visible_tests.value)

        folder_name = get_clustering_quality_multiple_data_variants_result_folder(
            results_type=ResultsType.dataset_description,
            overall_dataset_name=overall_ds_name,
            results_dir=data_dir,
            distance_measure="")

        # safe results
        df = mae_to_df(mae_results, lengths)
        df.to_csv(str(path.join(folder_name, cor + "_minimal_segment_lengths_mae_stats.csv")))

        # safe image
        file_name = get_image_results_path(folder_name, cor + '_minimal_segment_lengths.png')
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
