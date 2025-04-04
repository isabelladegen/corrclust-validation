import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, get_clustering_quality_multiple_data_variants_result_folder, ResultsType, \
    get_image_results_path, INTERVAL_HISTOGRAMS
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib, fontsize


def plot_gap_histograms(gaps_partial: [], gaps_sparse: [], unit: str = 'seconds', n_subject: int = 30,
                        backend=Backends.none.value):
    """
    Create histograms of time gaps for 70% and 10% data.

    Parameters:
    -----------
    :param gaps_partial : Time gaps for 70% data
    :param gaps_sparse: Time gaps for 10% data
    :param unit : 'seconds' or 'minutes'
    :param n_subject : int, how many subjects are the data from (will divide the counts for the freuquency by that
    """
    # Convert to minutes if specified
    if unit == 'minutes':
        gaps_partial = gaps_partial / 60
        gaps_sparse = gaps_sparse / 60
        x_label = 'Minutes'
    else:
        x_label = 'Seconds'

    # Setup plt
    reset_matplotlib(backend)

    # Create figure with two subplots in one row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = sns.color_palette("husl", n_colors=2)

    # Calculate statistics
    mean_70 = np.mean(gaps_partial)
    median_70 = np.median(gaps_partial)
    max_70 = np.max(gaps_partial)
    p95_70 = np.percentile(gaps_partial, 95)

    mean_10 = np.mean(gaps_sparse)
    median_10 = np.median(gaps_sparse)
    max_10 = np.max(gaps_sparse)
    p95_10 = np.percentile(gaps_sparse, 95)

    # For display filter gaps to 95%
    gaps_partial = np.array(gaps_partial)[gaps_partial <= p95_70]
    gaps_sparse = np.array(gaps_sparse)[gaps_sparse <= p95_10]

    def format_per_individual(x, pos):
        # First divide by the number of subjects
        x_per_individual = x / n_subject

        # Then apply the k-formatting
        if x_per_individual >= 1e6:
            return f'{x_per_individual * 1e-6:.0f}M'
        elif x_per_individual >= 1e3:
            return f'{x_per_individual * 1e-3:.0f}k'
        else:
            return f'{x_per_individual:.0f}'

    # Plot histogram for 70% data
    max_partial = int(np.ceil(p95_70))
    bins_partial = np.arange(0.5, max_partial + 1.5, 1)
    sns.histplot(gaps_partial,
                 ax=ax1,
                 kde=False,
                 bins=bins_partial,
                 color=colors[0],
                 edgecolor=colors[0],  # Same color as fill for border
                 linewidth=2,  # Thicker border for emphasis
                 alpha=0.5,
                 stat='count')
    ax1.set_title("Partial 70%", fontsize=fontsize, fontweight='bold')
    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax1.set_ylabel('Avg. Count', fontsize=fontsize)
    ax1.set_xticks(np.arange(1, max_partial + 1))

    # Add statistics annotations for 70% data
    stats_text_70 = f'Mean: {mean_70:.2f}\nMedian: {median_70:.1f}\n95th %-ile: {p95_70:.1f}\nMax: {max_70:.0f}'
    ax1.text(0.95, 0.95, stats_text_70, transform=ax1.transAxes,
             ha='right', va='top', fontsize=fontsize,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_per_individual))

    # Plot histogram for 10% data
    max_sparse = int(np.ceil(p95_10))
    bins_sparse = np.arange(0.5, max_sparse + 1.5, 1)
    sns.histplot(gaps_sparse,
                 ax=ax2,
                 kde=False,
                 bins=bins_sparse,
                 color=colors[1],
                 edgecolor=colors[1],  # Same color as fill for border
                 linewidth=2,  # Thicker border for emphasis
                 alpha=0.5,
                 stat='count')
    ax2.set_title("Sparse 10%", fontsize=fontsize, fontweight='bold')
    ax2.set_xlabel(x_label, fontsize=fontsize)
    ax2.set_ylabel('', fontsize=fontsize)
    ax2.set_xticks(np.arange(1, max_sparse + 1, 5))

    # Add statistics annotations for 10% data
    stats_text_10 = f'Mean: {mean_10:.2f}\nMedian: {median_10:.1f}\n95th %-ile: {p95_10:.1f}\nMax: {max_10:.0f}'
    ax2.text(0.95, 0.95, stats_text_10, transform=ax2.transAxes,
             ha='right', va='top', fontsize=fontsize,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_per_individual))

    plt.tight_layout()

    plt.show()

    return fig


if __name__ == "__main__":
    # violin plots for average ranking for each dataset in the N30
    backend = Backends.visible_tests.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    run_file = GENERATED_DATASETS_FILE_PATH
    overall_ds_name = "n30"

    data_type = SyntheticDataType.non_normal_correlated
    ds_p30 = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                            data_type=data_type,
                                            data_dir=IRREGULAR_P30_DATA_DIR, load_data=True)

    ds_p90 = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                            data_type=data_type,
                                            data_dir=IRREGULAR_P90_DATA_DIR, load_data=True)

    gaps_partial = ds_p30.all_time_gaps_in_seconds()
    gaps_sparse = ds_p90.all_time_gaps_in_seconds()
    fig = plot_gap_histograms(gaps_partial=gaps_partial, gaps_sparse=gaps_sparse, unit='seconds', backend=backend)

    if save_fig:
        folder = get_clustering_quality_multiple_data_variants_result_folder(
            results_type=ResultsType.internal_measures_calculation,
            overall_dataset_name=overall_ds_name,
            results_dir=root_result_dir,
            distance_measure='')
        # add an image results folder
        file_name = "_".join([data_type, INTERVAL_HISTOGRAMS])
        file_name = get_image_results_path(folder, file_name)
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
