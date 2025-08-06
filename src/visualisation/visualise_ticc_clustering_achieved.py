from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import SYNTHETIC_DATA_DIR, IRREGULAR_P90_DATA_DIR, ROOT_RESULTS_DIR, DataCompleteness
from src.utils.load_synthetic_data import SyntheticDataType, load_labels, load_ticc_labels
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize
from src.visualisation.visualise_corr_3d_scatter_multiple_variants import plot_data_2d, plot_data
from src.visualisation.visualise_corr_3d_scatter_plot import pattern_colours


def plot_subplots_for(run_name, relaxed_patterns, backend, additional_data=[]):
    reset_matplotlib(backend)
    fig = plt.figure(figsize=(13, 14))
    rows = 2
    columns = 2

    # Create gridspec with different row heights
    # Top row (2D plots) gets height ratio of 0.6, other rows get 1.0
    gs = GridSpec(rows, columns, height_ratios=[0.5, 1.0])

    # plot normal complete 2D
    ax1 = fig.add_subplot(gs[0, 0])
    data_type = SyntheticDataType.normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    normal_ticc_data = additional_data[0][[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data_2d(ax=ax1, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours, secondary_data=normal_ticc_data)
    ax1.set_title('Normal', weight='bold', fontsize=fontsize, pad=40)

    # plot nn complete 2D
    ax2 = fig.add_subplot(gs[0, 1])
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    nn_ticc_data = additional_data[1][[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data_2d(ax=ax2, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours, secondary_data=nn_ticc_data)
    ax2.set_title('Non-Normal', weight='bold', fontsize=fontsize, pad=40)

    # MIDDLE ROW - Original 3D plots (complete data)
    # plot normal complete 3D
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    data_type = SyntheticDataType.normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax4, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours, secondary_data=normal_ticc_data)

    # plot nn complete 3D
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax5, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours, secondary_data=nn_ticc_data)

    # row labels
    fig.text(0.02, 0.75, 'p₁₃ = 0', rotation=90, fontsize=fontsize, fontweight='bold', ha='center',
             va='center')


    plt.tight_layout(pad=2.0)

    # Add custom legend at bottom
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, label='Canonical patterns'),
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='CSTS'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8, label='TICC')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.show()
    return fig


if __name__ == "__main__":
    backend = Backends.visible_tests.value
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()
    root_results_dir = ROOT_RESULTS_DIR

    run_name = "unique-puddle-26"

    # Load ticc normal
    comp = DataCompleteness.complete
    normal_df = load_ticc_labels(run_id=run_name, data_type=SyntheticDataType.normal_correlated, completeness=comp)

    # load ticc non-normal
    non_normal_df = load_ticc_labels(run_id=run_name, data_type=SyntheticDataType.non_normal_correlated, completeness=comp)


    fig = plot_subplots_for(run_name, relaxed_patterns, backend, [normal_df, non_normal_df])
    fig.savefig('ticc_clustering_achieved_3D.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

