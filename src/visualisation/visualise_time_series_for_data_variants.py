import os
from os import path

import pandas as pd
from matplotlib import pyplot as plt

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR, base_dataset_result_folder_for_type, ROOT_RESULTS_DIR, \
    get_image_name_based_on_data_dir_and_data_type, ResultsType
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize

if __name__ == "__main__":
    run_name = "trim-fire-24"
    n_segments = 2
    n_obs = 10
    root_results_dir = ROOT_RESULTS_DIR

    data_dir = SYNTHETIC_DATA_DIR

    data_types = [
        SyntheticDataType.raw,
        SyntheticDataType.normal_correlated,
        SyntheticDataType.non_normal_correlated,
        SyntheticDataType.rs_1min
    ]
    titles = ['Raw', 'Normal', 'Non-normal', 'Downsampled']

    reset_matplotlib(backend=Backends.visible_tests.value)

    # Create one line grid
    fig = plt.figure(figsize=(25, 6))
    line_alpha = 0.8
    lw = 3
    ms = 10 # marker size

    # Create all 4 subplots first, then share manually
    ax1 = plt.subplot(1, 4, 1)
    ax2 = plt.subplot(1, 4, 2)
    ax3 = plt.subplot(1, 4, 3)
    ax4 = plt.subplot(1, 4, 4)

    # Manually share y-axis between ax3 and ax4 after plotting
    axes = [ax1, ax2, ax3, ax4]

    achieved_patterns = {}

    # Plot each subplot
    for idx, (ax, data_type, title) in enumerate(zip(axes, data_types, titles)):
        # Load data
        data_df, labels_df = load_synthetic_data(run_name, data_type, data_dir)
        segments = []
        patterns = []
        for segment_id in range(n_segments):
            start_idx = labels_df[SyntheticDataSegmentCols.start_idx].iloc[segment_id]
            end_idx = labels_df[SyntheticDataSegmentCols.end_idx].iloc[segment_id]
            pattern = labels_df[SyntheticDataSegmentCols.actual_correlation].iloc[segment_id]
            pattern = [round(i, 1) for i in pattern]  # round
            segment_data = data_df.iloc[start_idx:end_idx + 1]  # you need to include the end idx

            sub_segment = segment_data.iloc[:n_obs]
            segments.append(sub_segment)
            pt_str = str(pattern).replace('[', '(').replace(']', ')')
            patterns.append(pt_str)

        # save patterns
        achieved_patterns[idx] = patterns

        # plot time series
        data_to_plot = pd.concat(segments)
        data_to_plot.reset_index(drop=True, inplace=True)

        ax.plot(data_to_plot['iob'], marker='.', label=r'$v_1$', linewidth=lw, alpha=line_alpha, markersize=ms)
        ax.plot(data_to_plot['cob'], marker='.', label=r'$v_2$', linewidth=lw, alpha=line_alpha, markersize=ms)
        ax.plot(data_to_plot['ig'], marker='.', label=r'$v_3$', linewidth=lw, alpha=line_alpha, markersize=ms)

        # add regime changing line
        for i in range(1, n_segments):
            ax.axvline(x=i * n_obs, color='#DC143C', linestyle='-', alpha=1, linewidth=2)

        ax.set_title(title, fontsize=fontsize, fontweight='bold')

        # add datetime as x axis label
        # tick_positions = list(range(0, len(data_to_plot), 5))
        # tick_positions.append(data_to_plot.index[-1])
        # for i in range(1, n_segments):
        #     tick_positions.remove(i * n_obs)
        # tick_labels = [data_to_plot['datetime'].iloc[pos].strftime('%H:%M:%S') for pos in tick_positions]
        # ax.set_xticks(tick_positions)
        # ax.set_xticklabels([tick_labels], rotation=45, ha='center')
        ax.set_xticklabels([]) # no x ticks

        ax.set_xlim(0, n_segments * n_obs)
        ax.grid(True)

        # add legend for last subplot
        if idx == len(data_types) - 1:
            ax.legend(bbox_to_anchor=(1, 0.08), loc='lower right')

        if idx == 0:
            ax.set_ylabel('Complete 100%', fontsize=fontsize, fontweight='bold')

    # Pair 1: ax1 and ax2
    y_min_12 = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max_12 = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min_12, y_max_12)
    ax2.set_ylim(y_min_12, y_max_12)

    # Pair 2: ax3 and ax4
    y_min_34 = min(ax3.get_ylim()[0], ax4.get_ylim()[0])
    y_max_34 = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
    ax3.set_ylim(y_min_34, y_max_34)
    ax4.set_ylim(y_min_34, y_max_34)

    # Add correlation pattern text
    for ax_idx, ax in enumerate(axes):
        y_top = y_max_12 if ax_idx < 2 else y_max_34
        for i in range(n_segments):
            x_middle = i * n_obs + (n_obs / 2)  # Middle of each segment
            ax.text(x_middle, y_top * 0.98, achieved_patterns[ax_idx][i], ha='center', va='top', fontsize=fontsize,
                    fontweight='bold', c='grey')

    # Hide y-axis labels on ax2 and ax4 (show only on ax1 and ax3)
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])

    plt.tight_layout()
    plt.show()

    # save figure
    folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.dataset_description)
    folder = path.join(folder, "images")
    os.makedirs(folder, exist_ok=True)
    image_name = get_image_name_based_on_data_dir_and_data_type('_'.join([run_name, 'timeseries_plot_complete.png']),
                                                                data_dir,
                                                                '')
    fig.savefig(path.join(folder, image_name), dpi=300, bbox_inches='tight')
