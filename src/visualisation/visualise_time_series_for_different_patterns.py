import os
from os import path

import pandas as pd
from matplotlib import pyplot as plt

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import SYNTHETIC_DATA_DIR, base_dataset_result_folder_for_type, ROOT_RESULTS_DIR, \
    get_image_name_based_on_data_dir_and_data_type, ResultsType, VALID_ROOT_RESULTS_DIR
from src.utils.load_synthetic_data import SyntheticDataType, load_synthetic_data
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize

if __name__ == "__main__":
    run_name = "trim-fire-24"
    n_obs = 20
    root_results_dir = ROOT_RESULTS_DIR
    validity_results = VALID_ROOT_RESULTS_DIR

    data_dir = SYNTHETIC_DATA_DIR
    data_type = SyntheticDataType.normal_correlated

    reset_matplotlib(backend=Backends.visible_tests.value)

    pattern_ids_to_plot = [0, 13, 25, 18]

    # Load data
    data_df, labels_df = load_synthetic_data(run_name, data_type, data_dir)
    segments_data = []
    relaxed_patterns = []
    for pattern_id in pattern_ids_to_plot:
        row_idx = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == pattern_id].index[0]
        start_idx = labels_df[SyntheticDataSegmentCols.start_idx].iloc[row_idx]
        end_idx = start_idx + n_obs
        pattern = labels_df[SyntheticDataSegmentCols.actual_correlation].iloc[row_idx]
        pattern = [round(i, 1) for i in pattern]
        pattern = [0.0 if abs(x) == 0.0 else x for x in pattern]  # round
        # pt_str = str(pattern).replace('[', '').replace(']', '')
        relaxed_patterns.append(pattern)
        segment = data_df.iloc[start_idx:end_idx + 1]  # you need to include the end idx
        segments_data.append(segment)

    # plot each pattern
    for pattern_idx, pattern_id in enumerate(pattern_ids_to_plot):
        # Create one line grid
        fig, ax = plt.subplots(figsize=(8, 6))
        reset_matplotlib(backend=Backends.visible_tests.value)
        line_alpha = 1
        lw = 5
        ms = 1
        data_to_plot = segments_data[pattern_idx].copy()
        data_to_plot.reset_index(drop=True, inplace=True)

        # add offset so lines are not on top of each other
        line1 = \
            ax.plot(data_to_plot['iob'] + 0, marker='.', label=r'$v_1$', linewidth=lw, alpha=line_alpha, markersize=ms)[
                0]
        line2 = \
            ax.plot(data_to_plot['cob'] + 2, marker='.', label=r'$v_2$', linewidth=lw, alpha=line_alpha, markersize=ms)[
                0]
        line3 = \
            ax.plot(data_to_plot['ig'] + 4, marker='.', label=r'$v_3$', linewidth=lw, alpha=line_alpha, markersize=ms)[
                0]

        # Get colors
        c1, c2, c3 = line1.get_color(), line2.get_color(), line3.get_color()

        # Create colored title
        x_start = 0.35
        x_offset = x_start
        for i, (val, colors) in enumerate([
            (pattern[0], (c1, c2)),
            (pattern[1], (c1, c3)),
            (pattern[2], (c2, c3))
        ]):
            parts = str(val).split('.')
            ax.text(x_offset, 1.05, parts[0], transform=ax.transAxes, fontsize=fontsize,
                    fontweight='bold', ha='left', color=colors[0])
            x_offset += 0.035  # increased spacing
            ax.text(x_offset, 1.05, '.' + parts[1], transform=ax.transAxes, fontsize=fontsize,
                    fontweight='bold', ha='left', color=colors[1])
            x_offset += 0.05
            if i < 2:
                ax.text(x_offset, 1.05, ', ', transform=ax.transAxes, fontsize=fontsize,
                        fontweight='bold', ha='left', color='black')
                x_offset += 0.03

        ax.set_xticklabels([])  # no x ticks
        ax.set_yticklabels([])  # no y ticks

        ax.grid(False)
        ax.axis('off')

        plt.tight_layout()
        # save figure
        results_folder = path.join(validity_results, ResultsType.distance_measure_evaluation, 'images')
        os.makedirs(results_folder, exist_ok=True)

        plt.savefig(path.join(results_folder,
                              'normal_complete_ts_plot_pattern' + str(pattern_id) + '.png'), dpi=300,
                    bbox_inches='tight')
        plt.show()
