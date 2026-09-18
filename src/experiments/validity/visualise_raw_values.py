import os
from ast import literal_eval
from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.evaluation.distance_metric_evaluation import EvaluationCriteria, read_csv_of_raw_values_for_all_criteria, \
    DistanceMeasureCols
from src.experiments.validity.distance_measure_validity import DM_THRESHOLDS
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, IRREGULAR_P90_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, SYNTHETIC_DATA_DIR, ResultsType, VALID_ROOT_RESULTS_DIR, STRUCTURAL_2_CI_FILENAME, \
    distance_measure_evaluation_results_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize


def load_distance_raw_values(measures, run_names, data_type, data_dir, root_results_dir, pair):
    criteria = [EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.scale_free_inter_iii,
                EvaluationCriteria.disc_iii]

    raw_dfs = []
    for run_name in run_names:
        raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                  data_dir=data_dir,
                                                                  base_results_dir=root_results_dir)

        result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name, data_type=data_type,
                                                                 base_results_dir=root_results_dir, data_dir=data_dir)
        pair_df = pd.read_csv(path.join(result_dir, STRUCTURAL_2_CI_FILENAME))
        pair_df[DistanceMeasureCols.compared] = pair_df[DistanceMeasureCols.compared].apply(literal_eval)
        computed_row = (pair_df[pair_df[DistanceMeasureCols.compared] == pair]
                        .set_index(DistanceMeasureCols.type)[DistanceMeasureCols.mean_diff]
                        .reindex(measures))
        computed_row.name = EvaluationCriteria.inter_ii_values

        raw_dfs.append(pd.concat([raw_criteria_df.loc[criteria, measures],
                                  computed_row.to_frame().T]))

    return raw_dfs


def create_raw_box_plots(conditions_data, condition_names, distance_functions, criteria, colours, column_names,
                         dist_labels,
                         threshold_direction, figsize=(16, 12)):
    # Create figure
    reset_matplotlib(backend=Backends.none)
    n_rows = len(condition_names)
    fig, axes = plt.subplots(n_rows, len(criteria), figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    # figure out ranges
    y_ranges = {}
    for criterion in criteria:
        all_values = []
        for condition_data in conditions_data:
            for dist_func in distance_functions:
                values = [df.loc[criterion, dist_func] for df in condition_data]
                all_values.extend(values)
        y_ranges[criterion] = (min(all_values), max(all_values))

    # Plot each condition × criterion combination
    for row_idx, (condition_data, condition_name) in enumerate(zip(conditions_data, condition_names)):
        for col_idx, criterion in enumerate(criteria):
            ax = axes[row_idx, col_idx]

            # Extract data for this criterion across all subjects for each distance function
            plot_data = []
            for dist_func in distance_functions:
                values = [df.loc[criterion, dist_func] for df in condition_data]
                plot_data.append(values)

            # Create box plots
            bp = ax.boxplot(plot_data,
                            positions=range(len(distance_functions)),
                            widths=0.6,
                            patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker='_', markeredgecolor='black', markersize=8, linewidth=1.5))

            # Set consistent y-axis per column
            ymin, ymax = y_ranges[criterion]
            padding = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - padding, ymax + padding)

            # Set decimal precision based on column's max value
            if ymin < 0.1:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Colour boxes
            for patch, dist_func in zip(bp['boxes'], distance_functions):
                patch.set_facecolor(colours[dist_func])
                patch.set_alpha(0.7)

            threshold = DM_THRESHOLDS[criterion]
            ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            ylim = ax.get_ylim()
            alpha = 0.08
            if threshold_direction[criterion] == 'above':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='grey', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='green', zorder=0)
            elif threshold_direction[criterion] == 'below':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='green', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='grey', zorder=0)

            # Labels
            if row_idx == 0:
                ax.set_title(column_names[criterion], fontsize=fontsize, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(condition_name, fontsize=fontsize)

            # X-axis labels - only on bottom row
            ax.tick_params(axis='y', labelsize=fontsize)
            if row_idx == n_rows - 1:  # Last row only
                ax.set_xticks(range(len(distance_functions)))
                ax.set_xticklabels([dist_labels[df] for df in distance_functions], fontsize=fontsize)
            else:
                ax.set_xticks(range(len(distance_functions)))
                ax.set_xticklabels([])

            # Grid
            ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    return plt


dist_labels = {
    DistanceMeasures.l1_cor_dist: r'$L_1$',
    DistanceMeasures.l2_cor_dist: r'$L_2$',
    DistanceMeasures.l3_cor_dist: r'$L_3$',
    DistanceMeasures.l5_cor_dist: r'$L_5$',
    DistanceMeasures.l1_with_ref: r'$L_{\text{ref}_1}$',
    DistanceMeasures.linf_with_ref: r'$L_{\text{ref}_{\infty}}$',
    DistanceMeasures.dot_transform_l1: r'$L_{\text{dot}_1}$',
    DistanceMeasures.dot_transform_l2: r'$L_{\text{dot}_2}$',
    DistanceMeasures.linf_cor_dist: r'$L_{\infty}$',
    DistanceMeasures.log_frob_cor_dist: r'$\log F$',  # light red (invalid)
    DistanceMeasures.foerstner_cor_dist: r'$F$'  # pale red (invalid)
}

if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = ROOT_RESULTS_DIR

    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = IRREGULAR_P90_DATA_DIR
    partial = IRREGULAR_P30_DATA_DIR
    complete = SYNTHETIC_DATA_DIR

    all_distance_functions = [DistanceMeasures.l1_cor_dist,  # valid
                              DistanceMeasures.l3_cor_dist,
                              DistanceMeasures.l5_cor_dist,
                              DistanceMeasures.dot_transform_l2,
                              DistanceMeasures.linf_with_ref,  # invalid
                              DistanceMeasures.log_frob_cor_dist,
                              DistanceMeasures.foerstner_cor_dist
                              ]
    # take from tables in appendix which one to use for normal 100% and 10% data
    level_set_pair_for_min_diff = (2, 3)

    data_n_100 = load_distance_raw_values(all_distance_functions, run_names, normal, complete, root_result_dir,
                                          level_set_pair_for_min_diff)
    data_n_10 = load_distance_raw_values(all_distance_functions, run_names, normal, sparse, root_result_dir,
                                         level_set_pair_for_min_diff)
    data_nn_100 = load_distance_raw_values(all_distance_functions, run_names, non_normal, complete, root_result_dir,
                                           level_set_pair_for_min_diff)
    data_nn_10 = load_distance_raw_values(all_distance_functions, run_names, non_normal, sparse, root_result_dir,
                                          level_set_pair_for_min_diff)

    conditions_data = [data_n_100, data_n_10, data_nn_100, data_nn_10]
    # row names
    condition_names = [r'$\bf{Normal}$' + '\n100%', r'$\bf{Normal}$' + '\n10%']

    column_names = {
        EvaluationCriteria.scale_free_inter_i: r'1. $\Delta_{\text{Cliff}}$',
        EvaluationCriteria.inter_ii_values: r'2. $\overline{\gamma^{\mathfrak{L}_i}} - \overline{\gamma^{\mathfrak{L}_j}}$',
        EvaluationCriteria.scale_free_inter_iii: r'3. $\Delta_{\text{Cliff}}$',
        EvaluationCriteria.disc_iii: r'6. $F_1$',
    }

    # 'above': grey shades from the threshold up to the top of the axis, green shades from the bottom
    # up to the threshold. So values above the line fail, values below pass.
    # 'below': grey shades from the bottom of the axis up to the threshold, green shades from the threshold to the top.
    # Values below the line fail, values above pass.
    threshold_direction = {
        EvaluationCriteria.scale_free_inter_i: 'above',  # valid when < -0.8
        EvaluationCriteria.inter_ii_values: 'above',  # valid when < 0
        EvaluationCriteria.scale_free_inter_iii: 'below',  # valid when > 0.4
        EvaluationCriteria.disc_iii: 'below',  # valid when > 0.9
    }

    colours = {
        DistanceMeasures.l1_cor_dist: '#2E86AB',  # blue (valid)
        DistanceMeasures.l3_cor_dist: '#06A77D',  # teal (valid)
        DistanceMeasures.l5_cor_dist: '#D9A441',  # gold (valid)
        DistanceMeasures.dot_transform_l2: '#5E60CE',  # purple (valid)
        DistanceMeasures.linf_with_ref: '#C1121F',  # red (invalid)
        DistanceMeasures.log_frob_cor_dist: '#FFA07A',  # light red (invalid)
        DistanceMeasures.foerstner_cor_dist: '#FFB6B9'  # pale red (invalid)
    }

    # plot for passing
    criteria = [
        EvaluationCriteria.scale_free_inter_i,
        EvaluationCriteria.inter_ii_values,
        EvaluationCriteria.scale_free_inter_iii,
    ]
    create_raw_box_plots(conditions_data, condition_names, all_distance_functions[:5], criteria, colours,
                         column_names, dist_labels, threshold_direction, figsize=(16, 6))

    results_folder = path.join(VALID_ROOT_RESULTS_DIR, ResultsType.distance_measure_evaluation, 'images')
    os.makedirs(results_folder, exist_ok=True)

    filename1 = path.join(results_folder, 'structural_validity_lp_df.png')
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"saved {filename1}")

    # plot for failing
    criteria = [
        EvaluationCriteria.scale_free_inter_i,
        EvaluationCriteria.inter_ii_values,
        EvaluationCriteria.scale_free_inter_iii,
        EvaluationCriteria.disc_iii
    ]
    create_raw_box_plots([conditions_data[0]], [condition_names[0]], all_distance_functions[5:], criteria, colours,
                         column_names,
                         dist_labels, threshold_direction, figsize=(16, 3.5))
    filename2 = path.join(results_folder, 'structural_validity_corr_df.png')
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"saved {filename2}")
