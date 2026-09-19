import os
from os import path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.run_calculate_internal_measures_for_ground_truth import \
    read_ground_truth_clustering_quality_measures
from src.experiments.validity.icvi_validity import ICVIValCriteria
from src.experiments.validity.visualise_raw_values import dist_labels
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ResultsType, VALID_ROOT_RESULTS_DIR, get_data_dir, SYNTHETIC_DATA_DIR, \
    ROOT_REDUCED_SYNTHETIC_DATA_DIR, get_root_folder_for_reduced_cluster, ROOT_REDUCED_RESULTS_DIR, ROOT_RESULTS_DIR, \
    DataCompleteness
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def plot_errorbars_for_data(data_1, data_2, data_3, title, conditions, figsize, distance_measures,
                            quality_measures, invalid_distance_measures_by_condition_and_quality_measure,
                            invalid_label_color='#C1121F'):
    # Thresholds
    thresholds = {m: ICVIValCriteria.excellent_threshold_for(m) for m in quality_measures}
    y_limits = {
        ClusteringQualityMeasures.silhouette_score: {'min': 0.3, 'max': 1},
        ClusteringQualityMeasures.dbi: {'min': 0, 'max': 1},
        ClusteringQualityMeasures.vrc: {'min': 0, 'max': 40000},
        ClusteringQualityMeasures.pmb: {'min': 0, 'max': 800},
    }
    threshold_direction = {
        ClusteringQualityMeasures.silhouette_score: 'below',
        ClusteringQualityMeasures.dbi: 'above',
        ClusteringQualityMeasures.vrc: 'below',
        ClusteringQualityMeasures.pmb: 'below',
    }
    # Setup
    distances = [dist_labels[dm] for dm in distance_measures]
    colours = ['#8cbed8', '#629318', '#ca69ca']
    marker_size = 6
    cap_size = 2
    fontsize = 12
    n_cols = len(quality_measures) + 1  # +1 for the MAE column
    mae_col = n_cols - 1
    fig, axes = plt.subplots(len(conditions), n_cols, figsize=figsize,
                             gridspec_kw={'width_ratios': [1] * len(quality_measures) + [0.25]})
    x_positions = np.arange(len(distances))
    width = 0.25
    legend_row = len(conditions) - 1
    legend_col = len(quality_measures) - 1
    for row, condition in enumerate(conditions):
        for col, measure in enumerate(quality_measures):
            ax = axes[row, col]

            means_23 = [data_1[condition][measure][d][0] for d in distances]
            sds_23 = [data_1[condition][measure][d][1] for d in distances]

            means_11 = [data_2[condition][measure][d][0] for d in distances]
            sds_11 = [data_2[condition][measure][d][1] for d in distances]

            means_6 = [data_3[condition][measure][d][0] for d in distances]
            sds_6 = [data_3[condition][measure][d][1] for d in distances]

            ax.errorbar(x_positions - width, means_23, yerr=sds_23, fmt='o', capsize=cap_size,
                        color=colours[0], ecolor=colours[0], markersize=marker_size,
                        label=data_1['count'] if row == legend_row and col == legend_col else '')

            ax.errorbar(x_positions, means_11, yerr=sds_11, fmt='s', capsize=cap_size,
                        color=colours[1], ecolor=colours[1], markersize=marker_size,
                        label=data_2['count'] if row == legend_row and col == legend_col else '')

            ax.errorbar(x_positions + width, means_6, yerr=sds_6, fmt='^', capsize=cap_size,
                        color=colours[2], ecolor=colours[2], markersize=marker_size,
                        label=data_3['count'] if row == legend_row and col == legend_col else '')

            ymax = y_limits[measure]['max']
            ymin = y_limits[measure]['min']
            padding = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - padding, ymax + padding)

            threshold = thresholds[measure]
            ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            ylim = ax.get_ylim()
            alpha = 0.08
            if threshold_direction[measure] == 'above':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='grey', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='green', zorder=0)
            elif threshold_direction[measure] == 'below':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='green', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='grey', zorder=0)

            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_xticks(x_positions)
            invalid_distances = invalid_distance_measures_by_condition_and_quality_measure.get(condition, {}).get(
                measure, [])
            tick_texts = [d + ('*' if d in invalid_distances else '') for d in distances]
            ax.set_xticklabels(tick_texts, fontsize=fontsize)
            for tick_label, d in zip(ax.get_xticklabels(), distances):
                if d in invalid_distances:
                    tick_label.set_color(invalid_label_color)
            ax.grid(True, alpha=0.3, linestyle=':')

            if row == 0:
                ax.set_title(ClusteringQualityMeasures.get_display_name_for_measure(measure),
                             fontweight='bold', fontsize=fontsize)
            if col == 0:
                ax.set_ylabel(condition, fontweight='bold', fontsize=fontsize)

            ax.margins(y=0.15)

        # MAE column: narrow, no box, no x-axis — just a y-axis and title
        ground_truth_mae = {
            conditions[0]: (0.02, 0.02),
            conditions[1]: (0.03, 0.03),
        }
        ax_mae = axes[row, mae_col]
        mae_mean, mae_sd = ground_truth_mae[condition]
        ax_mae.errorbar([0], [mae_mean], yerr=[mae_sd], fmt='D', capsize=cap_size,
                        color='#333333', ecolor='#333333', markersize=marker_size)
        ax_mae.set_xlim(-0.5, 0.5)
        ax_mae.set_ylim(0, 1)
        ax_mae.set_xticks([])
        ax_mae.tick_params(axis='y', labelsize=fontsize)
        ax_mae.spines['top'].set_visible(False)
        ax_mae.spines['right'].set_visible(False)
        ax_mae.spines['bottom'].set_visible(False)
        if row == 0:
            ax_mae.set_title('MAE', fontweight='bold', fontsize=fontsize)

    # Legend: points are labelled on the PBM axes (legend_col), but drawn on the MAE axes
    handles, labels = axes[legend_row, legend_col].get_legend_handles_labels()
    axes[0, mae_col].legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 1.0),
                            fontsize=fontsize, framealpha=0.9, title=title, title_fontsize=fontsize)

    plt.tight_layout()


def load_ground_truth_quality_measures(data_dir_and_result_root_by_n_dropped, data_completeness, data_types,
                                       distance_measures, quality_measures, overall_dataset_name):
    results_by_count = {}
    for base_data_dir, result_root, count_label in data_dir_and_result_root_by_n_dropped.values():
        results_by_count[count_label] = {"count": count_label}
        for comp in data_completeness:
            data_dir = get_data_dir(base_data_dir, comp)
            for data_type in data_types:
                label = SyntheticDataType.display_name_for_type_and_data_dir(data_type, data_dir)
                stats = {m: {} for m in quality_measures}

                for distance_measure in distance_measures:
                    print(f"count_label={count_label}  root_results_dir={result_root}  "
                          f"data_dir={data_dir}  data_type={data_type}  distance_measure={distance_measure}")
                    df = read_ground_truth_clustering_quality_measures(
                        overall_ds_name=overall_dataset_name,
                        data_type=data_type,
                        root_results_dir=result_root,
                        data_dir=data_dir,
                        distance_measure=distance_measure,
                    )
                    for measure in quality_measures:
                        stats[measure][dist_labels[distance_measure]] = (df[measure].mean(), df[measure].std())

                results_by_count[count_label][label] = stats
    return results_by_count


if __name__ == "__main__":
    # legend title
    legend_t = 'Clusters'
    cond = ['Normal 100%', 'Normal 10%']
    root_reduced_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
    base_results_dir = ROOT_REDUCED_RESULTS_DIR

    distance_measures = [
        DistanceMeasures.l1_cor_dist,
        DistanceMeasures.l2_cor_dist,
        DistanceMeasures.l3_cor_dist,
        DistanceMeasures.l5_cor_dist,
        DistanceMeasures.l1_with_ref,
        DistanceMeasures.dot_transform_l2
    ]

    clustering_quality_measures = [
        ClusteringQualityMeasures.silhouette_score,
        ClusteringQualityMeasures.dbi,
        ClusteringQualityMeasures.vrc,
        ClusteringQualityMeasures.pmb
    ]
    overall_dataset_name = "n30"

    data_completeness = [
        DataCompleteness.complete,
        DataCompleteness.irregular_p90
    ]

    data_types = [
        SyntheticDataType.normal_correlated,
        # SyntheticDataType.non_normal_correlated
    ]

    n_dropped_clusters_to_count_label = {0: "23", 12: "11", 17: "6"}
    data_dir_and_result_root_by_n_dropped = {}
    for n_dropped, count_label in n_dropped_clusters_to_count_label.items():
        if n_dropped == 0:
            base_data_dir, result_root = SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR
        else:
            base_data_dir = get_root_folder_for_reduced_cluster(root_reduced_dir, n_dropped)
            result_root = get_root_folder_for_reduced_cluster(base_results_dir, n_dropped)
        data_dir_and_result_root_by_n_dropped[n_dropped] = (base_data_dir, result_root, count_label)

    results_by_count = load_ground_truth_quality_measures(
        data_dir_and_result_root_by_n_dropped=data_dir_and_result_root_by_n_dropped,
        data_completeness=data_completeness,
        data_types=data_types,
        distance_measures=distance_measures,
        quality_measures=clustering_quality_measures,
        overall_dataset_name=overall_dataset_name,
    )
    data_23 = results_by_count["23"]
    data_11 = results_by_count["11"]
    data_6 = results_by_count["6"]

    invalid_dms = {
        cond[0]: {
            ClusteringQualityMeasures.silhouette_score: [],
            ClusteringQualityMeasures.dbi: [],
            ClusteringQualityMeasures.vrc: [],
            ClusteringQualityMeasures.pmb: [
                dist_labels[DistanceMeasures.l1_cor_dist],
                dist_labels[DistanceMeasures.l2_cor_dist],
                dist_labels[DistanceMeasures.l3_cor_dist],
                dist_labels[DistanceMeasures.l5_cor_dist],
                dist_labels[DistanceMeasures.dot_transform_l2]],
        },
        cond[1]: {
            ClusteringQualityMeasures.silhouette_score: [
                dist_labels[DistanceMeasures.l1_with_ref]],
            ClusteringQualityMeasures.dbi: [
                dist_labels[DistanceMeasures.l1_cor_dist],
                dist_labels[DistanceMeasures.l1_with_ref]],
            ClusteringQualityMeasures.vrc: [],
            ClusteringQualityMeasures.pmb: [dist_labels[dm] for dm in distance_measures],
        }
    }

    plot_errorbars_for_data(data_23, data_11, data_6, legend_t, cond, (18, 6),
                            distance_measures=distance_measures,
                            quality_measures=clustering_quality_measures,
                            invalid_distance_measures_by_condition_and_quality_measure=invalid_dms)
    results_folder = path.join(VALID_ROOT_RESULTS_DIR, ResultsType.internal_measure_evaluation, 'images')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(path.join(results_folder, 'structural_test_1_3_clusters.png'), dpi=300, bbox_inches='tight')
    plt.show()
