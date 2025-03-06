import os
from os import path

import pandas as pd

from src.evaluation.distance_metric_evaluation import criteria_short_names
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    AVERAGE_RANK_DISTRIBUTION, HEATMAP_OF_RANKS, HEATMAP_OF_BEST_MEASURES_RAW_VALUES, get_irregular_folder_name_from, \
    DataCompleteness
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from src.visualisation.visualise_distance_measure_rank_distributions import heatmap_of_ranks, heatmap_of_raw_values

pattern_keys_ordered = ["complete, correlated", "partial, correlated", "sparse, correlated", "complete, non-normal",
                        "partial, non-normal", "sparse, non-normal", "complete, downsampled"]

all_variants_ordered = ["complete, raw", "partial, raw", "sparse, raw", ] + pattern_keys_ordered + [
    "partial, downsampled", "sparse, downsampled"]


def get_key_for_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def heatmap_for_all_variants(data_dirs, dataset_types, run_names, root_results_dir, distance_measures, overall_ds_name,
                             backend, save_fig=True):
    # build data to plot
    ranks_dfs = {}
    raw_value_dfs = {}
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)
            variant_desc = data_variant_description[(get_irregular_folder_name_from(data_dir), data_type)]
            # for average rank heatmap
            ranks_dfs[variant_desc] = interpretation.stats_for_average_ranks_across_all_runs().loc["50%"]
            # for raw value heatmap
            raw_value_dfs[variant_desc] = interpretation.median_raw_values

    # PLOT RANKING HEATMAP
    fig, rank_matrix = plot_ranking_heat_map(backend, ranks_dfs, pattern_keys_ordered)
    top_two_dist = rank_matrix.columns[:2].tolist()

    # PLOT RAW VALUE HEATMAP
    fig_raw = plot_raw_values_heat_map_for_top_two_measures(raw_value_dfs, top_two_dist, all_variants_ordered, backend)

    # save figures
    if save_fig:
        folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.distance_measure_evaluation)
        folder = path.join(folder, "images")
        os.makedirs(folder, exist_ok=True)
        fig.savefig(path.join(folder, HEATMAP_OF_RANKS), dpi=300, bbox_inches='tight')
        fig_raw.savefig(path.join(folder, HEATMAP_OF_BEST_MEASURES_RAW_VALUES), dpi=300, bbox_inches='tight')


def plot_raw_values_heat_map_for_top_two_measures(raw_value_dfs, highlight_cols, keys_ordered, backend):
    # remove keys not in keys ordered
    filtered_raw_value_dfs = {k: v for k, v in raw_value_dfs.items() if k in keys_ordered}

    # create matrix for raw values for top best ranked measures
    top_measures = [get_key_for_value(short_distance_measure_names, m) for m in highlight_cols]
    m1 = top_measures[0]
    m2 = top_measures[1]
    rows = []
    criteria = list(filtered_raw_value_dfs.values())[0].index
    for variant_name, df in filtered_raw_value_dfs.items():
        row_data = {}
        for criterion in criteria:
            row_data[f'{criteria_short_names[criterion]}:{short_distance_measure_names[m1]}'] = df.loc[criterion, m1]
            row_data[f'{criteria_short_names[criterion]}:{short_distance_measure_names[m2]}'] = df.loc[criterion, m2]
        rows.append(row_data)
    matrix_raw_values = pd.DataFrame(rows, index=list(filtered_raw_value_dfs.keys()))

    # sort df according to keys_ordered
    matrix_raw_values = matrix_raw_values.reindex(keys_ordered)

    fig = heatmap_of_raw_values(matrix_raw_values, backend=backend)
    return fig


def plot_ranking_heat_map(backend, ranks_series, keys_ordered, bar_label='Rank', low_is_best=True):
    """
    Plots a ranking heat map where distance measures are sorted by baseline variant
    :param backend:
    :param ranks_series: dictionary with key=variate description and value series with distances as index and median
    (or other) as value
    :param keys_ordered:
    :param bar_label:
    :param low_is_best:
    :return:
    """
    # remove keys not in keys ordered
    filtered_rank_dfs = {k: v for k, v in ranks_series.items() if k in keys_ordered}
    rank_matrix = pd.concat(filtered_rank_dfs).unstack(level=0).T
    # rename distance measures
    rank_matrix = rank_matrix.rename(columns=lambda x: short_distance_measure_names.get(x, x))
    # sort df according to keys_ordered
    rank_matrix = rank_matrix.reindex(keys_ordered)
    # sort columns by smallest for our baseline variant
    partial_nn = data_variant_description[(DataCompleteness.irregular_p30, SyntheticDataType.non_normal_correlated)]
    rank_matrix = rank_matrix.sort_values(by=partial_nn, axis=1, ascending=low_is_best)
    # for each row highlight best cell
    if low_is_best:
        best_col = rank_matrix.idxmin(axis=1)
    else:
        best_col = rank_matrix.idxmax(axis=1)
    highlight_rows = rank_matrix.index.tolist()
    highlight_cols = best_col.to_list()
    # plot heatmap
    fig = heatmap_of_ranks(rank_matrix, highlight_rows=highlight_rows, highlight_cols=highlight_cols, figsize=(15, 6),
                           low_is_best=low_is_best, bar_label=bar_label, backend=backend)
    return fig, rank_matrix


if __name__ == "__main__":
    # heatmap ov average ranking for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    backend = Backends.visible_tests.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    # this is an extensive list
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.l1_with_ref,  # lp norms with reference vector
                         DistanceMeasures.l2_with_ref,
                         DistanceMeasures.l3_with_ref,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrix
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    heatmap_for_all_variants(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                             root_results_dir=root_result_dir, distance_measures=distance_measures,
                             overall_ds_name="n30",
                             backend=backend, save_fig=save_fig)
