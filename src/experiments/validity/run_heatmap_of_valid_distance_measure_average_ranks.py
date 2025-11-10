import os
from os import path

import pandas as pd

from src.evaluation.distance_metric_evaluation import criteria_short_names
from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    HEATMAP_OF_RANKS, HEATMAP_OF_BEST_MEASURES_RAW_VALUES, DataCompleteness, get_data_completeness_from, \
    VALID_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description
from src.visualisation.run_heatmap_of_average_distance_measure_ranks import get_key_for_value, plot_ranking_heat_map
from src.visualisation.visualise_distance_measure_rank_distributions import heatmap_of_ranks, heatmap_of_raw_values

pattern_keys_ordered = ["complete, normal", "partial, normal", "sparse, normal", "complete, non-normal",
                        "partial, non-normal", "sparse, non-normal"]


#
# all_variants_ordered = ["complete, raw", "partial, raw", "sparse, raw", ] + pattern_keys_ordered


def heatmap_for_all_variants(data_dirs, dataset_types, run_names, root_results_dir, distance_measures, overall_ds_name,
                             backend, save_fig=True):
    # build data to plot
    ranks_dfs = {}
    mean_ranks_dfs = {}
    sd_ranks_dfs = {}
    raw_value_dfs = {}
    means_per_crit = []
    std_per_crit = []
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)
            variant_desc = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
            # for average rank heatmap
            ranks_dfs[variant_desc] = interpretation.stats_for_average_ranks_across_all_runs().loc["50%"]
            mean_ranks_dfs[variant_desc] = interpretation.stats_for_average_ranks_across_all_runs().loc["mean"]
            sd_ranks_dfs[variant_desc] = interpretation.stats_for_average_ranks_across_all_runs().loc["std"]
            # for raw value heatmap
            raw_value_dfs[variant_desc] = interpretation.median_raw_values
            # for write up
            for crit, per_crit in interpretation.stats_per_criterion_raw_ranks().items():
                df_mean = per_crit.loc["mean"].to_frame().T
                df_std = per_crit.loc["std"].to_frame().T
                df_mean.insert(0, 'Criterion', crit)
                df_mean.insert(0, 'Data Variant', variant_desc)
                df_std.insert(0, 'Criterion', crit)
                df_std.insert(0, 'Data Variant', variant_desc)
                means_per_crit.append(df_mean)
                std_per_crit.append(df_std)

    # PLOT RANKING HEATMAP
    fig, rank_matrix = plot_ranking_heat_map(backend, ranks_dfs, pattern_keys_ordered)

    folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.distance_measure_evaluation)
    # save rank matrix
    rank_matrix.to_csv(str(path.join(folder, "distance_measure_rank_matrix.csv")))
    # save mean and sd rank
    pd.concat(mean_ranks_dfs).unstack(level=0).T.to_csv(
        str(path.join(folder, "distance_measure_mean_ranks.csv")))
    pd.concat(sd_ranks_dfs).unstack(level=0).T.to_csv(str(path.join(folder, "distance_measure_std_ranks.csv")))
    # save per criterion rank
    pd.concat(means_per_crit).reset_index(drop=True).to_csv(
        str(path.join(folder, "distance_measure_rank_mean_per_crit.csv")))
    pd.concat(std_per_crit).reset_index(drop=True).to_csv(
        str(path.join(folder, "distance_measure_rank_std_per_crit.csv")))

    # save figures
    if save_fig:
        folder = path.join(folder, "images")
        os.makedirs(folder, exist_ok=True)
        fig.savefig(path.join(folder, HEATMAP_OF_RANKS), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # heatmap ov average ranking only for valid distance functions for each dataset in the N30
    # y = data variant, x = distance measure, lower ranks are better
    backend = Backends.visible_tests.value
    save_fig = True
    root_result_dir = VALID_ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    # valid list of distance measures
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         ]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    heatmap_for_all_variants(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                             root_results_dir=root_result_dir, distance_measures=distance_measures,
                             overall_ds_name="n30",
                             backend=backend, save_fig=save_fig)
