import os
from os import path

import pandas as pd

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    AVERAGE_RANK_DISTRIBUTION
from src.utils.distance_measures import DistanceMeasures, short_distance_measure_names
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_all_variants_average_rank_per_run import data_variant_description
from src.visualisation.visualise_distance_measure_rank_distributions import heatmap_of_ranks


def heatmap_for_all_variants(data_dirs, dataset_types, run_names, root_results_dir, distance_measures, overall_ds_name,
                             backend, save_fig=True):
    # build data to plot
    ranks_dfs = {}
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)
            variant_desc = data_variant_description[(data_dir, data_type)]
            ranks_dfs[variant_desc] = interpretation.stats_for_average_ranks_across_all_runs().loc["50%"]

    rank_matrix = pd.concat(ranks_dfs).unstack(level=0).T
    # rename distance measures
    rank_matrix = rank_matrix.rename(columns=lambda x: short_distance_measure_names.get(x, x))

    # plot heatmap
    # highlight baseline variant
    partial_nn = data_variant_description[(IRREGULAR_P30_DATA_DIR, SyntheticDataType.non_normal_correlated)]
    highlight_rows = [partial_nn]

    # find top 2 distance measure with min rank for partial_nn
    highlight_cols = rank_matrix.loc[partial_nn].sort_values(ascending=True).head(2).index.tolist()

    fig = heatmap_of_ranks(rank_matrix, highlight_rows=highlight_rows, highlight_cols=highlight_cols, backend=backend)

    # save figure
    if save_fig:
        folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.distance_measure_evaluation)
        folder = path.join(folder, "images")
        os.makedirs(folder, exist_ok=True)
        image_name = AVERAGE_RANK_DISTRIBUTION
        fig.savefig(path.join(folder, image_name), dpi=300, bbox_inches='tight')


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
