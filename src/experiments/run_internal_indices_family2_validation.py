import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, internal_measure_evaluation_dir_for, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description


def rank_distance_measures_by_raw_values_run_wilcox_signed_rank_tests(data_dirs: [],
                                                                      dataset_types: [],
                                                                      root_results_dir: str,
                                                                      distance_measures: [],
                                                                      internal_measures: [],
                                                                      overall_ds_name: str,
                                                                      alpha: float,
                                                                      bf_adjust: int,
                                                                      alternative: str, non_zero: float
                                                                      ):
    all_stats_results = []

    # calculate wilcox signed ranks per data variant
    for data_dir in data_dirs:
        for data_type in dataset_types:
            ga = InternalMeasureGroundTruthAssessment(overall_ds_name=overall_ds_name,
                                                      internal_measures=internal_measures,
                                                      distance_measures=distance_measures,
                                                      data_dir=data_dir,
                                                      data_type=data_type,
                                                      root_results_dir=root_results_dir)
            variant_desc = data_variant_description[(get_data_completeness_from(data_dir), data_type)]

            stats_results_df = ga.wilcoxons_signed_rank_step_down(alpha=alpha, bonferroni_adjust=bf_adjust,
                                                               alternative=alternative, non_zero=non_zero)

            # add data variant column
            stats_results_df.insert(0, "Data Variant", variant_desc)
            all_stats_results.append(stats_results_df)

    # create one df
    result = pd.concat(all_stats_results, ignore_index=True)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances measures included

    # save stats results
    result.to_csv(str(os.path.join(store_results_in, IAResultsCSV.distance_measures_stat_results_for_ground_truth)))


if __name__ == "__main__":
    # Calculates which distance measures are significantly better than others for given internal measures
    backend = Backends.none.value  #
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]
    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    # only the once that pass corr r>0.5
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,
                         DistanceMeasures.foerstner_cor_dist
                         ]

    rank_distance_measures_by_raw_values_run_wilcox_signed_rank_tests(data_dirs=data_dirs, dataset_types=dataset_types,
                                                                      overall_ds_name="n30",
                                                                      root_results_dir=root_result_dir,
                                                                      distance_measures=distance_measures,
                                                                      internal_measures=internal_measures,
                                                                      alpha=0.5,
                                                                      alternative="two-sided",
                                                                      bf_adjust=1,
                                                                      non_zero=0.0001
                                                                      )
