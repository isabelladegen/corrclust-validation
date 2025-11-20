from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_family2_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses: [], root_results_dir: str,
                                                        overall_ds_name: str, save_to_results_dir:str=''):
    if save_to_results_dir == '':
        save_to_results_dir == root_results_dir
    alpha = 0.05
    target_power = 0.8
    bonferroni_adjust = 1  # no adjustment due to hierarchical testing
    alternative = "less"  # one-sided confirmatory tests (h[2] better ranked than h[3]
    non_zero = 0.0001

    results = []
    # Calculate stats
    for h in prereg_hypotheses:
        # h e.g  (downsampled, sparse, internal measure, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist)
        data_type = h[0]
        data_dir = h[1]
        internal_measure = h[2]
        m1 = h[3]
        m2 = h[4]
        ga = InternalMeasureGroundTruthAssessment(overall_ds_name=overall_ds_name,
                                                  internal_measures=[internal_measure],
                                                  distance_measures=[m1, m2],
                                                  data_dir=data_dir,
                                                  data_type=data_type,
                                                  root_results_dir=root_results_dir)
        wilc_result_df = ga.wilcoxons_between(dm1=m1, dm2=m2, alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                              alternative=alternative, non_zero=non_zero, target_power=target_power)
        results.append(wilc_result_df)

    # Save result
    stats_df = pd.concat(results)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=save_to_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances measures included

    full_path = path.join(store_results_in, IAResultsCSV.family_2_results)
    stats_df.to_csv(str(full_path))


if __name__ == "__main__":
    # Confirm preregistered tests from exploratory phase for confirmatory data
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = CONF_ROOT_RESULTS_DIR

    downsampled = SyntheticDataType.rs_1min
    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = CONF_IRREGULAR_P90_DATA_DIR
    partial = CONF_IRREGULAR_P30_DATA_DIR
    complete = CONFIRMATORY_SYNTHETIC_DATA_DIR

    # preregistered hypotheses, sequential list of tuples
    # (data_type, data_dir, internal measure, distance measure 1, distance measure 2)
    # all written as x less than y
    hypotheses = [
        (normal, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l3_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (normal, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l3_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (downsampled, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (normal, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (non_normal, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (non_normal, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist,),
        (downsampled, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (normal, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref, DistanceMeasures.l3_cor_dist),
        (non_normal, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist,),
        (downsampled, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist,),
        (downsampled, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.l3_cor_dist,),
        (non_normal, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.linf_cor_dist,),
        (normal, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist, DistanceMeasures.linf_cor_dist,),
        (normal, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref, DistanceMeasures.l3_cor_dist,),
        (non_normal, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref,
         DistanceMeasures.l3_cor_dist,),
        (downsampled, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.l3_cor_dist,),
        (downsampled, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist,
         DistanceMeasures.l3_cor_dist,),
        (non_normal, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref,
         DistanceMeasures.l5_cor_dist,),
    ]

    # evaluate all hypotheses
    run_family2_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        root_results_dir=root_result_dir,
                                                        overall_ds_name=overall_dataset_name)
