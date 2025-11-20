from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.experiments.run_internal_measure_assessment_family4 import run_wilcox_signed_rank_for
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_family4_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses: [], root_results_dir: str,
                                                        overall_ds_name: str, run_names: [str],
                                                        save_results_dir: str = ''):
    if save_results_dir == '':
        save_results_dir = root_results_dir
    alpha = 0.05
    target_power = 0.8
    bonferroni_adjust = 1  # no adjustment due to hierarchical testing
    alternative = "greater"  # one-sided confirmatory tests (h[2] higher correlated than h[3])
    non_zero = 0.001

    results = []
    # Calculate stats
    for h in prereg_hypotheses:
        # h e.g  (downsampled, sparse, internal measure1, internal measure2, distance measure)
        data_type = h[0]
        data_dir = h[1]
        internal_measure1 = h[2]
        internal_measure2 = h[3]
        dm = h[4]
        df = run_wilcox_signed_rank_for(overall_ds_name=overall_ds_name, run_names=run_names,
                                        distance_measure=dm, data_type=data_type,
                                        data_dir=data_dir, results_dir=root_results_dir,
                                        internal_measure1=internal_measure1,
                                        internal_measure2=internal_measure2,
                                        alternative=alternative,
                                        non_zero=non_zero,
                                        alpha=alpha,
                                        bonferroni_adjust=bonferroni_adjust, target_power=target_power)
        df.insert(1, 'H', internal_measure1 + ">" + internal_measure2)
        results.append(df)

    # Save result
    stats_df = pd.concat(results)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=save_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances measures included

    full_path = path.join(store_results_in, IAResultsCSV.family_4)
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
    # (data_type, data_dir, internal measure1, internal measure2, distance measure)
    # all written as x less than y
    hypotheses = [
        (downsampled, complete, ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.silhouette_score,
         DistanceMeasures.l5_cor_dist),
        (downsampled, partial, ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.silhouette_score,
         DistanceMeasures.l5_cor_dist),
        (downsampled, sparse, ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.silhouette_score,
         DistanceMeasures.l5_cor_dist),
        (non_normal, sparse, ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.silhouette_score,
         DistanceMeasures.l5_cor_dist),
        (normal, sparse, ClusteringQualityMeasures.dbi, ClusteringQualityMeasures.silhouette_score,
         DistanceMeasures.l5_cor_dist),
    ]

    # evaluate all hypotheses
    run_family4_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        root_results_dir=root_result_dir,
                                                        overall_ds_name=overall_dataset_name,
                                                        run_names=run_names)
