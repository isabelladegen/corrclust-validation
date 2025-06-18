from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_cluster_segment_count_differences import \
    InternalMeasureGroundTruthClusterSegmentCount
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONFIRMATORY_SYNTHETIC_DATA_DIR, internal_measure_evaluation_dir_for, CONF_ROOT_REDUCED_RESULTS_DIR, \
    get_root_folder_for_reduced_cluster, get_data_dir, DataCompleteness, CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.visualisation.run_average_rank_visualisations import data_variant_description


def run_reduced_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses: [], reduced_root_results_dir: str,
                                                        full_root_results_dir: str,
                                                        overall_ds_name: str, test_type: str):
    alpha = 0.05
    target_power = 0.8
    bonferroni_adjust = 1  # no adjustment due to hierarchical testing
    alternative = "greater"  # one-sided confirmatory tests (h[2] higher values than h[3])
    non_zero = 0.0001

    results = []
    # Calculate stats
    for h in prereg_hypotheses:
        # h e.g   (non_normal, sparse, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        data_type = h[0]
        comp = h[1]
        internal_measure = h[2]
        cl1 = h[3]  # data dir part for count 1 dataset
        cl2 = h[4]  # data dir part for count 2 dataset
        dm = h[5]

        ga = InternalMeasureGroundTruthClusterSegmentCount(overall_ds_name=overall_ds_name,
                                                           internal_measures=[internal_measure],
                                                           distance_measure=dm,
                                                           data_dirs=[get_data_dir(cl1, comp), get_data_dir(cl2, comp)],
                                                           data_type=data_type,
                                                           reduced_root_result=reduced_root_results_dir,
                                                           original_root_result=full_root_results_dir)
        variant_desc = data_variant_description[(comp, data_type)]
        wilc_result_df = ga.wilcoxons_signed_rank_between_all_counts(alpha=alpha,
                                                                     bonferroni_adjust=bonferroni_adjust,
                                                                     alternative=alternative, non_zero=non_zero)[
            internal_measure]
        wilc_result_df.insert(0, 'Data variant', variant_desc)
        wilc_result_df.insert(1, 'Internal Measure', internal_measure)
        wilc_result_df['Distance Measure'] = dm
        results.append(wilc_result_df)

    # Save result
    stats_df = pd.concat(results).reset_index(drop=True)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=reduced_root_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # distances measures included in data

    full_path = path.join(store_results_in, test_type + '_' + IAResultsCSV.family_5_and_6_results)
    stats_df.to_csv(str(full_path))


if __name__ == "__main__":
    # Confirm preregistered tests from exploratory phase for confirmatory data
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    reduced_root_results_dir = CONF_ROOT_REDUCED_RESULTS_DIR  # read data from for reduced results
    full_data_root_results_dir = CONF_ROOT_RESULTS_DIR  # read data from for full dataset results

    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = DataCompleteness.irregular_p90
    partial = DataCompleteness.irregular_p30
    complete = DataCompleteness.complete

    dbi = ClusteringQualityMeasures.dbi
    swc = ClusteringQualityMeasures.silhouette_score

    # dirs for the reduced counts
    c23 = CONFIRMATORY_SYNTHETIC_DATA_DIR  # full dataset dropped 0 clusters
    c11 = get_root_folder_for_reduced_cluster(CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR,
                                              12)  # reduced dir dropped 12 clusters, kept 6
    c6 = get_root_folder_for_reduced_cluster(CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR,
                                             17)  # reduced dir dropped 17 clusters, kept 6

    # preregistered hypotheses, sequential list of tuples
    # (data_type, data_dir, internal measure, n_cl1, n_cl2, distance measure)
    # all written as x less than y
    hypotheses = [
        # Seq 1: normal, sparse, DBI₂₃ > DBI₆
        (normal, sparse, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 2: non-normal, sparse, DBI₂₃ > DBI₆
        (non_normal, sparse, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 3: normal, complete, DBI₂₃ > DBI₆
        (normal, complete, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 4: non-normal, complete, DBI₂₃ > DBI₆
        (non_normal, complete, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 5: normal, complete, DBI₂₃ > DBI₁₁
        (normal, complete, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 6: non-normal, complete, DBI₂₃ > DBI₁₁
        (non_normal, complete, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 7: non-normal, sparse, SWC₆ > SWC₂₃
        (non_normal, sparse, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 8: normal, partial, DBI₂₃ > DBI₆
        (normal, partial, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 9: normal, sparse, SWC₆ > SWC₂₃
        (normal, sparse, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 10: non-normal, partial, DBI₂₃ > DBI₆
        (non_normal, partial, dbi, c23, c6, DistanceMeasures.l5_cor_dist),
        # Seq 11: non-normal, sparse, SWC₁₁ > SWC₂₃
        (non_normal, sparse, swc, c11, c23, DistanceMeasures.l5_cor_dist),
        # Seq 12: non-normal, partial, SWC₆ > SWC₂₃
        (non_normal, partial, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 13: normal, partial, DBI₂₃ > DBI₁₁
        (normal, partial, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 14: non-normal, complete, SWC₆ > SWC₂₃
        (non_normal, complete, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 15: non-normal, partial, DBI₂₃ > DBI₁₁
        (non_normal, partial, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 16: normal, partial, SWC₆ > SWC₂₃
        (normal, partial, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 17: normal, complete, SWC₆ > SWC₂₃
        (normal, complete, swc, c6, c23, DistanceMeasures.l5_cor_dist),
        # Seq 18: normal, sparse, SWC₁₁ > SWC₂₃
        (normal, sparse, swc, c11, c23, DistanceMeasures.l5_cor_dist),
        # Seq 19: normal, sparse, DBI₂₃ > DBI₁₁
        (normal, sparse, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 20: non-normal, sparse, DBI₂₃ > DBI₁₁
        (non_normal, sparse, dbi, c23, c11, DistanceMeasures.l5_cor_dist),
        # Seq 21: normal, partial, DBI₁₁ > DBI₆
        (normal, partial, dbi, c11, c6, DistanceMeasures.l5_cor_dist),
        # Seq 22: non-normal, partial, DBI₁₁ > DBI₆
        (non_normal, partial, dbi, c11, c6, DistanceMeasures.l5_cor_dist),
        # Seq 23: normal, complete, SWC₁₁ > SWC₂₃
        (normal, complete, swc, c11, c23, DistanceMeasures.l5_cor_dist),
        # Seq 24: non-normal, complete, SWC₁₁ > SWC₂₃
        (non_normal, complete, swc, c11, c23, DistanceMeasures.l5_cor_dist),
        # Seq 25: normal, sparse, DBI₁₁ > DBI₆
        (normal, sparse, dbi, c11, c6, DistanceMeasures.l5_cor_dist),
        # Seq 26: non-normal, sparse, DBI₁₁ > DBI₆
        (non_normal, sparse, dbi, c11, c6, DistanceMeasures.l5_cor_dist),
    ]

    # evaluate all hypotheses
    run_reduced_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        reduced_root_results_dir=reduced_root_results_dir,
                                                        full_root_results_dir=full_data_root_results_dir,
                                                        overall_ds_name=overall_dataset_name,
                                                        test_type='cluster_count')
