from os import path

import pandas as pd

from src.experiments.confirmatory.run_family5_confirmation import run_reduced_wilcox_signed_rank_tests_for_hypotheses
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONFIRMATORY_SYNTHETIC_DATA_DIR, CONF_ROOT_REDUCED_RESULTS_DIR, DataCompleteness, \
    get_root_folder_for_reduced_segments, CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

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
    seg100 = CONFIRMATORY_SYNTHETIC_DATA_DIR  # full dataset dropped 0 clusters
    seg50 = get_root_folder_for_reduced_segments(CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR,
                                                 50)  # reduced dir dropped 12 clusters, kept 6
    seg25 = get_root_folder_for_reduced_segments(CONFIRMATORY_ROOT_REDUCED_SYNTHETIC_DATA_DIR,
                                                 75)  # reduced dir dropped 17 clusters, kept 6

    # preregistered hypotheses, sequential list of tuples
    # (data_type, data_dir, internal measure, n_seg1, n_seg2, distance measure)
    # all written as x less than y
    hypotheses = [
        # Seq 1: Normal, Complete (100%), SWC₁₀₀ > SWC₅₀
        (normal, complete, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 2: Non-normal, Complete (100%), SWC₁₀₀ > SWC₅₀
        (non_normal, complete, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 3: Normal, Complete (100%), SWC₅₀ > SWC₂₅
        (normal, complete, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 4: Non-normal, Complete (100%), SWC₅₀ > SWC₂₅
        (non_normal, complete, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 5: Normal, Partial (70%), SWC₁₀₀ > SWC₅₀
        (normal, partial, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 6: Non-normal, Partial (70%), SWC₁₀₀ > SWC₅₀
        (non_normal, partial, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 7: Normal, Partial (70%), SWC₅₀ > SWC₂₅
        (normal, partial, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 8: Non-normal, Partial (70%), SWC₅₀ > SWC₂₅
        (non_normal, partial, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 9: Normal, Sparse (10%), SWC₁₀₀ > SWC₅₀
        (normal, sparse, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 10: Non-normal, Sparse (10%), SWC₁₀₀ > SWC₅₀
        (non_normal, sparse, swc, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 11: Normal, Sparse (10%), SWC₅₀ > SWC₂₅
        (normal, sparse, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 12: Non-normal, Sparse (10%), SWC₅₀ > SWC₂₅
        (non_normal, sparse, swc, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 13: Normal, Sparse (10%), DBI₅₀ > DBI₂₅
        (normal, sparse, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 14: Non-normal, Sparse (10%), DBI₅₀ > DBI₂₅
        (non_normal, sparse, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 15: Normal, Partial (70%), DBI₅₀ > DBI₂₅
        (normal, partial, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 16: Non-normal, Partial (70%), DBI₅₀ > DBI₂₅
        (non_normal, partial, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 17: Normal, Complete (100%), DBI₅₀ > DBI₂₅
        (normal, complete, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 18: Non-normal, Complete (100%), DBI₅₀ > DBI₂₅
        (non_normal, complete, dbi, seg50, seg25, DistanceMeasures.l5_cor_dist),
        # Seq 19: Non-normal, Complete (100%), DBI₁₀₀ > DBI₅₀
        (non_normal, complete, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 20: Normal, Complete (100%), DBI₁₀₀ > DBI₅₀
        (normal, complete, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 21: Non-normal, Sparse (10%), DBI₁₀₀ > DBI₅₀
        (non_normal, sparse, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 22: Non-normal, Partial (70%), DBI₁₀₀ > DBI₅₀
        (non_normal, partial, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 23: Normal, Sparse (10%), DBI₁₀₀ > DBI₅₀
        (normal, sparse, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
        # Seq 24: Normal, Partial (70%), DBI₁₀₀ > DBI₅₀
        (normal, partial, dbi, seg100, seg50, DistanceMeasures.l5_cor_dist),
    ]

    # evaluate all hypotheses
    run_reduced_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        reduced_root_results_dir=reduced_root_results_dir,
                                                        full_root_results_dir=full_data_root_results_dir,
                                                        overall_ds_name=overall_dataset_name,
                                                        test_type='segment_count')
