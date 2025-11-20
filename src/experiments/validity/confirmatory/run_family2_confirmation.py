import pandas as pd

from src.experiments.confirmatory.run_family2_confirmation import run_family2_wilcox_signed_rank_tests_for_hypotheses
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    CONF_VALID_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Confirm preregistered tests from exploratory phase for confirmatory data
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = CONF_ROOT_RESULTS_DIR
    save_results_dir = CONF_VALID_ROOT_RESULTS_DIR

    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = CONF_IRREGULAR_P90_DATA_DIR
    partial = CONF_IRREGULAR_P30_DATA_DIR
    complete = CONFIRMATORY_SYNTHETIC_DATA_DIR

    dbi = ClusteringQualityMeasures.dbi
    swc = ClusteringQualityMeasures.silhouette_score

    l3 = DistanceMeasures.l3_cor_dist
    l5 = DistanceMeasures.l5_cor_dist

    # preregistered hypotheses, sequential list of tuples
    # (data_type, data_dir, internal measure, distance measure 1, distance measure 2)
    # all written as x less than y
    hypotheses = [
        (normal, complete, swc, l3, l5),
        (normal, partial, swc, l3, l5),
        (non_normal, complete, swc, l3, l5),
        (non_normal, partial, swc, l3, l5),
        (normal, sparse, swc, l3, l5),
        (non_normal, sparse, swc, l3, l5),
        (normal, sparse, dbi, l5, l3),  # DBI
        (non_normal, sparse, dbi, l5, l3),
        (non_normal, complete, dbi, l5, l3),
        (normal, partial, dbi, l5, l3),
        (normal, complete, dbi, l5, l3),
        (non_normal, partial, dbi, l5, l3),
    ]

    # evaluate all hypotheses
    run_family2_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        root_results_dir=root_result_dir,
                                                        overall_ds_name=overall_dataset_name,
                                                        save_to_results_dir=save_results_dir)
