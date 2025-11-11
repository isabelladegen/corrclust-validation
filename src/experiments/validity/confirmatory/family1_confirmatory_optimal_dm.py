import pandas as pd

from src.experiments.confirmatory.run_distance_measures_family1_confirmation import \
    run_wilcox_signed_rank_tests_for_hypotheses
from src.utils.configurations import CONFIRMATORY_SYNTHETIC_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, \
    CONF_VALID_ROOT_RESULTS_DIR, CONFIRMATORY_DATASETS_FILE_PATH, CONF_IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Test preregistered hypotheses
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = CONF_VALID_ROOT_RESULTS_DIR

    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = CONF_IRREGULAR_P90_DATA_DIR
    partial = CONF_IRREGULAR_P30_DATA_DIR
    complete = CONFIRMATORY_SYNTHETIC_DATA_DIR

    # preregistered family 1 hypotheses
    hypotheses = [
        (normal, complete, DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist),
        (normal, partial, DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist),
        (non_normal, complete, DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist),
        (non_normal, partial, DistanceMeasures.dot_transform_l2, DistanceMeasures.l1_cor_dist),
        (non_normal, sparse, DistanceMeasures.dot_transform_l2, DistanceMeasures.l3_cor_dist),
        (normal, sparse, DistanceMeasures.l3_cor_dist, DistanceMeasures.l1_cor_dist),
    ]

    # evaluate all hypotheses
    run_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses, run_names=run_names,
                                                root_results_dir=root_result_dir, overall_ds_name=overall_dataset_name,
                                                alternative='two-sided')
