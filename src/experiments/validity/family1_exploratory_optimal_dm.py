import pandas as pd

from src.experiments.confirmatory.run_distance_measures_family1_confirmation import \
    run_wilcox_signed_rank_tests_for_hypotheses
from src.utils.configurations import VALID_ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Create preregistration hypotheses
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = VALID_ROOT_RESULTS_DIR

    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = IRREGULAR_P90_DATA_DIR
    partial = IRREGULAR_P30_DATA_DIR
    complete = SYNTHETIC_DATA_DIR

    # hypotheses, sequential list of tuples (data_type, data_dir, measure 1, measure 2)
    # created from dm_hypothesis_by_overall_ranking.csv per data variant
    hypotheses = [
        (normal, complete, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
        (normal, partial, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
        (normal, sparse, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
        (non_normal, complete, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
        (non_normal, partial, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
        (non_normal, sparse, DistanceMeasures.l1_cor_dist, DistanceMeasures.dot_transform_l1),
    ]

    # evaluate all hypotheses
    run_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses, run_names=run_names,
                                                root_results_dir=root_result_dir, overall_ds_name=overall_dataset_name,
                                                alternative='two-sided')
