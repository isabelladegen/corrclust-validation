import pandas as pd

from src.experiments.run_distance_distance_metric_ranking import run_ranking_for
from src.experiments.run_distance_evaluation_raw_criteria import run_distance_evaluation_raw_criteria_for_ds
from src.utils.configurations import SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, VALID_ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    # Calculate raw criteria and rankings only for valid DM
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    # all variants but raw
    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    data_dirs = [SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR]

    root_results_dir = VALID_ROOT_RESULTS_DIR

    # all distance measures
    # valid list of distance measures
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         ]

    # 1. Calculate raw criteria for valid distance measures
    # Recalculation would not be required but given the root_results dir is where we read and safe to this is simpler
    run_distance_evaluation_raw_criteria_for_ds(data_dirs=data_dirs, dataset_types=data_types, run_names=run_names,
                                                root_result_dir=root_results_dir, distance_measures=distance_measures)

    # 2. Rank distance measures
    run_ranking_for(data_dirs=data_dirs, dataset_types=data_types, run_names=run_names,
                    root_result_dir=root_results_dir, distance_measures=distance_measures,
                    overall_ds_name=overall_dataset_name)
