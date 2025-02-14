import pandas as pd

from src.evaluation.distance_metric_evaluation import DistanceMetricEvaluation
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


def run_distance_evaluation_raw_criteria_for_ds(data_dirs: [str], dataset_types: [str], run_names: [str],
                                                root_result_dir: str,
                                                distance_measures: [str], backend=Backends.none.value):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in run_names:
                ev = DistanceMetricEvaluation(run_name=run_name, data_type=data_type, data_dir=data_dir,
                                              measures=distance_measures, backend=backend)
                ev.save_csv_of_raw_values_for_all_criteria(run_name=run_name, base_results_dir=root_result_dir)


if __name__ == "__main__":
    # configuration to calculate raw distance measure criteria for the N30 dataset
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

    run_distance_evaluation_raw_criteria_for_ds(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                                                root_result_dir=root_result_dir, distance_measures=distance_measures)
