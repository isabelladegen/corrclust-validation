from src.evaluation.distance_metric_evaluation import DistanceMetricEvaluation
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
    overall_ds_name = "n2"
    # run_distance_evaluation_raw_criteria_for_ds(overall_ds_name)
