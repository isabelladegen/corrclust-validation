from src.evaluation.distance_metric_ranking import read_csv_of_overall_rank_per_dataset, \
    read_csv_of_average_criteria_across_datasets
from src.utils.plots.matplotlib_helper_functions import Backends


class DistanceMetricInterpretation:
    def __init__(self, run_names: [str], overall_ds_name: str, data_type: str, data_dir: str, root_results_dir: str,
                 measures: [],
                 round_to: int = 3):
        self.run_name = run_names  # list of runs to load
        self.overall_ds_name = overall_ds_name
        self.data_type = data_type
        self.data_dir = data_dir
        self.root_results_dir = root_results_dir
        self.__measures = measures
        self.__round_to = round_to
        # x is runs, y is measures, cell value is average rank for run and measure across criterion
        self.average_rank_per_run = read_csv_of_overall_rank_per_dataset(self.overall_ds_name,
                                                                         self.data_type, self.data_dir,
                                                                         self.root_results_dir)[self.__measures]
        self.criteria_average_run_df = read_csv_of_average_criteria_across_datasets(self.overall_ds_name,
                                                                                    self.data_type, self.data_dir,
                                                                                    self.root_results_dir)[self.__measures]
