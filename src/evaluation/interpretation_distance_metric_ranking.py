import pandas as pd

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_ranking import read_csv_of_overall_rank_per_dataset, \
    read_csv_of_average_criteria_across_datasets, read_csv_of_ranks_for_all_criteria
from src.utils.configurations import RunInformationCols
from src.utils.plots.matplotlib_helper_functions import Backends


class DistanceMetricInterpretation:
    def __init__(self, run_names: [str], overall_ds_name: str, data_type: str, data_dir: str, root_results_dir: str,
                 measures: [],
                 round_to: int = 3):
        self.run_names = run_names  # list of runs to load
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
        # x is criteria, y is measures, cell value is average rank for that criteria and measure across runs
        self.criteria_average_run_df = read_csv_of_average_criteria_across_datasets(self.overall_ds_name,
                                                                                    self.data_type, self.data_dir,
                                                                                    self.root_results_dir)[
            self.__measures]
        # columns run_name, criteria, measure, rank - long df
        self.all_criteria_ranks_df = self.__melt_into_criteria_ranks_df()

    def __melt_into_criteria_ranks_df(self):
        melted_dfs = []
        for run_name in self.run_names:
            # load each run's criteria level ranking
            rank_df = read_csv_of_ranks_for_all_criteria(run_name, self.data_type, self.data_dir,
                                                         self.root_results_dir)
            # reset index
            rank_df = rank_df.reset_index(names=[DistanceMeasureCols.criterion])

            # change into long df with columns criteria, distance measure and rank
            columns = [DistanceMeasureCols.criterion]
            columns.extend(self.__measures)
            melted = pd.melt(rank_df[columns], id_vars=DistanceMeasureCols.criterion, var_name=DistanceMeasureCols.type, value_name=DistanceMeasureCols.rank)
            # add run name column
            melted[RunInformationCols.ds_name] = run_name

            # Append to our list
            melted_dfs.append(melted)

        # Combine all melted dataframes
        final_df = pd.concat(melted_dfs, ignore_index=True)

        # Reorder columns to desired format
        final_df = final_df[[RunInformationCols.ds_name, DistanceMeasureCols.criterion, DistanceMeasureCols.type,
                             DistanceMeasureCols.rank]]
        return final_df
