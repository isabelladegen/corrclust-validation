from dataclasses import dataclass
from os import path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_evaluation import EvaluationCriteria
from src.evaluation.distance_metric_ranking import read_csv_of_overall_rank_per_dataset, \
    read_csv_of_average_criteria_across_datasets, read_csv_of_ranks_for_all_criteria
from src.utils.configurations import RunInformationCols, distance_measure_evaluation_results_dir_for, \
    DISTANCE_MEASURE_EVALUATION_TOP_BOTTOM_MEASURES
from src.utils.distance_measures import short_distance_measure_names
from src.utils.stats import WilcoxResult


@dataclass
class DistanceInterpretation:
    top_rank: str = 'top avg'  # criteria averaged
    bottom_rank: str = 'bottom avg'  # criteria averaged
    raw_top_rank: str = 'top avg raw'  # across all criteria
    raw_bottom_rank: str = 'bottom avg raw'  # across all criteria averaged
    top_inter_i: str = "top c1"
    top_inter_ii: str = "top c2"
    top_inter_iii: str = "top c3"
    top_disc_i: str = "top c4"
    top_disc_ii: str = "top c5"
    top_disc_iii: str = "top c6"
    bottom_inter_i: str = "bottom c1"
    bottom_inter_ii: str = "bottom c2"
    bottom_inter_iii: str = "bottom c3"
    bottom_disc_i: str = "bottom c4"
    bottom_disc_ii: str = "bottom c5"
    bottom_disc_iii: str = "bottom c6"


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
        self.raw_criteria_ranks_df = self.__melt_into_criteria_ranks_df()

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
            melted = pd.melt(rank_df[columns], id_vars=DistanceMeasureCols.criterion, var_name=DistanceMeasureCols.type,
                             value_name=DistanceMeasureCols.rank)
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

    def stats_for_raw_criteria_ranks_across_all_runs(self):
        """
        Calculates descriptive statistics for the raw criteria ranks across all runs
        Input data 30 ranks for each of the 6 criterion
        :return: pd.Dataframe with rows descriptive statistics and columns distance measures
        """
        stats = self.raw_criteria_ranks_df.groupby(DistanceMeasureCols.type)[DistanceMeasureCols.rank].describe().round(
            self.__round_to)
        stats = stats.transpose()
        return stats

    def stats_for_average_ranks_across_all_runs(self):
        """
        Calculates descriptive statistics for the average ranks
        Input data: 30 average ranks across the criteria (this better weighs the criteria and gives less weight
        to single great criteria performance)
        :return: pd.Dataframe with rows descriptive statistics and columns distance measures
        """
        stats = self.average_rank_per_run.describe().round(self.__round_to)
        return stats

    def stats_per_criterion_raw_ranks(self):
        """
        Calculates descriptive statistics for the ranks across all runs per criterion using the 30 ranks per criterion
        :return: dictionary of key criterion value pd.Dataframe with rows descriptive statistics and columns
        distance measures
        """
        criteria = self.raw_criteria_ranks_df[DistanceMeasureCols.criterion].unique()
        result = {}
        for criterion in criteria:
            criterion_ranks = self.raw_criteria_ranks_df[
                self.raw_criteria_ranks_df[DistanceMeasureCols.criterion] == criterion]

            stats = criterion_ranks.groupby(DistanceMeasureCols.type)[DistanceMeasureCols.rank].describe().round(
                self.__round_to)
            stats = stats.transpose()
            result[criterion] = stats

        return result

    def top_and_bottom_x_distance_measures_ranks(self, x: int, save_results: bool = False):
        """Returns dataframe of top and bottom x distance measures calculated on raw criteria ranks. 30 ranks for each
        of the 6 criteria (180 ranks)
        :param x: number of x distance measures to include
        :param save_results: if true will save df to disk
        :return: pd.Dataframe with rows descriptive statistics and columns:
        - avg rank top x and bottom x measures (weighs each criterion equaly)
        - avg rank for top x and bottom x measures calculated from raw ranks
        - avg rank for top x and bottom x measure per criterion (using raw ranks)
        """
        avg_stats = self.stats_for_average_ranks_across_all_runs()
        avg_raw_stats = self.stats_for_raw_criteria_ranks_across_all_runs()
        criteria_stats = self.stats_per_criterion_raw_ranks()

        top_avg = []
        bottom_avg = []
        raw_top_avg = []
        raw_bottom_avg = []
        top_inter_i = []
        top_inter_ii = []
        top_inter_iii = []
        top_disc_i = []
        top_disc_ii = []
        top_disc_iii = []
        bottom_inter_i = []
        bottom_inter_ii = []
        bottom_inter_iii = []
        bottom_disc_i = []
        bottom_disc_ii = []
        bottom_disc_iii = []

        for stat in avg_raw_stats.index:
            if stat == 'count':  # skip count
                continue
            top_avg.append(", ".join(avg_stats.loc[stat].sort_values(ascending=True).head(x).index.tolist()))
            bottom_avg.append(", ".join(avg_stats.loc[stat].sort_values(ascending=False).head(x).index.tolist()))
            raw_top_avg.append(", ".join(avg_raw_stats.loc[stat].sort_values(ascending=True).head(x).index.tolist()))
            raw_bottom_avg.append(
                ", ".join(avg_raw_stats.loc[stat].sort_values(ascending=False).head(x).index.tolist()))
            top_inter_i.append(", ".join(criteria_stats[EvaluationCriteria.inter_i].loc[stat]
                                         .sort_values(ascending=True).head(x).index.tolist()))
            top_inter_ii.append(", ".join(criteria_stats[EvaluationCriteria.inter_ii].loc[stat]
                                          .sort_values(ascending=True).head(x).index.tolist()))
            top_inter_iii.append(", ".join(criteria_stats[EvaluationCriteria.inter_iii].loc[stat]
                                           .sort_values(ascending=True).head(x).index.tolist()))
            top_disc_i.append(", ".join(criteria_stats[EvaluationCriteria.disc_i].loc[stat]
                                        .sort_values(ascending=True).head(x).index.tolist()))
            top_disc_ii.append(", ".join(criteria_stats[EvaluationCriteria.disc_ii].loc[stat]
                                         .sort_values(ascending=True).head(x).index.tolist()))
            top_disc_iii.append(", ".join(criteria_stats[EvaluationCriteria.disc_iii].loc[stat]
                                          .sort_values(ascending=True).head(x).index.tolist()))
            bottom_inter_i.append(", ".join(criteria_stats[EvaluationCriteria.inter_i].loc[stat]
                                            .sort_values(ascending=False).head(x).index.tolist()))
            bottom_inter_ii.append(", ".join(criteria_stats[EvaluationCriteria.inter_ii].loc[stat]
                                             .sort_values(ascending=False).head(x).index.tolist()))
            bottom_inter_iii.append(", ".join(criteria_stats[EvaluationCriteria.inter_iii].loc[stat]
                                              .sort_values(ascending=False).head(x).index.tolist()))
            bottom_disc_i.append(", ".join(criteria_stats[EvaluationCriteria.disc_i].loc[stat]
                                           .sort_values(ascending=False).head(x).index.tolist()))
            bottom_disc_ii.append(", ".join(criteria_stats[EvaluationCriteria.disc_ii].loc[stat]
                                            .sort_values(ascending=False).head(x).index.tolist()))
            bottom_disc_iii.append(", ".join(criteria_stats[EvaluationCriteria.disc_iii].loc[stat]
                                             .sort_values(ascending=False).head(x).index.tolist()))

        result = pd.DataFrame(index=avg_raw_stats.index[1:].tolist(), data={  # skip count from index
            DistanceInterpretation.top_rank: top_avg,
            DistanceInterpretation.bottom_rank: bottom_avg,
            DistanceInterpretation.raw_top_rank: raw_top_avg,
            DistanceInterpretation.raw_bottom_rank: raw_bottom_avg,
            DistanceInterpretation.top_inter_i: top_inter_i,
            DistanceInterpretation.top_inter_ii: top_inter_ii,
            DistanceInterpretation.top_inter_iii: top_inter_iii,
            DistanceInterpretation.top_disc_i: top_disc_i,
            DistanceInterpretation.top_disc_ii: top_disc_ii,
            DistanceInterpretation.top_disc_iii: top_disc_iii,
            DistanceInterpretation.bottom_inter_i: bottom_inter_i,
            DistanceInterpretation.bottom_inter_ii: bottom_inter_ii,
            DistanceInterpretation.bottom_inter_iii: bottom_inter_iii,
            DistanceInterpretation.bottom_disc_i: bottom_disc_i,
            DistanceInterpretation.bottom_disc_ii: bottom_disc_ii,
            DistanceInterpretation.bottom_disc_iii: bottom_disc_iii
        })
        # rename distance measures w    ith shorter names
        for old, new in short_distance_measure_names.items():
            result = result.apply(lambda x: x.str.replace(old, new))

        if save_results:
            result_dir = distance_measure_evaluation_results_dir_for(run_name=self.overall_ds_name,
                                                                     data_type=self.data_type,
                                                                     base_results_dir=self.root_results_dir,
                                                                     data_dir=self.data_dir)
            result.to_csv(str(path.join(result_dir, str(x) + '_' + DISTANCE_MEASURE_EVALUATION_TOP_BOTTOM_MEASURES)))

        return result

    def statistical_validation_of_two_measures_based_on_average_ranking(self, measure1: str, measure2: str,
                                                                        non_zero: float = 0.001):
        """
        Calculates the Wilcoxon signed rank test between measure 1 and 2. The difference in ranks has to be greater
        than non-zero to be considered. Zero difference pairs are removed
        :returns WilcoxResult
        """
        m1_ranks = self.average_rank_per_run[measure1]
        m2_ranks = self.average_rank_per_run[measure2]

        # remove zeros to avoid approximation
        differences = np.array(m1_ranks) - np.array(m2_ranks)
        nonzero_diffs = differences[np.abs(differences) > non_zero]  # don't consider too small differences
        stats.wilcoxon(nonzero_diffs, mode='exact')

        result = stats.wilcoxon(x=nonzero_diffs, zero_method='wilcox', mode='exact')
        stats_res = WilcoxResult(result.statistic, result.pvalue, len(m1_ranks), len(nonzero_diffs))
        return stats_res


def read_top_bottom_distance_measure_result(x: int, overall_ds_name: str, data_type: str,
                                            base_results_dir: str, data_dir: str):
    """
    Reads csv from results for top and bottom distance measures
    :param x: how many top and bottom measures were listed
    :return pd.DataFrame: rows are the describe statistics and columns are the top, bottom for average rank and all
    criteria
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_ds_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = str(x) + '_' + DISTANCE_MEASURE_EVALUATION_TOP_BOTTOM_MEASURES

    full_path = path.join(result_dir, file_name)
    df = pd.read_csv(str(full_path), index_col=0)
    return df
