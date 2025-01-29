from dataclasses import dataclass
from os import path

import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria
from src.utils.configurations import distance_measure_evaluation_results_dir_for, \
    DISTANCE_MEASURE_EVALUATION_CRITERIA_RANKS_RESULTS_FILE, DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE, \
    DISTANCE_MEASURE_EVALUATION_AVERAGE_RANKS_PER_CRITERIA_RESULTS_FILE


@dataclass
class RankingStats:
    best: str = 'best'


class DistanceMetricRanking:
    def __init__(self, raw_criteria_data: {}, distance_measures: [], round_to: int = 3):
        """
        Class to rank distance metric both for one or multiple datasets. Rank 1 is best!
        :param raw_criteria_data: dictionary with key=ds-name and value= DistanceMeasureEvaluation df of raw values
        :param distance_measures: list of distance measures (see DistanceMeasures for valid values) that will be ranked
        we assume the raw_criteria_data includes the results for each measure, if it includes more, only the ones
        provided here will be ranked
        """
        self.raw_criteria_data = raw_criteria_data
        self.distance_measures = distance_measures
        self.__round_to = round_to
        # lookup dictionary with instruction how to rank the raw data based on value, rank 1 is best:
        # True: ascending -> higher values get higher ranks, i.e.lower values are better
        # False: descending -> lower values get higher ranks, i.e. higher values are better
        # 'boolean': special case for pass/fail criterion -> True best rank
        self.ranking_criteria = {
            EvaluationCriteria.inter_i: True,  # L0 closer to zero
            EvaluationCriteria.inter_ii: 'boolean',  # significant differences LS
            EvaluationCriteria.inter_iii: False,  # higher average rate of increase
            EvaluationCriteria.disc_i: False,  # higher overall entropy
            EvaluationCriteria.disc_ii: True,  # lower average LS entropy
            EvaluationCriteria.disc_iii: False,  # higher F1 score
        }

    def ranking_df_for_ds(self, run_name: str, root_results_dir: str = None, data_type: str = None,
                          data_dir: str = None):
        """
        Returns dataframe of criterion level ranks, key principle ranks and overall ranks for the distance_measure
        :param run_name: the run name you want to rank
        :param root_results_dir: if not None we save the df using that dir
        :param data_type: the data type, see SyntheticDataType
        :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
        :return pd.Dataframe with self.rankingCriteria as rows (and index) and columns the distance measures
        """
        raw_df = self.raw_criteria_data[run_name]
        # check that we have results for all distance measures (fails if not)
        error_msg = "Raw Criteria contains results for distance measures: " + str(
            raw_df.columns) + ". We want to evaluate measures: " + str(self.distance_measures)
        assert all(col in raw_df.columns for col in self.distance_measures), error_msg

        # find common ranking criteria
        criteria_to_rank = [x for x in raw_df.index if x in self.ranking_criteria]

        # build a new ranked df
        ranked_df = pd.DataFrame(index=criteria_to_rank, columns=raw_df.columns)

        for idx in ranked_df.index:
            what_is_best = self.ranking_criteria[idx]
            row_values = raw_df.loc[idx]

            if what_is_best == 'boolean':
                # Convert bools to int, then judge higher (=True) is better
                rankings = row_values.astype(int).rank(ascending=False, method='dense')
            else:  # for numeric rows use pandas ranking for series
                rankings = row_values.rank(ascending=what_is_best, method='dense')

            ranked_df.loc[idx] = rankings

        ranked_df = ranked_df.astype(float)
        if root_results_dir is not None:  # save results as csv
            result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name,
                                                                     data_type=data_type,
                                                                     base_results_dir=root_results_dir,
                                                                     data_dir=data_dir)
            ranked_df.to_csv(str(path.join(result_dir, DISTANCE_MEASURE_EVALUATION_CRITERIA_RANKS_RESULTS_FILE)))
        return ranked_df

    def calculate_overall_rank(self, overall_ds_name: str = "", root_results_dir: str = None, data_type: str = None,
                               data_dir: str = None):
        """
        Returns dataframe overall rank for each distance_measure per dataset
        :param overall_ds_name: name for the dataset overall for when we want to save the result for a dir, e.g. N30
        :param root_results_dir: if not None we save the df using that dir
        :param data_type: the data type, see SyntheticDataType
        :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
        :return pd.Dataframe with run_names as rows (and index) and columns the distance measures
        """
        # build a new overall rank df
        ranked_df = pd.DataFrame(index=self.raw_criteria_data.keys(), columns=self.distance_measures)

        for run_name in ranked_df.index:
            ds_ranks = self.ranking_df_for_ds(run_name, root_results_dir=root_results_dir, data_type=data_type,
                                              data_dir=data_dir)
            ranked_df.loc[run_name] = ds_ranks.mean()
        ranked_df = ranked_df.astype(float).round(decimals=self.__round_to)

        if root_results_dir is not None:  # save results as csv
            result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_ds_name,
                                                                     data_type=data_type,
                                                                     base_results_dir=root_results_dir,
                                                                     data_dir=data_dir)
            ranked_df.to_csv(str(path.join(result_dir, DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE)))

        return ranked_df

    def calculate_criteria_level_average_rank(self, overall_ds_name: str = "", root_results_dir: str = None,
                                              data_type: str = None, data_dir: str = None):
        """
        Returns dataframe of average rank for each measure and each criterion
        :param overall_ds_name: name for the dataset overall for when we want to save the result for a dir, e.g. n30
        :param root_results_dir: if not None, we save the df using that dir
        :param data_type: the data type, see SyntheticDataType
        :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
        :return pd.Dataframe with criteria as rows (and index, see EvaluationCriteria) and columns the distance measures
        each cell is the average rank across all the runs for that data variant. Last column is 'best' which is the
        name of the lowest ranked distance measure(s) per row
        """
        # 1. load all criteria level ranking for each run
        criteria_rankings = []
        for run_name in self.raw_criteria_data.keys():
            criteria_rankings.append(self.ranking_df_for_ds(run_name, root_results_dir=root_results_dir,
                                                            data_type=data_type, data_dir=data_dir))

        # 2. Stack all dataframes cells
        stacked = np.stack([df.values for df in criteria_rankings])

        # 3. Calculate average for each criterion (rows) and each measure (columns) across runs (stacks)
        avg_ranks = np.mean(stacked, axis=0).round(self.__round_to)

        # 4. Create new dataframe with index criteria and columns distance measures but values averages across runs
        average_ranking_df = pd.DataFrame(
            avg_ranks,
            index=criteria_rankings[0].index,
            columns=criteria_rankings[0].columns
        )

        # 5. Add best measure column
        average_ranking_df[RankingStats.best] = average_ranking_df.apply(
            lambda row: ', '.join(row.index[row == row.min()]),
            axis=1
        )

        # 6. Save results (optional)
        if root_results_dir is not None:
            result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_ds_name,
                                                                     data_type=data_type,
                                                                     base_results_dir=root_results_dir,
                                                                     data_dir=data_dir)
            average_ranking_df.to_csv(
                str(path.join(result_dir, DISTANCE_MEASURE_EVALUATION_AVERAGE_RANKS_PER_CRITERIA_RESULTS_FILE)))

        return average_ranking_df


def read_csv_of_ranks_for_all_criteria(run_name: str, data_type: str, data_dir: str,
                                       base_results_dir: str):
    """ Reads the raw criteria csv from the provided folder
      :param run_name: a name for the run, e.g. wandb run_name
      :param data_type: the data type, see SyntheticDataType
      :param base_results_dir: the directory for results, this is the main directory usually results or test results
      :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
      :returns pd.DataFrame: of the rank for the raw criteria values as row and distance measures as columns
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = DISTANCE_MEASURE_EVALUATION_CRITERIA_RANKS_RESULTS_FILE

    full_path = path.join(result_dir, file_name)
    df = pd.read_csv(str(full_path), index_col=0)
    return df.astype(float)


def read_csv_of_overall_rank_per_dataset(overall_run_name: str, data_type: str, data_dir: str,
                                         base_results_dir: str):
    """ Reads the overall rank csv from the provided folder
      :param overall_run_name: overall name such as, n30
      :param data_type: the data type, see SyntheticDataType
      :param base_results_dir: the directory for results, this is the main directory usually results or test results
      :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
      :returns pd.DataFrame: of the avg rank rows (run names), columns (distance metric), cells average rank accross
      criteria for each run and distance measure
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_run_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE

    full_path = path.join(result_dir, file_name)
    df = pd.read_csv(str(full_path), index_col=0)
    return df.astype(float)


def read_csv_of_average_criteria_across_datasets(overall_run_name: str, data_type: str, data_dir: str,
                                                 base_results_dir: str):
    """ Reads the average criteria rank csv from the provided folder
      :param overall_run_name: overall name such as, n30
      :param data_type: the data type, see SyntheticDataType
      :param base_results_dir: the directory for results, this is the main directory usually results or test results
      :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
      :returns pd.DataFrame: of the average rank rows (criterion - see EvaluationCriteria), columns
      (distance measures), cells average rank of that criterion for that distance measure across runs
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_run_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = DISTANCE_MEASURE_EVALUATION_AVERAGE_RANKS_PER_CRITERIA_RESULTS_FILE

    full_path = path.join(result_dir, file_name)
    df = pd.read_csv(str(full_path), index_col=0)
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(float)
    return df
