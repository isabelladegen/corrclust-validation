from os import path

import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria
from src.utils.configurations import distance_measure_evaluation_results_dir_for, \
    DISTANCE_MEASURE_EVALUATION_CRITERIA_RANKS_RESULTS_FILE, DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE


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
            EvaluationCriteria.stab_ii: True  # fewer inf and nans
        }

    def ranking_df_for_ds(self, run_name: str, root_results_dir: str = None, data_type: str = None,
                          data_dir: str = None):
        """
        Returns dataframe of criterion level ranks, key principle ranks and overall ranks for the distance_measure
        :param run_name: the run name you want to rank
        :param root_results_dir: if not None we save the df using that dir
        :param data_type: the data type, see SyntheticDataType
        :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
        :return pd.Dataframe with EvaluationCriteria as rows (and index) and columns the distance measures
        """
        raw_df = self.raw_criteria_data[run_name]
        # check that we have results for all distance measures (fails if not)
        error_msg = "Raw Criteria contains results for distance measures: " + str(
            raw_df.columns) + ". We want to evaluate measures: " + str(self.distance_measures)
        assert all(col in raw_df.columns for col in self.distance_measures), error_msg
        highest_rank = len(self.distance_measures)

        # build a new ranked df
        ranked_df = pd.DataFrame(index=raw_df.index, columns=raw_df.columns)

        for idx in raw_df.index:
            what_is_best = self.ranking_criteria[idx]
            row_values = raw_df.loc[idx]

            if what_is_best == 'boolean':
                # Convert bools to int, then judge higher (=True) is better but average
                rankings = row_values.astype(int).rank(ascending=False, method='average')
            else:  # for numeric rows use pandas ranking for series
                rankings = row_values.rank(ascending=what_is_best, method='average')

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

        for idx in ranked_df.index:
            ds_ranks = self.ranking_df_for_ds(idx, root_results_dir=root_results_dir, data_type=data_type,
                                              data_dir=data_dir)
            ranked_df.loc[idx] = ds_ranks.mean()
        ranked_df = ranked_df.astype(float).round(decimals=self.__round_to)

        if root_results_dir is not None:  # save results as csv
            result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_ds_name,
                                                                     data_type=data_type,
                                                                     base_results_dir=root_results_dir,
                                                                     data_dir=data_dir)
            ranked_df.to_csv(str(path.join(result_dir, DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE)))

        return ranked_df


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
    """ Reads the raw criteria csv from the provided folder
      :param overall_run_name: overall name such as, n30
      :param data_type: the data type, see SyntheticDataType
      :param base_results_dir: the directory for results, this is the main directory usually results or test results
      :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
      :returns pd.DataFrame: of the rank for the raw criteria values as row and distance measures as columns
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=overall_run_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = DISTANCE_MEASURE_EVALUATION_OVERALL_RANKS_RESULTS_FILE

    full_path = path.join(result_dir, file_name)
    df = pd.read_csv(str(full_path), index_col=0)
    return df.astype(float)
