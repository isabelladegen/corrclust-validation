import numpy as np
import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria


class DistanceMetricRanking:
    def __init__(self, raw_criteria_data: {}, distance_measures: []):
        """
        Class to rank distance metric both for one or multiple datasets. Rank 1 is best!
        :param raw_criteria_data: dictionary with key=ds-name and value= DistanceMeasureEvaluation df of raw values
        :param distance_measures: list of distance measures (see DistanceMeasures for valid values) that will be ranked
        we assume the raw_criteria_data includes the results for each measure, if it includes more, only the ones
        provided here will be ranked
        """
        self.raw_criteria_data = raw_criteria_data
        self.distance_measures = distance_measures
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

    def ranking_df_for_ds(self, a_ds_name):
        """
        Returns dataframe of criterion level ranks, key principle ranks and overall ranks for the distance_measure
        """
        raw_df = self.raw_criteria_data[a_ds_name]
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

        return ranked_df
