from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.utils.configurations import Aggregators
from src.utils.distance_measures import distance_calculation_method_for
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class EvaluationCriteria:
    inter_i: str = "Interpretability: L_0 close to zero"
    inter_ii: str = "Interpretability: proper level sets ordering"
    inter_iii: str = "Interpretability: rate of increase between level sets"
    disc_i: str = "Discriminative Power: overall RC"
    disc_ii: str = "Discriminative Power: overall CV"
    disc_iii: str = "Discriminative Power: macro F1 score"
    stab_i: str = "Stability: completed"
    stab_ii: str = "Stability: count of nan and inf distances"


class DistanceMetricEvaluation:
    def __init__(self, ds: DescribeSyntheticDataset, measures: [], backend: str = Backends.none.value,
                 round_to: int = 3):
        self.__ds = ds
        self.backend = backend
        self.__measures = measures
        self.__round_to = round_to
        # dictionary of key level set id and values list of tuples of pattern pairs
        self.level_sets = ds.level_sets
        self.level_set_indices = list(self.level_sets.keys())
        self.distances_df = self.__calculate_distances_df()
        self.per_level_set_distance_statistics_df = self.__calculate_per_level_sets_distance_statistics()
        self.ci_for_mean_differences = None

    def rate_of_increase_between_level_sets(self):
        """
        Calculates rate of increase level sets for each distance measure
        :return pd.DataFrame with columns: level_set_pair, rate of increase
        """
        measures = []
        ls = []
        rate_of_increase = []
        # adjacent level sets
        ls_pairs = [(self.level_set_indices[i], self.level_set_indices[i + 1]) for i in
                    range(len(self.level_set_indices) - 1)]

        for measure in self.__measures:
            for ls1, ls2 in ls_pairs:
                ls.append((ls1, ls2))
                measures.append(measure)

                # calculate rate of increase
                df = self.per_level_set_distance_statistics_df
                df_ms = df.loc[(df[DistanceMeasureCols.type] == measure)]

                mean_low = df_ms.loc[(df_ms[DistanceMeasureCols.level_set] == ls1)][Aggregators.mean].values[0]
                mean_high = df_ms.loc[(df_ms[DistanceMeasureCols.level_set] == ls2)][Aggregators.mean].values[0]

                rate = round(abs(mean_high - mean_low), self.__round_to)
                rate_of_increase.append(rate)

        return pd.DataFrame({
            DistanceMeasureCols.compared: ls,
            DistanceMeasureCols.type: measures,
            DistanceMeasureCols.rate_of_increase: rate_of_increase,
        })

    def raw_results_for_each_criteria(self, round_to: int = 3):
        """ Calculates the raw results for each distance measure and each criterion as described in the paper.
        Rows are the criteria (see EvaluationCriteria), columns are criteria followed by the distance measures.
        :returns pd.Dataframe
        """
        # setup columns, indices and empty row arrays for dataframe
        columns = self.__measures.copy()
        indices = [
            EvaluationCriteria.inter_i,
            EvaluationCriteria.inter_ii,
            EvaluationCriteria.inter_iii,
            EvaluationCriteria.disc_i,
            EvaluationCriteria.disc_ii,
            EvaluationCriteria.disc_iii,
            EvaluationCriteria.stab_i,
            EvaluationCriteria.stab_ii
        ]

        inter_i = []
        inter_ii = []
        inter_iii = []
        disc_i = []
        disc_ii = []
        disc_iii = []
        stab_i = []
        stab_ii = []

        # mean distances for level set 0 indexed by distance measure
        distances_for_level_set0 = self.per_level_set_distance_statistics_df.loc[
            self.per_level_set_distance_statistics_df[DistanceMeasureCols.level_set] == 0][
            [DistanceMeasureCols.type, Aggregators.mean]].round(3).set_index(DistanceMeasureCols.type, drop=True)

        # statistical diff between adjacent level sets
        ls_pairs = [(self.level_sets[i], self.level_sets[i + 1]) for i in range(len(self.level_sets) - 1)]
        ci_adjacent = self.ci_for_mean_differences.loc[
            (self.ci_for_mean_differences[DistanceMeasureCols.compared]).isin(ls_pairs)]

        # rate of increase df
        rate_of_increase = self.rate_of_increase_between_level_sets()
        average_rate = rate_of_increase.groupby(DistanceMeasureCols.type)[
            DistanceMeasureCols.rate_of_increase].mean().round(round_to)

        # for each distance measure calculate all criteria
        for measure in self.__measures:
            inter_i.append(distances_for_level_set0.loc[measure, Aggregators.mean])
            inter_ii.append(
                ci_adjacent.loc[(ci_adjacent[DistanceMeasureCols.type] == measure)][DistanceMeasureCols.stat_diff].eq(
                    'lower').all())
            inter_iii.append(average_rate.loc[measure])
            disc_i.append(0)
            disc_ii.append(0)
            disc_iii.append(0)
            stab_i.append(0)
            stab_ii.append(0)

        data = [
            inter_i,
            inter_ii,
            inter_iii,
            disc_i,
            disc_ii,
            disc_iii,
            stab_i,
            stab_ii,
        ]
        return pd.DataFrame(data=data, columns=columns, index=indices)

    def __calculate_per_level_sets_distance_statistics(self):
        """
        Calculates the per level sets distance stastistics
        :return pd.DataFrame with columns:
                DistanceMeasureCols.type  -> name of distance measure
                DistanceMeasureCols.level_set - 0, 1, 2 ... for all the different level sets
                pandas describe columns (mean, std, min, max, 25%, etc
        """
        dist_df = self.distances_df

        # calculate per level set statistics for each distance level (creates column multiindex df with measure name,
        # and stats
        stats_df = dist_df.groupby(DistanceMeasureCols.level_set)[self.__measures].describe()
        # rename the multiindex to not have to deal with panda's defaults
        stats_df.columns.names = [DistanceMeasureCols.type, None]

        # stack -> moves first level of multiindex (distance measure) as a column
        # reset_index(level=1) -> resets the column index to be proper columns for stats
        # reset_index -> makes the current level_set index a column
        # sort_values -> orders by distance measure and then level set
        # round -> to limit accuracy
        result = stats_df.stack(level=0).reset_index(level=1).reset_index().sort_values(
            [DistanceMeasureCols.type, DistanceMeasureCols.level_set]).round(self.__round_to)

        return result

    def __calculate_distances_df(self):
        """
        Calculate dataframe of all the MxL distances d(Ax,Py) in the dataset
        :return: pd.dataframe with columns:
                DistanceMeasureCols.segment_id -> segment id
                DistanceMeasureCols.canonical_pattern_id -> canonical pattern id for P_x (where A_x was generated from)
                DistanceMeasureCols.compared_to_pattern_id -> canonical pattern id for P_y
                DistanceMeasureCols.level_set -> id of the level set that P_x and P_y belong to
                Measure_names -> column for each measure in self.__measures and the distance for that measure
        """
        # 1. Get the empirical correlation of each segment, and it's canonical pattern and the pattern id
        # columns segment_id (index), pattern id, correlation_to_model, actual correlation
        segment_correlations = self.__ds.labels[
            [SyntheticDataSegmentCols.segment_id, SyntheticDataSegmentCols.pattern_id,
             SyntheticDataSegmentCols.correlation_to_model, SyntheticDataSegmentCols.actual_correlation]]
        segment_correlations.set_index(SyntheticDataSegmentCols.segment_id, inplace=True)

        # 2. Get a lookup of what level set which pattern_id pair belong to, this is key pattern pair (both ways round)
        # value level set
        level_set_lookup = {}
        for key, pattern_tuples_list in self.level_sets.items():
            for pattern_pairs in pattern_tuples_list:
                level_set_lookup[pattern_pairs] = key
                # add in both order to not need to worry
                level_set_lookup[(pattern_pairs[1], pattern_pairs[0])] = key

        @lru_cache(maxsize=None)
        def get_level_set_for_pattern_pairs(pattern_pair):
            return level_set_lookup.get(pattern_pair) or level_set_lookup.get((pattern_pair[1], pattern_pair[0]))

        # 3. Create df of unique canonical patterns and their correlation to model
        unique_patterns_df = segment_correlations[
            [SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.correlation_to_model]].drop_duplicates(
            subset=[SyntheticDataSegmentCols.pattern_id])
        n_patterns = len(unique_patterns_df)

        # 3. For each segment and each distance measure calculate the distances for all measures to all other segment_ids
        segment_ids = []
        p_x_ids = []
        p_y_ids = []
        resulting_distances = {measure: [] for measure in self.__measures}

        for seg_id in segment_correlations.index:
            # ax changes for each seg_id so we're only looping 100 times as we do the rest as vector
            a_x = segment_correlations.loc[seg_id, SyntheticDataSegmentCols.actual_correlation]
            p_x = segment_correlations.loc[seg_id, SyntheticDataSegmentCols.pattern_id]
            segment_ids.extend([seg_id] * n_patterns)
            p_x_ids.extend([p_x] * n_patterns)
            p_y_ids.extend(unique_patterns_df[SyntheticDataSegmentCols.pattern_id].to_list())

            for measure in self.__measures:
                calc_distance = distance_calculation_method_for(measure)
                # Calculate distances to all segments correlation_to_model (which are the canonical patterns)
                distances = unique_patterns_df[SyntheticDataSegmentCols.correlation_to_model].apply(
                    lambda x: calc_distance(a_x, x))
                # extend does not create a list of list but instead adds each element to the list
                resulting_distances[measure].extend(distances)

        # look up the level set for the pair compared
        level_sets = [get_level_set_for_pattern_pairs((px, py)) for px, py in zip(p_x_ids, p_y_ids)]

        # Create df from lists
        resulting_df = pd.DataFrame({
            DistanceMeasureCols.segment_id: segment_ids,
            DistanceMeasureCols.canonical_pattern_id: p_x_ids,
            DistanceMeasureCols.compared_to_pattern_id: p_y_ids,
            DistanceMeasureCols.level_set: level_sets
        })

        # add results from distance measures
        for measure, values in resulting_distances.items():
            resulting_df[measure] = values
        return resulting_df
