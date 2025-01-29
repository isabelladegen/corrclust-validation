import ast
from dataclasses import dataclass
from functools import lru_cache
from os import path

import numpy as np
import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.evaluation.distance_metric_assessment import DistanceMeasureCols, \
    calculate_ci_of_mean_differences_between_two_values_for_distance_measures
from src.evaluation.knn_for_synthetic_wrapper import KNNForSyntheticWrapper
from src.utils.configurations import Aggregators, distance_measure_evaluation_results_dir_for, \
    DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE
from src.utils.distance_measures import distance_calculation_method_for, DistanceMeasures
from src.utils.labels_utils import find_all_level_sets
from src.utils.load_synthetic_data import load_labels
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class EvaluationCriteria:
    inter_i: str = "Interpretability: L_0 close to zero"
    inter_ii: str = "Interpretability: levels sets sig different and correct order"
    inter_iii: str = "Interpretability: higher rate of increase between level sets"
    disc_i: str = "Discriminative Power: higher overall entropy"
    disc_ii: str = "Discriminative Power: lower level set entropy"
    disc_iii: str = "Discriminative Power: higher macro F1 score"
    stab_ii: str = "Stability: fewer nan and inf distances"


criteria_short_names = {
    EvaluationCriteria.inter_i: '1. L_0=0',
    EvaluationCriteria.inter_ii: '2. L_d diff',
    EvaluationCriteria.inter_iii: '3. L_d inc',
    EvaluationCriteria.disc_i: '4. H_D',
    EvaluationCriteria.disc_ii: '5. H_L',
    EvaluationCriteria.disc_iii: '6. F1',
}

# criteria for which lower values are better
inverse_criteria = [criteria_short_names[EvaluationCriteria.inter_i], criteria_short_names[EvaluationCriteria.disc_ii]]


class DistanceMetricEvaluation:
    def __init__(self, run_name: str, data_type: str, data_dir: str, measures: [], backend: str = Backends.none.value,
                 round_to: int = 3):
        self.backend = backend
        self.run_name = run_name
        self.data_type = data_type
        self.data_dir = data_dir
        self.__measures = measures
        self.__round_to = round_to
        self.__labels = load_labels(self.run_name, self.data_type, data_dir=data_dir)
        # dictionary of key level set id and values list of tuples of pattern pairs
        self.level_sets = find_all_level_sets(self.__labels)
        self.level_set_indices = list(self.level_sets.keys())
        # adjacent level sets
        self.adjacent_level_set_indices = [(self.level_set_indices[i], self.level_set_indices[i + 1]) for i in
                                           range(len(self.level_set_indices) - 1)]
        # calculate distances
        self.distances_df = self.__calculate_distances_df()
        # calculate statistics
        self.per_level_set_distance_statistics_df = self.__calculate_per_level_sets_distance_statistics()
        self.ci_for_mean_differences, self.alpha_for_level_set_ci = self.__calculate_ci_for_mean_differences_between_adjacent_level_sets()
        self.normalised_distance_df = self.__normalise_distances()

    def rate_of_increase_between_level_sets(self):
        """
        Calculates rate of increase level sets for each distance measure
        :return pd.DataFrame with columns: level_set_pair, rate of increase
        """
        measures = []
        ls = []
        rate_of_increase = []
        ls_pairs = self.adjacent_level_set_indices

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

    def calculate_level_set_shannon_entropy(self, n_bins: int = 50):
        """
        Calculates the Shannon's entropy for each distance measure per level set
        Measures with higher entropy have more diverse distances in the level set, measure with lower entropy have more
        similar distances
        :param n_bins: the number of bins used in the histogram. Each bin has the width 1/n_bins given we use
        normalised distances
        :returns pd.DataFrame with columns
            DistanceMeasureCols.level_set -> which level set the entropy is for
            a column per distance measure
        """
        level_sets = self.level_set_indices
        results = {measure: [] for measure in self.__measures}
        df = self.normalised_distance_df

        # calculate entropy for each measure
        for measure in self.__measures:
            for level_set in level_sets:
                distances = df.loc[(df[DistanceMeasureCols.level_set] == level_set), measure]
                entropy = self.calculate_shannon_entropy_of_distances(distances, n_bins)
                results[measure].append(entropy)

        # construct resulting dataframe
        results[DistanceMeasureCols.level_set] = level_sets
        df = pd.DataFrame(results)
        # reorder columns
        ordered = [DistanceMeasureCols.level_set]
        ordered.extend(self.__measures)
        df = df[ordered]
        return df

    def calculate_overall_shannon_entropy(self, n_bins: int = 50):
        """Calculates the Shannon's entropy for each distance measure overall distances
        Measures with higher entropy have more diverse distances, measure with lower entropy have more
        similar distances
        :param n_bins: the number of bins used in the histogram. Each bin has the width 1/n_bins given we use
        normalised distances
        :returns a dictionary with key=distance measure and value=entropy
        """
        results = {}

        # calculate entropy for each measure
        for measure in self.__measures:
            # finite normalised distances for measure (non nan or inf)
            distances = self.normalised_distance_df[measure]
            results[measure] = self.calculate_shannon_entropy_of_distances(distances, n_bins)

        return results

    def calculate_shannon_entropy_of_distances(self, distances: pd.Series, n_bins: int, range=(0, 1)):
        """Calculates the Shannon's entropy for the distances given
        :param distances: pd.Series of distances (if not normalised provide appropriate range for histogram)
        :param n_bins: how many bins
        :param range: optional, defaults to (0,1) for normalised distance
        :return shannon_entropy value
        """
        finite_dist = distances[np.isfinite(distances)]
        # Calculate histogram on normalised distances, hence range (0,1)
        hist, _ = np.histogram(finite_dist, bins=n_bins, range=range)
        # Calculate probabilities for non-empty bins (we only sum over the positive p(i) values)
        probs = hist[hist > 0] / len(finite_dist)
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        return round(entropy, self.__round_to)

    def raw_results_for_each_criteria(self, round_to: int = 3):
        """ Calculates the raw results for each distance measure and each criterion as described in the paper.
        Rows are the criteria (see EvaluationCriteria), columns are criteria followed by the distance measures.
        :returns pd.Dataframe with EvaluationCriteria as rows (and index) and columns the distance measures
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
            EvaluationCriteria.stab_ii
        ]

        inter_i = []
        inter_ii = []
        inter_iii = []
        disc_i = []
        disc_ii = []
        disc_iii = []
        stab_ii = []

        # mean distances for level set 0 indexed by distance measure
        distances_for_level_set0 = self.per_level_set_distance_statistics_df.loc[
            self.per_level_set_distance_statistics_df[DistanceMeasureCols.level_set] == 0][
            [DistanceMeasureCols.type, Aggregators.mean]].round(3).set_index(DistanceMeasureCols.type, drop=True)

        # rate of increase df
        rate_of_increase = self.rate_of_increase_between_level_sets()
        average_rate = rate_of_increase.groupby(DistanceMeasureCols.type)[
            DistanceMeasureCols.rate_of_increase].mean().round(round_to)

        # overall entropy df
        n_bins = 50
        overall_entropy = self.calculate_overall_shannon_entropy(n_bins=n_bins)

        # level set entropy values
        level_set_entropy = self.calculate_level_set_shannon_entropy(n_bins=n_bins)

        # 1NN x_test and y_true
        # true pattern that they should be predicted to
        y_true = self.__labels[SyntheticDataSegmentCols.pattern_id].to_numpy()
        # actual correlations achieved in the data
        x_test = np.array(self.__labels[SyntheticDataSegmentCols.actual_correlation].to_list())

        # count nan and infs
        stability_count = self.count_nan_inf_distance_for_measures()

        # for each distance measure calculate all criteria
        for measure in self.__measures:
            # Interpretability Criteria
            inter_i.append(distances_for_level_set0.loc[measure, Aggregators.mean])
            inter_ii.append(
                self.ci_for_mean_differences[self.ci_for_mean_differences[DistanceMeasureCols.type] == measure][
                    DistanceMeasureCols.stat_diff].eq('lower').all())
            inter_iii.append(average_rate.loc[measure])
            # Discriminative power Criteria
            disc_i.append(overall_entropy[measure])
            disc_ii.append(level_set_entropy[measure].mean().round(self.__round_to))
            # 1NN train and calculate f1
            knn_for_measure = KNNForSyntheticWrapper(measure=measure, n_neighbors=1, data_dir=self.data_dir,
                                                     backend=self.backend)
            knn_for_measure.predict(x_test, y_true)
            accuracy, precision, recall, f1 = knn_for_measure.evaluate_scores(average="macro", round_to=self.__round_to)
            disc_iii.append(f1)
            # Stability criteria
            stab_ii.append(stability_count[measure])

        data = [
            inter_i,
            inter_ii,
            inter_iii,
            disc_i,
            disc_ii,
            disc_iii,
            stab_ii,
        ]
        return pd.DataFrame(data=data, columns=columns, index=indices)

    def __calculate_per_level_sets_distance_statistics(self):
        """
        Calculates the per level sets distance statistics
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
        result = stats_df.stack(level=0, future_stack=True).reset_index(level=1).reset_index().sort_values(
            [DistanceMeasureCols.type, DistanceMeasureCols.level_set]).round(self.__round_to)

        return result

    def __calculate_distances_df(self):
        """
        Calculate dataframe of all the MxL distances d(Ax,Py) in the dataset
        :return: pd.dataframe with columns:
                DistanceMeasureCols.segment_id -> segment id
                DistanceMeasureCols.canonical_pattern_id -> canonical pattern id for P'_x (where A_x was generated from)
                DistanceMeasureCols.compared_to_pattern_id -> canonical pattern id for P_y
                DistanceMeasureCols.a_x -> segments correlation coefficients
                DistanceMeasureCols.p_x -> relaxed pattern correlation coefficients
                DistanceMeasureCols.level_set -> id of the level set that P_x and P_y belong to
                Measure_names -> column for each measure in self.__measures and the distance for that measure
        """
        # 1. Get the empirical correlation of each segment, and it's canonical pattern and the pattern id
        # columns segment_id (index), pattern id, correlation_to_model, actual correlation
        segment_correlations = self.__labels[
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
        unique_patterns_df = segment_correlations[SyntheticDataSegmentCols.pattern_id].drop_duplicates().to_frame()
        # find relaxed patterns for the ids to calculate distances to patterns that are valid correlations
        # to ensure that proper correlation distances are not wrongly punished
        relaxed_patterns = ModelCorrelationPatterns().relaxed_patterns()
        unique_patterns_df[SyntheticDataSegmentCols.correlation_to_model] = unique_patterns_df[
            SyntheticDataSegmentCols.pattern_id].map(relaxed_patterns)

        n_patterns = len(unique_patterns_df)

        # 4. For each segment and each distance measure calculate the distances for all measures to all other segment_ids
        segment_ids = []
        p_x_ids = []
        p_y_ids = []
        a_xs = []
        p_xs = []
        resulting_distances = {measure: [] for measure in self.__measures}

        for seg_id in segment_correlations.index:
            # ax changes for each seg_id so we're only looping 100 times as we do the rest as vector
            a_x = segment_correlations.loc[seg_id, SyntheticDataSegmentCols.actual_correlation]
            p_x = segment_correlations.loc[seg_id, SyntheticDataSegmentCols.pattern_id]
            segment_ids.extend([seg_id] * n_patterns)
            p_x_ids.extend([p_x] * n_patterns)
            p_y_ids.extend(unique_patterns_df[SyntheticDataSegmentCols.pattern_id].to_list())
            a_xs.extend([a_x] * n_patterns)
            p_xs.extend(unique_patterns_df[SyntheticDataSegmentCols.correlation_to_model].to_list())

            for measure in self.__measures:
                calc_distance = distance_calculation_method_for(measure)
                if seg_id == 1 and p_x == 1 and measure == DistanceMeasures.foerstner_cor_dist:
                    print("check")
                # Calculate distances to all segments correlation_to_model (which are the relaxed canonical patterns)
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
            DistanceMeasureCols.level_set: level_sets,
            DistanceMeasureCols.a_x: a_xs,
            DistanceMeasureCols.relaxed_p_x: p_xs
        })

        # add results from distance measures
        for measure, values in resulting_distances.items():
            resulting_df[measure] = values
        return resulting_df

    def __calculate_ci_for_mean_differences_between_adjacent_level_sets(self, alpha: float = 0.05,
                                                                        bonferroni: bool = True,
                                                                        two_tailed: bool = True):
        """
        Calculates ci of mean difference between adjacent level sets
        :returns pd.DataFrame with columns (below) and alpha value used for calc
            DistanceMeasureCols.compared -> tuple of the values compared
            DistanceMeasureCols.type -> distance measure name
            DistanceMeasureCols.stat_diff -> interpretation: overlap, lower, higher (comparing mean of value1 to value2)
            DistanceMeasureCols.mean_diff -> mean diff
            ConfidenceIntervalCols.ci_96lo -> lo ci (not necessary 95% as this depends on alpha)
            ConfidenceIntervalCols.ci_96hi -> high ci (not necessary 95% as this depends on alpha)
            ConfidenceIntervalCols.width -> diff of hi-low ci
            ConfidenceIntervalCols.standard_error -> standard_error for ci
        """
        stats = self.per_level_set_distance_statistics_df
        a, ci_mean_diff_df = calculate_ci_of_mean_differences_between_two_values_for_distance_measures(
            self.adjacent_level_set_indices, stats,
            DistanceMeasureCols.level_set,
            alpha,
            bonferroni, two_tailed)
        return ci_mean_diff_df, a

    def __normalise_distances(self):
        """Calculates the distances normalised to [0,1] for each measure, we round these to round_to (defaulted to 3
        decimals as a meaningful accuracy without over differentiation of small differences
        :returns df with same columns as distance_df but normalised distances
        """
        result = self.distances_df.copy()

        # vectorised calculation over all distance columns at the same time
        df = self.distances_df[self.__measures]
        result[self.__measures] = (df - df.min()) / (df.max() - df.min())

        return result.round(self.__round_to)

    def count_nan_inf_distance_for_measures(self):
        """Counts number of inf and nan distances for each distance measure overall distances
        :returns a dictionary with key=distance measure and value=count
        """
        results = {}

        for measure in self.__measures:
            measure_results = self.distances_df[measure]
            nan_count = measure_results.isna().sum()
            inf_count = np.isinf(measure_results).sum()
            results[measure] = nan_count + inf_count
        return results

    def save_csv_of_raw_values_for_all_criteria(self, run_name: str, base_results_dir: str):
        """ Saves the raw criteria values dataframe in the results folder as csv file"""
        result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name,
                                                                 data_type=self.data_type,
                                                                 base_results_dir=base_results_dir,
                                                                 data_dir=self.data_dir)
        results_df = self.raw_results_for_each_criteria(round_to=self.__round_to)
        results_df.to_csv(path.join(result_dir, DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE))


def read_csv_of_raw_values_for_all_criteria(run_name: str, data_type: str, data_dir: str,
                                            base_results_dir: str):
    """ Reads the raw criteria csv from the provided folder
      :param run_name: name for the run, e.g. wandb run_name
      :param data_type: the data type, see SyntheticDataType
      :param base_results_dir: the directory for results, this is the main directory usually results or test results
      :param data_dir: the directory from which the data was read to be able to add the irregular folder if required
      :returns pd.DataFrame: of the raw criteria values as row and distance measures as columns
    """
    result_dir = distance_measure_evaluation_results_dir_for(run_name=run_name,
                                                             data_type=data_type,
                                                             base_results_dir=base_results_dir,
                                                             data_dir=data_dir)
    file_name = DISTANCE_MEASURE_EVALUATION_CRITERIA_RESULTS_FILE
    full_path = path.join(result_dir, file_name)
    columns = pd.read_csv(full_path, nrows=0, index_col=0).columns
    pd.read_csv(full_path, index_col=0, converters={col: ast.literal_eval for col in columns})
    return pd.read_csv(full_path, index_col=0, converters={col: ast.literal_eval for col in columns})
