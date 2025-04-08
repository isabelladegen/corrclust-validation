from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import GeneralisedCols, SyntheticDataVariates
from src.utils.plots.matplotlib_helper_functions import Backends
from src.data_generation.generate_synthetic_correlated_data import GenerateData, calculate_spearman_correlation, \
    check_correlations_are_within_original_strength, calculate_pearson_correlation, calculate_kendalltau_correlation


@dataclass
class CorrType:
    spearman = "spearman"
    pearson = "pearson"
    kendall = "kendall"


def min_max_scaled_df(df: pd.DataFrame, scale_range: (), columns: []) -> pd.DataFrame:
    df_ = df.copy()
    d = len(columns)
    sz = df.shape[0]
    # X : array-like of shape (n_ts, sz, d) -> (1, n_observation, n_variates)
    x = df[columns].to_numpy().reshape(1, sz, d)

    scaler = TimeSeriesScalerMinMax(value_range=scale_range)
    x_scaled = scaler.fit_transform(x)
    x_scaled = x_scaled.reshape(sz, d)  # reshape back into 2d array

    for idx, col in enumerate(columns):
        df_[col] = x_scaled[:, idx]

    return df_


def recalculate_labels_df_from_data(data_df: pd.DataFrame, labels_df: pd.DataFrame, corr_type: str = CorrType.spearman):
    """
    Returns a new labels df with updated achieved correlation, correlations within tolerance and
    MAE and relaxed MAE columns calculated from the data_df provided
    """
    achieved_corrs = []
    within_tols = []
    # iterate through all segments and recalculate achieved correlation, keep data ordered by pattern_id
    for idx, row in labels_df.iterrows():
        start_idx = row[SyntheticDataSegmentCols.start_idx]
        end_idx = row[SyntheticDataSegmentCols.end_idx]
        length = row[SyntheticDataSegmentCols.length]

        # select data
        segment_df = data_df[SyntheticDataVariates.columns()].iloc[start_idx:end_idx + 1]
        segment = segment_df.to_numpy()
        assert segment.shape[0] == length, "Mistake with indexing dataframe"

        # calculated
        if corr_type == CorrType.spearman:
            achieved_cor = calculate_spearman_correlation(segment)
        elif corr_type == CorrType.pearson:
            achieved_cor = calculate_pearson_correlation(segment)
        elif corr_type == CorrType.kendall:
            achieved_cor = calculate_kendalltau_correlation(segment)
        else:
            assert False, "Unknown correlation type"
        within_tol = check_correlations_are_within_original_strength(row[SyntheticDataSegmentCols.correlation_to_model],
                                                                     achieved_cor)

        # store results
        achieved_corrs.append(achieved_cor)
        within_tols.append(within_tol)

    # update df
    new_labels_df = labels_df.copy()
    new_labels_df[SyntheticDataSegmentCols.actual_correlation] = achieved_corrs
    new_labels_df[SyntheticDataSegmentCols.actual_within_tolerance] = within_tols
    # calculate MAE per segment
    result = mean_absolute_error_from_labels_df(new_labels_df)
    new_labels_df[SyntheticDataSegmentCols.mae] = result[0]
    new_labels_df[SyntheticDataSegmentCols.relaxed_mae] = result[1]
    return new_labels_df


def mean_absolute_error_from_labels_df(labels_df: pd.DataFrame, round_to: int = 3):
    """Recalculate just the sum of absolute error_to_canonical_pattern between correlation to model and correlation achieved
    for each segment both to canonical pattern and relaxed pattern taken from correlation_patterns csv
    """
    # for the moment we get
    model_correlation_patterns = ModelCorrelationPatterns()
    canonical_patterns_lookup = model_correlation_patterns.canonical_patterns()
    relaxed_patterns_lookup = model_correlation_patterns.relaxed_patterns()

    n = len(labels_df.loc[0, SyntheticDataSegmentCols.correlation_to_model])
    canonical_patterns = np.array(
        labels_df[SyntheticDataSegmentCols.pattern_id].map(canonical_patterns_lookup).to_list())
    relaxed_patterns = np.array(labels_df[SyntheticDataSegmentCols.pattern_id].map(relaxed_patterns_lookup).to_list())
    achieved_correlations = np.array(labels_df[SyntheticDataSegmentCols.actual_correlation].to_list())
    error_canonical = np.round(np.sum(abs(np.array(canonical_patterns) - np.array(achieved_correlations)), axis=1) / n,
                               round_to)
    error_relaxed = np.round(np.sum(abs(np.array(relaxed_patterns) - np.array(achieved_correlations)), axis=1) / n,
                             round_to)
    return error_canonical, error_relaxed


@dataclass
class SyntheticDataSegmentCols:  # todo rename to labels cols
    old_regular_id = "old id"  # todo this really is a column of data not labels
    segment_id = "id"
    start_idx = "start idx"
    end_idx = "end idx"
    length = "length"
    pattern_id = "cluster_id"  # canonical pattern id
    correlation_to_model = "correlation to model"  # canonical pattern to model
    regularisation = "corr regularisation"
    actual_correlation = "correlation achieved"  # this is spearman correlation, this is A_x
    mae = "MAE"  # between canonical pattern and A_x
    relaxed_mae = "relaxed MAE"  # between relaxed canonical pattern and A_x
    actual_within_tolerance = "correlation achieved with tolerance"


def random_segment_lengths(short_segment_durations, long_segment_durations, n_segments, seed):
    """
       Generates a list of segment lengths to use for n_segments. The order of the segment length is random based
       on seed. We use each given segment length with a similar frequency. The short segments are used for 2/3 of the
       segments, the long segments are used for 1/3 of the segments
       :param short_segment_durations: list of short segments to use
       :param long_segment_durations: list of long segments to use
       :param n_segments: number of segments to generate
       :return: list of pattern_ids of length n_segments
       """
    np.random.seed(seed)

    n_short = len(short_segment_durations)  # number of short segment lengths
    n_long = len(long_segment_durations)  # number of long segment lengths
    n_short_segments = int(round((3 * n_segments) / 4, 0))  # make 3/4 of segments short
    n_long_segments = n_segments - n_short_segments  # make 1/4 of segments long

    # great least with approx same frequency for each segment length
    n_short_per_pattern = n_short_segments // n_short
    remainder_short = n_short_segments % n_short
    short_list = short_segment_durations * n_short_per_pattern
    if remainder_short:
        short_list.extend(np.random.choice(short_segment_durations, remainder_short, replace=False))

    n_long_per_pattern = n_long_segments // n_long
    remainder_long = n_long_segments % n_long
    long_list = long_segment_durations * n_long_per_pattern
    if remainder_long:
        long_list.extend(np.random.choice(long_segment_durations, remainder_long, replace=False))

    # combine the list
    result = short_list + long_list

    # shuffle at random (with seed set
    np.random.shuffle(result)
    return result


def random_list_of_patterns_for(pattern_ids_to_model: [], n_segments: int, seed: int):
    """
    Generates a list of patterns to use for n_segments. The order of the patterns is random.
    :param pattern_ids_to_model: unique list of pattern ids that should all be present in equal frequency
    :param n_segments: number of segments to generate
    :return: list of pattern_ids of length n_segments
    """
    n_patterns = len(pattern_ids_to_model)
    n_per_pattern = n_segments // n_patterns
    remainder = n_segments % n_patterns

    # create list of len n_segment starting with the even number of patterns making sure
    # each pattern appears approx the same number of times
    balanced_patterns = pattern_ids_to_model * n_per_pattern

    # add remaining patterns randomly if n_segments isn't perfectly divisible
    if remainder:
        np.random.seed(seed)
        balanced_patterns.extend(np.random.choice(pattern_ids_to_model, remainder, replace=False))

    # create a random ordered list where the patterns are never consecutive
    result = []
    previous_pattern = None
    for idx in range(n_segments):
        np.random.seed(seed + idx)
        random_selection = np.random.choice(balanced_patterns)

        # first two items are the same -> there will be no place to place the patter
        if previous_pattern == random_selection and len(result) == 1:
            for other_idx in range(n_segments):
                random_selection = np.random.choice(balanced_patterns)
                if random_selection != n_per_pattern:
                    break  # picked another pattern that we can place

        # run out of choices and they are still the same (this is at the end of the picking)
        if previous_pattern == random_selection:
            # insert random selection at the first index that does not cause a repetition
            inserted = False
            for result_idx in range(len(result)):
                if can_insert_at_idx_without_repetition(random_selection, result,
                                                        result_idx):  # can insert at result_index
                    previous_pattern = random_selection
                    result.insert(result_idx, random_selection)
                    balanced_patterns.remove(random_selection)
                    inserted = True
                    break
            if not inserted:
                raise ValueError("No valid pattern placement found that does not cause repetition")
        else:  # found a pattern and can use it
            previous_pattern = random_selection
            result.append(random_selection)
            balanced_patterns.remove(random_selection)

    return result


def can_insert_at_idx_without_repetition(item, insert_in_list, idx):
    if idx == 0 and insert_in_list[idx] != item:
        return True
    else:
        return insert_in_list[idx] != item and insert_in_list[idx - 1] != item


class SyntheticSegmentedData:
    def __init__(self, n_segments: int, n_variates: int,
                 distributions_for_variates: [],
                 distributions_args: [], distributions_kwargs: [], short_segment_durations: [],
                 long_segment_durations: [], patterns_to_model: {}, variate_names: [], cor_method: str = "loadings"):
        self.start_time = datetime(2017, 6, 23, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        self.time_delta = timedelta(seconds=1)  # sample at 1 second by default
        self.distributions_for_variates = distributions_for_variates
        self.distributions_args = distributions_args
        self.distributions_kwargs = distributions_kwargs
        self.short_segment_durations = short_segment_durations
        self.n_draw_short = 4  # take n_draw_short short then n_draw_long than n_draw_short
        self.n_draw_long = 1
        # what method to use to create the correlations (loadings or cholesky)
        self.cor_method = cor_method
        # how many times we reattempt to generate a segment to avoid failures
        self.long_segment_durations = long_segment_durations
        self.patterns_to_model = patterns_to_model
        self.variate_names = variate_names
        self.n_variates = n_variates
        self.n_segments = n_segments
        self.segment_data_generators: [GenerateData] = []
        self.non_normal_data_df = None  # correlated and distribution shifted, most AID like version
        self.non_normal_labels_df = None

    def generate(self, seed: int):
        """
        Generates a whole dataset of multiple segments using the settings given. For each segment the seed to generate
        observations is changed by adding the segment id to the given seed.
        :param seed: random seed
        :return: non normal correlated data (the raw, normal correlated, downsampled nn versions are also saved on
        the class but not returned)
        """
        self.segment_data_generators = []
        self.non_normal_data_df = None
        self.non_normal_labels_df = None

        generated_nn_df = None

        # lists to store results
        segment_ids = []
        start_indices = []
        end_indices = []
        pattern_ids = []
        correlations_to_model = []
        regularisations = []  # only for cholesky

        segment_start_idx = 0  # zero based indexing
        pattern_ids_to_model = list(self.patterns_to_model.keys())
        patterns_list = random_list_of_patterns_for(pattern_ids_to_model, self.n_segments, seed)
        create_segment_lengths = random_segment_lengths(self.short_segment_durations, self.long_segment_durations,
                                                        self.n_segments, seed)

        # create segments
        for segment_id in range(self.n_segments):
            # number of observations to created
            n_observations = create_segment_lengths[segment_id]

            pattern_id = patterns_list[segment_id]
            pattern = self.patterns_to_model[pattern_id]
            distributions = self.distributions_for_variates
            args = self.distributions_args
            kwargs = self.distributions_kwargs

            if isinstance(pattern, tuple):
                correlations = pattern[0]
                regularisation = pattern[1]
                # cholesky decomposition correlation needs regularisation term
                generator = GenerateData(n_observations, self.n_variates, correlations, distributions, args=args,
                                         kwargs=kwargs, regularisation=regularisation,
                                         method=self.cor_method)
            else:
                # loadings correlation data doesn't need regularisation term
                correlations = pattern
                regularisation = np.nan
                generator = GenerateData(n_observations, self.n_variates, correlations, distributions, args=args,
                                         kwargs=kwargs, method=self.cor_method)

            generator.generate(seed + segment_id)

            nn_df = pd.DataFrame(data=generator.non_normal_correlated_data, columns=self.variate_names)

            # safe results
            self.segment_data_generators.append(generator)

            # safe results to create labels df
            segment_ids.append(segment_id)
            start_indices.append(segment_start_idx)
            segment_end_idx = segment_start_idx + (n_observations - 1)  # -1 as 0 based
            end_indices.append(segment_end_idx)
            pattern_ids.append(pattern_id)
            correlations_to_model.append(correlations)

            if self.cor_method == "cholesky":
                regularisations.append(regularisation)

            # next segment start idx
            segment_start_idx = segment_end_idx + 1

            # append data to generated_df
            if generated_nn_df is None:
                generated_nn_df = nn_df
            else:
                generated_nn_df = pd.concat([generated_nn_df, nn_df], axis=0)

        # create nn_data
        generated_nn_df.reset_index(drop=True, inplace=True)
        nn_data_df = self.__add_timestamp(generated_nn_df)

        labels_dict = {
            SyntheticDataSegmentCols.segment_id: segment_ids,
            SyntheticDataSegmentCols.start_idx: start_indices,
            SyntheticDataSegmentCols.end_idx: end_indices,
            SyntheticDataSegmentCols.length: create_segment_lengths,
            SyntheticDataSegmentCols.pattern_id: pattern_ids,
            SyntheticDataSegmentCols.correlation_to_model: correlations_to_model,
        }
        if self.cor_method == "cholesky":
            labels_dict[SyntheticDataSegmentCols.regularisation] = regularisations

        nn_labels_df = pd.DataFrame(labels_dict)
        # calculate correlations achieved, within tol and mae
        self.non_normal_labels_df = recalculate_labels_df_from_data(nn_data_df, nn_labels_df)
        self.non_normal_data_df = nn_data_df

        return self.non_normal_data_df

    def plot_distribution_for_segment(self, segment_id: int, backend=Backends.none.value):
        """ Segments id start with 0 to n_segments-1"""
        title = "PDF and histogram of data for segment " + str(segment_id)
        return self.segment_data_generators[segment_id].plot_pdf_and_histogram(title=title, backend=backend)

    def plot_correlation_matrix_for_segment(self, segment_id: int, backend=Backends.none.value):
        """ Segments id start with 0 to n_segments-1"""
        title = "Spearman correlation matrix for segment " + str(segment_id)
        self.segment_data_generators[segment_id].plot_correlation_matrix(title=title, backend=backend)

    def __add_timestamp(self, generated_df):
        times = [self.start_time]
        for x in range(generated_df.shape[0] - 1):
            times.append(times[x] + self.time_delta)

        # add datetime column
        generated_df.insert(loc=0, column=GeneralisedCols.datetime, value=times)
        generated_df[GeneralisedCols.datetime] = pd.to_datetime(generated_df[GeneralisedCols.datetime])
        return generated_df

    def resample(self, rule: str):
        """
        Downsample original generated data with provided rule. Attention this was tested for rule 1min,
        might not work for all rules

        :param rule: pandas resample rule
        :return: pd.Dataframe
        """
        resampled = self.non_normal_data_df.resample(rule, on=GeneralisedCols.datetime).mean()
        # to make it consistent with the other df that the datetime is not automatically the index we reset the index
        self.resampled_data = resampled.reset_index()

        # calculate new cluster segment df for resampled data
        reduction_in_frequency = self.non_normal_data_df.shape[0] / self.resampled_data.shape[0]
        resampled_labels_df = self.non_normal_labels_df.copy()
        resampled_labels_df[SyntheticDataSegmentCols.length] = resampled_labels_df[
            SyntheticDataSegmentCols.length].div(reduction_in_frequency).astype(int)
        new_start_indices = [0]
        new_end_indices = []
        for previous_idx in range(resampled_labels_df.shape[0] - 1):
            previous_length = resampled_labels_df.iloc[previous_idx].length
            previous_start_idx = new_start_indices[previous_idx]
            new_idx = previous_start_idx + previous_length
            new_start_indices.append(new_idx)
            new_end_indices.append(new_idx - 1)
        # add last end index
        new_end_indices.append(new_start_indices[-1] + resampled_labels_df.iloc[-1].length - 1)

        # update labels data
        resampled_labels_df[SyntheticDataSegmentCols.start_idx] = new_start_indices
        resampled_labels_df[SyntheticDataSegmentCols.end_idx] = new_end_indices

        # update actual correlation and correlation within tolerance
        resampled_labels_df = recalculate_labels_df_from_data(resampled, resampled_labels_df)
        self.resampled_labels_df = resampled_labels_df

        return resampled, resampled_labels_df

    def raw_generated_data_labels_df(self):
        """
        Returns the normal data before it got correlated and distribution shifted
        """
        raw_data = [g.raw_data for g in self.segment_data_generators]
        stacked_data = np.vstack(raw_data)
        df = self.non_normal_data_df.copy()
        df[self.variate_names] = stacked_data
        labels_df = self.non_normal_labels_df.copy()
        labels_df = recalculate_labels_df_from_data(df, labels_df)
        return df, labels_df

    def normal_correlated_generated_data_labels_df(self):
        """
        Returns the normal and correlated data before it got distribution shifted
        """
        normal_cor_data = [g.normal_correlated_data for g in self.segment_data_generators]
        stacked_data = np.vstack(normal_cor_data)
        df = self.non_normal_data_df.copy()
        df[self.variate_names] = stacked_data
        labels_df = self.non_normal_labels_df.copy()
        labels_df = recalculate_labels_df_from_data(df, labels_df)
        return df, labels_df
