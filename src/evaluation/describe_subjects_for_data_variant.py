import ast
import re
from dataclasses import dataclass
from itertools import chain
from os import path

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols, \
    recalculate_labels_df_from_data, CorrType
from src.utils.configurations import SYNTHETIC_DATA_DIR, dataset_description_dir, \
    MULTIPLE_DS_SUMMARY_FILE, get_irregular_folder_name_from, base_dataset_result_folder_for_type, ResultsType, \
    RunInformationCols, get_data_completeness_from
from src.utils.labels_utils import calculate_n_segments_outside_tolerance_for, \
    calculate_n_segments_outside_tolerance_per_pattern_for
from src.utils.load_synthetic_data import SyntheticDataType, load_labels, load_synthetic_data
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.run_average_rank_visualisations import data_variant_description


@dataclass
class DistParams:
    method: str = 'method'
    args: str = "args"  # what's this representative?
    kwargs: str = "kwargs"
    median_args: str = "median_args"
    median_kwargs: str = "median_kwargs"
    min_args: str = "min_args"
    min_kwargs: str = "min_kwargs"
    max_args: str = "max_args"
    max_kwargs: str = "max-kwargs"


@dataclass
class SummaryStatistics:
    mae = "mae"  # mean of each labels file across datasets
    relaxed_mae = "relaxed_mae"  # mean of each labels file across datasets for relaxed canonical pattern
    overall_mae = "overall mae"  # all maes
    relaxed_overall_mae = "relaxed overall mae"  # all maes but to relaxed canonical pattern
    seg_outside_tol = "segments outside tolerance"
    observations = "observations"
    segments = "n segments"
    patterns = "n patterns"
    segment_lengths = "segment lengths"  # mean of each labels file across datasets
    overall_segment_lengths = "overall segment lengths"  # all segment lengths
    overall_interval_gaps = "interval gaps (sec)"  # all gaps between observations


def combine_all_ds_variations_multiple_description_summary_dfs(result_root_dir: str,
                                                               overall_ds_name: str,
                                                               dataset_types: [str],
                                                               data_dirs: str,
                                                               save_combined_results: bool = True):
    """Combines all the dataset description summaries for the given types and runs into one table
    :param result_root_dir: root dir for the results - will write to the dataset-description folder in this dir,
    create it if it does not exist
    :param overall_ds_name: just a string to identify the results
    :param dataset_types: list of dataset types to read, see SyntheticDataType
    :param data_dirs: list of str of directories from which the data was read
    :param save_combined_results: will save the combined resulting csv in dataset-description of the result root
    dir
    :return: MultiIndex pd.Dataframe, access results like e.g. combined_df['dataset_name']['n_segments']['mean']
    """
    results = {}
    for data_dir in data_dirs:
        irr_folder_str = get_irregular_folder_name_from(data_dir)
        for ds_type in dataset_types:
            folder = dataset_description_dir(overall_ds_name, ds_type, result_root_dir, data_dir)
            file_name = path.join(folder, MULTIPLE_DS_SUMMARY_FILE)
            df = pd.read_csv(file_name, index_col=0)
            name = SyntheticDataType.get_dataset_variation_name(ds_type, irr_folder_str)
            results[name] = df
    combined_result = pd.concat(results, axis=1)
    if save_combined_results:
        folder = base_dataset_result_folder_for_type(result_root_dir, ResultsType.dataset_description)
        combined_result.to_csv(str(path.join(folder, MULTIPLE_DS_SUMMARY_FILE)))
    return combined_result


def shorten_segments_to(length, labels_df):
    """
    Function to change start and end idx in labels to new length
    """
    new_labels_df = labels_df.copy()
    current_lengths = labels_df[SyntheticDataSegmentCols.length]
    assert all(current_lengths >= length), "Existing segments are already shorter than " + str(length)

    new_labels_df[SyntheticDataSegmentCols.length] = length
    # -1 because we've been following the convention to select the end_idx
    new_labels_df[SyntheticDataSegmentCols.end_idx] = new_labels_df[SyntheticDataSegmentCols.start_idx] + (length - 1)
    return new_labels_df


class DescribeSubjectsForDataVariant:
    """
    Use this class to describe the data for multiple subjects that are part of a data variant.
    The class can do both give overall statistics - e.g. when segment length is analysed from all segment lengths
    in all subjects in the collection - as well as variations across a subjects for which we usually have a single
    number that describes each subject and calculate statistics from that number across the subjects
    (if we did this for segment length we would calculate a mean segment length per subject - this is none sense for o
    ur data as the mean is the same for each subject).
    Overall analysis makes sense for example for segment length, MAE...
    Variations across ds makes sense for n observations, MAE (this time we see how much this varies), n segments
    outside tolerance...
    We will recalculate the number from the labels files
    """

    def __init__(self, wandb_run_file: str, overall_ds_name: str,
                 data_type: str = SyntheticDataType.non_normal_correlated, data_dir: str = SYNTHETIC_DATA_DIR,
                 load_data: bool = False, additional_corr: [str] = [], backend: str = Backends.none.value,
                 round_to: int = 3):
        """
        :param wandb_run_file: full path to the wandb run file
        :param overall_ds_name: a string for the ds - this is mainly used to safe results in
        :param data_type: type of data to load all datasets for, see SyntheticDataType for options
        :param data_dir: directory where the data is stored
        :param load_data: whether to just load the labels files (False - default) or the labels and data (True)
        :param additional_corr: list of which correlations to additionally calculate, Spearman is calculated by
        default, requires load_data to be TRUE
        :param backend: backend for matplotlib
        :param round_to: what to round the results to
        """
        self.__wandb_run_file = wandb_run_file
        self.__overall_ds_name = overall_ds_name
        self.__data_type = data_type
        self.__data_dir = data_dir
        self.__load_data = load_data
        self.__correlations = additional_corr
        self.__backend = backend
        self.__round_to = round_to
        self.variant_name = data_variant_description[(get_data_completeness_from(self.__data_dir), self.__data_type)]

        # dictionary with key run name and value labels file for the run
        self.__run_information = pd.read_csv(wandb_run_file, index_col=RunInformationCols.ds_name)
        self.run_names = self.__run_information.index.tolist()
        self.ts_variates = ast.literal_eval(self.__run_information.at[self.run_names[0], RunInformationCols.data_cols])
        self.label_dfs = {}
        self.data_dfs = {}
        if self.__load_data:  # load both labels and data
            for name in self.run_names:
                data_df, labels_df = load_synthetic_data(name, self.__data_type, self.__data_dir)
                self.data_dfs[name] = data_df
                self.label_dfs[name] = labels_df
        else:  # load just labels
            self.label_dfs = {name: load_labels(name, self.__data_type, self.__data_dir) for name in self.run_names}

        # dictionary of dictionary, first level keys CorrType.pearson and or CorrType.kendall
        self.other_corr_labels = {}
        # calculate labels for additional correlations
        for correlation in self.__correlations:
            results = {}
            for name, labels_df in self.label_dfs.items():
                labels_for_corr = recalculate_labels_df_from_data(self.data_dfs[name], labels_df,
                                                                  corr_type=correlation)
                results[name] = labels_for_corr
            self.other_corr_labels[correlation] = results

    def mae_stats(self, column: str):
        """ Return stats for mae
        This is to see variations across the different datasets, therefore we use the mean of each dataset
        :params columns SyntheticDataSegmentCols.mae or SyntheticDataSegmentCols.relaxed_mae
        :returns pandas describe series
        """
        mean_values = [label[column].mean() for label in self.label_dfs.values()]
        return pd.Series(mean_values).describe().round(3)

    def overall_mae_stats(self, column: str, corr_type=CorrType.spearman):
        """
        Return stats for mae overall, here we consider all segment's mae without calculating the mean
        first. This will give a better impression on how mae varies on a segment level.
        :params column to calculate mae from SyntheticDataSegmentCols.mae or SyntheticDataSegmentCols.relaxed_mae
        :corr_type: default Spearman, other can be specified if it was calculated when creating the class
        :returns pandas describe series
        """
        # list of list
        values = self.all_mae_values(column, corr_type=corr_type)
        values_flat = list(chain.from_iterable(values))
        return pd.Series(values_flat).describe().round(3)

    def overall_per_pattern_mae_stats(self, column: str, corr_type=CorrType.spearman):
        """
        Return stats for mae overall per pattern, here we consider all segment's mae without calculating the mean
        first. This will give a better impression on how mae varies on a segment level.
        :params column to calculate mae from SyntheticDataSegmentCols.mae or SyntheticDataSegmentCols.relaxed_mae
        :corr_type: default Spearman, other can be specified if it was calculated when creating the class
        :returns pandas describe df with rows for different patterns and columns for stats,
        access results like stats_df.loc[pattern_id, statistics]
        """
        # list of list
        pattern_id_mae_dfs = self.per_pattern_mae_values(column, corr_type=corr_type)
        combined_df = pd.concat(pattern_id_mae_dfs, ignore_index=True)
        stats_df = combined_df.groupby(SyntheticDataSegmentCols.pattern_id)[column].describe().round(3)

        # Sort by the '50%' column (median) in descending order
        stats_df = stats_df.sort_values(by='50%', ascending=False)
        return stats_df

    def all_mae_values(self, column: str, corr_type=CorrType.spearman):
        if corr_type == CorrType.spearman:
            values = [label[column] for label in self.label_dfs.values()]
        else:
            values = [label[column] for label in self.other_corr_labels[corr_type].values()]
        return values

    def per_pattern_mae_values(self, column: str, corr_type=CorrType.spearman):
        """Returns df for each subject with pattern id and column"""
        if corr_type == CorrType.spearman:
            values = [label[[SyntheticDataSegmentCols.pattern_id, column]] for label in self.label_dfs.values()]
        else:
            values = [label[[SyntheticDataSegmentCols.pattern_id, column]] for label in
                      self.other_corr_labels[corr_type].values()]
        return values

    def all_time_gaps_in_seconds(self):
        """
        Returns a list of all time gaps combined across all subjects for the data variant
        """
        dfs = self.data_dfs.values()
        all_gaps = []
        for df in dfs:
            # differences in time in seconds
            time_diffs = df['datetime'].diff().dropna().dt.total_seconds().values
            all_gaps.extend(time_diffs)
        return all_gaps

    def overall_time_gap_stats(self):
        """
        Return stats for overall time gaps, here we combine all subject's time gap
         :returns pandas describe series
        """
        # list of list
        values = self.all_time_gaps_in_seconds()
        stats = pd.Series(values).describe().round(3)
        return stats

    def n_segments_outside_tolerance_stats(self, corr_type=CorrType.spearman):
        """
        Return stats for n segment outside tolerance
        This is naturally a per dataset number
        :corr_type: default Spearman, other can be specified if it was calculated when creating the class
        :returns pandas describe series
        """
        if corr_type == CorrType.spearman:
            values = [calculate_n_segments_outside_tolerance_for(label) for label in self.label_dfs.values()]
        else:
            values = [calculate_n_segments_outside_tolerance_for(label) for label in
                      self.other_corr_labels[corr_type].values()]
        return pd.Series(values).describe().round(3)

    def per_pattern_n_segments_outside_tolerance_stats(self, corr_type=CorrType.spearman):
        """
        Return stats for n segment outside tolerance per pattern
        This is naturally a per dataset number
        :corr_type: default Spearman, other can be specified if it was calculated when creating the class
        :returns pandas describe series
        """
        if corr_type == CorrType.spearman:
            labels_dfs = self.label_dfs.values()
        else:
            labels_dfs = self.other_corr_labels[corr_type].values()

        per_pattern_within_tol_dfs = [calculate_n_segments_outside_tolerance_per_pattern_for(label) for label in labels_dfs]
        combined_df = pd.concat(per_pattern_within_tol_dfs, ignore_index=True)

        stats_df = combined_df.groupby(SyntheticDataSegmentCols.pattern_id)[
            SyntheticDataSegmentCols.n_outside_tolerance].describe().round(3)

        # Sort by the '50%' column in descending order
        stats_df = stats_df.sort_values(by='50%', ascending=False)
        return stats_df

    def observations_stats(self):
        """
        Return stats for n observations
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [label[SyntheticDataSegmentCols.length].sum() for label in self.label_dfs.values()]
        return pd.Series(values).describe().round(3)

    def n_patterns_stats(self):
        """
        Return stats for unique number of patterns, for the raw dataset there are none, but it will
        still use the ones that later will be modeled as this makes most sense overall
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [len(label[SyntheticDataSegmentCols.pattern_id].unique()) for label in self.label_dfs.values()]
        return pd.Series(values).describe().round(3)

    def n_segments_stats(self):
        """
        Return stats for n segments, this will be commonly 100 but might change for a few datasets
        This is naturally a per dataset number
        :returns pandas describe series
        """
        values = [label.shape[0] for label in self.label_dfs.values()]
        return pd.Series(values).describe().round(3)

    def segment_length_stats(self):
        """
        Return stats for segment length, here we first calculate the mean per dataset and then the statistics
        of the mean. For many variations the mean will be the same but this might not be true for all
        :returns pandas describe series
        """
        values = [label[SyntheticDataSegmentCols.length].mean() for label in self.label_dfs.values()]
        return pd.Series(values).describe().round(3)

    def overall_segment_length_stats(self):
        """
        Return stats for segment length overall, here we consider all segment lengths without calculating the mean
        first. This will give a better impression on how segment lengths vary.
        :returns pandas describe series
        """
        # list of list
        values = self.all_segment_lengths_values()
        values_flat = list(chain.from_iterable(values))
        return pd.Series(values_flat).describe().round(3)

    def overall_pattern_id_count_stats(self):
        """
        Return stats for pattern count overall, here we consider all pattern id without calculating the mean
        per subject first. We will translate the pattern id to counts to have more equivalent result
        :returns pandas describe series
        """
        # list of list
        values = self.all_pattern_id_values()
        counts = [subject.value_counts().values for subject in values]
        values_flat = list(chain.from_iterable(counts))
        return pd.Series(values_flat).describe().round(3)

    def all_segment_lengths_values(self):
        values = [label[SyntheticDataSegmentCols.length] for label in self.label_dfs.values()]
        return values

    def all_pattern_id_values(self):
        values = [label[SyntheticDataSegmentCols.pattern_id] for label in self.label_dfs.values()]
        return values

    def summary(self):
        """Returns summary pd.Dataframe of all statistics
        access like df[SummaryStatistics]['mean'], the second can be any value in pandas describe
        """
        return pd.DataFrame({
            SummaryStatistics.mae: self.mae_stats(SyntheticDataSegmentCols.mae),
            SummaryStatistics.relaxed_mae: self.mae_stats(SyntheticDataSegmentCols.relaxed_mae),
            SummaryStatistics.overall_mae: self.overall_mae_stats(SyntheticDataSegmentCols.mae),
            SummaryStatistics.relaxed_overall_mae: self.overall_mae_stats(SyntheticDataSegmentCols.relaxed_mae),
            SummaryStatistics.seg_outside_tol: self.n_segments_outside_tolerance_stats(),
            SummaryStatistics.observations: self.observations_stats(),
            SummaryStatistics.segments: self.n_segments_stats(),
            SummaryStatistics.patterns: self.n_patterns_stats(),
            SummaryStatistics.segment_lengths: self.segment_length_stats(),
            SummaryStatistics.overall_segment_lengths: self.overall_segment_length_stats(),
            SummaryStatistics.overall_interval_gaps: self.overall_time_gap_stats()
        })

    def save_summary(self, root_results_dir: str):
        """
        Saves summary df in the given root_result dir
        :param root_results_dir: directory where to save the results
        """
        df = self.summary()

        folder = dataset_description_dir(overall_dataset_name=self.__overall_ds_name, data_type=self.__data_type,
                                         root_results_dir=root_results_dir, data_dir=self.__data_dir)
        file_name = path.join(folder, MULTIPLE_DS_SUMMARY_FILE)
        df.to_csv(str(file_name))

    def get_data_as_xtrain(self, ds_name: str) -> np.array:
        """
        Returns the data for the given dataset name as 2d numpy array
        This is only possible if you loaded the data.
        """
        assert self.__load_data, "You need to load the data to be able to use this function. Set load_data to True!"
        return self.data_dfs[ds_name][self.ts_variates].to_numpy()

    def get_list_of_xtrain_of_all_datasets(self) -> []:
        """
        Returns a list of xtrain np arrays for all datasets loaded
        """
        return [self.get_data_as_xtrain(run_name) for run_name in self.run_names]

    def get_median_min_max_distribution_parameters(self):
        """
        Return median, min, max distribution parameters for the loaded datasets (from run_file)
        We assume that all datasets have the same columns (time series variates) and for the same time series variate
        use the same distribution, the parameters of that distribution might vary
        :return dictionary of dictionaries e.g:
        dist_params = {'iob':
            {
                'method': stats.genextreme,
                'args': (0.1,),  # shape parameter c
                'kwargs': {'loc': 2, 'scale': 3},  # location and scale
                'median_args': (0.1,),
                'median_kwargs': {'loc': 2, 'scale': 3},
                'min_args': (0.08,),
                'min_kwargs': {'loc': 1.8, 'scale': 2.8},
                'max_args': (0.12,),
                'max_kwargs': {'loc': 2.2, 'scale': 3.2}
            },
            'cob': {
                'method': ...
            }
            ...
        """
        # create an empty dict for each ts variate
        dist_params = {variate: {} for variate in self.ts_variates}

        # theoretical column names
        args_base_cols = [RunInformationCols().dist_args_for(variate) for variate in self.ts_variates]
        kwargs_base_cols = [RunInformationCols().dist_kwargs_for(variate) for variate in self.ts_variates]

        # find all columns that actually exist for each variate
        args_cols = {variate: [] for variate in self.ts_variates}  # if no args stays empty
        kwargs_cols = {variate: [] for variate in self.ts_variates}  # if no kwargs stays empty
        existing_cols = self.__run_information.columns
        # create dictionaries of dictionaries, here the values of the column dictionaries are empty lists
        # args: { 'iob' : {'distributions_args_iob' : (<tuples of args>)}...}
        # kwargs: { 'iob' : {'distributions_kwargs_iob.loc' : [values], 'distributions_kwargs_iob.scale': [values]}...}
        for idx, variate in enumerate(self.ts_variates):
            args_cols[variate] = {col: [] for col in existing_cols if col == args_base_cols[idx]}
            kwargs_cols[variate] = {col: [] for col in existing_cols if col.startswith(kwargs_base_cols[idx])}

        # load the data for each column into the empty lists
        for idx, variate in enumerate(self.ts_variates):
            arg_column_names = list(args_cols[variate].keys())
            # args - should just have one column with all the args
            error_msg = "We unexpectedly have multiple arg columns for variate: " + variate + ". Cols: " + str(
                arg_column_names)
            assert len(arg_column_names) <= 1, error_msg
            for col in arg_column_names:
                # these are lists of args
                args_cols[variate][col] = self.__run_information[col].apply(ast.literal_eval).to_list()
            # kwargs
            kwarg_column_names = list(kwargs_cols[variate].keys())
            for col in kwarg_column_names:
                # these are single numbers so we can just read
                kwargs_cols[variate][col] = self.__run_information[col].to_list()

        # get first datasets distribution - we assume the same dist is used for all datasets
        distributions = ast.literal_eval(self.__run_information.at[self.run_names[0], RunInformationCols.distribution])

        # for each variate calculate the values
        for idx, variate in enumerate(self.ts_variates):
            # distribution
            dist_params[variate][DistParams.method] = extract_distribution(distributions[idx])
            # min, max, median of distribution args
            # 2d numpy array with columns being the different args and rows being the different datasets
            args_for_variate = np.array(list(args_cols[variate].values())[0])
            dist_params[variate][DistParams.median_args] = tuple(np.median(args_for_variate, axis=0))
            dist_params[variate][DistParams.min_args] = tuple(np.min(args_for_variate, axis=0))
            dist_params[variate][DistParams.max_args] = tuple(np.max(args_for_variate, axis=0))
            # min, max, median of each kwargs
            kwargs_min = {}
            kwargs_max = {}
            kwargs_median = {}
            # go through all keywords and calculate min, max, median for each
            for col_name, values in kwargs_cols[variate].items():
                kw_name = col_name.split('.')[-1]
                kwargs_min[kw_name] = np.min(values)
                kwargs_max[kw_name] = np.max(values)
                kwargs_median[kw_name] = np.median(values)
            dist_params[variate][DistParams.median_kwargs] = kwargs_median
            dist_params[variate][DistParams.min_kwargs] = kwargs_min
            dist_params[variate][DistParams.max_kwargs] = kwargs_max

        return dist_params

    def name_for_worst_relaxed_mae(self, column=SyntheticDataSegmentCols.relaxed_mae):
        # get name and mae tuples
        name_mae = [(name, label[column].mean()) for name, label in self.label_dfs.items()]
        # find max mae and name
        name, mae = max(name_mae, key=lambda x: x[1])
        return name, round(mae, self.__round_to)

    def mean_mae_for_segment_lengths(self, lengths, cor_type=CorrType.spearman):
        result = {}
        result['mean'] = []
        result['50%'] = []
        result['25%'] = []
        result['75%'] = []
        if cor_type == CorrType.spearman:
            labels_dfs = self.label_dfs
        else:
            labels_dfs = self.other_corr_labels[cor_type]

        for length in lengths:
            # all means of all patterns for all subjects
            means = []
            for name, labels_df in labels_dfs.items():
                # 1. change labels start, end idx and length
                shorter_labels = shorten_segments_to(length, labels_df)

                # 2. recalculate rest of labels file from data
                updated_labels = recalculate_labels_df_from_data(self.data_dfs[name], shorter_labels,
                                                                 corr_type=cor_type)
                means.extend(updated_labels[SyntheticDataSegmentCols.relaxed_mae])
            # 3. Calculate mean for this length
            result['mean'].append(round(np.mean(means), self.__round_to))
            result['50%'].append(round(np.median(means), self.__round_to))
            result['25%'].append(round(np.percentile(means, 25), self.__round_to))
            result['75%'].append(round(np.percentile(means, 75), self.__round_to))
        return result

    def achieved_correlation_stats_for_pattern(self, pattern_id: int):
        """Calculates describe stats for the given pattern_id from the achieved correlations"""
        all_achieved_cors = []
        # read all correlations
        for labels_df in self.label_dfs.values():
            achieved_corr = labels_df[labels_df[SyntheticDataSegmentCols.pattern_id] == pattern_id][
                SyntheticDataSegmentCols.actual_correlation].to_list()
            all_achieved_cors.extend(achieved_corr)

        cors_df = pd.DataFrame(all_achieved_cors, columns=['c1', 'c2', 'c3'])
        desc = cors_df.describe().round(3)
        return desc


def extract_distribution(log_string: str):
    """Gets scipy distributio form log string"""
    match = re.search(r'scipy\.stats\._\w+distns\.(\w+)_gen', log_string)

    if match:
        dist_name = match.group(1)
        return getattr(stats, dist_name)
    else:
        assert False, "Could not match distribution in log string: " + log_string
