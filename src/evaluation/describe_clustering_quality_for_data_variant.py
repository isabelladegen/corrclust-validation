import itertools
from dataclasses import dataclass

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.evaluation.internal_measure_assessment import IAResultsCSV, InternalMeasureCols, \
    column_name_correlation_coefficient_for, read_internal_assessment_result_for
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import StatsCols
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


@dataclass
class IntSummaryCols:
    data_stage: str = "Generation Stage"
    data_completeness: str = "Completeness"
    gt: str = "GT"
    worst: str = "Worst"
    min: str = "Min"
    max: str = "Max"


class DescribeClusteringQualityForDataVariant:
    """
    Use this class to describe the clustering quality subjects that are part of a data variant.
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

    def __init__(self, wandb_run_file: str, overall_ds_name: str, data_type: str, data_dir: str, results_root_dir: str,
                 distance_measure: str, round_to: int = 3):
        """
        :param wandb_run_file: full path to the wandb run file
        :param overall_ds_name: a string for the ds - this is mainly used to safe results in
        :param data_type: type of data to load all datasets for, see SyntheticDataType for options
        :param data_dir: directory where the data is stored
        :param results_root_dir: directory where the results are stored
        :param distance_measure: what distance measure to load the results for
        :param round_to: what to round the results to
        """
        self.__wandb_run_file = wandb_run_file
        self.__overall_ds_name = overall_ds_name
        self.__data_type = data_type
        self.__data_dir = data_dir
        self.__results_root_dir = results_root_dir
        self.__distance_measure = distance_measure
        self.__round_to = round_to
        self.run_names = pd.read_csv(self.__wandb_run_file)['Name'].tolist()

        # load the various result data into dictionaries with key being run name and value being the results df

        # calculations results for the quality measures for each subject
        self.quality_measures_results = {
            name: read_clustering_quality_measures(overall_ds_name=self.__overall_ds_name, data_type=self.__data_type,
                                                   root_results_dir=self.__results_root_dir, data_dir=self.__data_dir,
                                                   distance_measure=self.__distance_measure, run_names=[name])[0] for
            name in self.run_names}

        # correlation summary (one df with one row per run)
        self.correlation_summary_results = read_internal_assessment_result_for(
            result_type=IAResultsCSV.correlation_summary,
            overall_dataset_name=self.__overall_ds_name,
            results_dir=self.__results_root_dir, data_type=self.__data_type,
            data_dir=self.__data_dir,
            distance_measure=self.__distance_measure)

        self.descriptive_corr_results = read_internal_assessment_result_for(
            result_type=IAResultsCSV.descriptive_statistics_measure_summary,
            overall_dataset_name=self.__overall_ds_name,
            results_dir=self.__results_root_dir, data_type=self.__data_type,
            data_dir=self.__data_dir,
            distance_measure=self.__distance_measure)

        self.paired_t_test_results = read_internal_assessment_result_for(
            result_type=IAResultsCSV.paired_t_test,
            overall_dataset_name=self.__overall_ds_name,
            results_dir=self.__results_root_dir, data_type=self.__data_type,
            data_dir=self.__data_dir,
            distance_measure=self.__distance_measure)

    def all_values_for_clustering_quality_measure(self, quality_measure: str):
        """
        Returns all values for the given clustering quality measure across all subjects
        :param quality_measure: see ClusteringQualityMeasures for options
        :return: Series of values
        """
        # create list of measures series
        combined_series = pd.concat([df.set_index(DescribeBadPartCols.name)[quality_measure] for df in
                                     self.quality_measures_results.values()])
        return combined_series

    def all_values_for_correlations_with_jaccard_index_for_quality_measure(self, quality_measure: str):
        """
        Returns all correlation coefficients for the given clustering quality measure correlated to the
        Jaccard index across all subjects
        :param quality_measure: see ClusteringQualityMeasures for options (cannot be Jaccard Index)
        :return: Series of values
        """
        msg = "Provide an other quality measure that is compared to the Jaccard Index"
        assert quality_measure != ClusteringQualityMeasures.jaccard_index, msg
        # create list of measures series with subject name as index
        col_name = column_name_correlation_coefficient_for(quality_measure)
        result = self.correlation_summary_results.set_index(InternalMeasureCols.name)[col_name]
        return result

    def mean_sd_correlation_for(self, quality_measures: [str]):
        """
        Returns mean (sd) of correlation coefficients for the given quality measures
        :param quality_measures: which measures to return
        :return: pd.Dataframe with one row and columns: data generation stage, data completeness, quality measure
        """
        row = {
            IntSummaryCols.data_stage: [SyntheticDataType.get_display_name_for_data_type(self.__data_type)],
            IntSummaryCols.data_completeness: [get_row_name_from(self.__data_dir)]
        }

        # Add mean(sd) for each measure
        for measure in quality_measures:
            col_name = column_name_correlation_coefficient_for(measure)
            values = self.descriptive_corr_results[col_name]
            mean = values['mean']
            sd = values['std']
            row[measure] = [f"{mean:.2f} ({sd:.2f})"]

        # Create the summary dataframe
        summary_df = pd.DataFrame(row)

        return summary_df

    def p_value_and_effect_size_of_correlation_for(self, quality_measures: [str]):
        """
        Returns p_value (effect size) of correlation between the given quality measures and their correlation with
        the Jaccard Index
        :param quality_measures: which measures to return
        :return: pd.Dataframe with one row and columns: data generation stage, data completeness, tuple of quality
        measures compared
        """
        row = {
            IntSummaryCols.data_stage: [SyntheticDataType.get_display_name_for_data_type(self.__data_type)],
            IntSummaryCols.data_completeness: [get_row_name_from(self.__data_dir)]
        }

        # create column names for measure pairs to compare (don't know which order they will be so very messy)
        two_combinations = list(itertools.combinations(quality_measures, 2))
        measures_corr_col_names = {
            measure: InternalMeasureCols.persons_r + ' ' + measure + ", " + ClusteringQualityMeasures.jaccard_index for
            measure in
            quality_measures}
        col_names = {}
        for m1, m2 in two_combinations:
            m1_cor_name = measures_corr_col_names[m1]
            m2_cor_name = measures_corr_col_names[m2]
            name1 = m1_cor_name + ' vs ' + m2_cor_name
            name2 = m2_cor_name + ' vs ' + m1_cor_name
            if name1 in self.paired_t_test_results.columns:
                col_names[(m1, m2)] = name1
            if name2 in self.paired_t_test_results.columns:
                col_names[(m2, m1)] = name2

        # Add p_value (effect size) for each measure pair
        for measures, col_name in col_names.items():
            values = self.paired_t_test_results[col_name]
            p_value = values[StatsCols.p_value]
            d = values[StatsCols.effect_size]
            row[measures] = [f"{p_value:.2f} ({d:.2f})"]

        # Create the summary dataframe
        summary_df = pd.DataFrame(row)

        return summary_df

    def mean_sd_measure_values_for_ground_truth_and_lowest_jaccard_index(self, quality_measures: [str]):
        """
        Returns mean values and sd for ground truth and lowest jaccard index segmented clustering for the given quality
        measures
        :param quality_measures: which measures to return
        :return: pd.Dataframe with one row and columns: data generation stage, data completeness, quality measure's mean
        value (SD) for ground truth and worst segmented clustering
        """
        # get all subjects ground truth row and worst row (dfs are sorted by jaccard index so last row is best)
        ground_truth_df = pd.DataFrame([df[quality_measures].iloc[-1] for df in self.quality_measures_results.values()])
        worst_df = pd.DataFrame([df[quality_measures].iloc[0] for df in self.quality_measures_results.values()])

        # calculate mean and std
        mean_sd_gt_df = ground_truth_df.agg(['mean', 'std'])
        mean_sd_worst_df = worst_df.agg(['mean', 'std'])

        # build as a single row for all measures
        row = {
            IntSummaryCols.data_stage: [SyntheticDataType.get_display_name_for_data_type(self.__data_type)],
            IntSummaryCols.data_completeness: [get_row_name_from(self.__data_dir)]
        }

        # Add mean(sd) for each measure for ground truth and for worst partition
        for measure in quality_measures:
            gt_mean = mean_sd_gt_df.loc['mean', measure]
            gt_sd = mean_sd_gt_df.loc['std', measure]
            row[(measure, IntSummaryCols.gt)] = [f"{gt_mean:.2f} ({gt_sd:.2f})"]

            worst_mean = mean_sd_worst_df.loc['mean', measure]
            worst_sd = mean_sd_worst_df.loc['std', measure]
            row[(measure, IntSummaryCols.worst)] = [f"{worst_mean:.2f} ({worst_sd:.2f})"]

        # Create the summary dataframe
        summary_df = pd.DataFrame(row)

        return summary_df

    def summary_benchmark_df(self):
        """
        Calculates the statistics given in stats for quality measures across all subjects and returns a pandas DataFrame
        """
        stats = ['mean', 'std', 'median',  lambda x: x.quantile(0.25),  lambda x: x.quantile(0.75), 'min', 'max']
        combined_df = pd.concat(self.quality_measures_results.values(), keys=self.quality_measures_results.keys())

        # group by same mistakes introduced
        grouped_stats = combined_df.groupby(
            [DescribeBadPartCols.n_obs_shifted, DescribeBadPartCols.n_wrong_clusters]).agg({
            ClusteringQualityMeasures.jaccard_index: stats,
            ClusteringQualityMeasures.silhouette_score: stats,
            ClusteringQualityMeasures.dbi: stats,
            DescribeBadPartCols.errors: stats  # relaxed mae
        })

        # make n_obs_shifted and n_wrong_clusters regular columns
        grouped_stats = grouped_stats.reset_index().round(decimals=self.__round_to)

        # order by highest jaccard index first
        grouped_stats = grouped_stats.sort_values(by=(ClusteringQualityMeasures.jaccard_index, 'mean'), ascending=False)

        return grouped_stats
