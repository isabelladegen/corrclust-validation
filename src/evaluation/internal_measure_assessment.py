import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.evaluation.describe_bad_partitions import default_internal_measures, default_external_measures, \
    DescribeBadPartCols
from src.evaluation.run_cluster_quality_measures_calculation import run_internal_measure_calculation_for_dataset
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, ROOT_RESULTS_DIR, \
    internal_measure_assessment_dir_for, SYNTHETIC_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import standardized_effect_size_of_mean_difference, calculate_hi_lo_difference_ci, \
    ConfidenceIntervalCols


@dataclass
class IAResultsCSV:
    correlation_summary: str = "correlation_summary.csv"
    effect_size_difference_worst_best: str = "effect_size_difference_of_worst_to_best_partition.csv"
    descriptive_statistics_measure_summary: str = "descriptive_statistics_internal_measures_correlation.csv"
    ci_of_differences_between_measures: str = "ci_differences_between_internal_measure_correlation.csv"


def get_full_filename_for_results_csv(full_results_dir: str, csv_filename: str):
    """ Returns the full filename to read or save a csv result
    :param full_results_dir: the full path where to save the file or read the file from
    :param csv_filename: the full filename to save or read, see InternalMeasureAssessmentCSV for options
    :returns the full path to the csv file
    """
    return os.path.join(full_results_dir, csv_filename)


@dataclass
class InternalMeasureCols:
    name: str = "name"
    partitions: str = "N_pi+1"
    persons_r: str = 'r'
    p_value: str = 'P'
    effect_size: str = 'd'


class InternalMeasureAssessment:
    def __init__(self, distance_measure: str, dataset_results: [pd.DataFrame],
                 internal_measures: [str] = default_internal_measures,
                 external_measures: [str] = default_external_measures, round_to: int = 3):
        self.distance_measure = distance_measure
        self.dataset_results = dataset_results
        self.__internal_measures = internal_measures
        self.__external_measures = external_measures
        self.measures_combinations = list(itertools.product(self.__internal_measures, self.__external_measures))
        self.measures_combinations_col_names = [pair[0] + ', ' + pair[1] for pair in self.measures_combinations]
        self.measures_corr_col_names = [InternalMeasureCols.persons_r + ' ' + item for item in
                                        self.measures_combinations_col_names]
        self.measures_p_col_names = [InternalMeasureCols.p_value + ' ' + item for item in
                                     self.measures_combinations_col_names]

        self.__comparing_internal_measures = list(itertools.combinations(self.measures_corr_col_names, 2))
        self.compare_internal_measures_cols = [item[0] + ' vs ' + item[1] for item in
                                               self.__comparing_internal_measures]
        self.__round_to = round_to

        # calculate correlations between all combinations of internal and external measures
        names = []
        partitions = []
        correlations = {col_names: [] for col_names in self.measures_corr_col_names}
        p_values = {col_names: [] for col_names in self.measures_p_col_names}

        # calculate correlations for each dataset and each measure pair
        for ds in self.dataset_results:
            name = ds.iloc[0][DescribeBadPartCols.name]
            n_partition = ds.shape[0]  # partitions including ground truth

            # calculate correlations for each pair
            for p_idx, pair in enumerate(self.measures_combinations):
                m1_values = ds[pair[0]].to_numpy()
                m2_values = ds[pair[1]].to_numpy()
                stat_result = pearsonr(m1_values, m2_values)
                cor = stat_result.statistic
                p = stat_result.pvalue
                # update result
                correlations[self.measures_corr_col_names[p_idx]].append(round(cor, round_to))
                p_values[self.measures_p_col_names[p_idx]].append(round(p, round_to))

            # update results
            names.append(name)
            partitions.append(n_partition)

        # add p_values dict to correlations
        correlations.update(p_values)
        corr_df = pd.DataFrame(correlations)
        corr_df.insert(loc=0, column=InternalMeasureCols.name, value=names, allow_duplicates=True)
        corr_df.insert(loc=1, column=InternalMeasureCols.partitions, value=partitions, allow_duplicates=True)

        self.correlation_summary = corr_df

    def effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(self, internal_measure: str,
                                                                         worst_ranked_by: str, z=1.96):
        """Calculates Cohen's d effect size and CI (with z=1.96 this is the 95% CI) of the differences in mean of the
        ground truth and worst partition. The mean is calculated across the N_D datasets for the provided
        internal_measure. The worst partition is judged by the lowest number for worst_ranked_by measure provided
        returns: effect_size, lo_ci, hi_ci, standard_error
        """
        n1 = n2 = len(self.dataset_results)
        gts = []
        worsts = []

        # get gt and worst value
        for ds in self.dataset_results:
            # value for ground truth
            gts.append(ds.iloc[0][internal_measure])
            ds_sorted = ds.sort_values(by=worst_ranked_by, ascending=True)  # lowest value first
            # worst value for internal measure
            worsts.append(ds_sorted.iloc[0][internal_measure])

        gts = np.array(gts)
        worsts = np.array(worsts)

        m1 = gts.mean()
        m2 = worsts.mean()
        s1 = gts.std()
        s2 = worsts.std()
        effect_size = standardized_effect_size_of_mean_difference(n1, n2, s1, s2, m1, m2)
        lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z)
        return effect_size, lo_ci, hi_ci, standard_error

    def descriptive_statistics_for_internal_measures_correlation(self):
        """ Calculates the descriptive stats for each internal measure and it's correlation to the external measures
        and returns a pd.Dataframe with the internal measure v external measure as column names and the descriptive
        statistics as rows.
        """
        return self.correlation_summary[self.measures_corr_col_names].describe().round(2)

    def ci_of_differences_between_internal_measure_correlations(self, z=1.96):
        """ Calculates the CI of mean difference between each of the internal measures correlation.
        the rows are indexed by lo, hi ci and standard error, the columns are the different internal measures combinations
        """
        df = self.descriptive_statistics_for_internal_measures_correlation()

        mean = df.loc['mean']
        count = df.loc['count']
        std = df.loc['std']

        # measures that we need to compared
        compare = self.__comparing_internal_measures

        names = []
        lo_cis = []
        hi_cis = []
        standard_errors = []

        for idx, measure_pair in enumerate(compare):
            m1 = mean[measure_pair[0]]
            m2 = mean[measure_pair[1]]
            n1 = count[measure_pair[0]]
            n2 = count[measure_pair[1]]
            s1 = std[measure_pair[0]]
            s2 = std[measure_pair[1]]

            lo_ci, hi_ci, standard_error = calculate_hi_lo_difference_ci(n1, n2, s1, s2, m1, m2, z)

            names.append(self.compare_internal_measures_cols[idx])
            lo_cis.append(lo_ci)
            hi_cis.append(hi_ci)
            standard_errors.append(standard_error)

        result = pd.DataFrame({
            InternalMeasureCols.name: names,
            ConfidenceIntervalCols.ci_96lo: lo_cis,
            ConfidenceIntervalCols.ci_96hi: hi_cis,
            ConfidenceIntervalCols.standard_error: standard_errors,
        })
        result = result.set_index(keys=InternalMeasureCols.name).T.round(self.__round_to)
        return result

    def differences_between_worst_and_best_partition(self):
        """Calculates effect sizes and ci of the difference in correlation between the worst and best partition for
        each internal measure.
        """
        names = []
        effect_sizes = []
        lo_cis = []
        hi_cis = []
        standard_errors = []
        for internal_measure in self.__internal_measures:
            effect_size, lo_ci, hi_ci, standard_error = self.effect_size_and_ci_of_difference_of_means_gt_and_worst_partition(
                internal_measure=internal_measure,
                worst_ranked_by=ClusteringQualityMeasures.jaccard_index)
            names.append(internal_measure)
            effect_sizes.append(effect_size)
            lo_cis.append(lo_ci)
            hi_cis.append(hi_ci)
            standard_errors.append(standard_error)

        result = pd.DataFrame({InternalMeasureCols.name: names, InternalMeasureCols.effect_size: effect_sizes,
                               ConfidenceIntervalCols.ci_96lo: lo_cis, ConfidenceIntervalCols.ci_96hi: hi_cis,
                               ConfidenceIntervalCols.standard_error: standard_errors})
        return result.round(self.__round_to)


def assess_internal_measures(overall_dataset_name: str, run_names: [str], data_type: str,
                             root_results_dir: str, data_dir: str,
                             distance_measure: str,
                             internal_measures: [str], n_clusters=0, n_segments=0):
    # load all the internal measure calculation summaries
    partitions = run_internal_measure_calculation_for_dataset(overall_ds_name=overall_dataset_name, run_names=run_names,
                                                              data_type=data_type,
                                                              results_dir=root_results_dir, data_dir=data_dir,
                                                              internal_measures=internal_measures,
                                                              distance_measure=distance_measure,
                                                              n_dropped_clusters=n_clusters,
                                                              n_dropped_segments=n_segments)

    ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                   internal_measures=internal_measures)

    store_results_in = internal_measure_assessment_dir_for(
        overall_dataset_name=overall_dataset_name,
        data_type=data_type,
        results_dir=root_results_dir, data_dir=data_dir,
        distance_measure=distance_measure,
        drop_segments=n_segments, drop_clusters=n_clusters)

    # correlation summary
    ia.correlation_summary.to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.correlation_summary))

    # effect size between difference of mean correlation of worst and gt
    ia.differences_between_worst_and_best_partition().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.effect_size_difference_worst_best))

    # descriptive statistics
    ia.descriptive_statistics_for_internal_measures_correlation().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.descriptive_statistics_measure_summary))

    # 95% CI of differences in mean correlation between internal measures
    ia.ci_of_differences_between_internal_measure_correlations().to_csv(
        get_full_filename_for_results_csv(store_results_in, IAResultsCSV.ci_of_differences_between_measures))


def run_internal_measure_assessment_datasets(overall_ds_name: str,
                                             run_names: [str],
                                             distance_measure: str = DistanceMeasures.l1_cor_dist,
                                             data_type: str = SyntheticDataType.non_normal_correlated,
                                             data_dir: str = SYNTHETIC_DATA_DIR,
                                             results_dir: str = ROOT_RESULTS_DIR,
                                             internal_measures: [str] = [ClusteringQualityMeasures.silhouette_score,
                                                                          ClusteringQualityMeasures.pmb],
                                             n_dropped_clusters: [int] = [],
                                             n_dropped_segments: [int] = [],
                                             ):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to assess
    :param n_dropped_clusters: list of the number of clusters to drop in each run, if empty then we run the assessment
    on all the cluster
    :param n_dropped_segments: list of the number of segments to drop in each run, if empty then we run the assessment
    using all segments
    """
    # decide which assessment to run
    if len(n_dropped_clusters) == 0 and len(n_dropped_segments) == 0:

        assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                 data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                 distance_measure=distance_measure,
                                 internal_measures=internal_measures)
    else:
        # run evaluation for all dropped clusters and for all dropped segments separately
        # for this we just do clusters first
        for n_clus in n_dropped_clusters:
            assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                     data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                     distance_measure=distance_measure,
                                     internal_measures=internal_measures, n_clusters=n_clus)
        # and second we do segments
        for n_seg in n_dropped_segments:
            assess_internal_measures(overall_dataset_name=overall_ds_name, run_names=run_names,
                                     data_type=data_type, root_results_dir=results_dir, data_dir=data_dir,
                                     distance_measure=distance_measure,
                                     internal_measures=internal_measures, n_segments=n_seg)


if __name__ == "__main__":
    overall_ds_name = "n2"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    run_internal_measure_assessment_datasets(overall_ds_name, run_names=run_names)
