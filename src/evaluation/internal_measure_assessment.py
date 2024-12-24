import itertools
import os
from dataclasses import dataclass
from os import path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.configurations import ROOT_DIR
from src.stats import standardized_effect_size_of_mean_difference, calculate_hi_lo_difference_ci, ConfidenceIntervalCols
from src.synthetic_assessment.distance_metric_assessment import DistanceMeasureCols
from src.synthetic_data.describe_bad_partitions import default_external_measures, default_internal_measures, \
    DescribeBadPartCols, DescribeBadPartitions


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
                worst_ranked_by=DescribeBadPartCols.jaccard_index)
            names.append(internal_measure)
            effect_sizes.append(effect_size)
            lo_cis.append(lo_ci)
            hi_cis.append(hi_ci)
            standard_errors.append(standard_error)

        result = pd.DataFrame({InternalMeasureCols.name: names, InternalMeasureCols.effect_size: effect_sizes,
                               ConfidenceIntervalCols.ci_96lo: lo_cis, ConfidenceIntervalCols.ci_96hi: hi_cis,
                               ConfidenceIntervalCols.standard_error: standard_errors})
        return result.round(self.__round_to)


def save_results(distance_measure: str, partitions: [pd.DataFrame], internal_measures: [str], directory: str):
    ia = InternalMeasureAssessment(distance_measure=distance_measure, internal_measures=internal_measures,
                                   dataset_results=partitions)

    # correlation summary
    ia.correlation_summary.to_csv(directory + 'correlation_summary_30_n_d_67_partitions.csv')

    # effect size between difference of mean correlation of worst and gt
    ia.differences_between_worst_and_best_partition().to_csv(
        directory + 'effect_size_difference_of_worst_to_best_partition.csv')

    # descriptive statistics
    ia.descriptive_statistics_for_internal_measures_correlation().to_csv(
        directory + 'descriptive_statistics_internal_measures_correlation.csv')

    # 95% CI of differences in mean correlation between internal measures
    ia.ci_of_differences_between_internal_measure_correlations().to_csv(
        directory + 'ci_differences_between_internal_measure_correlation.csv')


def evaluate_all_partitions(dir_name: str, distance_measure: str, internal_measures: [str], n_clusters=None,
                            n_segments=None):
    partitions = []
    for ds_name in generated_ds:
        print(ds_name)
        # we don't vary the seed so all datasets will select the same clusters and segments
        sum_df = DescribeBadPartitions(ds_name=ds_name, distance_measure=distance_measure,
                                       internal_measures=internal_measures, n_segments=n_segments,
                                       n_clusters=n_clusters, test_run=False).summary_df.copy()
        sum_df.to_csv(directory + ds_name + '_measures_summary.csv')
        partitions.append(sum_df)

    save_results(distance_measure=distance_measure, partitions=partitions, internal_measures=internal_measures,
                 directory=dir_name)


if __name__ == "__main__":
    # assess all 30 ds
    csv_file = path.join(ROOT_DIR, 'experiments/evaluate/csv/synthetic-data/wandb_export_30_ds-creation.csv')
    generated_ds = pd.read_csv(csv_file)['Name'].tolist()

    subdirectory_names = {
        DistanceMeasureCols.l1_with_ref: "L1ref/",
        DistanceMeasureCols.l2_with_ref: "L2ref/",
        DistanceMeasureCols.l1_cor_dist: "L1/",
        DistanceMeasureCols.l2_cor_dist: "L2/",
    }

    # configurations
    distance_measure = DistanceMeasureCols.l1_with_ref
    directory = "tables/" + subdirectory_names[distance_measure]
    internal_measures = [DescribeBadPartCols.silhouette_score, DescribeBadPartCols.pmb]
    # if both lists are empty, we do the standard evaluation as before
    n_clusters = [2, 4, 8, 12, 16, 20]  # add the number of clusters
    n_segments = [10, 20, 40, 60, 80]  # add the number of segments

    # for this we just do clusters
    for n_clus in n_clusters:
        dir_name = directory + "Clusters_" + str(n_clus) + "/"
        # create dir if it does not exist
        os.makedirs(dir_name, exist_ok=True)
        # do evaluation
        evaluate_all_partitions(dir_name, distance_measure, internal_measures, n_clusters=n_clus)

    # we also do segments
    for n_seg in n_segments:
        dir_name = directory + "Segments_" + str(n_seg) + "/"
        # create dir if it does not exist
        os.makedirs(dir_name, exist_ok=True)
        # do evaluation
        evaluate_all_partitions(dir_name, distance_measure, internal_measures, n_segments=n_seg)

    # if both lists are empty we do the evaluations as before with all clusters and all segments
    if len(n_clusters) == 0 and len(n_segments) == 0:
        evaluate_all_partitions(directory, distance_measure, internal_measures)
