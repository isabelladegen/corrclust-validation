import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV, InternalMeasureAssessment, InternalMeasureCols
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_completeness_from
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import calculate_wilcox_signed_rank, StatsCols
from src.visualisation.run_average_rank_visualisations import data_variant_description


def run_wilcox_signed_rank_for(overall_ds_name: str, run_names: [str], distance_measure: str,
                               data_type: str, data_dir: str, results_dir: str, internal_measure1: str,
                               internal_measure2: str, alternative: str, non_zero: float):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measure: name of distance measure to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measure1: measure 1
    :param internal_measure2: measure 2
    :param alternative: which alternative to use to assess the measure
    :param non_zero: non-zero value to control what differences are considered 0
    :return wilcox result
    """
    # load all the internal measure calculation summaries
    partitions = read_clustering_quality_measures(overall_ds_name=overall_ds_name, data_type=data_type,
                                                  root_results_dir=results_dir, data_dir=data_dir,
                                                  distance_measure=distance_measure, run_names=run_names)
    ia = InternalMeasureAssessment(distance_measure=distance_measure, dataset_results=partitions,
                                   internal_measures=[internal_measure1, internal_measure2], )

    correlation_summary = ia.correlation_summary.copy()

    # invert correlations for DBI
    if internal_measure1 == ClusteringQualityMeasures.dbi or internal_measure2 == ClusteringQualityMeasures.dbi:
        dbi_cols = [col for col in correlation_summary.columns if ClusteringQualityMeasures.dbi in col]
        # turn copy warning off given that we work on a copy of the df
        with pd.option_context('mode.chained_assignment', None):
            # for DBI where lower values are better we need to invert the correlation coefficients for a fair comparison
            correlation_summary[dbi_cols] = correlation_summary[dbi_cols].multiply(-1)

    # perform wilcox test
    m1_coefficients = correlation_summary[internal_measure1]
    m2_coefficients = correlation_summary[internal_measure2]

    # calculate statistic
    wilc_result = calculate_wilcox_signed_rank(m1_coefficients, m2_coefficients, non_zero,
                                               alternative=alternative)
    return wilc_result


if __name__ == "__main__":
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]

    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    alternative = "two-sided"
    alpha = 0.05
    non_zero = 0.001
    bonferroni_adjust = 1

    # TODO select the measures for this
    distance_measure = None

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    results = {}

    for data_dir in data_dirs:
        for data_type in dataset_types:
            print(" , Dataset type: " + data_type + ", Compactness: " + data_dir)
            wilx_results = run_wilcox_signed_rank_for(overall_ds_name="n30", run_names=run_names,
                                                      distance_measure=distance_measure, data_type=data_type,
                                                      data_dir=data_dir, results_dir=root_result_dir,
                                                      internal_measure1=internal_measures[0],
                                                      internal_measure2=internal_measures[2],
                                                      alternative=alternative,
                                                      non_zero=non_zero)
            variant_desc = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
            results[variant_desc] = wilx_results

    # create one df of results
    data_variants = []
    measures = []
    p_values = []
    statistics = []
    effect_sizes = []
    powers = []
    alphas = []
    nz_pairs = []
    significants = []
    names = ",".join(internal_measures)
    for data_variant, wilc_result in results.items():
        data_variants.append(data_variant)
        names.append(names)
        p_values.append(wilc_result.pvalue)
        statistics.append(wilc_result.statistic)
        effect_sizes.append(wilc_result.effect_size(alternative=alternative))
        powers.append(
            wilc_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust, alternative=alternative))
        alphas.append(alpha)
        nz_pairs.append(wilc_result.non_zero)
        significants.append(wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))

    overall_df = pd.DataFrame({
        "Data Variant": data_variants,
        InternalMeasureCols.name: names,
        StatsCols.is_significant: significants,
        StatsCols.p_value: p_values,
        StatsCols.statistic: statistics,
        StatsCols.effect_size: effect_sizes,
        StatsCols.achieved_power: powers,
        StatsCols.alpha: alphas,
        StatsCols.none_zero_pairs: nz_pairs
    })
    overall_df = overall_df.set_index(keys=InternalMeasureCols.name).T.round(2)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_result_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # fixed to one

    # save stats results
    overall_df.to_csv(str(os.path.join(store_results_in, distance_measure + "_" + IAResultsCSV.family_4_tests)))
