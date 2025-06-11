import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV, InternalMeasureAssessment, InternalMeasureCols
from src.evaluation.internal_measure_ground_truth_assessment import GroupAssessmentCols, \
    internal_measure_lower_values_best
from src.experiments.run_cluster_quality_measures_calculation import read_clustering_quality_measures
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import calculate_wilcox_signed_rank, StatsCols
from src.visualisation.run_average_rank_visualisations import data_variant_description


def stats_for_correlation_ranks_across_all_runs(correlation_values_for_im: pd.DataFrame, internal_measure: str,
                                                round_to: int = 3):
    """ Calculates ranks of dm columns for the given internal measure and returns stats for ranks
    Expects DBI to have been inverted to positive correlations
    """
    # Rank across columns (axis=1) for each row
    ranked_df = correlation_values_for_im.rank(axis=1, method='dense', ascending=False)
    return ranked_df.describe().round(round_to)


def wilcoxons_signed_rank_correlation_step_down(correlations_for_internal_measure, internal_measure,
                                                alpha: float = 0.05,
                                                bonferroni_adjust: int = 1,
                                                alternative: str = 'two-sided',
                                                non_zero: float = 0.001):
    """
    Calculates for each internal measure the wilcoxon's signed rank test between different distance measures in a
    step-down fashion until it finds a significant difference between two measures correlation
    :param correlations_for_internal_measure: raw values  with columns distance measures and values correlation coefficient
    :param internal_measure: clustering quality measure
    :param alpha: significance level
    :param bonferroni_adjust: divide alpha by this to adjust for multiplicity
    :param alternative: which alternative to test
    :param non_zero: what differences to consider zero
    :return: series for internal measure columns wilcoxon results
    """
    # raw values with columns distance measures and values correlation coefficient
    values = correlations_for_internal_measure

    # rank the distance measures
    # dm are tested in order of rank (lowest is best)
    # df with rows=describe stats, columns= distance measures,
    # cells=rank stats for that distance measure across all subjects
    ranks = stats_for_correlation_ranks_across_all_runs(values, internal_measure)

    ordered_dm = ranks.loc['mean'].sort_values().index.tolist()
    dm_1 = ordered_dm[0]
    measure1_raw = values[dm_1]

    dms_tested = []
    wilc_result = None

    # cycle through all dm until significant difference is found
    for dm_2 in ordered_dm[1:]:
        dms_tested.append(dm_2)
        measure2_raw = values[dm_2]
        wilc_result = calculate_wilcox_signed_rank(measure1_raw, measure2_raw, non_zero,
                                                   alternative=alternative)
        # stop if significant
        if wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust):
            break

    results_dict = {
        "Internal Measure": internal_measure,
        "Best ranked dm": dm_1,
        "compared to dm": dm_2,
        GroupAssessmentCols.effect_size: wilc_result.effect_size(alternative=alternative),
        GroupAssessmentCols.non_zero_pairs: wilc_result.non_zero,
        GroupAssessmentCols.is_significat: wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust),
        GroupAssessmentCols.p_value: wilc_result.p_value,
        GroupAssessmentCols.achieved_power: wilc_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                                       alternative=alternative),
        "Non sig dm": dms_tested,
        "Not tested dms": list(set(ordered_dm[1:]) - set(dms_tested)),
        GroupAssessmentCols.alpha: alpha,
        GroupAssessmentCols.statistic: wilc_result.statistic,
    }

    # create dataframe
    series = pd.Series(results_dict)

    return series


def run_wilcox_signed_rank_for(overall_ds_name: str, run_names: [str], distance_measures: [str],
                               data_type: str, data_dir: str, results_dir: str,
                               internal_measures: [str], alternative: str, non_zero: float, bonferroni_adjust: int,
                               alpha: float):
    """ Runs the internal measure assessment on all ds in the csv files of the generated runs
    :param overall_ds_name: a name for the dataset we're using e.g. n30 or n2
    :param run_names: list of run_names to load (subjects)
    :param distance_measures: names of distance measures to run assessment for
    :param data_type: which datatype to use see SyntheticDataType
    :param data_dir: where to read the data from
    :param results_dir: directory where to store the results, it will use a subdirectory based on the distance measure,
    and the data type
    :param internal_measures: list of internal measures to examine
    :param alternative: which alternative to use to assess the measure
    :param non_zero: non-zero value to control what differences are considered 0
    :param bonferroni_adjust: how much bonferroni adjustment to use
    :param alpha: how much significant differences we should consider
    :return wilcox results as a list of series for each internal measure
    """
    overall_df = calculate_correlation_summary_for(data_dir, data_type, distance_measures, internal_measures,
                                                   overall_ds_name, results_dir, run_names)

    # calculate step down wilcox signed rank for each internal measure
    wilx_results = []
    variant_desc = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
    for im in internal_measures:
        wilx = wilcoxons_signed_rank_correlation_step_down(overall_df[im], im, alpha, bonferroni_adjust, alternative,
                                                           non_zero)

        wilx = pd.concat([pd.Series({'Data Variant': variant_desc}), wilx])
        wilx.name = (variant_desc, im)
        wilx_results.append(wilx)

    return wilx_results


def calculate_correlation_summary_for(data_dir, data_type, distance_measures, internal_measures, overall_ds_name,
                                      results_dir, run_names):
    # dictionary with key = dm and values correlation f
    correlations = {}
    for dm in distance_measures:
        # load all the internal measure calculation summaries
        partitions = read_clustering_quality_measures(overall_ds_name=overall_ds_name, data_type=data_type,
                                                      root_results_dir=results_dir, data_dir=data_dir,
                                                      distance_measure=dm, run_names=run_names)
        ia = InternalMeasureAssessment(distance_measure=dm, dataset_results=partitions,
                                       internal_measures=internal_measures, )
        # keep relevant r columns and names
        correlation_summary = ia.correlations_for_statistical_tests()

        # rename columns to measures
        rename_to_measures = {next((s for s in ia.measures_corr_col_names if im in s), None): im for im in
                              internal_measures}
        correlation_summary = correlation_summary.rename(columns=rename_to_measures)
        correlation_summary.set_index(InternalMeasureCols.name, inplace=True)
        correlations[dm] = correlation_summary
    # create one big dataframe, with index run_name and columns multiindex first level internal measure,
    # second level internal dm
    overall_df = pd.concat(correlations, axis=1).swaplevel(axis=1)
    return overall_df


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

    # only the once that pass corr r>0.5
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,
                         DistanceMeasures.foerstner_cor_dist
                         ]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    # list of series of wilcoxons results for each internal measures and data variant
    results = []
    for data_dir in data_dirs:
        for data_type in dataset_types:
            print("Dataset type: " + data_type + ", Compactness: " + data_dir)
            wilx_results = run_wilcox_signed_rank_for(overall_ds_name="n30", run_names=run_names,
                                                      distance_measures=distance_measures, data_type=data_type,
                                                      data_dir=data_dir, results_dir=root_result_dir,
                                                      internal_measures=internal_measures,
                                                      alternative=alternative,
                                                      non_zero=non_zero,
                                                      bonferroni_adjust=bonferroni_adjust,
                                                      alpha=alpha)
            results.extend(wilx_results)

    # store result
    overall_df = pd.concat(results, axis=1).T.reset_index(drop=True)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_result_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # fixed to one

    # save stats results
    overall_df.to_csv(str(os.path.join(store_results_in, IAResultsCSV.family_3)))
