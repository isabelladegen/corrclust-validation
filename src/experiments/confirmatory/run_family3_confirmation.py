from os import path

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.evaluation.internal_measure_ground_truth_assessment import InternalMeasureGroundTruthAssessment, \
    GroupAssessmentCols, hypotheses_string
from src.experiments.run_internal_measure_assessment_family3 import calculate_correlation_summary_for
from src.utils.clustering_quality_measures import ClusteringQualityMeasures

from src.utils.configurations import CONFIRMATORY_DATASETS_FILE_PATH, CONF_ROOT_RESULTS_DIR, \
    CONF_IRREGULAR_P90_DATA_DIR, CONF_IRREGULAR_P30_DATA_DIR, CONFIRMATORY_SYNTHETIC_DATA_DIR, \
    internal_measure_evaluation_dir_for, get_data_completeness_from
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import StatsCols, calculate_wilcox_signed_rank
from src.visualisation.run_average_rank_visualisations import data_variant_description


def wilcoxons_between(dm1: str, dm2: str, raw_values: pd.DataFrame, internal_measure: str, data_desc: str, alpha: float = 0.05,
                      target_power: float = 0.8, bonferroni_adjust: int = 1, alternative: str = 'two-sided',
                      non_zero: float = 0.0001):
    """
    Calculates  the wilcoxon's signed rank between dm1 and dm2.
    :param dm1: distance measure 1
    :param dm2: distance measure 2
    :param raw_values: raw data values pandas dataframe with double column index, level 1 internal measure, level 2 distance measure
    :param internal_measure: internal validity index
    :param data_desc: data variant description string
    :param alpha: significance level
    :param target_power: to calculate n required to achieve target power, e.g 0.8 for 80%
    :param bonferroni_adjust: divide alpha by this to adjust for multiplicity
    :param alternative: which alternative to test
    :param non_zero: what differences to consider zero
    :return: df of wilxocon's signed rank result
    """
    # results to build dataframe
    data_variants = []
    internal_measures = []
    hypotheses = []
    is_significances = []
    effect_sizes = []
    p_values = []
    nz_pairs = []
    achieved_powers = []
    statistics = []
    alphas_used = []
    n_target_powers = []

    # df of columns are distance measures, values are the scores for the runs
    values = raw_values[internal_measure]

    wilc_result = calculate_wilcox_signed_rank(values[dm1], values[dm2], non_zero, alternative=alternative)
    es = wilc_result.effect_size(alternative=alternative)
    data_variants.append(data_desc)
    internal_measures.append(internal_measure)
    hypotheses.append(hypotheses_string(dm1, dm2, effect_size=es, alternative=alternative))
    p_values.append(wilc_result.p_value)
    nz_pairs.append(wilc_result.non_zero)
    statistics.append(wilc_result.statistic)
    effect_sizes.append(es)
    achieved_powers.append(wilc_result.achieved_power(alpha=alpha, bonferroni_adjust=bonferroni_adjust,
                                                      alternative=alternative))
    alphas_used.append(alpha)
    n_target_powers.append(
        wilc_result.sample_size_for_power(target_power=target_power, alternative=alternative, alpha=alpha,
                                          bonferroni_adjust=bonferroni_adjust))
    is_significances.append(wilc_result.is_significant(alpha=alpha, bonferroni_adjust=bonferroni_adjust))

    results_dict = {
        "Data Variant": data_variants,
        "Internal Measure": internal_measures,
        "H": hypotheses,
        GroupAssessmentCols.p_value: p_values,
        GroupAssessmentCols.effect_size: effect_sizes,
        GroupAssessmentCols.non_zero_pairs: nz_pairs,
        GroupAssessmentCols.achieved_power: achieved_powers,
        GroupAssessmentCols.is_significat: is_significances,
        GroupAssessmentCols.alpha: alphas_used,
        GroupAssessmentCols.statistic: statistics,
        StatsCols.n_for_power_80: n_target_powers,
    }

    # create dataframe
    df = pd.DataFrame(results_dict)

    return df


def run_family3_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses: [], root_results_dir: str,
                                                        overall_ds_name: str, run_names: [str]):
    alpha = 0.05
    target_power = 0.8
    bonferroni_adjust = 1  # no adjustment due to hierarchical testing
    alternative = "greater"  # one-sided confirmatory tests (h[2] higher correlated than h[3])
    non_zero = 0.001

    results = []
    # Calculate stats
    for h in prereg_hypotheses:
        # h e.g  (downsampled, sparse, internal measure, DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist)
        data_type = h[0]
        data_dir = h[1]
        internal_measure = h[2]
        m1 = h[3]
        m2 = h[4]
        overall_df = calculate_correlation_summary_for(data_dir, data_type, [m1, m2], [internal_measure],
                                                       overall_ds_name, root_results_dir, run_names)
        data_variant = data_variant_description[(get_data_completeness_from(data_dir), data_type)]
        wilc_result_df = wilcoxons_between(m1, m2, overall_df, internal_measure, data_variant, alpha, target_power, bonferroni_adjust,
                                           alternative,
                                           non_zero)

        results.append(wilc_result_df)

    # Save result
    stats_df = pd.concat(results)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=root_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # all distances measures included

    full_path = path.join(store_results_in, IAResultsCSV.family_3_results)
    stats_df.to_csv(str(full_path))


if __name__ == "__main__":
    # Confirm preregistered tests from exploratory phase for confirmatory data
    overall_dataset_name = "n30"
    run_names = pd.read_csv(CONFIRMATORY_DATASETS_FILE_PATH)['Name'].tolist()
    root_result_dir = CONF_ROOT_RESULTS_DIR

    downsampled = SyntheticDataType.rs_1min
    non_normal = SyntheticDataType.non_normal_correlated
    normal = SyntheticDataType.normal_correlated
    sparse = CONF_IRREGULAR_P90_DATA_DIR
    partial = CONF_IRREGULAR_P30_DATA_DIR
    complete = CONFIRMATORY_SYNTHETIC_DATA_DIR

    # preregistered hypotheses, sequential list of tuples
    # (data_type, data_dir, internal measure, distance measure 1, distance measure 2)
    # all written as x less than y
    hypotheses = [
        (downsampled, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.foerstner_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (downsampled, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.foerstner_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (downsampled, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.foerstner_cor_dist,
         DistanceMeasures.linf_cor_dist),
        (non_normal, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (non_normal, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (normal, complete, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (non_normal, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (normal, sparse, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (normal, partial, ClusteringQualityMeasures.silhouette_score, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (downsampled, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref,
         DistanceMeasures.l1_with_ref),
        (downsampled, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_with_ref,
         DistanceMeasures.l1_with_ref),
        (normal, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist, DistanceMeasures.l5_cor_dist),
        (normal, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist, DistanceMeasures.l5_cor_dist),
        (non_normal, complete, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (non_normal, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l1_with_ref),
        (
            downsampled, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.l5_cor_dist,
            DistanceMeasures.l1_cor_dist),
        (non_normal, partial, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist,
         DistanceMeasures.l5_cor_dist),
        (normal, sparse, ClusteringQualityMeasures.dbi, DistanceMeasures.linf_cor_dist, DistanceMeasures.l3_cor_dist),
    ]

    # evaluate all hypotheses
    run_family3_wilcox_signed_rank_tests_for_hypotheses(prereg_hypotheses=hypotheses,
                                                        root_results_dir=root_result_dir,
                                                        overall_ds_name=overall_dataset_name,
                                                        run_names=run_names)
