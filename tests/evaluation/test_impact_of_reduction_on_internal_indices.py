from hamcrest import *

from src.evaluation.impact_of_reduction_on_internal_indices import ImpactReductionOnInternalIndices, ReductionType, \
    get_col_name_reduction_internal_corr, get_col_reductions_compared
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_REDUCED_RESULTS_DIR, ROOT_RESULTS_DIR, ROOT_REDUCED_SYNTHETIC_DATA_DIR, \
    SYNTHETIC_DATA_DIR, DataCompleteness
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.stats import StatsCols

internal_measure = ClusteringQualityMeasures.silhouette_score
distance_measure = DistanceMeasures.l1_cor_dist
root_reduced_results_dir = ROOT_REDUCED_RESULTS_DIR
root_reduced_data_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR
root_results_dir = ROOT_RESULTS_DIR
unreduced_data_dir = SYNTHETIC_DATA_DIR
overall = "n30"


def test_combines_all_correlation_results_for_index_and_all_reductions_into_one_df_for_clusters():
    n_dropped = [12, 17]
    reduction_type = ReductionType.clusters
    ir = ImpactReductionOnInternalIndices(overall_ds_name=overall,
                                          reduced_root_result_dir=root_reduced_results_dir,
                                          unreduced_root_result_dir=root_results_dir,
                                          data_type=SyntheticDataType.normal_correlated,
                                          root_reduced_data_dir=root_reduced_data_dir,
                                          unreduced_data_dir=unreduced_data_dir,
                                          data_completeness=DataCompleteness.irregular_p30,
                                          n_dropped=n_dropped, reduction_type=reduction_type,
                                          internal_measure=internal_measure,
                                          distance_measure=distance_measure)

    df = ir.correlations_df

    # loads all the correlation values for each subject and each reduction including none reduction
    assert_that(df.shape, is_((30, len(n_dropped) + 2)))

    # check values for n_dropped 17
    n_dropped_17_vals = df[
        get_col_name_reduction_internal_corr(internal_measure=internal_measure, n_dropped=17,
                                             reduction_type=reduction_type)]
    assert_that(n_dropped_17_vals.iloc[0], is_(0.984))
    assert_that(n_dropped_17_vals.iloc[1], is_(0.532))


def test_combines_all_correlation_results_for_index_and_all_reductions_into_one_df_for_segments():
    n_dropped = [50, 75]
    reduction_type = ReductionType.segments
    ir = ImpactReductionOnInternalIndices(overall_ds_name=overall,
                                          reduced_root_result_dir=root_reduced_results_dir,
                                          unreduced_root_result_dir=root_results_dir,
                                          data_type=SyntheticDataType.normal_correlated,
                                          root_reduced_data_dir=root_reduced_data_dir,
                                          unreduced_data_dir=unreduced_data_dir,
                                          data_completeness=DataCompleteness.irregular_p90,
                                          n_dropped=n_dropped, reduction_type=reduction_type,
                                          internal_measure=internal_measure,
                                          distance_measure=distance_measure)

    df = ir.correlations_df

    # loads all the correlation values for each subject and each reduction including none reduction
    assert_that(df.shape, is_((30, len(n_dropped) + 2)))

    # check values for n_dropped 17
    n_dropped_75_vals = df[
        get_col_name_reduction_internal_corr(internal_measure=internal_measure, n_dropped=75,
                                             reduction_type=reduction_type)]
    assert_that(n_dropped_75_vals.iloc[0], is_(0.984))
    assert_that(n_dropped_75_vals.iloc[1], is_(0.532))

    # calculate paired t test between the different reductions
    paired_t_test = ir.paired_samples_t_test_on_fisher_transformed_correlation_coefficients()
    # rows is the statistics, cols is the reductions compared
    assert_that(paired_t_test.shape, is_((5, 3)))
    results = paired_t_test[get_col_reductions_compared(0, 75, reduction_type)]
    assert_that(results.loc[StatsCols.p_value], is_(0))
    assert_that(results.loc[StatsCols.achieved_power], is_(0))
    assert_that(results.loc[StatsCols.effect_size], is_(0))
