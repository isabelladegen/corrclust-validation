import pytest
from hamcrest import *

from src.data_generation.create_synthetic_data_wandb import SyntheticDataConfig, one_synthetic_creation_run, \
    load_synthetic_data, SyntheticDataSets, SyntheticFileTypes, SyntheticGroundTruthFiles
from src.evaluation.clustering_result import SegmentValueClusterResult, cluster_col

from src.utils.wandb_utils import set_test_configurations


def test_wandb_synthetic_data_creation_with_loadings_correlation_method():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 10  # reduces time
    config.min_max_scaled = False
    config.do_ground_truth_analysis = True  # just to get it running
    config.do_distribution_fit = True

    # evaluation is None if the run fails
    evaluation, gt_evaluation = one_synthetic_creation_run(config)
    assert_that(evaluation.no_clusters, is_(10))
    assert_that(gt_evaluation.covering_score, is_(1.0))  # perfect covering score for ground truth

    # check data was not scaled
    for variate in config.columns:
        assert_that(evaluation.segment_value_results.df[[variate]].max()[0], is_(greater_than(config.value_range[1])))


def test_wandb_synthetic_data_creation_works_for_cholesky_method_too():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 5  # reduces time
    config.correlation_model = "cholesky"
    config.min_max_scaled = False
    config.do_ground_truth_analysis = True  # just to get it running
    config.do_distribution_fit = True
    config.max_repetitions = 1  # reduce time

    # evaluation is None if the run fails
    evaluation, gt_evaluation = one_synthetic_creation_run(config)
    assert_that(evaluation.no_clusters, is_(5))
    assert_that(gt_evaluation.covering_score, is_(1.0))  # perfect covering score for ground truth

    # check data was not scaled
    for variate in config.columns:
        assert_that(evaluation.segment_value_results.df[[variate]].max()[0], is_(greater_than(config.value_range[1])))


def test_wandb_can_run_synthetic_data_analysis_on_scaled_data():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 10  # reduces time
    config.min_max_scaled = True
    config.do_ground_truth_analysis = False  # just to get it running
    config.do_distribution_fit = False

    evaluation, gt_evaluation = one_synthetic_creation_run(config)
    assert_that(evaluation.no_clusters, is_(10))

    # check data was scaled
    for variate in evaluation.segment_value_results.scaled_columns:
        assert_that(round(evaluation.segment_value_results.df[[variate]].min()[0], 2), is_(config.value_range[0]))
        assert_that(round(evaluation.segment_value_results.df[[variate]].max()[0], 2), is_(config.value_range[1]))


@pytest.mark.skip(reason="take too long as it runs on the full dataframe")
def test_can_run_just_analysis_on_existing_data():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 10  # reduces time
    config.min_max_scaled = True
    config.do_ground_truth_analysis = False  # do it on smaller dataset
    config.do_distribution_fit = False
    config.do_logcovfrob_silhouette_analysis = False  # this is the huge dataset so let's not rerun this
    config.do_euclidian_silhouette_analysis = False  # this is the huge dataset so let's not rerun this

    # providing data will skip generation
    generated_df, generated_segment_df, gt_labels = load_synthetic_data(SyntheticDataSets.v_test)
    assert_that(generated_df.shape[0], is_(greater_than(1000000)))
    assert_that(generated_segment_df.shape[0], is_(100))

    evaluation, gt_evaluation = one_synthetic_creation_run(config, generated_df, generated_segment_df)
    assert_that(evaluation.no_clusters, is_(23))
    assert_that(gt_evaluation.covering_score, is_(1.0))  # perfect covering score for ground truth

    # check data was scaled
    for variate in evaluation.segment_value_results.scaled_columns:
        assert_that(round(evaluation.segment_value_results.df[[variate]].min()[0], 2), is_(config.value_range[0]))
        assert_that(round(evaluation.segment_value_results.df[[variate]].max()[0], 2), is_(config.value_range[1]))


def test_can_run_analysis_on_downsampled_data_by_reloading_it():
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.max_repetitions = 1  # reduces time
    config.number_of_segments = 10  # reduces time
    config.min_max_scaled = True
    config.do_ground_truth_analysis = True  # just to get it running
    config.do_distribution_fit = True

    # providing data will skip generation
    generated_df, generated_segment_df, gt_labels = load_synthetic_data(SyntheticDataSets.v_test_1min_sampling)
    assert_that(generated_df.shape[0], is_(greater_than(100)))
    assert_that(generated_segment_df.shape[0], is_(100))

    evaluation, gt_evaluation = one_synthetic_creation_run(config, generated_df, generated_segment_df)
    assert_that(evaluation.no_clusters, is_(23))
    assert_that(gt_evaluation.covering_score, is_(1.0))  # perfect covering score for ground truth

    # check data was scaled
    for variate in evaluation.segment_value_results.scaled_columns:
        assert_that(round(evaluation.segment_value_results.df[[variate]].min()[0], 2), is_(config.value_range[0]))
        assert_that(round(evaluation.segment_value_results.df[[variate]].max()[0], 2), is_(config.value_range[1]))


def test_can_run_logcorrsil_analysis_and_clustering_jaccard_analysis():
    # load existing data instead of generating to safe time
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.dataset_to_load = SyntheticDataSets.perfect_run_1min_sampling  # downsampled data is smaller
    config.data_type = SyntheticFileTypes.data
    config.min_max_scaled = False
    config.do_ground_truth_analysis = False
    config.do_distribution_fit = False
    config.do_logcorrfrob_silhouette_analysis = True
    config.do_logcovfrob_silhouette_analysis = True
    config.do_euclidian_silhouette_analysis = False
    config.calculate_clustering_jaccard_coefficient = True
    config.plot_distribution_profile_of_segment = False
    config.plot_resulting_segments = False
    config.plot_agp_like_graphs = False
    config.plot_time_heatmaps = False
    config.cov_regularisation = 0.0

    # providing data will skip generation
    generated_df, generated_segment_df, gt_labels = load_synthetic_data(run_id=config.dataset_to_load,
                                                                        data_type=config.data_type)
    # check data has been loaded correctly
    assert_that(generated_df.shape[0], is_(20440))
    assert_that(generated_segment_df.shape[0], is_(100))

    evaluation, gt_evaluation = one_synthetic_creation_run(config, generated_df, generated_segment_df)

    # check that run in general worked
    assert_that(evaluation.no_clusters, is_(23))

    # check for actual measures
    gt_segment_value_result = SegmentValueClusterResult.create_from_segment_df(gt_labels,
                                                                               generated_df,
                                                                               generated_df[config.columns].to_numpy(),
                                                                               config.columns)

    # external measure
    assert_that(evaluation.clustering_jaccard_coeff(list(gt_segment_value_result.df[cluster_col])), is_(1.0))

    # internal silhouette
    assert_that(evaluation.number_of_segments_with_valid_log_corr_frobenius_distance(), is_(100))
    assert_that(evaluation.log_corr_frobenius_silhouette_avg(), is_(0.611))
    assert_that(evaluation.log_cov_frobenius_silhouette_avg(), is_(0.441))
    assert_that(len(evaluation.log_corr_frobenius_silhouette_scores()), is_(23))
    assert_that(len(evaluation.log_cov_frobenius_silhouette_scores()), is_(23))
    evaluation.plot_log_corr_frobenius_silhouette_analysis()  # can plot

    # internal pmb
    assert_that(evaluation.logcovfrob_pmb(), is_(1.972))
    assert_that(evaluation.logcorrfrob_pmb(), is_(6.189))
    assert_that(evaluation.foerstner_cov_pmb(), is_(3.795))
    assert_that(evaluation.foerstner_corr_pmb(), is_(5.754))


def test_can_calculate_measures_with_uncorrelated_ground_truth_labels():
    # here only 000 segments will be correct
    # load existing data instead of generating to safe time
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.dataset_to_load = SyntheticDataSets.perfect_run_1min_sampling  # downsampled data is smaller
    config.data_type = SyntheticFileTypes.data
    config.gt_labels = SyntheticGroundTruthFiles.uncorrelated_splendid_sunset_1min

    config.min_max_scaled = False
    config.do_ground_truth_analysis = False
    config.do_distribution_fit = False
    config.do_logcorrfrob_silhouette_analysis = False
    config.do_logcovfrob_silhouette_analysis = False
    config.do_euclidian_silhouette_analysis = False
    config.calculate_clustering_jaccard_coefficient = True
    config.log_figures_local = False
    config.cov_regularisation = 0.0

    # providing data will skip generation
    generated_df, generated_segment_df, gt_labels = load_synthetic_data(run_id=config.dataset_to_load,
                                                                        data_type=config.data_type,
                                                                        gt_labels=config.gt_labels)
    # check data has been loaded correctly
    assert_that(generated_df.shape[0], is_(20440))
    assert_that(generated_segment_df.shape[0], is_(100))
    assert_that(gt_labels.shape[0], is_(1))  # no clusters everything is uncorrelated

    evaluation, gt_evaluation = one_synthetic_creation_run(config, generated_df, generated_segment_df)

    # check that run worked
    assert_that(evaluation.no_clusters, is_(23))

    # check jaccard is very low
    gt_segment_value_result = SegmentValueClusterResult.create_from_segment_df(gt_labels,
                                                                               generated_df,
                                                                               generated_df[config.columns].to_numpy(),
                                                                               config.columns)
    assert_that(evaluation.clustering_jaccard_coeff(list(gt_segment_value_result.df[cluster_col])), is_(0.051))


def test_continuous_evaluation_if_one_measure_does_no_calculate():
    # load existing data that throws a foerstner error for splendid-sunset normal correlated
    config = SyntheticDataConfig()
    set_test_configurations(config)
    config.dataset_to_load = SyntheticDataSets.splendid_sunset
    config.data_type = SyntheticFileTypes.normal_correlated_data
    config.min_max_scaled = False
    config.do_ground_truth_analysis = False
    config.do_distribution_fit = False
    config.do_logcorrfrob_silhouette_analysis = True
    config.do_logcovfrob_silhouette_analysis = True
    config.calculate_foerstner_distance = True
    config.do_euclidian_silhouette_analysis = False
    config.plot_distribution_profile_of_segment = False
    config.plot_resulting_segments = False
    config.plot_agp_like_graphs = False
    config.plot_time_heatmaps = False
    config.cov_regularisation = 0.0  # this would need regularisation of 0.00015

    # providing data will skip generation
    generated_df, generated_segment_df, gt_labels = load_synthetic_data(config.dataset_to_load, config.data_type)
    # check data has been loaded correctly
    assert_that(generated_df.shape[0], greater_than(1000000))
    assert_that(generated_segment_df.shape[0], is_(100))

    evaluation, gt_evaluation = one_synthetic_creation_run(config, generated_df, generated_segment_df)

    # check that run in general worked
    assert_that(evaluation.no_clusters, is_(23))

    # check for actual measures
    assert_that(evaluation.number_of_segments_with_valid_log_corr_frobenius_distance(), is_(75))
    assert_that(evaluation.log_corr_frobenius_silhouette_avg(), is_(0.956))
    assert_that(evaluation.log_cov_frobenius_silhouette_avg(), is_(0.668))
    assert_that(calling(evaluation.foerstner_silhouette_avg), raises(ValueError))
