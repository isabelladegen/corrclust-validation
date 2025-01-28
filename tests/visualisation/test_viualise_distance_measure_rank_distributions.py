import pandas as pd
from hamcrest import *

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import IRREGULAR_P30_DATA_DIR, ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, \
    IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_distance_measure_rank_distributions import \
    violin_plots_of_average_rank_per_distance_measure, violin_plot_grids_per_criteria_for_distance_measure

root_results_dir = ROOT_RESULTS_DIR
measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist, DistanceMeasures.log_frob_cor_dist,
            DistanceMeasures.foerstner_cor_dist]
overall_ds_name = "n30"
run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
backend = Backends.visible_tests.value

partial_non_normal = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                  data_type=SyntheticDataType.non_normal_correlated,
                                                  data_dir=IRREGULAR_P30_DATA_DIR,
                                                  root_results_dir=root_results_dir,
                                                  measures=measures)

sparse_non_normal = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                 data_type=SyntheticDataType.non_normal_correlated,
                                                 data_dir=IRREGULAR_P90_DATA_DIR,
                                                 root_results_dir=root_results_dir,
                                                 measures=measures)


def test_plots_violin_plots_of_average_rank_per_distance_measure():
    measure = 'Distribution of Average Ranks for the partial, non-normal data variant'
    fig = violin_plots_of_average_rank_per_distance_measure(partial_non_normal.average_rank_per_run, title=measure,
                                                            backend=backend)

    assert_that(fig, is_(not_none()))

    measure = 'Distribution of Average Ranks for the sparse, non-normal data variant'
    fig = violin_plots_of_average_rank_per_distance_measure(sparse_non_normal.average_rank_per_run, title=measure,
                                                            backend=backend)
    assert_that(fig, is_(not_none()))


def test_plots_grid_of_violin_plots_per_criterion():
    fig = violin_plot_grids_per_criteria_for_distance_measure(partial_non_normal.raw_criteria_ranks_df,
                                                              title="Partial, non-normal",
                                                              backend=backend)
    assert_that(fig, is_(not_none()))

    fig = violin_plot_grids_per_criteria_for_distance_measure(sparse_non_normal.raw_criteria_ranks_df,
                                                              title="Sparse, non-normal",
                                                              backend=backend)
    assert_that(fig, is_(not_none()))
