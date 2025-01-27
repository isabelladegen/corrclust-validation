import pandas as pd

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_distance_measure_rank_distributions import \
    violin_plots_of_average_rank_per_distance_measure

title_dictionary = {
    (SYNTHETIC_DATA_DIR, SyntheticDataType.raw): "complete, raw data variant",
    (SYNTHETIC_DATA_DIR, SyntheticDataType.normal_correlated): "complete, correlated data variant",
    (SYNTHETIC_DATA_DIR, SyntheticDataType.non_normal_correlated): "complete, non-normal data variant",
    (SYNTHETIC_DATA_DIR, SyntheticDataType.rs_1min): "complete, downsampled data variant",
    (IRREGULAR_P30_DATA_DIR, SyntheticDataType.raw): "partial, raw data variant",
    (IRREGULAR_P30_DATA_DIR, SyntheticDataType.normal_correlated): "partial, correlated data variant",
    (IRREGULAR_P30_DATA_DIR, SyntheticDataType.non_normal_correlated): "partial, non-normal data variant",
    (IRREGULAR_P30_DATA_DIR, SyntheticDataType.rs_1min): "partial, downsampled data variant",
    (IRREGULAR_P90_DATA_DIR, SyntheticDataType.raw): "sparse, raw data variant",
    (IRREGULAR_P90_DATA_DIR, SyntheticDataType.normal_correlated): "sparse, correlated data variant",
    (IRREGULAR_P90_DATA_DIR, SyntheticDataType.non_normal_correlated): "sparse, non-normal data variant",
    (IRREGULAR_P90_DATA_DIR, SyntheticDataType.rs_1min): "sparse, downsampled data variant",
}


def violin_plots_for(data_dirs, dataset_types, run_names, root_results_dir, distance_measures, overall_ds_name,
                     backend):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)
            title = "Distribution of Average Ranks for the " + title_dictionary[(data_dir, data_type)]
            fig = violin_plots_of_average_rank_per_distance_measure(interpretation.average_rank_per_run,
                                                                    title=title,
                                                                    backend=backend)


if __name__ == "__main__":
    # violin plots for average ranking for each dataset in the N30
    backend = Backends.visible_tests.value
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    # this is an extensive list
    distance_measures = [DistanceMeasures.l1_cor_dist,  # lp norms
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.linf_cor_dist,
                         DistanceMeasures.l1_with_ref,  # lp norms with reference vector
                         DistanceMeasures.l2_with_ref,
                         DistanceMeasures.l3_with_ref,
                         DistanceMeasures.l5_with_ref,
                         DistanceMeasures.linf_with_ref,
                         DistanceMeasures.dot_transform_l1,  # dot transform + lp norms
                         DistanceMeasures.dot_transform_l2,
                         DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist,  # correlation metrix
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    violin_plots_for(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                     root_results_dir=root_result_dir, distance_measures=distance_measures, overall_ds_name="n30",
                     backend=backend)
