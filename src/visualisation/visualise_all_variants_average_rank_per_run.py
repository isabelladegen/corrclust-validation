import os
from os import path

import pandas as pd

from src.evaluation.interpretation_distance_metric_ranking import DistanceMetricInterpretation
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, base_dataset_result_folder_for_type, ResultsType, \
    get_image_name_based_on_data_dir, AVERAGE_RANK_DISTRIBUTION, get_image_name_based_on_data_dir_and_data_type, \
    CRITERIA_RANK_DISTRIBUTION
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_distance_measure_rank_distributions import \
    violin_plots_of_average_rank_per_distance_measure, violin_plot_grids_per_criteria_for_distance_measure

data_variant_description = {
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
                     backend, save_fig=True):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            interpretation = DistanceMetricInterpretation(run_names=run_names, overall_ds_name=overall_ds_name,
                                                          data_type=data_type,
                                                          data_dir=data_dir,
                                                          root_results_dir=root_results_dir,
                                                          measures=distance_measures)
            variant_desc = data_variant_description[(data_dir, data_type)]
            title = "Distribution of Average Ranks for the " + variant_desc
            fig = violin_plots_of_average_rank_per_distance_measure(interpretation.average_rank_per_run,
                                                                    title=title,
                                                                    backend=backend)
            criteria_title = "Distribution of Ranks for the " + variant_desc
            criteria_fig = violin_plot_grids_per_criteria_for_distance_measure(interpretation.all_criteria_ranks_df,
                                                                               title=criteria_title,
                                                                               backend=backend)
            if save_fig:
                folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.distance_measure_evaluation)
                folder = path.join(folder, "images")
                os.makedirs(folder, exist_ok=True)
                image_name = get_image_name_based_on_data_dir_and_data_type(AVERAGE_RANK_DISTRIBUTION, data_dir,
                                                                            data_type)
                fig.savefig(path.join(folder, image_name), dpi=300, bbox_inches='tight')

                image_name_criteria = get_image_name_based_on_data_dir_and_data_type(CRITERIA_RANK_DISTRIBUTION,
                                                                                     data_dir,
                                                                                     data_type)
                criteria_fig.savefig(path.join(folder, image_name_criteria), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # violin plots for average ranking for each dataset in the N30
    backend = Backends.visible_tests.value
    save_fig = True
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
                     backend=backend, save_fig=save_fig)
