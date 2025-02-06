import os
from os import path

import pandas as pd

from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, get_image_name_based_on_data_dir_and_data_type, \
    internal_measure_evaluation_dir_for, PARTITIONS_QUALITY_DESCRIPTION
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.partitions_visualisation import PartitionVisualisation


def partition_visualisation(data_dirs, dataset_types, run_names, root_results_dir, distance_measure, overall_ds_name,
                            backend, save_fig=True):
    columns = [ClusteringQualityMeasures.jaccard_index,
               DescribeBadPartCols.n_wrong_clusters,
               DescribeBadPartCols.n_obs_shifted]
    for data_dir in data_dirs:
        for data_type in dataset_types:
            partition_vis = PartitionVisualisation(overall_dataset_name=overall_ds_name, data_type=data_type,
                                                   root_results_dir=root_results_dir, data_dir=data_dir,
                                                   distance_measure=distance_measure, run_names=run_names)

            fig = partition_vis.plot_multiple_quality_measures(columns=columns, backend=backend)

            if save_fig:
                folder = internal_measure_evaluation_dir_for(
                    overall_dataset_name=overall_ds_name,
                    data_type=data_type,
                    results_dir=root_results_dir, data_dir=data_dir,
                    distance_measure=distance_measure)
                folder = path.join(folder, "images")
                os.makedirs(folder, exist_ok=True)
                image_name = get_image_name_based_on_data_dir_and_data_type(PARTITIONS_QUALITY_DESCRIPTION, data_dir,
                                                                            data_type)
                fig.savefig(path.join(folder, image_name), dpi=300, bbox_inches='tight')


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

    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l1_with_ref,
                         DistanceMeasures.foerstner_cor_dist]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    partition_visualisation(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                            root_results_dir=root_result_dir, distance_measure=distance_measures[0],
                            overall_ds_name="n30",
                            backend=backend, save_fig=save_fig)
