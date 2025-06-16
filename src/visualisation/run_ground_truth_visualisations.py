from os import path

import pandas as pd

from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, internal_measure_evaluation_dir_for
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_clustering_quality_measures_for_multiple_data_variants import \
    VisualiseGroundTruthClusteringQualityMeasuresForDataVariants


def ground_truth_visualisations(data_dirs: [str], data_types: [str], root_results_dir: str,
                                distance_measures: [str], clustering_quality_measures: [str],
                                overall_ds_name: str, backend: str, save_fig=True):
    gtv = VisualiseGroundTruthClusteringQualityMeasuresForDataVariants(overall_ds_name=overall_ds_name,
                                                                       dataset_types=data_types,
                                                                       data_dirs=data_dirs,
                                                                       result_root_dir=root_results_dir,
                                                                       internal_measures=clustering_quality_measures,
                                                                       distance_measures=distance_measures,
                                                                       backend=backend)
    data = gtv.all_variants_ground_truth
    all_dfs = []
    for completeness in data.keys():
        for type_key in data[completeness].keys():
            for internal_measure in data[completeness][type_key].keys():
                # Get the current dataframe
                df = data[completeness][type_key][internal_measure]

                # Calculate descriptive statistics for each column
                desc_stats = df.describe().round(2)

                # Reset index to make statistics a column
                desc_stats = desc_stats.reset_index()
                desc_stats = desc_stats.rename(columns={'index': 'Statistic'})

                # Add the dictionary key columns
                desc_stats.insert(0, 'Internal Measure', internal_measure)
                desc_stats.insert(0, 'Type', type_key)
                desc_stats.insert(0, 'Completeness', completeness)

                # Add to list
                all_dfs.append(desc_stats)

    # Concatenate all dataframes
    final_df = pd.concat(all_dfs, ignore_index=True)
    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type="",  # all
        results_dir=root_results_dir, data_dir="",  # all
        distance_measure="")  # all
    full_path = path.join(store_results_in, "descriptive_statistics_for_ground_truth_clusterings.csv")
    final_df.to_csv(str(full_path))

    gtv.ci_mean_ground_truth_for_quality_measures(save_fig=save_fig)


if __name__ == "__main__":
    # confidence interval
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measures = [DistanceMeasures.l1_cor_dist, DistanceMeasures.l2_cor_dist, DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist, DistanceMeasures.linf_cor_dist, DistanceMeasures.l1_with_ref,
                         DistanceMeasures.l5_with_ref, DistanceMeasures.dot_transform_linf,
                         DistanceMeasures.log_frob_cor_dist, DistanceMeasures.foerstner_cor_dist]

    ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
                                root_results_dir=root_result_dir, distance_measures=distance_measures,
                                clustering_quality_measures=[ClusteringQualityMeasures.silhouette_score,
                                                             ClusteringQualityMeasures.dbi],
                                overall_ds_name=overall_ds_name,
                                backend=backend, save_fig=save_fig)

    # ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
    #                             root_results_dir=root_result_dir, distance_measures=distance_measures,
    #                             clustering_quality_measures=[ClusteringQualityMeasures.vrc],
    #                             overall_ds_name=overall_ds_name,
    #                             backend=backend, save_fig=save_fig)
    #
    # ground_truth_visualisations(data_dirs=data_dirs, data_types=dataset_types,
    #                             root_results_dir=root_result_dir, distance_measures=distance_measures,
    #                             clustering_quality_measures=[ClusteringQualityMeasures.pmb],
    #                             overall_ds_name=overall_ds_name,
    #                             backend=backend, save_fig=save_fig)
