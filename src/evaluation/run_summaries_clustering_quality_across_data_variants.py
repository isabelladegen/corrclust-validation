from collections import defaultdict
from os import path

import pandas as pd

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant, \
    IntSummaryCols
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, get_clustering_quality_multiple_data_variants_result_folder, \
    ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import get_row_name_from


def summarise_clustering_quality(data_dirs: [str], data_types: [str], run_file: str, root_results_dir: str,
                                 distance_measures: [str], clustering_quality_measures: [str], overall_ds_name: str):
    all_dfs = []
    for data_type in data_types:
        for data_dir in data_dirs:
            # turn row data into dictionary grouping all distance measures into same row
            row_data = {}
            for distance_measure in distance_measures:
                describe = DescribeClusteringQualityForDataVariant(wandb_run_file=run_file,
                                                                   overall_ds_name=overall_ds_name,
                                                                   data_type=data_type, data_dir=data_dir,
                                                                   results_root_dir=root_results_dir,
                                                                   distance_measure=distance_measure)
                df = describe.mean_sd_correlation_for(quality_measures=clustering_quality_measures)
                # group all distance measure to the same row data
                row_data[IntSummaryCols.data_stage] = df[IntSummaryCols.data_stage].iloc[0]
                row_data[IntSummaryCols.data_completeness] = df[IntSummaryCols.data_completeness].iloc[0]
                for quality_measure in clustering_quality_measures:
                    row_data[(quality_measure, distance_measure)] = df[quality_measure].iloc[0]

            all_dfs.append(row_data)

    mean_corr_df = pd.DataFrame(all_dfs)

    # sort the rows
    data_stage_order = [
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.normal_correlated),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.rs_1min),
        SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.raw)
    ]

    completeness_order = [
        get_row_name_from(IRREGULAR_P30_DATA_DIR),
        get_row_name_from(IRREGULAR_P90_DATA_DIR),
        get_row_name_from(SYNTHETIC_DATA_DIR)
    ]

    mean_corr_df = mean_corr_df.sort_values(
        by=[IntSummaryCols.data_stage, IntSummaryCols.data_completeness],
        ascending=[True, True],
        key=lambda x: pd.Categorical(x,
                                     categories=data_stage_order if x.name == IntSummaryCols.data_stage else completeness_order)
    )
    folder = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.internal_measure_evaluation,
        overall_dataset_name=overall_ds_name,
        results_dir=root_results_dir,
        distance_measure="")
    file_name = path.join(folder, IAResultsCSV.mean_correlation_data_variant)
    mean_corr_df.to_csv(str(file_name))


if __name__ == "__main__":
    # violin plots for all clustering quality measures for each dataset in the N30 and each distance measure
    # backend = Backends.visible_tests.value
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
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

    clustering_quality_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    run_file = GENERATED_DATASETS_FILE_PATH

    summarise_clustering_quality(data_dirs=data_dirs, data_types=dataset_types, run_file=run_file,
                                 root_results_dir=root_result_dir, distance_measures=distance_measures,
                                 clustering_quality_measures=clustering_quality_measures,
                                 overall_ds_name=overall_ds_name)
