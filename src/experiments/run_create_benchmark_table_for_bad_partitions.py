from os import path

import pandas as pd

from src.evaluation.describe_clustering_quality_for_data_variant import DescribeClusteringQualityForDataVariant
from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, get_clustering_quality_multiple_data_variants_result_folder, \
    ResultsType
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends
from src.visualisation.visualise_multiple_data_variants import get_row_name_from

if __name__ == "__main__":
    """
    Creates a summary table for all data variants that serves as a benchmark table. The table includes:
    - n observation shifted
    - n clusters miss identified
    - statistics for SCW and DBI both calculated with L5 dist (as validated to give consistently good results)
    """
    backend = Backends.none.value
    save_fig = True
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    run_file = GENERATED_DATASETS_FILE_PATH
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measure = DistanceMeasures.l5_cor_dist
    clustering_quality_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    dfs = []
    # calculate df for each data variant
    for data_type in dataset_types:
        for data_dir in data_dirs:
            describe = DescribeClusteringQualityForDataVariant(wandb_run_file=run_file,
                                                               overall_ds_name=overall_ds_name,
                                                               data_type=data_type, data_dir=data_dir,
                                                               results_root_dir=root_result_dir,
                                                               distance_measure=distance_measure)
            generation_stage = SyntheticDataType.get_display_name_for_data_type(data_type)
            completeness_level = get_row_name_from(data_dir)
            df_for_variant = describe.summary_benchmark_df()
            df_for_variant.insert(0, "completeness", completeness_level)
            df_for_variant.insert(0, "generation stage", generation_stage)
            dfs.append(df_for_variant)

    # combine result into one df
    result = pd.concat(dfs, ignore_index=True)

    # save result
    folder = get_clustering_quality_multiple_data_variants_result_folder(
        results_type=ResultsType.internal_measure_evaluation,
        overall_dataset_name=overall_ds_name,
        results_dir=root_result_dir,
        distance_measure="")
    file_name = path.join(folder, distance_measure + "_" + IAResultsCSV.benchmark_summary)
    result.to_csv(str(file_name))
