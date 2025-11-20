import os

import pandas as pd

from src.evaluation.internal_measure_assessment import IAResultsCSV
from src.experiments.run_internal_measure_assessment_family3 import run_wilcox_signed_rank_for
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, internal_measure_evaluation_dir_for, \
    SYNTHETIC_DATA_DIR, ROOT_RESULTS_DIR, IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR, VALID_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_ds_name = "n30"
    root_result_dir = ROOT_RESULTS_DIR
    save_results_dir = VALID_ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated]

    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    alternative = "two-sided"
    alpha = 0.05
    non_zero = 0.001
    bonferroni_adjust = 1

    # valid
    distance_measures = [DistanceMeasures.l1_cor_dist,
                         DistanceMeasures.l2_cor_dist,
                         DistanceMeasures.l3_cor_dist,
                         DistanceMeasures.l5_cor_dist,
                         DistanceMeasures.dot_transform_l2
                         ]

    internal_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()

    # list of series of wilcoxons results for each internal measures and data variant
    results = []
    for data_dir in data_dirs:
        for data_type in dataset_types:
            print("Dataset type: " + data_type + ", Compactness: " + data_dir)
            wilx_results = run_wilcox_signed_rank_for(overall_ds_name="n30", run_names=run_names,
                                                      distance_measures=distance_measures, data_type=data_type,
                                                      data_dir=data_dir, results_dir=root_result_dir,
                                                      internal_measures=internal_measures,
                                                      alternative=alternative,
                                                      non_zero=non_zero,
                                                      bonferroni_adjust=bonferroni_adjust,
                                                      alpha=alpha)
            results.extend(wilx_results)

    # store result
    overall_df = pd.concat(results, axis=1).T.reset_index(drop=True)

    store_results_in = internal_measure_evaluation_dir_for(
        overall_dataset_name=overall_ds_name,
        data_type='',  # all datatypes as rows
        results_dir=save_results_dir,
        data_dir='',  # all data data comp as rows
        distance_measure='')  # fixed to one

    # save stats results
    overall_df.to_csv(str(os.path.join(store_results_in, IAResultsCSV.family_3)))
