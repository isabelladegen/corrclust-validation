import os
from collections import Counter

import pandas as pd

from src.evaluation.distance_metric_evaluation import read_csv_of_raw_values_for_all_criteria
from src.evaluation.distance_metric_ranking import DistanceMetricRanking
from src.utils.configurations import ROOT_RESULTS_DIR, GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, \
    IRREGULAR_P30_DATA_DIR, IRREGULAR_P90_DATA_DIR
from src.utils.distance_measures import DistanceMeasures
from src.utils.load_synthetic_data import SyntheticDataType


def run_ranking_for(data_dirs: [str], dataset_types: [str], run_names: [str], root_result_dir: str,
                    distance_measures: [str], overall_ds_name: str):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            # 1. load all raw_criteria_data for the data_dir and data_type
            raw_criteria_data = {}
            for run_name in run_names:
                raw_criteria_df = read_csv_of_raw_values_for_all_criteria(run_name=run_name, data_type=data_type,
                                                                          data_dir=data_dir,
                                                                          base_results_dir=root_result_dir)
                raw_criteria_data[run_name] = raw_criteria_df
            # 2. rank (this also saves the per criterion ranking as well as the overall)
            ranker = DistanceMetricRanking(raw_criteria_data, distance_measures)
            overall_rank = ranker.calculate_overall_rank(overall_ds_name=overall_ds_name,
                                                         root_results_dir=root_result_dir,
                                                         data_type=data_type, data_dir=data_dir)

            # 3. Calculate most frequent min measure in overall rank
            min_ranks = overall_rank.min(axis=1)  # per row min
            # a dict {'run_name':[list of min ranked measures (columns]}
            min_results = {a_run: overall_rank.columns[overall_rank.loc[a_run] == min_ranks[a_run]].tolist()
                           for a_run in overall_rank.index}
            # count most frequent best ranked measure
            counts = Counter()
            for cols in min_results.values():
                counts.update(cols)
            max_freq = max(counts.values())
            best_measures = [(col, counts[col]) for col in overall_rank.columns if counts[col] == max_freq]
            print_dir = os.path.basename(os.path.normpath(data_dir))
            print(print_dir + " type: " + data_type + " -> Best measure(s): " + str(best_measures))


if __name__ == "__main__":
    # configuration to both per criteria ranking and overall ranking for each dataset in the N30
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

    run_ranking_for(data_dirs=data_dirs, dataset_types=dataset_types, run_names=run_names,
                    root_result_dir=root_result_dir, distance_measures=distance_measures, overall_ds_name="n30")
