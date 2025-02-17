from os import path

import pandas as pd

from src.data_generation.create_reduced_datasets import CreateReducedDatasets
from src.data_generation.wandb_create_synthetic_data import save_data_labels_to_file
from src.utils.configurations import GENERATED_DATASETS_FILE_PATH, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, ROOT_REDUCED_SYNTHETIC_DATA_DIR, get_root_folder_for_reduced_cluster, \
    get_root_folder_for_reduced_segments, get_data_dir, DataCompleteness
from src.utils.load_synthetic_data import SyntheticDataType

if __name__ == "__main__":
    overall_dataset_name = "n30"
    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    seed = 600

    data_types = [SyntheticDataType.normal_correlated, SyntheticDataType.non_normal_correlated]
    completeness = [DataCompleteness.complete, DataCompleteness.irregular_p30, DataCompleteness.irregular_p90]

    root_reduced_data_dir = ROOT_REDUCED_SYNTHETIC_DATA_DIR

    # drop 50% and 75% of clusters and segments
    n_dropped_clusters = [12, 17]
    n_dropped_segments = [50, 75]

    for comp in completeness:
        for data_type in data_types:
            data_dir = get_data_dir(SYNTHETIC_DATA_DIR, comp)  # to read data
            rd = CreateReducedDatasets(run_names=run_names, data_type=SyntheticDataType.normal_correlated,
                                       data_dir=data_dir, drop_n_clusters=n_dropped_clusters,
                                       drop_n_segments=n_dropped_segments, base_seed=seed)
            # reduced clusters label and data dfs
            reduced_clusters_labels = rd.reduced_labels_patterns
            reduced_clusters_data = rd.reduced_data_patterns

            # reduced segments label and data dfs
            reduced_segments_labels = rd.reduced_labels_segments
            reduced_segments_data = rd.reduced_data_segments

            # save df of clusters kept
            df = pd.DataFrame([(dropped_clust, name, list_clusters_selected)
                               for dropped_clust, inner_dict in rd.selected_patterns.items()
                               for name, list_clusters_selected in inner_dict.items()],
                              columns=['n dropped clusters', 'name', 'selected clusters'])
            df.to_csv(path.join(root_reduced_data_dir, comp + '_' + data_type + "_overview_of_clusters_selected.csv"))

            # save df of segments kept
            df = pd.DataFrame([(dropped_seg, name, list_segments_selected)
                               for dropped_seg, inner_dict in rd.selected_segments.items()
                               for name, list_segments_selected in inner_dict.items()],
                              columns=['n dropped segments', 'name', 'selected segments'])
            df.to_csv(path.join(root_reduced_data_dir, comp + '_' + data_type + "_overview_of_segments_selected.csv"))

            # save reduced datasets for dropped clusters
            for n_dropped_cluster in n_dropped_clusters:
                folder = get_root_folder_for_reduced_cluster(root_reduced_data_dir, n_dropped_cluster)
                folder = get_data_dir(folder, comp)
                for run_name in run_names:
                    save_data_labels_to_file(folder,
                                             data_type,
                                             reduced_clusters_data[n_dropped_cluster][run_name],
                                             reduced_clusters_labels[n_dropped_cluster][run_name],
                                             run_name)

            # save reduced datasets for dropped segments
            for n_dropped_seg in n_dropped_segments:
                folder = get_root_folder_for_reduced_segments(root_reduced_data_dir, n_dropped_seg)
                folder = get_data_dir(folder, comp)
                for run_name in run_names:
                    save_data_labels_to_file(folder,
                                             data_type,
                                             reduced_segments_data[n_dropped_seg][run_name],
                                             reduced_segments_labels[n_dropped_seg][run_name],
                                             run_name)
