from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.utils.configurations import ROOT_RESULTS_DIR, SYNTHETIC_DATA_DIR, IRREGULAR_P30_DATA_DIR, \
    IRREGULAR_P90_DATA_DIR, GENERATED_DATASETS_FILE_PATH, dataset_description_dir, DENDROGRAM_IMAGE
from src.utils.distance_measures import DistanceMeasures
from src.utils.labels_utils import calculate_distance_matrix_for
from src.utils.load_synthetic_data import SyntheticDataType, load_labels
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib


def plot_dendrogram_for_datasets(root_result_dir, dataset_types, data_dirs, run_names, distance_measure, linkage_method,
                                 save_fig, backend, threshold=1.5):
    for data_dir in data_dirs:
        for data_type in dataset_types:
            for run_name in run_names:
                # load gt labels
                labels_df = load_labels(run_name, data_type, data_dir)
                # calculate distance matrix
                distance_matrix = calculate_distance_matrix_for(labels_df, distance_measure)
                # calculate linkage matrix
                linkage_matrix = linkage(distance_matrix, method=linkage_method)

                # plot diagram
                reset_matplotlib(backend)
                fig = plt.figure(figsize=(20, 8))

                # create dendrogram
                dend = dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10, color_threshold=threshold)

                # Get leaf order (the order of segment ids)
                leaf_order = dend['leaves']
                leaves_color = dend['leaves_color_list']

                # assigned cluster in dendrogram in order of segment
                clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
                # Put cluster id in leave order
                ordered_clusters = [clusters[seg_id] for seg_id in leaf_order]

                # Add pattern labels at center of each cluster
                segs_in_cluster = []
                positions = []
                current_cl_id = ordered_clusters[0]
                x_scale = plt.xlim()[1]/len(leaf_order)
                for idx, segment_id in enumerate(leaf_order):
                    if ordered_clusters[idx] == current_cl_id:  # same cluster as before
                        segs_in_cluster.append(segment_id)
                        positions.append(idx)
                    else:
                        # write previous cluster id and start new one
                        add_pattern_text_for(idx, labels_df, leaves_color, positions, segs_in_cluster, x_scale)

                        # start new lists for next cluster
                        current_cl_id = ordered_clusters[idx]
                        segs_in_cluster = [segment_id]
                        positions = [idx]
                # write pattern for last cluster
                add_pattern_text_for(idx, labels_df, leaves_color, positions, segs_in_cluster, x_scale)

                # color leaves label too
                ax = plt.gca()
                tick_labels = ax.get_xticklabels()

                for (label, color) in zip(tick_labels, leaves_color):
                    label.set_color(color)


                plt.grid(which='minor', visible=False)
                plt.grid(axis='x', visible=False)
                plt.tight_layout()
                plt.show()

                if save_fig:
                    folder = dataset_description_dir("images", data_type, root_result_dir, data_dir)
                    image_name = path.join(folder, '_'.join([run_name, distance_measure, linkage_method, DENDROGRAM_IMAGE]))
                    fig.savefig(image_name, dpi=300, bbox_inches='tight')


def add_pattern_text_for(idx, labels_df, leaves_color, positions, segs_in_cluster, x_scale):
    center_x = (np.mean(positions) + 0.5) * x_scale
    patterns = labels_df.iloc[segs_in_cluster][SyntheticDataSegmentCols.pattern_id]
    assert patterns.nunique() == 1, "Segments from different correlation patterns grouped"
    canonical_pattern = labels_df.iloc[segs_in_cluster[0]][SyntheticDataSegmentCols.correlation_to_model]
    ax = plt.gca()
    pattern_string = str(canonical_pattern).replace('[', '(').replace(']', ')')
    ax.text(center_x, -0.5, pattern_string, ha='center', va='top', fontsize=12, color=leaves_color[idx - 1])


if __name__ == "__main__":
    backend = Backends.none.value
    save_fig = True
    root_result_dir = ROOT_RESULTS_DIR
    dataset_types = [SyntheticDataType.raw,
                     SyntheticDataType.normal_correlated,
                     SyntheticDataType.non_normal_correlated,
                     SyntheticDataType.rs_1min]
    data_dirs = [SYNTHETIC_DATA_DIR,
                 IRREGULAR_P30_DATA_DIR,
                 IRREGULAR_P90_DATA_DIR]

    distance_measure = DistanceMeasures.linf_cor_dist # keeps 0 alone
    linkage_method = 'average' # average less noise sensitive

    run_names = pd.read_csv(GENERATED_DATASETS_FILE_PATH)['Name'].tolist()
    plot_dendrogram_for_datasets(root_result_dir, dataset_types, data_dirs, run_names, distance_measure, linkage_method,
                                 save_fig, backend)
