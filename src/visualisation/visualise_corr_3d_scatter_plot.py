import os
from os import path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import SYNTHETIC_DATA_DIR, IRREGULAR_P90_DATA_DIR, base_dataset_result_folder_for_type, \
    ROOT_RESULTS_DIR, ResultsType, get_image_name_based_on_data_dir_and_data_type, AVERAGE_RANK_DISTRIBUTION
from src.utils.load_synthetic_data import SyntheticDataType, load_labels
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends
from tests.evaluation.test_impact_of_reduction_on_internal_indices import root_results_dir


def draw_coordinate_planes(ax, fontsize):
    # XY plane at z=0
    xy_outline = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]])
    ax.plot(xy_outline[:, 0], xy_outline[:, 1], xy_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # XZ plane at y=0
    xz_outline = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 0, -1]])
    ax.plot(xz_outline[:, 0], xz_outline[:, 1], xz_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # YZ plane at x=0
    yz_outline = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1], [0, -1, -1]])
    ax.plot(yz_outline[:, 0], yz_outline[:, 1], yz_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # Coordinate axes - same strength as frame
    ax.plot([-1, 1], [0, 0], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], 'gray', linewidth=2, alpha=0.5)

    # Add axis labels at the positive ends
    ax.text(1.15, -0.05, -0.05, r'$v_1, v_2$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, 1.15, 0.05, r'$v_1, v_3$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, -0.05, 1.15, r'$v_2, v_3$', fontsize=fontsize + 4, color='black')

    # XZ plane at y=0 with fill
    xz_x, xz_z = np.meshgrid([-1, 1], [-1, 1])
    xz_y = np.zeros_like(xz_x)
    ax.plot_surface(xz_x, xz_y, xz_z, alpha=0.3, color='lightgray')

    # YZ plane at x=0 with fill
    yz_y, yz_z = np.meshgrid([-1, 1], [-1, 1])
    yz_x = np.zeros_like(yz_y)
    ax.plot_surface(yz_x, yz_y, yz_z, alpha=0.1, color='lightgray')


def draw_cube_wireframe(ax, fontsize):
    # Define cube vertices
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # top face
    ]

    # Define edges (which vertices to connect)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
    ]

    for edge in edges:
        # square box
        start, end = vertices[edge[0]], vertices[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'gray', linewidth=1, alpha=0.5)

    # Add coordinate labels at cube corners
    corner_labels = [
        (-1, -1, -1, '(-1,-1,-1)'),
        (1, -1, -1, '(1,-1,-1)'),
        (1, 1, -1, '(1,1,-1)'),
        (-1, 1, -1, '(-1,1,-1)'),
        (-1, -1, 1, '(-1,-1,1)'),
        (1, -1, 1, '(1,-1,1)'),
        (1, 1, 1, '(1,1,1)'),
        (-1, 1, 1, '(-1,1,1)')
    ]

    for x, y, z, label in corner_labels:
        ax.text(x * 1.09, y * 1.09, z * 1.09, label, fontsize=fontsize, ha='center', va='center', color='grey')


def plot_data(data, ref_patterns, backend=Backends.none.value):
    """
    Plot correlation patterns in 3d qube grid data is dictionary of all the correlations wth pattern ids as keys
    and correlation coefficient vectors as value
    :param data: list of tuples with value 0 pattern id and value 1 correlation vector
    :param ref_patterns: dictionary with key pattern id and value relaxed pattern vector
    :return:
    """
    reset_matplotlib(backend)
    fontsize = 15
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Remove all default grids and ticks and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Draw coordinate plane outlines and fill some planes for visual effect
    draw_coordinate_planes(ax, fontsize)

    # Draw cube wireframe
    draw_cube_wireframe(ax, fontsize)

    # Create distinct colors
    distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#008080',
                       '#a65628', '#f781bf', '#999999', '#1b9e77', '#d95f02', '#7570b3',
                       '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666', '#8dd3c7',
                       '#800000', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69',
                       '#fccde5', '#bc80bd']

    distinct_cmap = ListedColormap(distinct_colors)

    # Plot data
    corr_vectors = [segment[1] for segment in data]
    corr_colour_id = [segment[0] for segment in data]
    corr_vectors = np.array(corr_vectors)
    x, y, z = corr_vectors[:, 0], corr_vectors[:, 1], corr_vectors[:, 2]

    ax.scatter(x, y, z, c=corr_colour_id, cmap=distinct_cmap, s=20, edgecolors='none', marker='o', alpha=1, vmin=0, vmax=25)

    # Plot relaxed canonical patterns
    extreme_points = np.array(list(ref_patterns.values()))
    pattern_colour_id = list(ref_patterns.keys())
    ax.scatter(extreme_points[:, 0], extreme_points[:, 1], extreme_points[:, 2],
               c=pattern_colour_id, cmap=distinct_cmap, facecolor='none', s=60, alpha=0.3, marker='o', vmin=0, vmax=25)

    # Add text annotations for each extreme point
    for id, corr in ref_patterns.items():
        pattern_colour = distinct_cmap(id / 25)  # Normalize to 0-1 range
        x, y, z = corr[0], corr[1], corr[2]

        # Paint mini coordinate system for points in panes
        non_zero_count = sum(1 for coord in corr if abs(coord) > 0)
        if non_zero_count == 2:
            # Draw mini-axes for confusing patterns
            arr_length = 0.15
            arr_width = 2
            arr_alpha = 1
            arr_head = 0.3

            if abs(x) > 0.1:  # X-direction arrow
                arrow_dir = 1 if x > 0 else -1
                ax.quiver(x, y, z, arrow_dir * arr_length, 0, 0,
                          color=pattern_colour, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

            if abs(y) > 0.1:  # Y-direction arrow
                arrow_dir = 1 if y > 0 else -1
                ax.quiver(x, y, z, 0, arrow_dir * arr_length, 0,
                          color=pattern_colour, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

            if abs(z) > 0.1:  # Z-direction arrow
                arrow_dir = 1 if z > 0 else -1
                ax.quiver(x, y, z, 0, 0, arrow_dir * arr_length,
                          color=pattern_colour, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

            # Add dot at origin to hide line joins
            ax.scatter([x], [y], [z], c=[pattern_colour], s=10, marker='o', alpha=arr_alpha,
                       edgecolors='none')  # No edge to keep it clean

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Clean background - remove default panes completely
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)

    # Remove the black frame lines completely
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0

    # ensure 1,1,1 corner of cube is front middle
    ax.view_init(elev=35, azim=40)

    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def save_figure(fig, run_name, data_dir, data_type):
    folder = base_dataset_result_folder_for_type(root_results_dir, ResultsType.dataset_description)
    folder = path.join(folder, "images")
    os.makedirs(folder, exist_ok=True)
    image_name = get_image_name_based_on_data_dir_and_data_type('_'.join([run_name, '3d_pattern_plot.png']), data_dir,
                                                                data_type)
    fig.savefig(path.join(folder, image_name), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    backend = Backends.visible_tests.value
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()
    root_results_dir = ROOT_RESULTS_DIR

    run_name = "trim-fire-24"

    # plot normal complete
    data_type = SyntheticDataType.normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    fig = plot_data(data=data, ref_patterns=relaxed_patterns, backend=backend)
    save_figure(fig, run_name, data_dir, data_type)

    # plot nn complete
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    fig = plot_data(data=data, ref_patterns=relaxed_patterns, backend=backend)
    save_figure(fig, run_name, data_dir, data_type)

    # plot complete downsampled
    data_type = SyntheticDataType.rs_1min
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    fig = plot_data(data=data, ref_patterns=relaxed_patterns, backend=backend)
    save_figure(fig, run_name, data_dir, data_type)

    # plot normal sparse
    data_type = SyntheticDataType.normal_correlated
    data_dir = IRREGULAR_P90_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    fig = plot_data(data=data, ref_patterns=relaxed_patterns, backend=backend)
    save_figure(fig, run_name, data_dir, data_type)


