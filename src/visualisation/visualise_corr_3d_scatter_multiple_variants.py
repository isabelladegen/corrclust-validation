import os
from os import path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.configurations import SYNTHETIC_DATA_DIR, IRREGULAR_P90_DATA_DIR, base_dataset_result_folder_for_type, \
    ROOT_RESULTS_DIR, ResultsType, get_image_name_based_on_data_dir_and_data_type, AVERAGE_RANK_DISTRIBUTION
from src.utils.load_synthetic_data import SyntheticDataType, load_labels
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize
from src.visualisation.visualise_corr_3d_scatter_plot import pattern_colours
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
    ax.text(1.15, -0.05, -0.05, r'$p_{12}$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, 1.15, 0.05, r'$p_{13}$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, -0.05, 1.15, r'$p_{23}$', fontsize=fontsize + 4, color='black')

    # XZ plane at y=0 with fill
    xz_x, xz_z = np.meshgrid([-1, 1], [-1, 1])
    xz_y = np.zeros_like(xz_x)
    ax.plot_surface(xz_x, xz_y, xz_z, alpha=0.3, color='#E0EFFF')

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
        # Highlight (1,1,1) corner in teal as reference point
        color = '#008080' if (x, y, z) == (1, 1, 1) else 'grey'
        weight = 'bold' if (x, y, z) == (1, 1, 1) else 'normal'
        ax.text(x * 1.09, y * 1.09, z * 1.05, label, fontsize=fontsize, ha='center', va='center', color=color,
                weight=weight, zorder=float('inf'))

def plot_data_2d(ax, data, ref_patterns, patterns_colours, secondary_data=None):
    """
    Plot correlation patterns in 2D showing only p_{12} and p_{23} plane
    Data filtered to show only patterns where p_{13} < 0.01
    """
    fontsize = 18

    # Remove all default grids and ticks and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Draw coordinate axes - same style as 3D version
    ax.axhline(y=0, color='gray', linewidth=2, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=2, alpha=0.5)

    # Add axis labels at the positive ends - same style as 3D
    ax.text(1.1, -0.05, r'$p_{12}$', fontsize=fontsize + 4, color='black')
    ax.text(0.05, 1.1, r'$p_{23}$', fontsize=fontsize + 4, color='black')

    # Set limits to match 3D version
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    # Force equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add subtle square outline to match cube style
    square_outline = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(square_outline[:, 0], square_outline[:, 1], 'gray', linewidth=1, alpha=0.5)
    ax.fill(square_outline[:, 0], square_outline[:, 1], color='#E0EFFF', alpha=0.3)

    distinct_cmap = ListedColormap(patterns_colours)

    # Plot filtered reference patterns
    filtered_ref_patterns = {k: v for k, v in ref_patterns.items() if abs(v[1]) < 0.01}
    if filtered_ref_patterns:
        extreme_points = np.array(list(filtered_ref_patterns.values()))
        pattern_colour_id = list(filtered_ref_patterns.keys())
        ax.scatter(extreme_points[:, 0], extreme_points[:, 2],
                   c=pattern_colour_id, cmap=distinct_cmap, facecolor='none', s=100, alpha=0.4, marker='o', vmin=0,
                   vmax=25)

    # Filter data by reference pattern
    filtered_data = [(pattern_id, corr_vec) for pattern_id, corr_vec in data if
                     pattern_id in filtered_ref_patterns.keys()]

    if filtered_data:
        corr_vectors = [segment[1] for segment in filtered_data]
        corr_colour_id = [segment[0] for segment in filtered_data]
        corr_vectors = np.array(corr_vectors)
        x, y = corr_vectors[:, 0], corr_vectors[:, 2]  # Only p_{12} and p_{23}

        ax.scatter(x, y, c=corr_colour_id, cmap=distinct_cmap, s=60, edgecolors='none', marker='x', alpha=1, vmin=0,
                   vmax=25)

    # plot secondary data
    if secondary_data is not None:
        sec_filtered = [(pattern_id, corr_vec) for pattern_id, corr_vec in secondary_data if
                         pattern_id in filtered_ref_patterns.keys()]

        if sec_filtered:
            corr_vectors = [segment[1] for segment in sec_filtered]
            corr_colour_id = [segment[0] for segment in sec_filtered]
            corr_vectors = np.array(corr_vectors)
            x, y = corr_vectors[:, 0], corr_vectors[:, 2]  # Only p_{12} and p_{23}

            ax.scatter(x, y, c=corr_colour_id, cmap=distinct_cmap, s=60, edgecolors='none', marker='^', alpha=1, vmin=0,
                       vmax=25)



def plot_data(ax, data, ref_patterns, patterns_colours, secondary_data=None):
    """
    Plot correlation patterns in 3d qube grid data is dictionary of all the correlations wth pattern ids as keys
    and correlation coefficient vectors as value
    :param ax: axis
    :param data: list of tuples with value 0 pattern id and value 1 correlation vector
    :param ref_patterns: dictionary with key pattern id and value relaxed pattern vector
    :param patterns_colours: dictionary with key pattern id and value colour
    :return:
    """
    fontsize = 18
    # Remove all default grids and ticks and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.set_proj_type('ortho')

    # Draw coordinate plane outlines and fill some planes for visual effect
    draw_coordinate_planes(ax, fontsize)

    # Draw cube wireframe
    draw_cube_wireframe(ax, fontsize)

    # Force equal aspect ratio to prevent distortion
    ax.set_box_aspect([1, 1, 1])

    # Force matplotlib to respect equal scaling
    ax.set_aspect('equal', adjustable='box')

    # Additional scaling enforcement
    ax.pbaspect = [1, 1, 1]

    distinct_cmap = ListedColormap(patterns_colours)

    # Plot data
    corr_vectors = [segment[1] for segment in data]
    corr_colour_id = [segment[0] for segment in data]
    corr_vectors = np.array(corr_vectors)
    x, y, z = corr_vectors[:, 0], corr_vectors[:, 1], corr_vectors[:, 2]

    ax.scatter(x, y, z, c=corr_colour_id, cmap=distinct_cmap, s=60, edgecolors='none', marker='x', alpha=1)

    # Plot secondary data
    if secondary_data is not None:
        corr_vectors = [segment[1] for segment in secondary_data]
        corr_colour_id = [segment[0] for segment in secondary_data]
        corr_vectors = np.array(corr_vectors)
        x, y, z = corr_vectors[:, 0], corr_vectors[:, 1], corr_vectors[:, 2]

        ax.scatter(x, y, z, c=corr_colour_id, cmap=distinct_cmap, s=60, edgecolors='none', marker='^', alpha=1)


    # Plot relaxed canonical patterns
    extreme_points = np.array(list(ref_patterns.values()))
    pattern_colour_id = list(ref_patterns.keys())
    ax.scatter(extreme_points[:, 0], extreme_points[:, 1], extreme_points[:, 2],
               c=pattern_colour_id, cmap=distinct_cmap, facecolor='none', s=60, alpha=0.4, marker='o', vmin=0, vmax=25)

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


def plot_subplots_for(run_name, relaxed_patterns, backend):
    reset_matplotlib(backend)
    fig = plt.figure(figsize=(20, 21))  # Increased height for 3 rows
    rows = 3  # Changed from 2 to 3
    columns = 3

    # Create gridspec with different row heights
    # Top row (2D plots) gets height ratio of 0.6, other rows get 1.0
    gs = GridSpec(rows, columns, height_ratios=[0.5, 1.0, 1.0])

    # plot normal complete 2D
    ax1 = fig.add_subplot(gs[0, 0])
    data_type = SyntheticDataType.normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data_2d(ax=ax1, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)
    ax1.set_title('Normal', weight='bold', fontsize=fontsize, pad=40)

    # plot nn complete 2D
    ax2 = fig.add_subplot(gs[0, 1])
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data_2d(ax=ax2, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)
    ax2.set_title('Non-Normal', weight='bold', fontsize=fontsize, pad=40)

    # plot complete downsampled 2D
    ax3 = fig.add_subplot(gs[0, 2])
    data_type = SyntheticDataType.rs_1min
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data_2d(ax=ax3, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)
    ax3.set_title('Downsampled', weight='bold', fontsize=fontsize, pad=40)

    # MIDDLE ROW - Original 3D plots (complete data)
    # plot normal complete 3D
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    data_type = SyntheticDataType.normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax4, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # plot nn complete 3D
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax5, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # plot complete downsampled 3D
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')
    data_type = SyntheticDataType.rs_1min
    data_dir = SYNTHETIC_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax6, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # BOTTOM ROW - Original 3D plots (sparse data)
    # plot normal sparse 3D
    ax7 = fig.add_subplot(gs[2, 0], projection='3d')
    data_type = SyntheticDataType.normal_correlated
    data_dir = IRREGULAR_P90_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax7, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # plot non-normal sparse 3D
    ax8 = fig.add_subplot(gs[2, 1], projection='3d')
    data_type = SyntheticDataType.non_normal_correlated
    data_dir = IRREGULAR_P90_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax8, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # plot downsampled sparse 3D
    ax9 = fig.add_subplot(gs[2, 2], projection='3d')
    data_type = SyntheticDataType.rs_1min
    data_dir = IRREGULAR_P90_DATA_DIR
    labels_df = load_labels(run_name, data_type, data_dir)
    data = labels_df[[SyntheticDataSegmentCols.pattern_id, SyntheticDataSegmentCols.actual_correlation]].values.tolist()
    plot_data(ax=ax9, data=data, ref_patterns=relaxed_patterns, patterns_colours=pattern_colours)

    # row labels
    fig.text(0.02, 0.85, 'Complete, p₁₃ = 0', rotation=90, fontsize=fontsize, fontweight='bold', ha='center',
             va='center')

    fig.text(0.02, 0.58, 'Complete', rotation=90, fontsize=fontsize, fontweight='bold', ha='center', va='center')

    fig.text(0.02, 0.21, 'Sparse', rotation=90, fontsize=fontsize, fontweight='bold', ha='center', va='center')

    plt.tight_layout(pad=2.0)
    plt.show()
    return fig


if __name__ == "__main__":
    backend = Backends.visible_tests.value
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()
    root_results_dir = ROOT_RESULTS_DIR

    run_name = "trim-fire-24"

    fig = plot_subplots_for(run_name, relaxed_patterns, backend)
    fig.savefig('csts_clustering_achieved_3D.png', dpi=300, bbox_inches='tight', pad_inches=0.2)

