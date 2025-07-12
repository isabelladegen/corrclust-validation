import numpy as np
from matplotlib import pyplot as plt

from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns


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
    ax.plot([-1, 1], [0, 0], [0, 0], 'gray', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], 'gray', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], 'gray', linewidth=1, alpha=0.5)

    # Add axis labels at the positive ends
    ax.text(1.15, 0.05, 0.05, 'X', fontsize=fontsize, color='black')
    ax.text(0.05, 1.15, 0.05, 'Y', fontsize=fontsize, color='black')
    ax.text(0.05, 0.05, 1.15, 'Z', fontsize=fontsize, color='black')


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
        ax.text(x * 1.09, y * 1.09, z * 1.09, label, fontsize=fontsize, ha='center', va='center', color='black')


def plot_data(data, ref_patterns):
    """
    Plot correlation patterns in 3d qube grid data is dictionary of all the correlations wth pattern ids as keys
    and correlation coefficient vectors as value
    :param data: list of tuples with value 0 pattern id and value 1 correlation vector
    :param ref_patterns: dictionary with key pattern id and value relaxed pattern vector
    :return:
    """
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

    # Plot data
    vectors = data
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    colors = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Distance from origin
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=60, alpha=0.8)

    # Plot relaxed canonical patterns
    extreme_points = np.array(list(ref_patterns.values()))
    ax.scatter(extreme_points[:, 0], extreme_points[:, 1], extreme_points[:, 2],
               c='darkgray', s=100, alpha=0.5, marker='o', label='Relaxed Canonical Patterns')

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
    ax.view_init(elev=30, azim=30)

    # Add colorbar with limited height
    cbar = plt.colorbar(scatter, ax=ax, label='Distance from Origin', shrink=0.6)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()

    # Your vectors
    vectors = [[1, 1, 0], [0.7, 0.7, 0.8], [-1, -1, -1], [0, 1, -1], [1, 0, 1], [-0.8, 0.2, 0.9]]
    vectors = np.array(vectors)

    plot_data(data=vectors, ref_patterns=relaxed_patterns)
