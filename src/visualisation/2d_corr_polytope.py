import numpy as np
import matplotlib.pyplot as plt

from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize
from src.visualisation.visualise_corr_3d_scatter_plot import pattern_colours


def plot_polytope_plane_intersections_with_patterns(relaxed_patterns, pattern_colours):
    """
    Plot all three 2D cross-sections where the correlation polytope intersects coordinate planes
    """

    # Create figure with 3 subplots in one row
    reset_matplotlib(Backends.visible_tests.value)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Unit square
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]

    # Cross-section 1: v2v3 = 0 (XY plane)
    paint_cross_section(ax1, pattern_colours, relaxed_patterns, x_label=r'$p_{12}$', y_label=r'$p_{13}$',
                        title=r'$p_{23} = 0$', x_index=0, y_index=1, zero_index=2)

    # Cross-section 2: v1v3 = 0 (XZ plane)
    paint_cross_section(ax2, pattern_colours, relaxed_patterns, x_label=r'$p_{12}$', y_label=r'$p_{23}$',
                        title=r'$p_{13} = 0$', x_index=0, y_index=2, zero_index=1)


    # Cross-section 3: v1v2 = 0 (YZ plane)
    paint_cross_section(ax3, pattern_colours, relaxed_patterns, x_label=r'$p_{13}$', y_label=r'$p_{23}$',
                        title=r'$p_{12} = 0$', x_index=1, y_index=2, zero_index=0)

    plt.tight_layout()
    plt.show()
    return fig


def paint_cross_section(ax, pattern_colours, relaxed_patterns, x_label, y_label, title, x_index, y_index, zero_index):
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Unit square
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]

    ax.fill(circle_x, circle_y, color='#008080', alpha=0.3)
    ax.plot(circle_x, circle_y, color='#008080', linewidth=3)
    ax.plot(square_x, square_y, 'gray', linewidth=2, alpha=0.5, linestyle='--')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)

    # Add patterns that fall on this cross-section (v2v3 = 0)
    for pattern_id, pattern in relaxed_patterns.items():
        p_x = pattern[x_index]
        p_y = pattern[y_index]
        p_zero = pattern[zero_index]
        if abs(p_zero) < 0.01:  # Pattern falls on this cross-section
            color = pattern_colours[pattern_id] if pattern_id < len(pattern_colours) else '#000000'
            ax.scatter(p_x, p_y, c=color, s=100, marker='o', alpha=1, zorder=100)

            ax.annotate(str(pattern_id), (p_x, p_y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=15, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgrey'))


# Example usage:
if __name__ == "__main__":
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()

    fig = plot_polytope_plane_intersections_with_patterns(relaxed_patterns, pattern_colours)
    fig.savefig('2d-polytop-cross-sections-patterns.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
