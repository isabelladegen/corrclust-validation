import numpy as np
from matplotlib import pyplot as plt

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.visualisation.visualise_corr_3d_scatter_plot import pattern_colours


def calculate_viewing_direction(elev, azim):
    """Calculate the viewing direction vector from elevation and azimuth angles"""
    # Convert degrees to radians
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Calculate viewing direction (where the camera is looking FROM)
    x = np.cos(elev_rad) * np.cos(azim_rad)
    y = np.cos(elev_rad) * np.sin(azim_rad)
    z = np.sin(elev_rad)

    return np.array([x, y, z])


def get_polytope_boundary_points(resolution=60):
    """Get all boundary points of the correlation polytope"""
    rho_range = np.linspace(-1, 1, resolution)
    boundary_points = []

    print(f"Calculating boundary points with resolution {resolution}...")

    for rho12 in rho_range:
        for rho13 in rho_range:
            a = -1
            b = 2 * rho12 * rho13
            c = 1 - rho12 ** 2 - rho13 ** 2

            discriminant = b ** 2 - 4 * a * c

            if discriminant >= 0:
                rho23_1 = (-b + np.sqrt(discriminant)) / (2 * a)
                rho23_2 = (-b - np.sqrt(discriminant)) / (2 * a)

                for rho23 in [rho23_1, rho23_2]:
                    if -1 <= rho23 <= 1:
                        boundary_points.append([rho12, rho13, rho23])

    if not boundary_points:
        print("No boundary points found!")
        return None

    boundary_points = np.array(boundary_points)
    print(f"Found {len(boundary_points)} boundary points")
    return boundary_points


def classify_front_back_points(boundary_points, reference_elev=35, reference_azim=40,
                               depth_threshold_percentile=60):
    """Classify points as front or back using a fixed reference viewing direction"""
    # Calculate reference viewing direction
    reference_view_direction = calculate_viewing_direction(reference_elev, reference_azim)

    # Project all points onto the reference viewing direction to get their "depth"
    depths = np.dot(boundary_points, reference_view_direction)

    # Separate front and back based on depth percentile
    depth_threshold = np.percentile(depths, depth_threshold_percentile)

    front_mask = depths >= depth_threshold
    back_mask = depths < depth_threshold

    return front_mask, back_mask


def draw_coordinate_planes_fallback(ax, fontsize):
    """Fallback implementation of coordinate planes"""
    # XY plane at z=0
    xy_outline = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]])
    ax.plot(xy_outline[:, 0], xy_outline[:, 1], xy_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # XZ plane at y=0
    xz_outline = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 0, -1]])
    ax.plot(xz_outline[:, 0], xz_outline[:, 1], xz_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # YZ plane at x=0
    yz_outline = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1], [0, -1, -1]])
    ax.plot(yz_outline[:, 0], yz_outline[:, 1], yz_outline[:, 2], 'grey', linewidth=0.8, alpha=0.3)

    # Coordinate axes
    ax.plot([-1, 1.1], [0, 0], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [-1, 1.1], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1.1], 'gray', linewidth=2, alpha=0.5)
    # Add arrows at +1 ends
    # X-axis arrow
    ax.quiver(1, 0, 0, 0.2, 0, 0, color='gray', alpha=0.7,
              arrow_length_ratio=0.3, linewidth=2)

    # Y-axis arrow
    ax.quiver(0, 1, 0, 0, 0.2, 0, color='gray', alpha=0.7,
              arrow_length_ratio=0.3, linewidth=2)

    # Z-axis arrow
    ax.quiver(0, 0, 1, 0, 0, 0.2, color='gray', alpha=0.7,
              arrow_length_ratio=0.3, linewidth=2)

    # XZ plane at y=0 with fill
    xz_x, xz_z = np.meshgrid([-1, 1], [-1, 1])
    xz_y = np.zeros_like(xz_x)
    ax.plot_surface(xz_x, xz_y, xz_z, alpha=0.3, color='lightgray')

    # YZ plane at x=0 with fill
    yz_y, yz_z = np.meshgrid([-1, 1], [-1, 1])
    yz_x = np.zeros_like(yz_y)
    ax.plot_surface(yz_x, yz_y, yz_z, alpha=0.1, color='lightgray')


def draw_cube_wireframe_fallback(ax, fontsize):
    """Fallback implementation of cube wireframe with corner labels"""
    # Define cube vertices
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # top face
    ]

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
    ]

    for edge in edges:
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


def setup_single_subplot(ax, fontsize):
    """Setup a single subplot with coordinate system and cube"""
    # Basic setup
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.set_proj_type('ortho')

    # Draw coordinate system
    draw_coordinate_planes_fallback(ax, fontsize)
    draw_cube_wireframe_fallback(ax, fontsize)

    # Add axis labels at the positive ends
    ax.text(1.15, -0.05, -0.05, r'$p_{12}$', fontsize=fontsize, color='black', zorder=float('inf'))
    ax.text(0.1, 1.15, 0.05, r'$p_{13}$', fontsize=fontsize, color='black', zorder=float('inf'))
    ax.text(0.1, -0.05, 1.15, r'$p_{23}$', fontsize=fontsize, color='black', zorder=float('inf'))

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Force equal aspect ratio to prevent distortion
    ax.set_box_aspect([1, 1, 1])

    # Force matplotlib to respect equal scaling
    ax.set_aspect('equal', adjustable='box')

    # Additional scaling enforcement
    ax.pbaspect = [1, 1, 1]

    # Clean styling
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


def plot_polytope_4_views(resolution=60, alpha_front=0.6, alpha_back=0.2):
    """
    Plot correlation polytope from 6 different viewing angles with consistent coloring and relaxed patterns
    """
    fontsize = 18

    # Setup patterns and colors in main function
    patterns = ModelCorrelationPatterns()
    relaxed_patterns = patterns.relaxed_patterns()

    # Create 3x2 subplot layout
    reset_matplotlib(Backends.visible_tests.value)
    fig = plt.figure(figsize=(20, 14))

    # Define the 6 viewing angles
    base_elev = 35
    base_azim = 40
    viewing_angles = [
        (base_elev, base_azim),  # Original view
        (base_elev, base_azim - 90),  # -90 degrees around z
        (base_elev - 50, base_azim - 60),  # -60 degrees around y (looking up from below)
        (base_elev, base_azim - 180),  # -180 degrees around z
        (base_elev, base_azim - 270),  # -270 degrees around z
        (base_elev - 20, base_azim - 70)  # -60 degrees around y, -90 around z
    ]

    # Get boundary points once
    boundary_points = get_polytope_boundary_points(resolution)
    if boundary_points is None:
        print("Failed to generate boundary points")
        return None

    # Classify points as front/back using reference view (first viewing angle)
    front_mask, back_mask = classify_front_back_points(
        boundary_points,
        reference_elev=viewing_angles[0][0],
        reference_azim=viewing_angles[0][1]
    )

    front_points = boundary_points[front_mask]
    back_points = boundary_points[back_mask]

    print(f"Front points: {len(front_points)}, Back points: {len(back_points)}")

    # Create each subplot
    for i, (elev, azim) in enumerate(viewing_angles):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')

        # Setup the subplot
        setup_single_subplot(ax, fontsize)

        # Plot back surface first (so it's behind front surface)
        if len(back_points) > 0:
            ax.scatter(back_points[:, 0], back_points[:, 1], back_points[:, 2],
                       c='#B0C4DE', s=4, alpha=alpha_back)

        # Plot front surface on top
        if len(front_points) > 0:
            ax.scatter(front_points[:, 0], front_points[:, 1], front_points[:, 2],
                       c="#008080", s=12, alpha=alpha_front, linewidth=0.1)

        # Add relaxed patterns (all patterns since it's 3D)
        for pattern_id, pattern in relaxed_patterns.items():
            p_x, p_y, p_z = pattern[0], pattern[1], pattern[2]
            color = pattern_colours[pattern_id] if pattern_id < len(pattern_colours) else '#000000'
            ax.scatter(p_x, p_y, p_z, c=color, s=100, marker='o', alpha=1, zorder=1000)

            # Paint mini coordinate system for points in planes
            non_zero_count = sum(1 for coord in pattern if abs(coord) > 0)
            if non_zero_count == 2:
                # Draw mini-axes for patterns in planes
                arr_length = 0.15
                arr_width = 2
                arr_alpha = 1
                arr_head = 0.3

                if abs(p_x) > 0.1:  # X-direction arrow
                    arrow_dir = 1 if p_x > 0 else -1
                    ax.quiver(p_x, p_y, p_z, arrow_dir * arr_length, 0, 0,
                              color=color, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

                if abs(p_y) > 0.1:  # Y-direction arrow
                    arrow_dir = 1 if p_y > 0 else -1
                    ax.quiver(p_x, p_y, p_z, 0, arrow_dir * arr_length, 0,
                              color=color, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

                if abs(p_z) > 0.1:  # Z-direction arrow
                    arrow_dir = 1 if p_z > 0 else -1
                    ax.quiver(p_x, p_y, p_z, 0, 0, arrow_dir * arr_length,
                              color=color, arrow_length_ratio=arr_head, linewidth=arr_width, alpha=arr_alpha)

                # Add dot at origin to hide line joins
                ax.scatter([p_x], [p_y], [p_z], c=[color], s=10, marker='o', alpha=arr_alpha,
                           edgecolors='none', zorder=1001)

        # Set the viewing angle
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=2.0)
    plt.show()
    return fig


# Example usage:
if __name__ == "__main__":
    fig = plot_polytope_4_views(
        resolution=80
    )
    fig.savefig('polytope-3D-6angles-with-patterns-arrows.png', dpi=300, bbox_inches='tight', pad_inches=0.2)