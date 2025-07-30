import numpy as np
from matplotlib import pyplot as plt

from src.visualisation.visualise_corr_3d_scatter_plot import draw_coordinate_planes, draw_cube_wireframe


def calculate_viewing_direction(elev, azim):
    """Calculate the viewing direction vector from elevation and azimuth angles"""
    # Convert degrees to radians
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Calculate viewing direction (where the camera is looking FROM)
    # In matplotlib's coordinate system:
    x = np.cos(elev_rad) * np.cos(azim_rad)
    y = np.cos(elev_rad) * np.sin(azim_rad)
    z = np.sin(elev_rad)

    return np.array([x, y, z])


def estimate_surface_normal(point, neighbors, k=5):
    """Estimate surface normal at a point using nearby neighbors"""
    if len(neighbors) < 3:
        return np.array([0, 0, 1])  # Default normal

    # Find k closest neighbors
    distances = np.linalg.norm(neighbors - point, axis=1)
    closest_indices = np.argsort(distances)[:min(k, len(neighbors))]
    closest_points = neighbors[closest_indices]

    if len(closest_points) < 3:
        return np.array([0, 0, 1])

    # Fit a plane through the closest points using SVD
    centered_points = closest_points - np.mean(closest_points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]  # Last row of V is the normal vector
        return normal / np.linalg.norm(normal)
    except:
        return np.array([0, 0, 1])


def add_front_back_polytope_boundary(ax, resolution=60, alpha_front=0.9, alpha_back=0.2,
                                     elev=35, azim=40, depth_threshold_percentile=60):
    """Add polytope boundary with front surface red, back surface grey"""

    # Create a grid for the boundary surface
    rho_range = np.linspace(-1, 1, resolution)
    boundary_points = []

    print(f"Calculating boundary points with resolution {resolution}...")

    # Find all boundary points (same as your working code)
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
        return False

    boundary_points = np.array(boundary_points)
    print(f"Found {len(boundary_points)} boundary points")

    # Calculate viewing direction
    view_direction = calculate_viewing_direction(elev, azim)

    # Method 1: Simple depth-based separation
    # Project all points onto the viewing direction to get their "depth"
    depths = np.dot(boundary_points, view_direction)

    # Separate front and back based on depth percentile
    depth_threshold = np.percentile(depths, depth_threshold_percentile)

    front_mask = depths >= depth_threshold
    back_mask = depths < depth_threshold

    front_points = boundary_points[front_mask]
    back_points = boundary_points[back_mask]

    print(f"Front points: {len(front_points)}, Back points: {len(back_points)}")

    # Plot back surface first (so it's behind front surface)
    if len(back_points) > 0:
        ax.scatter(back_points[:, 0], back_points[:, 1], back_points[:, 2],
                   c='lightgrey', s=4, alpha=alpha_back, label='Back Surface')

    # Plot front surface on top
    if len(front_points) > 0:
        ax.scatter(front_points[:, 0], front_points[:, 1], front_points[:, 2],
                   c='red', s=12, alpha=alpha_front, label='Front Surface', edgecolors='darkred', linewidth=0.1)

    return True


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
    ax.plot([-1, 1], [0, 0], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], 'gray', linewidth=2, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], 'gray', linewidth=2, alpha=0.5)

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
        ax.text(x * 1.09, y * 1.09, z * 1.09, label, fontsize=fontsize, ha='center', va='center', color='grey')


def plot_visible_polytope_surface(draw_coordinate_planes=None, draw_cube_wireframe=None,
                                  elev=35, azim=40, resolution=60):
    """
    Plot polytope with only visible front surface highlighted
    """
    fontsize = 15
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Basic setup
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Draw coordinate system - use provided functions or fallbacks
    if draw_coordinate_planes:
        draw_coordinate_planes(ax, fontsize)
    else:
        draw_coordinate_planes_fallback(ax, fontsize)

    if draw_cube_wireframe:
        draw_cube_wireframe(ax, fontsize)
    else:
        draw_cube_wireframe_fallback(ax, fontsize)

    # Add visible polytope surface
    success = add_front_back_polytope_boundary(ax, resolution=resolution, elev=elev, azim=azim)

    if not success:
        print("Failed to create polytope surface")

    # Add axis labels at the positive ends
    ax.text(1.15, -0.05, -0.05, r'$v_1, v_2$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, 1.15, 0.05, r'$v_1, v_3$', fontsize=fontsize + 4, color='black')
    ax.text(0.1, -0.05, 1.15, r'$v_2, v_3$', fontsize=fontsize + 4, color='black')

    # Set view and limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=elev, azim=azim)

    # Clean styling (matching your original)
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

    plt.tight_layout()
    plt.show()
    return fig


# Example usage:
if __name__ == "__main__":
    test_data = [(0, [0.5, 0.3, 0.2]), (1, [0.8, 0.7, 0.9]), (2, [-0.5, 0.4, -0.3])]
    test_patterns = {0: [0.7, 0.7, 0.71], 1: [1, 1, 1], 2: [-0.7, 0.7, -0.7]}

    fig = plot_visible_polytope_surface(
        draw_coordinate_planes=draw_coordinate_planes,
        draw_cube_wireframe=draw_cube_wireframe,
        elev=35, azim=40,
        resolution=80
    )
