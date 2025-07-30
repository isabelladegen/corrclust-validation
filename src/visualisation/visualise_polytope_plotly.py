import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends


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


def add_cube_wireframe(fig, row, col):
    """Add cube wireframe to a subplot"""
    # Define cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # top face
    ])

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
    ]

    # Draw edges
    for edge in edges:
        start, end = vertices[edge[0]], vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines',
            line=dict(color='lightgray', width=3),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

    # Add corner labels
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
        color = '#008080' if (x, y, z) == (1, 1, 1) else 'gray'
        fig.add_trace(go.Scatter3d(
            x=[x * 1.15], y=[y * 1.15], z=[z * 1.15],
            mode='text',
            text=[label],
            textfont=dict(color=color, size=10),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)


def add_coordinate_planes(fig, row, col):
    """Add coordinate planes to a subplot"""
    # XY plane at z=0
    xy_outline = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]])
    fig.add_trace(go.Scatter3d(
        x=xy_outline[:, 0], y=xy_outline[:, 1], z=xy_outline[:, 2],
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    # XZ plane at y=0
    xz_outline = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 0, -1]])
    fig.add_trace(go.Scatter3d(
        x=xz_outline[:, 0], y=xz_outline[:, 1], z=xz_outline[:, 2],
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    # YZ plane at x=0
    yz_outline = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1], [0, -1, -1]])
    fig.add_trace(go.Scatter3d(
        x=yz_outline[:, 0], y=yz_outline[:, 1], z=yz_outline[:, 2],
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    # Coordinate axes
    fig.add_trace(go.Scatter3d(
        x=[-1, 1], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-1, 1], z=[0, 0],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-1, 1],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    # Add filled coordinate planes
    # XZ plane at y=0
    xz_x, xz_z = np.meshgrid([-1, 1], [-1, 1])
    xz_y = np.zeros_like(xz_x)
    fig.add_trace(go.Surface(
        x=xz_x, y=xz_y, z=xz_z,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        opacity=0.3,
        showscale=False,
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    # YZ plane at x=0
    yz_y, yz_z = np.meshgrid([-1, 1], [-1, 1])
    yz_x = np.zeros_like(yz_y)
    fig.add_trace(go.Surface(
        x=yz_x, y=yz_y, z=yz_z,
        colorscale=[[0, 'lightgray'], [1, 'lightgray']],
        opacity=0.1,
        showscale=False,
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)


def add_axis_labels(fig, row, col):
    """Add axis labels to a subplot"""
    # Add axis labels at the positive ends
    fig.add_trace(go.Scatter3d(
        x=[1.25], y=[0], z=[0],
        mode='text',
        text=['v₁, v₂'],
        textfont=dict(color='black', size=12),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[1.25], z=[0],
        mode='text',
        text=['v₁, v₃'],
        textfont=dict(color='black', size=12),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.25],
        mode='text',
        text=['v₂, v₃'],
        textfont=dict(color='black', size=12),
        showlegend=False,
        hoverinfo='skip'
    ), row=row, col=col)


def create_camera_dict(elev, azim):
    """Convert elevation and azimuth to Plotly camera parameters"""
    # Convert to radians
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Calculate camera position (eye)
    r = 3  # distance from center
    x = r * np.cos(elev_rad) * np.cos(azim_rad)
    y = r * np.cos(elev_rad) * np.sin(azim_rad)
    z = r * np.sin(elev_rad)

    return dict(
        eye=dict(x=x, y=y, z=z),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )


def plot_polytope_6_views_plotly(resolution=60, alpha_front=0.6, alpha_back=0.2):
    """
    Plot correlation polytope from 6 different viewing angles using Plotly
    """
    # Define the 6 viewing angles
    base_elev = 35
    base_azim = 40
    viewing_angles = [
        (base_elev, base_azim),  # Original view
        (base_elev, base_azim - 90),  # -90 degrees around z
        (base_elev - 55, base_azim - 60),  # looking up from below
        (base_elev, base_azim - 180),  # -180 degrees around z
        (base_elev, base_azim - 270),  # -270 degrees around z
        (base_elev - 90, base_azim - 90)  # looking straight up
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

    # Create subplots
    reset_matplotlib(Backends.visible_tests.value)
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=None,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # Define subplot positions
    subplot_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    # Create each subplot
    for i, ((elev, azim), (row, col)) in enumerate(zip(viewing_angles, subplot_positions)):
        # Add coordinate planes and cube wireframe
        add_coordinate_planes(fig, row, col)
        add_cube_wireframe(fig, row, col)
        add_axis_labels(fig, row, col)

        # Plot back surface first (so it's behind front surface)
        if len(back_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=back_points[:, 0], y=back_points[:, 1], z=back_points[:, 2],
                mode='markers',
                marker=dict(
                    color='#B0C4DE',
                    size=3,
                    opacity=alpha_back
                ),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

        # Plot front surface on top
        if len(front_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=front_points[:, 0], y=front_points[:, 1], z=front_points[:, 2],
                mode='markers',
                marker=dict(
                    color='#008080',
                    size=4,
                    opacity=alpha_front
                ),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

    # Update layout for each subplot
    scene_updates = {}
    for i, (elev, azim) in enumerate(viewing_angles):
        scene_name = f'scene{i + 1}' if i > 0 else 'scene'
        scene_updates[scene_name] = dict(
            xaxis=dict(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=False, visible=False),
            camera=create_camera_dict(elev, azim),
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1)
        )

    # Update overall layout
    fig.update_layout(
        **scene_updates,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        width=1600,
        height=1000,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


# Example usage:
if __name__ == "__main__":
    fig = plot_polytope_6_views_plotly(resolution=80)
    if fig:
        fig.show()

        # To save as high-quality PNG for thesis:
        fig.write_image("polytope_6_views.png", width=1600, height=1000, scale=3)  # 300 DPI equivalent