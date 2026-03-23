import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image
import io

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}\usepackage{amsmath}"

# ── Colour scale ──────────────────────────────────────────────
CORR_COLORS = [
    (166 / 255, 77 / 255, 121 / 255),
    (210 / 255, 166 / 255, 188 / 255),
    (1.0, 1.0, 1.0),
    (135 / 255, 192 / 255, 196 / 255),
    (14 / 255, 128 / 255, 136 / 255),
]
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_diverging",
    list(zip([0.0, 0.25, 0.5, 0.75, 1.0], CORR_COLORS)),
)
TEAL = CORR_COLORS[4]
BERRY = CORR_COLORS[0]
LIGHT_MAUVE = (0.84, 0.85, 0.88)
GREY_MAUVE = (0.55, 0.58, 0.68)
STEEL_BLUE = (0.27, 0.39, 0.60)
AMBER = (0.85, 0.65, 0.13)
TERRACOTTA = (0.76, 0.43, 0.28)
INDIGO = (0.29, 0.32, 0.55)
PETROL_BLUE = (0.10, 0.42, 0.55)
DUSTY_MAUVE = (0.55, 0.45, 0.50)
BLUE_LIGHT_GREY = (0.62, 0.67, 0.77)
MAUVE = (0.68, 0.47, 0.63)
DEEP_MAUVE = (0.55, 0.35, 0.53)
VIVID_EMERALD = (0.16, 0.65, 0.48)

ELLIPTOPE_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
])

# ── Global drawing constants ──────────────────────────────────
AXIS_LW = 5
CUBE_LW = 3.5
ARROW_SIZE = 0.045
ARROW_LENGTH = 0.12
FS = 30

HIDDEN_CORNERS = {(-1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, -1, -1), (1, 1, -1), (-1, -1, 1), (0, -1, -1), (0, 1, -1),
                  (0, -1, 1)}

ALL_CORNERS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]

AXIS_LABELS = {
    "x": (r"\textbf{$\boldsymbol{\rho_{12}}$}", (1.28, -0.05, -0.05)),
    "y": (r"\textbf{$\boldsymbol{\rho_{13}}$}", (0.08, 1.28, 0.05)),
    "z": (r"\textbf{$\boldsymbol{\rho_{23}}$}", (0.08, -0.05, 1.28)),
}

VIEW_3D = (25, -40)
VIEW_FRONT = (0, 0)


# ── Shared geometry helpers ───────────────────────────────────
def _plane_coords(c1, c2, normal_axis="x"):
    """Map 2D coordinates (c1, c2) into 3D, placing zeros on normal_axis."""
    z = np.zeros_like(c1)
    if normal_axis == "x":
        return z, c1, c2
    elif normal_axis == "y":
        return c1, z, c2
    elif normal_axis == "z":
        return c1, c2, z


def _vertex_dist_normalised(pts):
    """Min Euclidean distance to nearest elliptope vertex, normalised to [0,1].
    pts: array of shape (..., 3)."""
    orig_shape = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    d = np.full(len(flat), np.inf)
    for v in ELLIPTOPE_VERTICES:
        d = np.minimum(d, np.sqrt(np.sum((flat - v) ** 2, axis=1)))
    mx = np.max(d)
    result = d / mx if mx > 0 else d
    return result.reshape(orig_shape)


def _fade_color(base_color, dist, fade):
    """Blend base_color toward white by distance. Returns (N, 4) RGBA."""
    r = np.zeros((len(dist), 4))
    for ch in range(3):
        r[:, ch] = base_color[ch] + (1.0 - base_color[ch]) * dist * fade
    return r


# ── Label helpers ─────────────────────────────────────────────
def _rgb_tex(v):
    if v == 0:
        return "0,0,0"
    rgba = CORR_CMAP((v + 1) / 2)
    return f"{rgba[0]:.3f},{rgba[1]:.3f},{rgba[2]:.3f}"


def _corner_label(vals):
    parts = []
    for v in vals:
        rgb = _rgb_tex(v)
        parts.append(r"\textcolor[rgb]{" + rgb + r"}{\textbf{" + f"{v:g}" + r"}}")
    return r"\textsf{" + r"\textbf{,\,}".join(parts) + r"}"


# ── Drawing primitives ────────────────────────────────────────
def _colored_line_1d(ax, pts, varying_vals, lw=CUBE_LW):
    n = len(pts)
    segments = [[pts[i], pts[i + 1]] for i in range(n - 1)]
    midpoints = (varying_vals[:-1] + varying_vals[1:]) / 2
    colors = CORR_CMAP((midpoints + 1) / 2)
    lc = Line3DCollection(segments, colors=colors, linewidths=lw)
    ax.add_collection3d(lc)


def _draw_arrow(ax, tip, axis, colour=TEAL):
    d = ARROW_LENGTH
    s = ARROW_SIZE
    if axis == "x":
        verts = [[tip, 0, 0], [tip - d, s, 0], [tip - d, -s, 0]]
    elif axis == "y":
        verts = [[0, tip, 0], [0, tip - d, s], [0, tip - d, -s]]
    elif axis == "z":
        verts = [[0, 0, tip], [0, s, tip - d], [0, -s, tip - d]]
    ax.add_collection3d(Poly3DCollection([verts], color=colour, zorder=10000))


def _draw_axis(ax, axis, n=100, coloured=True):
    t_color = np.linspace(-1, 1, n)
    t_pos = np.linspace(-1, 1.06, n)
    zeros = np.zeros(n)
    if axis == "x":
        pts = np.column_stack([t_pos, zeros, zeros])
    elif axis == "y":
        pts = np.column_stack([zeros, t_pos, zeros])
    elif axis == "z":
        pts = np.column_stack([zeros, zeros, t_pos])
    if coloured:
        _colored_line_1d(ax, pts, t_color, lw=AXIS_LW)
        arrow_color = TEAL
    else:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="grey", lw=AXIS_LW, alpha=0.8)
        arrow_color = "grey"
    _draw_arrow(ax, 1.18, axis, arrow_color)


def draw_cube_wireframe(ax, n=80, coloured=True):
    t = np.linspace(-1, 1, n)
    for axis in ("x", "y", "z"):
        for fix1 in (-1, 1):
            for fix2 in (-1, 1):
                if axis == "x":
                    pts = np.column_stack([t, np.full(n, fix1), np.full(n, fix2)])
                elif axis == "y":
                    pts = np.column_stack([np.full(n, fix1), t, np.full(n, fix2)])
                elif axis == "z":
                    pts = np.column_stack([np.full(n, fix1), np.full(n, fix2), t])
                if coloured:
                    _colored_line_1d(ax, pts, t, lw=CUBE_LW)
                else:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="grey", lw=CUBE_LW, alpha=0.5)


def _draw_square_outline(ax, normal_axis="x", color="darkgray", lw=2, alpha=0.5):
    sq = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]])
    ax.plot(*_plane_coords(sq[:, 0], sq[:, 1], normal_axis),
            color=color, lw=lw, alpha=alpha, zorder=8)


def draw_grey_plane(ax, normal_axis="x", alpha=0.08, color="gray",
                    pane_filled=True):
    """Draw a reference plane. pane_filled=False draws outline only."""
    if pane_filled:
        g = np.array([[-1, 1], [-1, 1]])
        ax.plot_surface(*_plane_coords(g, g.T, normal_axis),
                        alpha=alpha, color=color, shade=False)
    _draw_square_outline(ax, normal_axis=normal_axis, color=color, lw=1.5, alpha=1.0)


def _draw_filled_circle(ax, normal_axis="x", alpha=0.4, color="darkgray",
                        outline_color=PETROL_BLUE, outline_lw=3):
    r_grid = np.linspace(0, 1, 50)
    t_grid = np.linspace(0, 2 * np.pi, 100)
    R, T = np.meshgrid(r_grid, t_grid)
    C1, C2 = R * np.cos(T), R * np.sin(T)
    ax.plot_surface(*_plane_coords(C1, C2, normal_axis),
                    alpha=alpha, color=color, shade=False, zorder=5)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(*_plane_coords(np.cos(theta), np.sin(theta), normal_axis),
            color=outline_color, lw=outline_lw, zorder=10)


# ── Layout ────────────────────────────────────────────────────
def draw_axes_and_labels(ax, axes=("x", "y", "z"), coloured=True):
    for axis in axes:
        _draw_axis(ax, axis, coloured=coloured)
    for axis in axes:
        text, pos = AXIS_LABELS[axis]
        ax.text(*pos, text, fontsize=FS, color="black", zorder=10000)


def draw_corner_labels(ax, corners=None):
    if corners is None:
        corners = ALL_CORNERS
    for c in corners:
        if c in HIDDEN_CORNERS:
            continue
        label = _corner_label(c)
        ax.text(c[0] * 1.15, c[1] * 1.15, c[2] * 1.10, label,
                fontsize=FS, ha="center", va="center", zorder=10000)


def style_ax(ax, elev=25, azim=-60):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_proj_type("ortho")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.set_box_aspect([1, 1, 1])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("white")
        a.pane.set_alpha(0)
        a.line.set_color("white")
        a._axinfo["tick"]["inward_factor"] = 0
        a._axinfo["tick"]["outward_factor"] = 0
    ax.view_init(elev=elev, azim=azim)


def make_fig(draw_fn, save_path=None):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")
    draw_fn(ax)
    plt.subplots_adjust(left=-0.15, right=1.15, top=1.15, bottom=-0.15)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    pad_inches=0, transparent=True)
    plt.show()
    return fig


# ── Elliptope surface (unused, kept for reference) ────────────
def _get_elliptope_sheets(n=120):
    """Upper and lower sheets of det(R)=0 boundary, parameterised by (ρ₁₂, ρ₁₃)."""
    rho12 = np.linspace(-1, 1, n)
    rho13 = np.linspace(-1, 1, n)
    R12, R13 = np.meshgrid(rho12, rho13)

    disc = (1 - R12 ** 2) * (1 - R13 ** 2)
    disc = np.clip(disc, 0, None)
    sqrt_disc = np.sqrt(disc)

    upper = R12 * R13 + sqrt_disc
    lower = R12 * R13 - sqrt_disc

    return R12, R13, upper, lower


def _make_surface_colors(R12, alpha_dark=0.6, alpha_light=0.15,
                         color_dark=TEAL, color_light=(0.7, 0.8, 0.8),
                         split_at=0.0):
    """RGBA facecolor array: dark where R12 < split_at, light elsewhere."""
    rows, cols = R12.shape
    fc = np.zeros((rows - 1, cols - 1, 4))
    mid_r12 = (R12[:-1, :-1] + R12[1:, 1:]) / 2

    dark_mask = mid_r12 < split_at

    fc[dark_mask, 0] = color_dark[0]
    fc[dark_mask, 1] = color_dark[1]
    fc[dark_mask, 2] = color_dark[2]
    fc[dark_mask, 3] = alpha_dark

    fc[~dark_mask, 0] = color_light[0]
    fc[~dark_mask, 1] = color_light[1]
    fc[~dark_mask, 2] = color_light[2]
    fc[~dark_mask, 3] = alpha_light

    return fc


def _make_surface_colors_two_tone(R12, R13, Z,
                                  color_a=GREY_MAUVE, color_b=(0.55, 0.45, 0.50),
                                  alpha_near=0.7, alpha_far=0.1,
                                  split_at=0.0):
    """Two colours split by ρ₁₂, both fading from vertices to white mid-surface."""
    rows, cols = R12.shape
    fc = np.zeros((rows - 1, cols - 1, 4))

    mid_x = (R12[:-1, :-1] + R12[1:, 1:]) / 2
    mid_y = (R13[:-1, :-1] + R13[1:, 1:]) / 2
    mid_z = (Z[:-1, :-1] + Z[1:, 1:]) / 2

    pts = np.stack([mid_x, mid_y, mid_z], axis=-1)
    t = _vertex_dist_normalised(pts)

    mask_a = mid_x < split_at

    for ch in range(3):
        fc[mask_a, ch] = color_a[ch] * (1 - t[mask_a]) + 1.0 * t[mask_a]
        fc[~mask_a, ch] = color_b[ch] * (1 - t[~mask_a]) + 1.0 * t[~mask_a]

    fc[..., 3] = alpha_near * (1 - t) + alpha_far * t

    return fc


def _make_surface_colors_depth(R12, R13, Z,
                               color=GREY_MAUVE,
                               alpha=0.7,
                               fade_strength=0.8,
                               split_color=None, split_at=0.0,
                               split_fade_strength=None):
    """Colour fades toward white at centre. Alpha stays constant (no holes)."""
    rows, cols = R12.shape
    fc = np.zeros((rows - 1, cols - 1, 4))

    mid_x = (R12[:-1, :-1] + R12[1:, 1:]) / 2
    mid_y = (R13[:-1, :-1] + R13[1:, 1:]) / 2
    mid_z = (Z[:-1, :-1] + Z[1:, 1:]) / 2
    pts = np.stack([mid_x, mid_y, mid_z], axis=-1)

    t = _vertex_dist_normalised(pts)

    if split_color is not None:
        s_fade = split_fade_strength if split_fade_strength is not None else fade_strength
        mask = mid_x < split_at
        for ch in range(3):
            fc[mask, ch] = split_color[ch] + (1.0 - split_color[ch]) * t[mask] * s_fade
            fc[~mask, ch] = color[ch] + (1.0 - color[ch]) * t[~mask] * fade_strength
    else:
        for ch in range(3):
            fc[..., ch] = color[ch] + (1.0 - color[ch]) * t * fade_strength

    fc[..., 3] = alpha

    return fc


def _draw_elliptope(ax, n=120, color=GREY_MAUVE,
                    alpha=0.7, fade_strength=0.8,
                    split_color=None, split_at=0.0, split_fade_strength=None,
                    edge_color="lightgray", edge_lw=0.3):
    R12, R13, upper, lower = _get_elliptope_sheets(n)

    for Z, zo in [(upper, 2), (lower, 1)]:
        fc = _make_surface_colors_depth(R12, R13, Z, color,
                                        alpha, fade_strength,
                                        split_color, split_at,
                                        split_fade_strength)
        surf = ax.plot_surface(R12, R13, Z, facecolors=fc, shade=False, zorder=zo)
        surf.set_edgecolors(edge_color)
        surf.set_linewidth(edge_lw)


# ── Elliptope wireframe ───────────────────────────────────────
def _draw_elliptope_curve(ax, pts, is_split_side,
                          color, fade_strength, alpha,
                          split_color, split_fade_strength, split_alpha, lw):
    """Draw a single elliptope curve with per-segment colour from vertex distance."""
    if len(pts) < 2:
        return
    dist = _vertex_dist_normalised(pts)
    mid_dist = (dist[:-1] + dist[1:]) / 2

    if is_split_side and split_color is not None:
        fc = _fade_color(split_color, mid_dist, split_fade_strength)
        fc[:, 3] = split_alpha
    else:
        fc = _fade_color(color, mid_dist, fade_strength)
        fc[:, 3] = alpha

    segments = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    lc = Line3DCollection(segments, colors=fc, linewidths=lw)
    ax.add_collection3d(lc)


def draw_elliptope_wireframe(ax, n_slices=40, n_pts=200,
                             color=GREY_MAUVE, split_color=None, split_at=0.0,
                             lw=1.0, alpha=0.6, split_alpha=0.8,
                             fade_strength=0.6, split_fade_strength=0.3):
    """Elliptope as a coloured wireframe mesh. No surfaces, no artefacts."""

    rho12_vals = np.linspace(-1, 1, n_slices)
    rho13_vals = np.linspace(-1, 1, n_slices)
    t = np.linspace(-1, 1, n_pts)

    curve_kw = dict(color=color, fade_strength=fade_strength, alpha=alpha,
                    split_color=split_color, split_fade_strength=split_fade_strength,
                    split_alpha=split_alpha, lw=lw)

    # Family 1: slices at fixed ρ₁₂ (horizontal ellipses)
    for r12 in rho12_vals:
        disc = (1 - r12 ** 2) * (1 - t ** 2)
        valid = disc >= 0
        if not np.any(valid):
            continue
        t_v = t[valid]
        sqrt_d = np.sqrt(disc[valid])
        base = r12 * t_v
        is_split = r12 < split_at

        pts_u = np.column_stack([np.full(len(t_v), r12), t_v, base + sqrt_d])
        _draw_elliptope_curve(ax, pts_u, is_split, **curve_kw)

        pts_l = np.column_stack([np.full(len(t_v), r12), t_v, base - sqrt_d])
        _draw_elliptope_curve(ax, pts_l, is_split, **curve_kw)

    # Family 2: slices at fixed ρ₁₃ (vertical curves)
    for r13 in rho13_vals:
        disc = (1 - t ** 2) * (1 - r13 ** 2)
        valid = disc >= 0
        if not np.any(valid):
            continue
        t_v = t[valid]
        sqrt_d = np.sqrt(disc[valid])
        base = t_v * r13
        split_mask = t_v < split_at

        pts_u = np.column_stack([t_v, np.full(len(t_v), r13), base + sqrt_d])
        for is_split, mask in [(True, split_mask), (False, ~split_mask)]:
            p = pts_u[mask]
            if len(p) >= 2:
                _draw_elliptope_curve(ax, p, is_split, **curve_kw)

        pts_l = np.column_stack([t_v, np.full(len(t_v), r13), base - sqrt_d])
        for is_split, mask in [(True, split_mask), (False, ~split_mask)]:
            p = pts_l[mask]
            if len(p) >= 2:
                _draw_elliptope_curve(ax, p, is_split, **curve_kw)


# ── Figures ───────────────────────────────────────────────────
def figure_1_cube(save_path=None):
    def draw(ax):
        style_ax(ax, *VIEW_3D)
        draw_grey_plane(ax, "x", alpha=0.4, color="darkgray")
        draw_grey_plane(ax, "z", alpha=0.4, color="lightgray")
        draw_cube_wireframe(ax)
        draw_axes_and_labels(ax)
        draw_corner_labels(ax)

    return make_fig(draw, save_path)


def figure_2_circle(save_path=None, circle_color=PETROL_BLUE):
    def draw(ax):
        style_ax(ax, *VIEW_3D)
        draw_grey_plane(ax, "x", alpha=0.4, color="darkgray")
        draw_grey_plane(ax, "z", alpha=0.4, color="lightgray")
        draw_axes_and_labels(ax)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.zeros(200), np.cos(theta), np.sin(theta),
                color=circle_color, lw=3, zorder=10)

    return make_fig(draw, save_path)


def figure_3_circle_front(save_path=None, circle_color=PETROL_BLUE):
    def draw(ax):
        style_ax(ax, *VIEW_FRONT)
        _draw_filled_circle(ax, normal_axis="x", outline_color=circle_color)
        _draw_square_outline(ax, normal_axis="x")
        draw_axes_and_labels(ax, axes=("y", "z"))
        face_corners = [(0, y, z) for y in (-1, 1) for z in (-1, 1)]
        draw_corner_labels(ax, corners=face_corners)

    return make_fig(draw, save_path)


# Constants
ELLIPTOPE_LW = 1.0
ELLIPTOPE_ALPHA = 0.6
ELLIPTOPE_SPLIT_ALPHA = 0.8
N_SLICES = 70
SPLIT_COLOUR = PETROL_BLUE
LIGHT_SIDE_COLOUR = BLUE_LIGHT_GREY


def figure_4_elliptope(save_path=None, elev=25, azim=-50):
    def draw(ax):
        style_ax(ax, elev, azim)
        draw_elliptope_wireframe(ax, n_slices=N_SLICES,
                                 color=LIGHT_SIDE_COLOUR,
                                 split_color=SPLIT_COLOUR,
                                 split_at=0.0,
                                 lw=ELLIPTOPE_LW,
                                 alpha=ELLIPTOPE_ALPHA,
                                 split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                                 fade_strength=0.6,
                                 split_fade_strength=0.5)
        draw_grey_plane(ax, "x", alpha=0.15, color="darkgray")
        draw_grey_plane(ax, "z", alpha=0.08, color="lightgray")
        draw_cube_wireframe(ax, coloured=False)
        draw_axes_and_labels(ax, coloured=False)
        draw_corner_labels(ax)

    return make_fig(draw, save_path)


def figure_4_elliptope_gif(save_path="elliptope_rotation.gif",
                           n_frames=72, elev=25):
    """Rotating GIF: full 360° around z-axis."""
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")

    style_ax(ax, elev, -50)
    draw_elliptope_wireframe(ax, n_slices=N_SLICES,
                             color=LIGHT_SIDE_COLOUR,
                             split_color=SPLIT_COLOUR,
                             split_at=0.0,
                             lw=ELLIPTOPE_LW,
                             alpha=ELLIPTOPE_ALPHA,
                             split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                             fade_strength=0.6,
                             split_fade_strength=0.5)
    draw_grey_plane(ax, "x", alpha=0.15, color="darkgray")
    draw_grey_plane(ax, "z", alpha=0.08, color="lightgray")
    draw_cube_wireframe(ax, coloured=False)
    draw_axes_and_labels(ax, coloured=False)
    draw_corner_labels(ax)
    plt.tight_layout(pad=1.0)

    frames = []
    for i in range(n_frames):
        azim = -50 + (360 * i / n_frames)
        ax.view_init(elev=elev, azim=azim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    pad_inches=0.3, transparent=False, facecolor="white")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=150, loop=0, optimize=True)
    print(f"Saved {save_path} ({len(frames)} frames)")


if __name__ == "__main__":
    figure_1_cube("img/fig1_cube.png")
    figure_2_circle("img/fig2_circle.png")
    figure_3_circle_front("img/fig3_circle_front.png")
    figure_4_elliptope("img/fig4_elliptope.png")
    figure_4_elliptope_gif("img/elliptope_rotation.gif")
