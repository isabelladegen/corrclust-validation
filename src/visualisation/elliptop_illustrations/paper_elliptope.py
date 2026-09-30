# ── Paper figure config (edit here) ───────────────────────────
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

from src.visualisation.elliptop_illustrations.elliptope_with_pattern_animation import CANONICAL_POS, PATTERN_VALID, \
    RELAXED_POS, INVALID_CANONICAL_POS
from src.visualisation.elliptop_illustrations.trustworthy_ai_talk_figures import _draw_axis, style_ax, draw_grey_plane, \
    draw_cube_wireframe, draw_corner_labels, VIVID_EMERALD, TERRACOTTA, SPLIT_COLOUR, N_SLICES, ELLIPTOPE_LW, \
    LIGHT_SIDE_COLOUR, ELLIPTOPE_ALPHA, ELLIPTOPE_SPLIT_ALPHA, LineBuffer, ALL_CORNERS, HIDDEN_CORNERS, \
    draw_elliptope_rings

PAPER_DOT_COLOUR = "#1A2E61"
PAPER_HIGHLIGHT_COLOUR = "#66B8EB"  # adjusted / on-curve patterns
PAPER_CIRCLE_COLOUR = SPLIT_COLOUR  # darker half of the elliptope
PAPER_CIRCLE_LW = 2
PAPER_DOT_SIZE = 900
PAPER_CROSS_COLOUR = "darkgrey"
PAPER_CROSS_SIZE = 900
PANEL_SIZE = 14  # inches per panel, matches the single-panel figures
PANEL_OVERHANG = 0.15  # same zoom as make_fig's subplots_adjust(-0.15, 1.15)
LABEL_COLOUR = "black"
AXIS_LABEL_FS = 36
CORNER_LABEL_FS = 42
PANEL_V_FRAC = 0.92  # figure height as a fraction of panel_size; lower = tighter

V_AXIS_LABELS = {
    "x": (r"$\boldsymbol{v_{12}}$", (1.28, -0.05, -0.05)),
    "y": (r"$\boldsymbol{v_{13}}$", (0.08, 1.28, 0.05)),
    "z": (r"$\boldsymbol{v_{23}}$", (0.08, -0.05, 1.28)),
}

# ── Pattern subsets (aligned with RELAXED_POS) ────────────────
VALID_CANONICAL_POS = CANONICAL_POS[PATTERN_VALID]
IS_ADJUSTED = np.any(~np.isclose(VALID_CANONICAL_POS, RELAXED_POS), axis=1)
IS_ON_CIRCLE = (np.isclose(RELAXED_POS[:, 0], 0)
                & np.isclose(np.hypot(RELAXED_POS[:, 1], RELAXED_POS[:, 2]),
                             1, atol=0.02))
IS_HALF = RELAXED_POS[:, 0] <= 1e-9  # patterns with v12 <= 0
INVALID_HALF = INVALID_CANONICAL_POS[:, 0] < 0


# ── Drawing helpers ───────────────────────────────────────────
def _draw_corner_labels_v(ax, corners=None, fontsize=CORNER_LABEL_FS,
                          color=LABEL_COLOUR):
    for c in (corners or ALL_CORNERS):
        if c in HIDDEN_CORNERS:
            continue
        label = "$" + r",\,".join(f"{v:g}" for v in c) + "$"
        ax.text(c[0] * 1.15, c[1] * 1.15, c[2] * 1.10, label,
                fontsize=fontsize, color=color,
                ha="center", va="center", zorder=10000)


def _draw_axes_and_v_labels(ax, axes=("x", "y", "z"), buf=None):
    for axis in axes:
        _draw_axis(ax, axis, coloured=False, buf=buf)
    for axis in axes:
        text, pos = V_AXIS_LABELS[axis]
        ax.text(*pos, text, fontsize=AXIS_LABEL_FS, color=LABEL_COLOUR, zorder=10000)


def _draw_circle(ax, color=PAPER_CIRCLE_COLOUR, lw=PAPER_CIRCLE_LW, buf=None):
    theta = np.linspace(0, 2 * np.pi, 300)
    pts = np.column_stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)])
    if buf is not None:
        buf.add_polyline(pts, color, lw)
    else:
        ax.plot(*pts.T, color=color, lw=lw, zorder=10)


def _draw_scaffold(ax, elev, azim, pane_filled=True, buf=None):
    style_ax(ax, elev, azim)
    draw_grey_plane(ax, "x", alpha=0.15, color="darkgray", pane_filled=pane_filled, buf=buf)
    draw_grey_plane(ax, "z", alpha=0.08, color="lightgray", pane_filled=pane_filled, buf=buf)
    draw_cube_wireframe(ax, coloured=False, buf=buf)
    _draw_axes_and_v_labels(ax, buf=buf)
    draw_corner_labels(ax)


def _draw_patterns(ax, positions, highlight_mask, dot_color, highlight_color,
                   dot_size, depthshade=True):
    """Single scatter call so depth shading is normalised across all dots."""
    colours = np.where(highlight_mask[:, None],
                       np.array(to_rgb(highlight_color)),
                       np.array(to_rgb(dot_color)))
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colours, s=dot_size, marker="o",
               zorder=10000, depthshade=depthshade)


def _draw_crosses(ax, positions, cross_color, cross_size,
                  edgecolor="none", edgelw=0):
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=[cross_color], s=cross_size, marker="X",
               edgecolors=edgecolor, linewidths=edgelw,
               zorder=10000, depthshade=False)


# ── Three-panel figure ────────────────────────────────────────
def figure_paper_patterns(elev=25, azim=-50,
                          dot_color=PAPER_DOT_COLOUR,
                          highlight_color=PAPER_HIGHLIGHT_COLOUR,
                          circle_color=PAPER_CIRCLE_COLOUR,
                          circle_lw=PAPER_CIRCLE_LW,
                          dot_size=PAPER_DOT_SIZE,
                          cross_color=PAPER_CROSS_COLOUR,
                          cross_size=PAPER_CROSS_SIZE,
                          pane_filled=True,
                          lw=ELLIPTOPE_LW,
                          light_color=LIGHT_SIDE_COLOUR, split_color=SPLIT_COLOUR,
                          alpha=ELLIPTOPE_ALPHA, split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                          wf_fade_strength=0.6, wf_split_fade_strength=0.5,
                          panel_size=PANEL_SIZE, overhang=PANEL_OVERHANG, v_frac=PANEL_V_FRAC):
    # (a) all 27 naive patterns on the cube
    def panel_a(ax, buf):
        _draw_scaffold(ax, elev, azim, pane_filled, buf)
        _draw_patterns(ax, CANONICAL_POS,
                       np.zeros(len(CANONICAL_POS), dtype=bool),
                       dot_color, highlight_color, dot_size)

    # (b) cube + circle, only the patterns lying on the circle
    def panel_b(ax, buf):
        _draw_scaffold(ax, elev, azim, pane_filled, buf)
        draw_elliptope_rings(ax, x_range=(-1.0, 0.0),
                             color=light_color, split_color=split_color,
                             lw=lw, alpha=alpha, split_alpha=split_alpha,
                             fade_strength=wf_fade_strength,
                             split_fade_strength=wf_split_fade_strength, buf=buf)
        _draw_circle(ax, circle_color, circle_lw, buf)
        _draw_patterns(ax, RELAXED_POS[IS_HALF], IS_ADJUSTED[IS_HALF],
                       dot_color, highlight_color, dot_size)
        _draw_crosses(ax, INVALID_CANONICAL_POS[INVALID_HALF], cross_color, cross_size)

    # (c) elliptope + circle, adjusted patterns highlighted, invalid crossed
    def panel_c(ax, buf):
        _draw_scaffold(ax, elev, azim, pane_filled, buf)
        draw_elliptope_rings(ax, x_range=(-1.0, 1.0),
                             color=light_color, split_color=split_color,
                             lw=lw, alpha=alpha, split_alpha=split_alpha,
                             fade_strength=wf_fade_strength,
                             split_fade_strength=wf_split_fade_strength, buf=buf)
        _draw_circle(ax, circle_color, circle_lw, buf)
        _draw_patterns(ax, RELAXED_POS, IS_ADJUSTED,
                       dot_color, highlight_color, dot_size)
        _draw_crosses(ax, INVALID_CANONICAL_POS, cross_color, cross_size)

    n = 3  # number of pannels
    fig = plt.figure(figsize=(n * panel_size, panel_size * v_frac))
    w = 1 / n
    h = (1 + 2 * overhang) / v_frac
    for i, draw in enumerate((panel_a, panel_b, panel_c)):
        rect = [i * w - overhang * w, (1 - h) / 2, w * (1 + 2 * overhang), h]
        ax = fig.add_axes(rect, projection="3d")
        ax.set_facecolor("none")
        ax.computed_zorder = False  # use explicit zorders: fills 1, lines 2, dots/text 10000
        buf = LineBuffer()
        draw(ax, buf)
        buf.flush(ax, zorder=2)

    plt.show()
    return fig


if __name__ == "__main__":
    save_path = "img/correlation-elliptope-with-canonical-patterns.png"
    fig = figure_paper_patterns(
        elev=25,
        azim=-50,
        dot_color=PAPER_DOT_COLOUR,
        highlight_color=PAPER_HIGHLIGHT_COLOUR,
        circle_color=PAPER_CIRCLE_COLOUR,
        circle_lw=PAPER_CIRCLE_LW,
        dot_size=PAPER_DOT_SIZE,
        cross_color=PAPER_CROSS_COLOUR,
        cross_size=PAPER_CROSS_SIZE,
        pane_filled=True,
        lw=ELLIPTOPE_LW,
        light_color=LIGHT_SIDE_COLOUR,
        split_color=SPLIT_COLOUR,
        alpha=ELLIPTOPE_ALPHA,
        split_alpha=ELLIPTOPE_SPLIT_ALPHA,
        wf_fade_strength=0.6,
        wf_split_fade_strength=0.5,
        panel_size=PANEL_SIZE,
        overhang=PANEL_OVERHANG,
        v_frac=PANEL_V_FRAC,
    )

    fig.savefig(save_path, dpi=300, pad_inches=0, transparent=True)
