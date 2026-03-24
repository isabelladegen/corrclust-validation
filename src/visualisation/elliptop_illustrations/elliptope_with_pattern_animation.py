# ── Pattern data ──────────────────────────────────────────────
import io

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from src.visualisation.elliptop_illustrations.trustworthy_ai_talk_figures import draw_corner_labels, \
    draw_axes_and_labels, draw_cube_wireframe, draw_grey_plane, BERRY, draw_elliptope_wireframe, style_ax, \
    make_fig, SPLIT_COLOUR, LIGHT_SIDE_COLOUR, N_SLICES, ELLIPTOPE_LW, \
    ELLIPTOPE_ALPHA, ELLIPTOPE_SPLIT_ALPHA, VIVID_EMERALD

PATTERN_TABLE = [
    ((0, 0, 0), (0, 0, 0), True),  # 0
    ((0, 0, 1), (0, 0, 1), True),  # 1
    ((0, 0, -1), (0, 0, -1), True),  # 2
    ((0, 1, 0), (0, 1, 0), True),  # 3
    ((0, 1, 1), (0, 0.71, 0.7), True),  # 4
    ((0, 1, -1), (0, 0.71, -0.7), True),  # 5
    ((0, -1, 0), (0, -1, 0), True),  # 6
    ((0, -1, 1), (0, -0.71, 0.7), True),  # 7
    ((0, -1, -1), (0, -0.71, -0.7), True),  # 8
    ((1, 0, 0), (1, 0, 0), True),  # 9
    ((1, 0, 1), (0.71, 0, 0.7), True),  # 10
    ((1, 0, -1), (0.71, 0, -0.7), True),  # 11
    ((1, 1, 0), (0.71, 0.7, 0), True),  # 12
    ((1, 1, 1), (1, 1, 1), True),  # 13
    ((1, 1, -1), None, False),  # 14
    ((1, -1, 0), (0.71, -0.7, 0), True),  # 15
    ((1, -1, 1), None, False),  # 16
    ((1, -1, -1), (1, -1, -1), True),  # 17
    ((-1, 0, 0), (-1, 0, 0), True),  # 18
    ((-1, 0, 1), (-0.71, 0, 0.7), True),  # 19
    ((-1, 0, -1), (-0.71, 0, -0.7), True),  # 20
    ((-1, 1, 0), (-0.71, 0.7, 0), True),  # 21
    ((-1, 1, 1), None, False),  # 22
    ((-1, 1, -1), (-1, 1, -1), True),  # 23
    ((-1, -1, 0), (-0.71, -0.7, 0), True),  # 24
    ((-1, -1, 1), (-1, -1, 1), True),  # 25
    ((-1, -1, -1), None, False),  # 26
]

CANONICAL_POS = np.array([p[0] for p in PATTERN_TABLE], dtype=float)
RELAXED_POS = np.array([p[1] for p in PATTERN_TABLE if p[2]], dtype=float)
PATTERN_VALID = np.array([p[2] for p in PATTERN_TABLE])
INVALID_CANONICAL_POS = CANONICAL_POS[~PATTERN_VALID]


# ── Pattern drawing ───────────────────────────────────────────
def _draw_pattern_dots(ax, positions, dot_color=VIVID_EMERALD,
                       dot_size=120, depthshade=False):
    """Draw a set of pattern dots, all same style."""
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=[dot_color], s=dot_size, marker='o',
               edgecolors=None, zorder=10000, depthshade=depthshade)


def _draw_invalid_crosses(ax, positions, cross_color="darkgrey",
                          cross_size=150, edgecolor='black', edgelw=0.5):
    """Draw X markers at invalid pattern positions."""
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=[cross_color], s=cross_size, marker='X',
               edgecolors=edgecolor, linewidths=edgelw,
               zorder=10000, depthshade=False)


# ── Rotation renderer ─────────────────────────────────────────
def _render_rotation(draw_fn, n_frames=72, elev=25, start_azim=-50, dpi=100):
    """Render one full 360° rotation, return list of PIL Images."""
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="3d")
    draw_fn(ax)
    plt.tight_layout(pad=1.0)

    frames = []
    for i in range(n_frames):
        azim = start_azim + (360 * i / n_frames)
        ax.view_init(elev=elev, azim=azim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.3, transparent=False, facecolor="white")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)
    return frames


DOT_COLOUR = VIVID_EMERALD
DOT_SIZE = 500
CROSS_COLOUR = "darkgrey"
CROSS_SIZE = 500


# ── Figure 5a: static, 27 naive patterns on cube ─────────────
def figure_5a_patterns_on_cube(save_path=None, elev=25, azim=-50,
                               n_slices=N_SLICES, lw=ELLIPTOPE_LW,
                               light_color=LIGHT_SIDE_COLOUR, split_color=SPLIT_COLOUR,
                               alpha=ELLIPTOPE_ALPHA, split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                               wf_fade_strength=0.6, wf_split_fade_strength=0.5,
                               pane_filled=True,
                               dot_color=DOT_COLOUR, dot_size=DOT_SIZE):
    def draw(ax):
        style_ax(ax, elev, azim)
        # draw_elliptope_wireframe(ax, n_slices=n_slices,
        #                          color=light_color, split_color=split_color,
        #                          split_at=0.0, lw=lw,
        #                          alpha=alpha, split_alpha=split_alpha,
        #                          fade_strength=wf_fade_strength,
        #                          split_fade_strength=wf_split_fade_strength)
        draw_grey_plane(ax, "x", alpha=0.15, color="darkgray",
                        pane_filled=pane_filled)
        draw_grey_plane(ax, "z", alpha=0.08, color="lightgray",
                        pane_filled=pane_filled)
        draw_cube_wireframe(ax, coloured=False)
        draw_axes_and_labels(ax, coloured=False)
        draw_corner_labels(ax)
        _draw_pattern_dots(ax, CANONICAL_POS, dot_color=dot_color,
                           dot_size=dot_size, depthshade=True)

    return make_fig(draw, save_path)


# ── Figure 5b: static, 23 valid + 4 crossed at corners ───────
def figure_5b_patterns_adjusted(save_path=None, elev=25, azim=-50,
                                n_slices=N_SLICES, lw=ELLIPTOPE_LW,
                                light_color=LIGHT_SIDE_COLOUR, split_color=SPLIT_COLOUR,
                                alpha=ELLIPTOPE_ALPHA, split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                                wf_fade_strength=0.6, wf_split_fade_strength=0.5,
                                pane_filled=True,
                                dot_color=DOT_COLOUR, dot_size=DOT_SIZE,
                                cross_color=CROSS_COLOUR, cross_size=CROSS_SIZE):
    def draw(ax):
        style_ax(ax, elev, azim)
        draw_elliptope_wireframe(ax, n_slices=n_slices,
                                 color=light_color, split_color=split_color,
                                 split_at=0.0, lw=lw,
                                 alpha=alpha, split_alpha=split_alpha,
                                 fade_strength=wf_fade_strength,
                                 split_fade_strength=wf_split_fade_strength)
        draw_grey_plane(ax, "x", alpha=0.15, color="darkgray",
                        pane_filled=pane_filled)
        draw_grey_plane(ax, "z", alpha=0.08, color="lightgray",
                        pane_filled=pane_filled)
        draw_cube_wireframe(ax, coloured=False)
        draw_axes_and_labels(ax, coloured=False)
        draw_corner_labels(ax)
        _draw_pattern_dots(ax, RELAXED_POS, dot_color=dot_color,
                           dot_size=dot_size, depthshade=True)
        _draw_invalid_crosses(ax, INVALID_CANONICAL_POS,
                              cross_color=cross_color, cross_size=cross_size)

    return make_fig(draw, save_path)


# ── Figure 5c: animated, final state rotating ─────────────────
def figure_5_patterns_gif(save_path=None,
                          n_frames=72, elev=25,
                          n_slices=N_SLICES, lw=ELLIPTOPE_LW,
                          light_color=LIGHT_SIDE_COLOUR, split_color=SPLIT_COLOUR,
                          alpha=ELLIPTOPE_ALPHA, split_alpha=ELLIPTOPE_SPLIT_ALPHA,
                          wf_fade_strength=0.6, wf_split_fade_strength=0.5,
                          pane_filled=False,
                          dot_color=DOT_COLOUR, dot_size=DOT_SIZE,
                          dpi=100, duration=150):
    def draw_final(ax):
        style_ax(ax, elev, -50)
        draw_elliptope_wireframe(ax, n_slices=n_slices,
                                 color=light_color, split_color=split_color,
                                 split_at=0.0, lw=lw,
                                 alpha=alpha, split_alpha=split_alpha,
                                 fade_strength=wf_fade_strength,
                                 split_fade_strength=wf_split_fade_strength)
        draw_grey_plane(ax, "x", alpha=0.15, color="darkgray",
                        pane_filled=pane_filled)
        draw_grey_plane(ax, "z", alpha=0.08, color="lightgray",
                        pane_filled=pane_filled)
        draw_cube_wireframe(ax, coloured=False)
        draw_axes_and_labels(ax, coloured=False)
        draw_corner_labels(ax)
        _draw_pattern_dots(ax, RELAXED_POS, dot_color=dot_color,
                           dot_size=dot_size)

    frames = _render_rotation(draw_final, n_frames, elev, dpi=dpi)

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"Saved {save_path} ({len(frames)} frames)")


if __name__ == "__main__":
    figure_5a_patterns_on_cube("img/fig5a_patterns_on_cube.png")
    figure_5b_patterns_adjusted("img/fig5b_patterns_adjusted.png")
    # figure_5_patterns_gif("img/fig5_elliptope_patterns_animated.gif")
