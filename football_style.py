"""football_style.py — the look of the charts: dark pitch with markings, smooth heat maps, momentum curves.

The pitch lies along the sideways axis (x, metres) with the camera at the bottom edge, as in the video:
the near touchline is at the bottom, the far touchline at the top. Its outline is estimated from where
players were seen (the goal lines and touchlines are never visible as such), so the markings are drawn to
fit that box in standard proportions (a 105 x 68 m pitch scaled to the box). Pass a box to override.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

BG = "#1c1c1e"
PANEL = "#262628"
TEXT = "#f2f2f2"
MUTED = "#9a9a9e"
LINE = "#6f6f74"
TEAM_A = "#d9392b"          # red, like the inspiration
TEAM_B = "#ececec"          # white

Box = Tuple[float, float, float, float]      # x0, x1, y0, y1  (metres)


def style_axes(ax, facecolor: str = PANEL):
    ax.set_facecolor(facecolor)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def estimate_box(x: np.ndarray, y: np.ndarray, pct: float = 1.0, pad: float = 1.5) -> Box:
    """Pitch rectangle from the spread of the sightings (goal lines left/right, touchlines near/far)."""
    x0, x1 = np.percentile(x, [pct, 100 - pct])
    y0, y1 = np.percentile(y, [pct, 100 - pct])
    return float(x0 - pad), float(x1 + pad), float(max(0.0, y0 - pad)), float(y1 + pad)


def centre_box(box: Box) -> Box:
    """Camera on the halfway line: the pitch is symmetrical about the camera (x = 0). Both goal lines are as far out as the
    farther one was seen, so a goal end that players rarely reached is not cut off."""
    h = max(abs(box[0]), abs(box[1]))
    return -h, h, box[2], box[3]


def draw_pitch(ax, box: Box, line: str = LINE, lw: float = 1.3, pitch: Tuple[float, float] = (105.0, 68.0)):
    """Outline, halfway line, centre circle, penalty and goal areas, scaled to the box."""
    from matplotlib.patches import Arc, Circle, Rectangle
    x0, x1, y0, y1 = box
    L, W = x1 - x0, y1 - y0
    sx, sy = L / pitch[0], W / pitch[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.add_patch(Rectangle((x0, y0), L, W, fill=False, ec=line, lw=lw, zorder=3))
    ax.plot([cx, cx], [y0, y1], color=line, lw=lw, zorder=3)
    ax.add_patch(Ellipse_((cx, cy), 9.15 * sx, 9.15 * sy, line, lw))
    ax.scatter([cx], [cy], s=8, c=line, zorder=3)
    for sgn, xe in ((1, x0), (-1, x1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            ax.add_patch(Rectangle((xe if sgn == 1 else xe - depth * sx, cy - half * sy), depth * sx, 2 * half * sy,
                                   fill=False, ec=line, lw=lw, zorder=3))
        ax.scatter([xe + sgn * 11 * sx], [cy], s=8, c=line, zorder=3)
    ax.set_xlim(x0 - 0.04 * L, x1 + 0.04 * L)
    ax.set_ylim(y0 - 0.06 * W, y1 + 0.06 * W)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def Ellipse_(centre, rx, ry, line, lw):
    from matplotlib.patches import Ellipse
    return Ellipse(centre, 2 * rx, 2 * ry, fill=False, ec=line, lw=lw, zorder=3)


def heat_cmap():
    """Transparent -> green -> yellow -> orange -> red."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pitchheat", [
        (0.00, (0.18, 0.80, 0.44, 0.0)), (0.12, (0.18, 0.80, 0.44, 0.35)), (0.35, (0.95, 0.85, 0.20, 0.65)),
        (0.65, (0.96, 0.55, 0.15, 0.82)), (1.00, (0.85, 0.15, 0.12, 0.95))])


def heat_image(x: np.ndarray, y: np.ndarray, box: Box, cell: float = 1.0, sigma: float = 3.0):
    """Smoothed density on the box, scaled so the busiest spot is 1. -> (image (ny, nx), extent)."""
    from scipy.ndimage import gaussian_filter
    x0, x1, y0, y1 = box
    nx, ny = max(2, int((x1 - x0) / cell)), max(2, int((y1 - y0) / cell))
    h, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[x0, x1], [y0, y1]])
    h = gaussian_filter(h, sigma / cell)
    top = np.percentile(h[h > 0], 99.5) if (h > 0).any() else 1.0
    return np.clip(h.T / max(top, 1e-9), 0, 1), [x0, x1, y0, y1]


def draw_heat(ax, x: np.ndarray, y: np.ndarray, box: Box, sigma: float = 3.0):
    img, ext = heat_image(x, y, box, sigma=sigma)
    ax.imshow(img, origin="lower", extent=ext, cmap=heat_cmap(), vmin=0, vmax=1, aspect="equal",
              interpolation="bilinear", zorder=2)


def smooth(v: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.nan_to_num(v), sigma)


def draw_momentum(ax, minutes: np.ndarray, value: np.ndarray, colour_pos: str = TEAM_A, colour_neg: str = TEAM_B,
                  marks: Sequence[Tuple[float, str]] = (), ylim: Optional[float] = None):
    """Mirrored area: above the line the first team has the play, below the line the second."""
    v = np.asarray(value, float)
    ax.fill_between(minutes, 0, np.where(v > 0, v, 0), color=colour_pos, lw=0, zorder=2)
    ax.fill_between(minutes, 0, np.where(v < 0, v, 0), color=colour_neg, lw=0, zorder=2)
    ax.axhline(0, color=LINE, lw=1.0, zorder=3)
    top = ylim or max(0.2, float(np.nanmax(np.abs(v))) * 1.1)
    ax.set_ylim(-top, top)
    ax.set_xlim(float(minutes[0]), float(minutes[-1]))
    for m, label in marks:
        ax.axvline(m, color=LINE, lw=1.0, ls=(0, (1, 4)), zorder=1)
        ax.text(m, -top * 0.97, label, color=TEXT, fontsize=10, ha="center", va="bottom", fontweight="bold")
    ax.set_yticks([])
    ax.grid(False)


# ── team colours that match the shirts ───────────────────────────────────────

def _hsv(bgr):
    import colorsys
    b, g, r = [float(v) / 255.0 for v in bgr]
    return colorsys.rgb_to_hsv(r, g, b)


def colour_name(bgr) -> str:
    """white / grey / black / red / orange / yellow / green / blue / purple, from the mean shirt colour."""
    h, s, v = _hsv(bgr)
    if s < 0.25:
        return "white" if v > 0.45 else "black"
    d = h * 360.0
    return ("red" if d < 15 or d >= 345 else "orange" if d < 40 else "yellow" if d < 70 else
            "green" if d < 170 else "blue" if d < 260 else "purple")


def display_colour(bgr) -> str:
    """The shirt colour as it should be drawn on the dark background: whites stay white, dark shirts are
    lifted to a clearly visible version of the same hue (navy -> bright blue)."""
    import colorsys
    h, s, v = _hsv(bgr)
    if s < 0.25:
        return "#ececec" if v > 0.45 else "#8e8e93"
    r, g, b = colorsys.hsv_to_rgb(h, max(s, 0.62), max(v, 0.88))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
