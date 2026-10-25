"""
The look of the single map figures (average maps, group maps, GMM maps, statistic maps, difference maps ...), made to be placed as vector graphics
into a multi-panel figure: every one has the same size, the same line widths and the same fonts, so the files can be put side by side at one scale.

  - the figure is a fixed canvas (CANVAS_MM) with a transparent background in the svg (the png is white), text stays text (Nimbus Sans, 7 pt numbers,
    9 pt label);
  - the brain (the outline's bounding box) is fitted into a fixed slot of the canvas, centred, with the aspect ratio kept;
  - the outline is black and dotted (round dots of one line width, separated by two line widths) and 2 mm wide in the reference drawing;
  - the unlabelled scale bar is a black bar inside the outline's bounding box, close to its lower right corner;
  - the colourbar has a black border and ticks, right of the brain and centred on the canvas, with its axis label rotated 90 degrees (reading upward).

All sizes are given in `reference' millimetres (a hand-made reference drawing of an average map with its colourbar, in which the numbers are 6.22 mm
high) and multiplied with MM_PER_REFERENCE_MM so that the numbers are FONT_PT points. Change the constants here to change every map.

  fig, ax = map_canvas(contour)                    the figure and the axes of the brain (data coordinates of `contour`, equal aspect, no frame)
  decorate_map(fig, ax, contour, ...)               outline, scale bar, colourbar and label
  save_map(fig, path)                               png (white) and svg (transparent) next to each other
  add_title(fig, text)                              an optional title above the brain (centred over the outline): TITLE_PT bold, in the blank space above the outline and, where
                                                    that is not enough, above the canvas; that much blank space at the bottom of the canvas is cut off, so the titled map is
                                                    as high as the untitled one (unless the title needs more than the bottom has): the brain keeps its place and size.
                                                    save_map(fig, path, title=text) saves only the titled map; with TITLE_OF set (a function path -> text or None) save_map writes
                                                    the usual file and, next to it, <name>_title.png/.svg for the maps that have a title: the existing maps are not changed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.patches import Rectangle

FONT_PT = 8            # the numbers of the colourbar
TITLE_PT = 8           # the optional title above the map (bold)
TITLE_LINE_MM = TITLE_PT * 25.4 / 72 * 1.2   # its line pitch
TITLE_GAP_MM = 1.2     # between the last line and the outline
TITLE_OF = None        # function(path) -> title text or None: maps for which it gives a title are also saved as <name>_title.png/.svg (see save_map)
LABEL_PT = 9           # the colourbar's label (shrunk when it is longer than the canvas is high)
FONT_FAMILY = ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"]
MM_PER_REFERENCE_MM = 0.39704   # the size of everything but the text (the brain is about 52 mm high); the reference drawing's numbers are 6.2197 mm high

# in reference millimetres (the reference drawing, page coordinates, y downward from the top of the page)
TILE_HEIGHT_MM = 60.0   # the canvas is as high as one tile of a multi-panel figure: the map keeps its size, the tile around it is transparent
_REFERENCE_PAGE_HEIGHT = 131.196
CANVAS_WIDTH_MM = 56.0
CANVAS = (CANVAS_WIDTH_MM / MM_PER_REFERENCE_MM, TILE_HEIGHT_MM / MM_PER_REFERENCE_MM)
_DY = (CANVAS[1] - _REFERENCE_PAGE_HEIGHT) / 2   # the reference drawing centred in the (higher) tile
SLOT = (0.902, 0.963 + _DY, 94.613, 129.318)    # x, y (top), width, height of the box the outline's bounding box is fitted in
WITHOUT_COLOURBAR_FRACTION = (0.902 + 94.613 + 0.9) / CANVAS[0]   # the part of the canvas's width that is left when the colourbar is hidden
BRAIN_ALIGN = "right"   # a brain narrower than the slot: "right" = against the right edge of the slot (the colourbar is as close to every brain as to the widest one; the blank space is on the
                        # left, where a figure layout cuts it off), "centre" = in the middle of the slot (as before)
OUTLINE_WIDTH = 2.0
OUTLINE_DASHES = (1.0, 2.0)                      # dot, gap in outline widths (round caps add one width to the dot)
SCALEBAR_HEIGHT = 3.0
SCALEBAR_INSET = (4.3, 4.34)                     # from the right and the bottom of the outline's bounding box (if the outline is in the way the bar moves left, then below the brain)
SCALEBAR_CLEARANCE = 2.5                         # the bar keeps this distance from the outline
COLOURBAR_X, COLOURBAR_TOP = 100.745, 15.995 + _DY
COLOURBAR_WIDTH, COLOURBAR_HEIGHT = 6.2197, 99.03
COLOURBAR_BORDER = 0.691                         # the border and the ticks
COLOURBAR_TICK = 2.533
COLOURBAR_NUMBER_GAP = 1.555
LABEL_X = 130.49                                 # the label's baseline

CANVAS_MM = tuple(v * MM_PER_REFERENCE_MM for v in CANVAS)
_PT_PER_MM = 72 / 25.4


def _mm(value):
    return value * MM_PER_REFERENCE_MM


def map_canvas(contour, invert_y=False):
    """(figure, axes): the brain's axes fitted into the slot, limits = bounding box of `contour`, equal aspect, no frame, transparent."""
    contour = np.asarray(contour, dtype=float)
    with plt.rc_context({"font.family": FONT_FAMILY}):
        fig = plt.figure(figsize=(CANVAS_MM[0] / 25.4, CANVAS_MM[1] / 25.4))
    x0, x1 = contour[:, 0].min(), contour[:, 0].max()
    y0, y1 = contour[:, 1].min(), contour[:, 1].max()
    slot_x, slot_top, slot_w, slot_h = (_mm(v) for v in SLOT)
    scale = min(slot_w / (x1 - x0), slot_h / (y1 - y0))   # mm per data unit
    width, height = (x1 - x0) * scale, (y1 - y0) * scale
    left = slot_x + (slot_w - width) / (1 if BRAIN_ALIGN == "right" else 2)
    bottom = CANVAS_MM[1] - (slot_top + slot_h) + (slot_h - height) / 2   # from the bottom of the canvas
    ax = fig.add_axes([left / CANVAS_MM[0], bottom / CANVAS_MM[1], width / CANVAS_MM[0], height / CANVAS_MM[1]])
    ax.set_xlim(x0, x1)
    ax.set_ylim((y1, y0) if invert_y else (y0, y1))
    ax.set_aspect("equal")
    ax.axis("off")
    fig._brain_box_mm = (left, bottom, width, height)
    fig._contour_mm = np.column_stack([left + (contour[:, 0] - x0) * scale, bottom + ((y1 - contour[:, 1]) if invert_y else (contour[:, 1] - y0)) * scale])   # the outline in mm on the canvas
    fig._mm_per_unit = scale
    return fig, ax


def _ticks_of(vmin, vmax, ticks):
    if ticks is None:
        ticks = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    ticks = np.asarray(ticks, dtype=float)
    return ticks[(ticks >= vmin - 1e-9 * abs(vmax - vmin)) & (ticks <= vmax + 1e-9 * abs(vmax - vmin))]


def _formatter(ticks):
    steps = np.diff(np.sort(ticks))
    step = steps[steps > 0].min() if (steps > 0).any() else 1.0
    decimals = int(max(0, min(6, -np.floor(np.log10(step) + 1e-9))))
    return ticker.FuncFormatter(lambda value, _: f"{(0.0 if abs(value) < 10 ** -(decimals + 3) * 0.5 else value):.{decimals}f}")


def outline_line(ax, contour, colour="black"):
    """The dotted outline."""
    contour = np.asarray(contour, dtype=float)
    closed = np.vstack([contour, contour[:1]])
    width_pt = _mm(OUTLINE_WIDTH) * _PT_PER_MM
    return ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=width_pt, ls=(0, OUTLINE_DASHES), dash_capstyle="round",
                   solid_joinstyle="round", clip_on=False, zorder=5)[0]


def _brain_height_mm(fig):
    return fig._brain_box_mm[3]


def scale_bar(fig, length, colour="black"):
    """A bar of `length` (data units of the map) near the lower right corner of the brain's bounding box (no label). Where the outline is in the way (a narrow brain
    fills the corner) the bar moves left along the bottom and higher up the side; if there is no room, it goes to the right, under the colourbar (the free corner of the canvas),
    and only then below the brain (where the canvas would get higher)."""
    from matplotlib.path import Path as Polygon
    from matplotlib.transforms import Bbox
    left, bottom, width, _ = fig._brain_box_mm
    long_mm, high_mm, gap = length * fig._mm_per_unit, _mm(SCALEBAR_HEIGHT), _mm(SCALEBAR_CLEARANCE)
    outline = Polygon(fig._contour_mm)
    right_edge = left + width - _mm(SCALEBAR_INSET[0])
    inset = _mm(SCALEBAR_INSET[1])
    colourbar_right = _mm(COLOURBAR_X + COLOURBAR_WIDTH)
    colourbar_bottom = CANVAS_MM[1] - _mm(COLOURBAR_TOP) - _mm(COLOURBAR_HEIGHT)
    rise = [(bottom + inset + k * _mm(1.0), right_edge) for k in range(int(0.45 * _brain_height_mm(fig) / _mm(1.0)))]   # beside the lower edge and, if the outline is in the way, higher up
    corner = [(y, colourbar_right) for y in np.arange(bottom + inset, colourbar_bottom - 2 * gap - high_mm - 1.0, 0.5)]            # under the colourbar, to the right of the lowest part of the brain
    placed = None
    for y, right in rise + corner + [(bottom, right_edge), (bottom - inset, right_edge)]:   # last: below the brain (the canvas gets higher only there)
        x = right - long_mm
        while x >= left - 1e-9:
            if not outline.intersects_bbox(Bbox([[x - gap, y - gap], [x + long_mm + gap, y + high_mm + gap]]), filled=True):
                placed = (x, y)
                break
            x -= 0.5
        if placed:
            break
    x, y = placed or (right_edge - long_mm, bottom - inset)
    fig.add_artist(Rectangle((x / CANVAS_MM[0], y / CANVAS_MM[1]), long_mm / CANVAS_MM[0], high_mm / CANVAS_MM[1], transform=fig.transFigure, fc=colour, ec="none"))   # in figure fractions: it moves with everything else when the saved canvas is cut or grown (a title)


def colourbar(fig, mappable, vmin, vmax, label="", ticks=None):
    """The colourbar right of the brain, in the style of the reference drawing; returns the colorbar."""
    with plt.rc_context({"font.family": FONT_FAMILY}):
        top = CANVAS_MM[1] - _mm(COLOURBAR_TOP)
        height = _mm(COLOURBAR_HEIGHT)
        cax = fig.add_axes([_mm(COLOURBAR_X) / CANVAS_MM[0], (top - height) / CANVAS_MM[1], _mm(COLOURBAR_WIDTH) / CANVAS_MM[0], height / CANVAS_MM[1]])
        marks = _ticks_of(vmin, vmax, ticks)
        cax.set_gid("colourbar")   # svg layers that can be hidden afterwards (the colourbar of all but one map of a row)
        bar = fig.colorbar(mappable, cax=cax, ticks=marks)
        bar.ax.yaxis.set_major_formatter(_formatter(marks))
        border_pt = _mm(COLOURBAR_BORDER) * _PT_PER_MM
        bar.outline.set_linewidth(border_pt)
        bar.outline.set_edgecolor("black")
        bar.solids.set_edgecolor("face")
        bar.solids.set_linewidth(0.2)
        bar.solids.set_rasterized(False)
        cax.tick_params(axis="y", length=_mm(COLOURBAR_TICK) * _PT_PER_MM, width=border_pt, labelsize=FONT_PT,
                        pad=_mm(COLOURBAR_NUMBER_GAP) * _PT_PER_MM, colors="black")
        if label:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            numbers_right = max((tick.get_window_extent(renderer).x1 for tick in cax.get_yticklabels() if tick.get_text()), default=0) / fig.dpi * 25.4
            colourbar_label(fig, label, numbers_right)
    return bar


def colourbar_label(fig, label, numbers_right=0.0):
    """The label bottom-to-top at the reference position (further right when the numbers of the colourbar reach to `numbers_right` mm), centred on the colourbar;
    its font shrinks when it would not fit the canvas."""
    centre = (CANVAS_MM[1] - _mm(COLOURBAR_TOP) - _mm(COLOURBAR_HEIGHT) / 2) / CANVAS_MM[1]
    with plt.rc_context({"font.family": FONT_FAMILY}):
        x = max(_mm(LABEL_X), numbers_right + 1.0 + 0.78 * LABEL_PT / _PT_PER_MM)   # the label's glyphs reach about 0.78 of the font size left of its baseline
        text = fig.text(x / CANVAS_MM[0], centre, label, rotation=90, ha="center", va="baseline", rotation_mode="anchor", fontsize=LABEL_PT)
        text.set_gid("colourbar_label")
        renderer = fig.canvas.get_renderer()
        room = (CANVAS_MM[1] - 2) / 25.4 * fig.dpi
        size = LABEL_PT
        while text.get_window_extent(renderer).height > room and size > 5:
            size -= 0.5
            text.set_fontsize(size)
    return text


def decorate_map(fig, ax, contour, mappable=None, vmin=None, vmax=None, label="", ticks=None, scalebar=None):
    """Outline, scale bar and (with a `mappable`) the colourbar with its label."""
    outline_line(ax, contour)
    if scalebar:
        scale_bar(fig, scalebar)
    if mappable is not None:
        return colourbar(fig, mappable, vmin, vmax, label, ticks)
    return None


def add_title(fig, title):
    """A title above the brain of a map made with map_canvas: `title` (lines separated by newlines) in TITLE_PT bold, centred over the outline, its last line TITLE_GAP_MM above
    the outline. Returns the height (mm) by which the title reaches above the canvas (0 if it fits into the blank space above the outline): save the figure that much higher
    (save_map does)."""
    left, _, width, _ = fig._brain_box_mm
    outline_top = fig._contour_mm[:, 1].max() + _mm(OUTLINE_WIDTH) / 2
    lines = str(title).split("\n")
    with plt.rc_context({"font.family": FONT_FAMILY}):
        for k, line in enumerate(lines):
            text = fig.text((left + width / 2) / CANVAS_MM[0], (outline_top + TITLE_GAP_MM + (len(lines) - 1 - k) * TITLE_LINE_MM) / CANVAS_MM[1], line, ha="center",
                            va="baseline", fontsize=TITLE_PT, fontweight="bold")
            text.set_gid("title")
    first_top = outline_top + TITLE_GAP_MM + (len(lines) - 1) * TITLE_LINE_MM + 0.73 * TITLE_PT * 25.4 / 72   # the top of the first line's capitals
    bottoms = [fig._contour_mm[:, 1].min() - _mm(OUTLINE_WIDTH) / 2] + [a.get_y() * CANVAS_MM[1] for a in fig.artists if isinstance(a, Rectangle)] + [ax.get_position().y0 * CANVAS_MM[1] for ax in fig.axes]
    fig._free_bottom_mm = max(0.0, min(bottoms) - 0.6)   # the blank space below the brain, the scale bar and the colourbar: the title takes it before the canvas grows
    return max(0.0, first_top + 0.4 - CANVAS_MM[1])


def _write(fig, path, dpi, extra_mm=0.0):
    """png (white background) and, next to it, svg (transparent, text as text) - unless a figure style already writes the svg with every png. `extra_mm`: the canvas is
    that much higher at the top (a title that does not fit into it); as far as the canvas has blank space at the bottom (fig._free_bottom_mm, set by add_title), that is cut off,
    so a title usually does not make the map higher than its canvas."""
    from matplotlib.transforms import Bbox
    path = Path(path)
    trim = min(extra_mm, getattr(fig, "_free_bottom_mm", 0.0))
    keep = {"bbox_inches": Bbox.from_extents(0, trim / 25.4, CANVAS_MM[0] / 25.4, (CANVAS_MM[1] + extra_mm) / 25.4)} if extra_mm > 0 else {}
    fig.savefig(path, dpi=dpi, facecolor="white", **keep)
    if path.suffix.lower() == ".png" and not getattr(type(fig).savefig, "_with_svg", False):
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path.with_suffix(".svg"), format="svg", transparent=True, **keep)


def save_map(fig, path, dpi=600, title=None):
    """The map as png (white background) and svg (transparent, text as text) at `path`. `title`: save the map WITH this title at `path` (see add_title). Without it, and if
    TITLE_OF gives a title for `path`, the usual files are written and next to them <name>_title.png/.svg with the title."""
    path = Path(path)
    text = title or (TITLE_OF(path) if TITLE_OF else None)
    if title is None:
        _write(fig, path, dpi)
    if text:
        _write(fig, path if title else path.with_name(path.stem + "_title" + path.suffix), dpi, add_title(fig, text))
