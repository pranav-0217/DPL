"""Shared chart style: colour-blind-safe palette, fonts and layout helpers."""

import matplotlib as mpl

# --- categorical slots (fixed order, never cycled) ---
S1_BLUE = "#2a78d6"
S2_ORANGE = "#eb6834"
S3_AQUA = "#1baf7a"

# --- diverging pair (polarity: better <-> worse) ---
DIV_GOOD = "#2a78d6"
DIV_MID = "#f0efec"
DIV_BAD = "#d03b3b"

# --- chrome and ink ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",          # solid hairlines, never dashed
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "figure.constrained_layout.use": False,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def layout(fig, top=0.80, bottom=0.22, left=0.10, right=0.97, **kw):
    """Reserve figure margins up front instead of using a tight bounding box."""
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, **kw)


def caption(fig, text, y=0.035):
    fig.text(0.5, y, text, ha="center", va="center", fontsize=8.5, color=MUTED)


def title_block(fig, title, subtitle=None, x=0.015, y=0.955):
    """Draw the headline and subtitle in figure coordinates."""
    fig.text(x, y, title, fontsize=14.5, fontweight="bold", color=INK,
             ha="left", va="top")
    if subtitle:
        fig.text(x, y - 0.075, subtitle, fontsize=10, color=INK_2,
                 ha="left", va="top")


def panel_title(ax, text):
    """Heading above one panel of a multi-panel figure."""
    ax.text(0, 1.04, text, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=INK, ha="left", va="bottom")
