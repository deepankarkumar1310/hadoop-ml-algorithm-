"""Shared figure style for every project chart (matplotlib, static PNG).

Palette roles (validated reference palette, light surface):
    EXPECTED   muted gray  - districts that follow the expected pattern
    ANOM_A     orange      - developed yet vulnerable
    ANOM_B     blue        - underdeveloped yet resilient
Three classes only, which keeps scatter plots inside the all-pairs-safe set;
marker shape is a second encoding so identity never rests on colour alone.
Diverging scales are blue <-> red with a gray midpoint; sequential is blue.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

EXPECTED, ANOM_A, ANOM_B = MUTED, ORANGE, BLUE

QUADRANT_STYLE = {
    "High dev / Low vul":  dict(color=EXPECTED, marker="o", label="Expected: developed, less vulnerable"),
    "Low dev / High vul":  dict(color=EXPECTED, marker="s", label="Expected: less developed, vulnerable"),
    "High dev / High vul": dict(color=ANOM_A, marker="^", label="Anomaly A: developed yet vulnerable"),
    "Low dev / Low vul":   dict(color=ANOM_B, marker="D", label="Anomaly B: less developed yet resilient"),
}

DIVERGING = LinearSegmentedColormap.from_list(
    "project_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a32323"])
SEQUENTIAL = LinearSegmentedColormap.from_list("project_seq", SEQ)


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")
