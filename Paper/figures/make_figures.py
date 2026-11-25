"""
Column-width figures (252 bp, 9 pt Times, two per row) from figures/data/*.csv.
Labels are placed by measurement: every candidate position is tested against
all drawn lines, markers, error bars, bars and other labels in display
coordinates, and the build fails if any label still collides.
  python make_figures.py      -> fig_*.pdf, label_check.txt
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection      # noqa: E402
from matplotlib.path import Path as MplPath                             # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402
from matplotlib.text import Text                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
W = 3.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "hatch.linewidth": 0.5,
    "axes.facecolor": "#EBEBEB", "axes.linewidth": 0.0, "axes.labelcolor": "#1A1A1A",
    "xtick.color": "#4D4D4D", "ytick.color": "#4D4D4D", "xtick.labelcolor": "#4D4D4D", "ytick.labelcolor": "#4D4D4D",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "lines.linewidth": 1.1, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": True, "axes.grid.axis": "both", "grid.color": "white", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.pad_inches": 0.02, "legend.frameon": False})

STYLE = {
    "LACP":    dict(color="#377EB8", marker="o", ls="-"),
    "CC-LACP": dict(color="#E41A1C", marker="D", ls="-"),
    "CC-CP":   dict(color="#FF7F00", marker="v", ls=(0, (4, 1.5))),
    "ACI":     dict(color="#4DAF4A", marker="^", ls=(0, (1.5, 1.2))),
    "ACI-CC":  dict(color="#A65628", marker="s", ls=(0, (4, 1.5, 1.2, 1.5))),
    "SCP":     dict(color="#B8B8B8", marker="s", ls="-"),
    "ACI-tuned": dict(color="#9FD49D", marker="^", ls=(0, (1.5, 1.2))),
    "ACI-LACP": dict(color="#1B5E20", marker="P", ls=(0, (4, 1.5, 1.2, 1.5))),
    "Width-matched SCP": dict(color="white", marker="s", ls="-"),
}
SHOW = {"ACI-LACP": "ACI on LACP", "ACI-CC": "ACI on CC-CP"}
KEYBOX = dict(boxstyle="round,pad=0.35", fc="white", ec="none", lw=0.0)
SPAN = "#D4D4D4"          # event window shading on the grey panel


def gg(ax, minor_x=True, minor_y=True):
    """ggplot-type panel: white minor grid halfway between major lines, no minor ticks."""
    from matplotlib.ticker import AutoMinorLocator
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="minor", color="white", lw=0.35)
    ax.tick_params(which="minor", length=0)
REPORT = []
FAILED = []


# ======================================================================
# collision-checked label placement
# ======================================================================
def _seg_rect(a, b, r):
    """True if segment a-b intersects rectangle r = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = r
    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
        return True
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


class Placer:
    def __init__(self, fig, ax, soft=()):
        self.fig, self.ax, self.soft = fig, ax, set(soft)
        fig.canvas.draw()
        self.r = fig.canvas.get_renderer()
        self.labels = []
        self.extra = []          # (kind, geometry) obstacles added by labels (leaders, boxes)

    def obstacles(self):
        ax = self.ax
        segs, pts = [], []
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_gid() in self.soft:
                continue
            xy = ln.get_transform().transform(ln.get_xydata())
            ok = np.isfinite(xy).all(1)
            if ln.get_linestyle() not in ("None", "none", "", " "):
                for a, b, oa, ob in zip(xy[:-1], xy[1:], ok[:-1], ok[1:]):
                    if oa and ob:
                        segs.append((a, b))
            if ln.get_marker() not in (None, "None", "none", ""):
                rad = ln.get_markersize() * self.fig.dpi / 72 / 2 + 1.0
                pts += [(p, rad) for p in xy[ok]]
        self.polys = []
        for col in ax.collections:
            if isinstance(col, LineCollection) and col.get_gid() not in self.soft:
                for s in col.get_segments():
                    d = col.get_transform().transform(np.asarray(s))
                    segs += list(zip(d[:-1], d[1:]))
            elif isinstance(col, PolyCollection) and col.get_gid() not in self.soft:
                # confidence bands: their outline and interior are obstacles
                for pth in col.get_paths():
                    d = col.get_transform().transform(pth.vertices)
                    segs += list(zip(d[:-1], d[1:]))
                    self.polys.append(MplPath(d))
        rects = []
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_gid() not in self.soft:
                e = p.get_window_extent(self.r)
                rects.append((e.x0, e.y0, e.x1, e.y1))
        return segs, pts, rects

    def collides(self, bb, segs, pts, rects, pad=1.5):
        r = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
        axb = self.ax.get_window_extent(self.r)
        if bb.x0 < axb.x0 + 1 or bb.x1 > axb.x1 - 1 or bb.y0 < axb.y0 + 1 or bb.y1 > axb.y1 - 1:
            return "outside"
        for a, b in segs:
            if _seg_rect(a, b, r):
                return "line"
        for (c, rad) in pts:
            cx = min(max(c[0], r[0]), r[2])
            cy = min(max(c[1], r[1]), r[3])
            if (cx - c[0]) ** 2 + (cy - c[1]) ** 2 < rad ** 2:
                return "marker"
        for q in rects + [(l.x0, l.y0, l.x1, l.y1) for l in self.labels]:
            if not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3]):
                return "box"
        centre = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        for poly in getattr(self, "polys", []):
            if poly.contains_point(centre):
                return "band"
        return None

    def place(self, text, xy, color, cands=None, coords="data", leader=False, **kw):
        ax = self.ax
        if cands is None:
            cands = [(rad * np.cos(t), rad * np.sin(t)) for rad in (5, 8, 12, 16)
                     for t in np.deg2rad([30, -30, 150, -150, 0, 180, 90, -90, 60, -60, 120, -120])]
        segs, pts, rects = self.obstacles()
        whys = []
        for dx, dy in cands:
            ha = "left" if dx > 0.5 else ("right" if dx < -0.5 else "center")
            va = "bottom" if dy > 0.5 else ("top" if dy < -0.5 else "center")
            ap = None
            if leader and np.hypot(dx, dy) > 9:
                ap = dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0.5, shrinkB=2.5)
            t = ax.annotate(text, xy=xy, xycoords=coords, xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, color=color, arrowprops=ap, **kw)
            t.update_positions(self.r)
            bb = Text.get_window_extent(t, self.r)
            why = self.collides(bb, segs, pts, rects)
            if why is None and ap is not None:
                # the leader must not cross lines either
                a = ax.transData.transform(xy) if coords == "data" else None
                c = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
                if a is not None and any(_seg_dist(a, p, q) > 3 and _cross(p, q, a, c) for p, q in segs):
                    why = "leader"
            if why is None:
                self.labels.append(bb)
                return t
            whys.append(why)
            t.remove()
        raise RuntimeError(f"no free position for label {text!r}: {pd.Series(whys).value_counts().to_dict()}")

    def on_series(self, text, xs, ys, color, **kw):
        """Direct label next to its own curve: try anchors along the curve
        (segment midpoints, then vertices) with a ring of small offsets."""
        anchors = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:])]
        anchors += list(zip(xs, ys))
        for radii in ((5, 8, 11), (15, 20, 26)):
            ring = [(r * np.cos(t), r * np.sin(t)) for r in radii
                    for t in np.deg2rad([90, -90, 60, -60, 120, -120, 30, -30, 150, -150, 0, 180])]
            for a in anchors:
                try:
                    return self.place(text, a, color, cands=ring, leader=radii[0] > 9, **kw)
                except RuntimeError:
                    continue
        raise RuntimeError(f"no free position along the curve for {text!r}")

    def box(self, text, corners=("ul", "ur", "ll", "lr"), **kw):
        pos = {"ul": (0.03, 0.97, "left", "top"), "ur": (0.97, 0.97, "right", "top"),
               "ll": (0.03, 0.03, "left", "bottom"), "lr": (0.97, 0.03, "right", "bottom"),
               "uc": (0.5, 0.97, "center", "top"), "cl": (0.03, 0.5, "left", "center")}
        segs, pts, rects = self.obstacles()
        for c in corners:
            x, y, ha, va = pos[c]
            t = self.ax.text(x, y, text, transform=self.ax.transAxes, ha=ha, va=va, bbox=KEYBOX,
                             linespacing=1.2, **kw)
            bb = t.get_window_extent(self.r)
            if self.collides(bb, segs, pts, rects, pad=2.5) is None:
                self.labels.append(bb)
                return t
            t.remove()
        raise RuntimeError("no free corner for key box")


def end_labels(fig, ax, items, x_text, gap_pt=10.0, leader=True):
    """Direct labels in a column at x_text (data units): each label sits at the
    height of its curve end, spread apart by at least gap_pt, with a thin
    leader to the curve end."""
    fig.canvas.draw()
    to_disp = ax.transData.transform
    inv = ax.transData.inverted().transform
    gap = gap_pt * fig.dpi / 72
    ys = sorted([(to_disp((x, y))[1], lab, x, y, c) for lab, x, y, c in items])
    adj = [ys[0][0]]
    for yd, *_ in ys[1:]:
        adj.append(max(yd, adj[-1] + gap))
    ax_bb = ax.get_window_extent()
    over = adj[-1] - (ax_bb.y1 - gap / 2)
    if over > 0:
        adj = [a - over for a in adj]
        for i in range(len(adj) - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1] - gap)
    xd = to_disp((x_text, 0))[0]
    P = Placer(fig, ax, soft=("soft",))
    segs, pts, rects = P.obstacles()
    for (yd, lab, x, y, c), ya in zip(ys, adj):
        ap = dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1.0, shrinkB=3.0) if leader else None
        # nudge vertically (up to 6 pt) until the text clears lines, markers and earlier labels
        for dy in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
            yt = inv((xd, ya + dy * fig.dpi / 72))[1]
            t = ax.annotate(lab, xy=(x, y), xytext=(x_text, yt), textcoords="data", ha="left", va="center",
                            color=c, arrowprops=ap)
            t.update_positions(P.r)
            bb = Text.get_window_extent(t, P.r)
            if P.collides(bb, segs, pts, rects, pad=0.6) is None:
                P.labels.append(bb)
                break
            t.remove()
        else:
            raise RuntimeError(f"no free end-label position for {lab!r}")


def _seg_dist(a, p, q):
    p, q, a = np.asarray(p, float), np.asarray(q, float), np.asarray(a, float)
    d = q - p
    t = 0.0 if not d.any() else float(np.clip(np.dot(a - p, d) / np.dot(d, d), 0, 1))
    return float(np.hypot(*(p + t * d - a)))


def _cross(p, q, a, c):
    def o(u, v, w):
        return np.sign((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))
    return o(p, q, a) != o(p, q, c) and o(a, c, p) != o(a, c, q)


def audit(name, fig, axes):
    """Final check: no text box overlaps a line, marker, bar or another text box."""
    fig.canvas.draw()
    bad = 0
    for ax in axes:
        P = Placer(fig, ax, soft=("soft",))
        segs, pts, rects = P.obstacles()
        texts = [t for t in ax.texts if t.get_visible() and t.get_gid() != "panel" and t.get_text().strip()]
        boxes = [Text.get_window_extent(t, P.r) for t in texts]
        for i, (t, bb) in enumerate(zip(texts, boxes)):
            P.labels = [b for j, b in enumerate(boxes) if j != i]
            why = P.collides(bb, segs, pts, rects, pad=0.5)
            if why is not None:
                bad += 1
                REPORT.append(f"  {name}: '{t.get_text()[:30]}' collides ({why})")
    REPORT.append(f"{name}: {bad} collisions")
    if bad:
        FAILED.append(name)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)


def lines_with_ci(ax, x, v, lo, hi, m, xoff=0.0, band=False, **kw):
    st = STYLE[m]
    if band:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, lw=0, gid="band")
        ax.plot(x, v, color=st["color"], lw=1.2, ls=st["ls"], **kw)
    else:
        eb = ax.errorbar(np.asarray(x) + xoff, v, yerr=[np.asarray(v) - lo, np.asarray(hi) - v], color=st["color"],
                         marker=st["marker"], ms=3.8, mfc=st["color"], mec=st["color"], capsize=1.8,
                         elinewidth=0.6, lw=1.1, **kw)
        eb[0].set_linestyle(st["ls"])


def save_fixed(fig, name):
    """Fixed page size (no tight bounding box), so figures in one row align."""
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)


def errbar_overlaps(name, fig, axes, min_gap_pt=3.0):
    """Reviewer 3: no two error bars of different series may overlap. Collects every
    vertical error-bar segment (LineCollection from errorbar) in points and flags pairs
    from different series closer than min_gap_pt horizontally with overlapping y ranges."""
    fig.canvas.draw()
    k = 72.0 / fig.dpi
    bad = 0
    for ax in axes:
        segs = []
        for sid, col in enumerate(c for c in ax.collections if isinstance(c, LineCollection)):
            for s in col.get_segments():
                d = col.get_transform().transform(np.asarray(s)) * k
                if len(d) == 2 and abs(d[0, 0] - d[1, 0]) < 0.01:
                    segs.append((sid, d[0, 0], min(d[:, 1]), max(d[:, 1])))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                if a[0] != b[0] and abs(a[1] - b[1]) < min_gap_pt and not (a[3] < b[2] or b[3] < a[2]):
                    bad += 1
    REPORT.append(f"{name}: {bad} overlapping error-bar pairs (gap < {min_gap_pt} pt)")
    if bad:
        FAILED.append(name + " (error bars)")


# ======================================================================
# figures, review round 1: column width (252 bp) at 9 pt, two per row across the page;
# frames in absolute inches so the two figures of a row align
# ======================================================================
W = 252 / 72.27
H_TALL, H_ROW = 2.50, 1.72
HATCH = {"LACP": "", "CC-LACP": "////", "CC-CP": "\\\\\\\\", "ACI": "....", "ACI-LACP": "xxxx", "SCP": "",
         "Width-matched SCP": "////"}
LEGEND_KW = dict(loc="upper center", handlelength=1.6, columnspacing=1.1, handletextpad=0.4, labelspacing=0.25,
                 borderaxespad=0.0)


def frame(fig, h, top_rows, bottom_in=0.40, left_in=0.50, right_in=0.07, hspace_in=None, nrows=1, tag=False):
    """Axes frame in inches: legend band of top_rows rows at the top."""
    top_in = 0.06 + 0.175 * top_rows + 0.06 + (0.13 if tag else 0.0)
    kw = dict(left=left_in / W, right=1 - right_in / W, bottom=bottom_in / h, top=1 - top_in / h)
    if hspace_in is not None and nrows > 1:
        ax_h = (h - top_in - bottom_in - hspace_in * (nrows - 1)) / nrows
        kw["hspace"] = hspace_in / ax_h
    fig.subplots_adjust(**kw)


def bar(ax, x, v, lo, hi, w, m, label, scale=1.0):
    st = STYLE[m]
    fc = st["color"] if m != "Width-matched SCP" else STYLE["SCP"]["color"]
    ec = "white"
    ax.bar(x, scale * v, w, color=fc, edgecolor=ec, hatch=HATCH.get(m, ""), lw=0, label=label, zorder=2)
    ax.errorbar(x, scale * v, yerr=[scale * (v - lo), scale * (hi - v)], fmt="none", ecolor="#222222",
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)


def panel_tag(ax, s):
    ax.text(0.0, 1.03, s, transform=ax.transAxes, va="bottom", ha="left", gid="panel")


def fig_households():
    d = pd.read_csv(os.path.join(DATA, "fig2_pv_loss_households.csv"))
    fig, axs = plt.subplots(2, 1, figsize=(W, H_TALL))
    order = ["LACP", "CC-LACP", "CC-CP", "ACI"]
    for ax, panel, methods, groups, ylab, scale in [
            (axs[0], "a", order, ["H3", "H4", "H6", "Pooled"], "Log width ratio", 1),
            (axs[1], "b", ["CC-LACP", "CC-CP", "ACI"], ["H3", "H4", "H6", "Mean"], "Score vs. LACP (%)", 100)]:
        wbar = 0.82 / 4
        for m in methods:
            g = d[(d.panel == panel) & (d.method == m)].set_index("household").loc[groups]
            k = order.index(m)
            bar(ax, np.arange(len(groups)) + (k - 1.5) * wbar, g.value.values, g.lo.values, g.hi.values,
                wbar * 0.92, m, m, scale)
        ax.axhline(0, color="#4D4D4D", lw=0.6, zorder=2)
        gg(ax, minor_x=False)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylab)
        panel_tag(ax, f"({panel})")
        ax.set_xlim(-0.55, len(groups) - 0.45)
    axs[0].set_ylim(-0.2, 1.6)
    axs[0].set_yticks([0, 0.5, 1.0, 1.5])
    axs[1].set_ylim(-40, 50)
    axs[1].set_yticks([-25, 0, 25])
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=2, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_TALL), **LEGEND_KW)
    axs[1].set_xlabel("Household")
    frame(fig, H_TALL, 2, bottom_in=0.40, hspace_in=0.32, nrows=2, tag=True)
    P = Placer(fig, axs[0])
    P.box("PV loss, 180 events\nper household", corners=("ul",))
    audit("fig_households", fig, axs)
    errbar_overlaps("fig_households", fig, axs)
    save_fixed(fig, "fig_households.pdf")


def fig_eventtime():
    d = pd.read_csv(os.path.join(DATA, "fig_eventtime.csv"))
    fig, axs = plt.subplots(2, 1, figsize=(W, H_TALL), sharex=True)
    order = ["LACP", "CC-LACP", "CC-CP", "ACI"]
    bars_at = {"LACP": [-45, 15, 75, 135], "CC-LACP": [-30, 30, 90, 150], "CC-CP": [-15, 45, 105, 165],
               "ACI": [0, 60, 120, 180]}
    for ax in axs:
        ax.axvspan(0, 120, color=SPAN, lw=0, zorder=0, gid="soft")
        gg(ax)
    for ax, panel in [(axs[0], "width"), (axs[1], "coverage")]:
        for m in order:
            g = d[(d.panel == panel) & (d.method == m)].sort_values("x")
            st = STYLE[m]
            ax.plot(g.x, g.value, color=st["color"], ls=st["ls"], lw=1.2, zorder=3)
            s = g[g.x.isin(bars_at[m])]
            ax.errorbar(s.x, s.value, yerr=[s.value - s.lo, s.hi - s.value], fmt=st["marker"], ms=3.6,
                        color=st["color"], mfc=st["color"], mec=st["color"], elinewidth=0.7, capsize=1.4,
                        capthick=0.7, zorder=4)
    axs[0].axhline(0, color="#4D4D4D", lw=0.6)
    axs[0].set_ylim(-0.16, 0.76)
    axs[0].set_ylabel("Log width ratio")
    panel_tag(axs[0], "(a)")
    gp = d[d.panel == "coverage_placebo"].groupby("x").value.mean()
    axs[1].plot(gp.index, gp.values, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=2)
    axs[1].axhline(0.9, color="#1A1A1A", lw=0.6, ls=(0, (1, 1.5)))
    axs[1].set_ylabel("Coverage")
    axs[1].set_ylim(0.42, 0.99)
    axs[1].set_xlabel("Minutes since onset")
    panel_tag(axs[1], "(b)")
    for a in axs:
        a.set_xlim(-62, 182)
        a.set_xticks([-60, 0, 60, 120, 180])
    h = [plt.Line2D([], [], color=STYLE[m]["color"], lw=1.2, ls=STYLE[m]["ls"], marker=STYLE[m]["marker"], ms=3.6)
         for m in order]
    h += [plt.Line2D([], [], color="#555555", lw=0.9, ls=(0, (3, 2))),
          plt.Line2D([], [], color="#1A1A1A", lw=0.6, ls=(0, (1, 1.5)))]
    fig.legend(h, order + ["no injection", "0.90"], ncol=3, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_TALL),
               **LEGEND_KW)
    frame(fig, H_TALL, 2, bottom_in=0.40, hspace_in=0.32, nrows=2, tag=True)
    audit("fig_eventtime", fig, axs)
    errbar_overlaps("fig_eventtime", fig, axs)
    save_fixed(fig, "fig_eventtime.pdf")


def fig_severity():
    d = pd.read_csv(os.path.join(DATA, "fig4_pv_severity.csv"))
    fig, ax = plt.subplots(figsize=(W, H_ROW))
    order = ["LACP", "CC-LACP", "CC-CP", "ACI"]
    off = dict(zip(order, (-0.18, -0.06, 0.06, 0.18)))
    for m in order:
        g = d[d.method == m].sort_values("severity")
        x = np.arange(3) + off[m]
        st = STYLE[m]
        eb = ax.errorbar(x, g.value, yerr=[g.value - g.lo, g.hi - g.value], color=st["color"], marker=st["marker"],
                         ms=3.8, mfc=st["color"], mec=st["color"], capsize=1.6, capthick=0.7, elinewidth=0.7,
                         lw=1.1, zorder=3)
        eb[0].set_linestyle(st["ls"])
    ax.axhline(0, color="#4D4D4D", lw=0.6)
    gg(ax, minor_x=False)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["25", "50", "75"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.16, 0.92)
    ax.set_xlabel(r"PV-loss severity $s$ (%)")
    ax.set_ylabel("Log width ratio")
    h = [plt.Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], marker=STYLE[m]["marker"], ms=3.8, lw=1.1)
         for m in order]
    fig.legend(h, order, ncol=4, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_ROW), **LEGEND_KW)
    frame(fig, H_ROW, 1, bottom_in=0.40)
    P = Placer(fig, ax, soft=("soft",))
    P.box("PV loss, $d$ and $h$ pooled,\n180 events per severity", corners=("ul", "uc"))
    audit("fig_severity", fig, [ax])
    errbar_overlaps("fig_severity", fig, [ax])
    save_fixed(fig, "fig_severity.pdf")


def fig_bands():
    d = pd.read_csv(os.path.join(DATA, "fig_bands.csv"))
    methods = ["SCP", "Width-matched SCP", "LACP", "CC-CP", "CC-LACP", "ACI"]
    bands = ["zero", "moderate", "high"]
    fig, ax = plt.subplots(figsize=(W, H_ROW))
    n = len(methods)
    wbar = 0.86 / n
    for i, m in enumerate(methods):
        g = d[d.method == m].set_index("band").loc[bands]
        bar(ax, np.arange(3) + (i - (n - 1) / 2) * wbar, g.picp.values, g.lo.values, g.hi.values, wbar * 0.92, m,
            "Width-matched" if m == "Width-matched SCP" else m)
    ax.axhline(0.9, color="#1A1A1A", lw=0.7, ls=(0, (1, 1.5)), zorder=2)
    gg(ax, minor_x=False)
    sh = d[d.method == "SCP"].set_index("band").loc[bands, "share"]
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"$\\tilde\\rho_t=0$\n({100 * sh['zero']:.0f}% of targets)",
                        f"moderate\n({100 * sh['moderate']:.0f}%)", f"high\n({100 * sh['high']:.0f}%)"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.72, 0.98)
    ax.set_ylabel("Coverage")
    ax.set_xlim(-0.5, 2.5)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_ROW), **LEGEND_KW)
    frame(fig, H_ROW, 2, bottom_in=0.36)
    P = Placer(fig, ax, soft=("soft",))
    P.place("target 0.90", (1.5, 0.9), "#1A1A1A",
            cands=[(dx, dy) for dy in (3, 5) for dx in (0, -20, 20, -40, 40, -60, 60)])
    audit("fig_bands", fig, [ax])
    errbar_overlaps("fig_bands", fig, [ax])
    save_fixed(fig, "fig_bands.pdf")


def fig_tradeoff():
    d = pd.read_csv(os.path.join(DATA, "fig3_parameter_sweep.csv"))
    XL, YL = (-2.5, 18.5), (-6.0, 22.0)
    fig, ax = plt.subplots(figsize=(W, H_ROW))
    fams = ["CC-CP", "ACI", "ACI-CC", "CC-LACP"]
    fam_lab = {"CC-CP": r"CC-CP ($\beta$)", "ACI": r"ACI ($\gamma$)",
               "ACI-CC": r"ACI on CC-CP ($\gamma$)", "CC-LACP": r"CC-LACP ($\beta$)"}
    for fam in fams:
        g = d[d.family == fam].sort_values("param")
        x, y = 100 * g.clean_rel.values, 100 * g.inside_rel.values
        st = STYLE[fam]
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.1, zorder=3, clip_on=True)
        inside = (x >= XL[0]) & (x <= XL[1]) & (y >= YL[0]) & (y <= YL[1])
        ax.plot(x[inside], y[inside], ls="none", marker=st["marker"], ms=3.8, mfc="white", mec=st["color"],
                mew=0.9, zorder=4)
        s = g[g.selected]
        ax.plot(100 * s.clean_rel, 100 * s.inside_rel, ls="none", marker=st["marker"], ms=5.6, mfc=st["color"],
                mec="#111111", mew=0.6, zorder=5)
    ax.plot([0], [0], ls="none", marker="o", ms=5.6, mfc=STYLE["LACP"]["color"], mec="#111111", mew=0.6, zorder=6)
    ax.axhline(0, color="#4D4D4D", lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.axvline(0, color="#4D4D4D", lw=0.5, ls=(0, (2, 2)), zorder=1)
    gg(ax)
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_xlabel("Normal-operation score vs. LACP (%)")
    ax.set_ylabel("In-event score vs. LACP (%)")
    h = [plt.Line2D([], [], color=STYLE[f]["color"], ls=STYLE[f]["ls"], lw=1.1, marker=STYLE[f]["marker"], ms=3.8,
                    mfc="white", mec=STYLE[f]["color"], mew=0.9) for f in ["CC-CP", "CC-LACP", "ACI", "ACI-CC"]]
    h += [plt.Line2D([], [], ls="none", marker="s", ms=5.6, mfc="#9A9A9A", mec="#111111", mew=0.6),
          plt.Line2D([], [], ls="none", marker="o", ms=5.6, mfc=STYLE["LACP"]["color"], mec="#111111", mew=0.6)]
    labs = [fam_lab[f] for f in ["CC-CP", "CC-LACP", "ACI", "ACI-CC"]] + ["selected", "LACP"]
    fig.legend([h[0], h[1], h[4], h[2], h[3], h[5]], [labs[0], labs[1], labs[4], labs[2], labs[3], labs[5]],
               ncol=3, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_ROW), **{**LEGEND_KW, "columnspacing": 0.7})
    frame(fig, H_ROW, 2, bottom_in=0.40)
    for fam in fams:
        g = d[d.family == fam].sort_values("param")
        x, y = 100 * g.clean_rel.values, 100 * g.inside_rel.values
        out = ~((x >= XL[0]) & (x <= XL[1]) & (y >= YL[0]) & (y <= YL[1]))
        if not out.any():
            continue
        k = int(np.argmax(out))
        x0, y0, x1, y1 = x[k - 1], y[k - 1], x[k], y[k]
        ts = [(XL[1] - x0) / (x1 - x0)] if x1 > XL[1] else []
        ts += [(YL[1] - y0) / (y1 - y0)] if y1 > YL[1] else []
        t = min(ts) * 0.985
        xe, ye = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        ax.annotate("", xy=(xe, ye), xytext=(xe - 0.06 * (x1 - x0), ye - 0.06 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color=STYLE[fam]["color"], lw=1.0, mutation_scale=7))
    audit("fig_tradeoff", fig, [ax])
    save_fixed(fig, "fig_tradeoff.pdf")


def fig_horizon():
    d = pd.read_csv(os.path.join(DATA, "fig_horizon.csv"))
    d = d[d.types == "all"]
    methods = ["CC-LACP", "CC-CP", "ACI", "ACI-LACP"]
    hz = [15, 30, 60]
    fig, ax = plt.subplots(figsize=(W, H_ROW))
    wbar = 0.8 / len(methods)
    for i, m in enumerate(methods):
        g = d[d.method == m].set_index("horizon").loc[hz]
        bar(ax, np.arange(len(hz)) + (i - (len(methods) - 1) / 2) * wbar, g.value.values, g.lo.values,
            g.hi.values, wbar * 0.92, m, SHOW.get(m, m), scale=100)
    ax.axhline(0, color="#4D4D4D", lw=0.6, zorder=2)
    gg(ax, minor_x=False)
    ax.set_xticks(range(len(hz)))
    ax.set_xticklabels([f"{h} min" for h in hz])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(hz) - 0.5)
    ax.set_ylim(-10, 29)
    ax.set_yticks([-10, 0, 10, 20])
    ax.set_xlabel("Forecast horizon $h$")
    ax.set_ylabel("In-event score vs. LACP (%)")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, ncol=2, bbox_to_anchor=(0.5 + 0.215 / W, 1 - 0.06 / H_ROW), **LEGEND_KW)
    frame(fig, H_ROW, 2, bottom_in=0.40)
    audit("fig_horizon", fig, [ax])
    errbar_overlaps("fig_horizon", fig, [ax])
    save_fixed(fig, "fig_horizon.pdf")


def selftest_errbar():
    """The overlap check must flag two error bars of different series at the same x."""
    fig, ax = plt.subplots(figsize=(W, H_ROW))
    ax.errorbar([1.0], [1.0], yerr=[[0.2], [0.2]], fmt="none")
    ax.errorbar([1.01], [1.05], yerr=[[0.2], [0.2]], fmt="none")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    n0 = len(FAILED)
    errbar_overlaps("selftest", fig, [ax])
    plt.close(fig)
    assert len(FAILED) == n0 + 1, "overlap check missed a known overlap"
    FAILED.pop()
    REPORT.pop()
    REPORT.append("self-test: overlap check flags a known overlap")


if __name__ == "__main__":
    selftest_errbar()
    fig_households()
    fig_eventtime()
    fig_severity()
    fig_bands()
    fig_tradeoff()
    fig_horizon()
    open(os.path.join(HERE, "label_check.txt"), "w").write("\n".join(REPORT) + "\n")
    print("\n".join(REPORT))
    if FAILED:
        raise SystemExit(f"label collisions in {FAILED}")
