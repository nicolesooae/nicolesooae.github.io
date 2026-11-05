"""
Plotting helpers for SYDE 572 Assignment 3. These only draw: pca.py computes the numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

CLASS_COLORS = ["#2f5bd3", "#d9602b", "#2a9d6f"]
MARKERS = ["o", "s", "^"]
POINT = "#1c1b19"
MUTED = "#8a877f"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#c9c6bd",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ebe9e3",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, Z, y, classes, **kw):
    for i, c in enumerate(classes):
        ax.scatter(*[Z[y == i, j] for j in range(Z.shape[1])], s=kw.get("s", 22), color=CLASS_COLORS[i],
                   marker=MARKERS[i], alpha=0.8, edgecolor="white", linewidth=0.4, label=c)


# ---------- Part 1 ----------
def plot_part1(p1, fname):
    D, mu, w = p1["data"], p1["mean"], p1["w"]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    t = np.linspace(-5, 6, 2)
    ax.plot(mu[0] + t * w[0], mu[1] + t * w[1], color=MUTED, lw=1.5, ls="--", label="first PC through the mean")
    ax.scatter(D[:, 0], D[:, 1], s=50, color=CLASS_COLORS[0], zorder=3, label="data")
    ax.scatter(*mu, s=90, marker="x", color=POINT, zorder=4, label="mean (0, 1)")
    for (p, (z, rec, err)), col in zip(p1["points"].items(), [CLASS_COLORS[1], CLASS_COLORS[2]]):
        ax.scatter(*p, s=60, color=col, marker="s", zorder=4, label=f"{p}: error {err:.3f}")
        ax.scatter(*rec, s=60, facecolor="white", edgecolor=col, lw=2, zorder=4)
        ax.plot([p[0], rec[0]], [p[1], rec[1]], color=col, lw=2)
    ax.set_xlim(-3, 4.5)
    ax.set_ylim(-2, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Part 1: first PC and reconstructions")
    ax.legend(loc="upper left")
    _save(fig, fname)


# ---------- Part 2 ----------
def plot_1d(X, y, names, classes, fname):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for j, ax in enumerate(axes.ravel()):
        bins = np.linspace(X[:, j].min(), X[:, j].max(), 26)
        for i, c in enumerate(classes):
            ax.hist(X[y == i, j], bins=bins, color=CLASS_COLORS[i], alpha=0.55, label=c)
        ax.set_xlabel(f"{names[j]} (cm)")
        ax.set_ylabel("count")
        ax.set_title(names[j].capitalize())
    axes[0, 0].legend()
    _save(fig, fname)


def plot_2d(X, y, names, classes, fname):
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for (a, b), ax in zip(pairs, axes.ravel()):
        _scatter(ax, X[:, [a, b]], y, classes)
        ax.set_xlabel(f"{names[a]} (cm)")
        ax.set_ylabel(f"{names[b]} (cm)")
    axes[0, 0].legend()
    _save(fig, fname)


def plot_3d(X, y, names, classes, triples, fname):
    fig = plt.figure(figsize=(13, 6))
    for k, t in enumerate(triples):
        ax = fig.add_subplot(1, len(triples), k + 1, projection="3d")
        _scatter(ax, X[:, list(t)], y, classes, s=26)
        ax.set_xlabel(names[t[0]])
        ax.set_ylabel(names[t[1]])
        ax.set_zlabel(names[t[2]])
        ax.view_init(elev=22, azim=-60)
        ax.set_box_aspect(None, zoom=0.85)          # keep the z label inside the figure
        ax.set_title(" / ".join(names[i] for i in t), fontsize=11)
    fig.axes[0].legend(loc="upper left")
    _save(fig, fname)


# ---------- Part 3 ----------
def plot_pc1(Z, y, classes, fname):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    bins = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 31)
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(Z))
    for i, c in enumerate(classes):
        ax.hist(Z[y == i, 0], bins=bins, color=CLASS_COLORS[i], alpha=0.55, label=c)
        ax2.scatter(Z[y == i, 0], i + jitter[y == i], s=14, color=CLASS_COLORS[i], marker=MARKERS[i], alpha=0.8)
    ax.set_ylabel("count")
    ax.set_title("Iris projected onto the first principal component")
    ax.legend()
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(classes)
    ax2.set_xlabel("PC1 score (cm)")
    _save(fig, fname)


def plot_pc12(Z, y, classes, fname):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _scatter(ax, Z[:, :2], y, classes, s=30)
    ax.set_xlabel("PC1 (cm)")
    ax.set_ylabel("PC2 (cm)")
    ax.set_title("Iris on the top two principal components")
    ax.legend()
    _save(fig, fname)


def plot_pc123(Z, y, classes, fname):
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    _scatter(ax, Z[:, :3], y, classes, s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=20, azim=-55)
    ax.set_box_aspect(None, zoom=0.82)
    ax.set_title("Iris on the top three principal components")
    ax.legend(loc="upper left")
    _save(fig, fname)


# ---------- Part 4 ----------
def plot_scree(cum_frac, fname):
    cum = 100 * np.array(cum_frac)
    each = np.diff(np.concatenate([[0], cum]))
    k = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(k, each, color=CLASS_COLORS[0], alpha=0.75, width=0.55, label="variance explained by each PC")
    ax.plot(k, cum, color=CLASS_COLORS[1], lw=2, marker="o", label="cumulative")
    ax.axhline(50, color=MUTED, ls="--", lw=1.2, label="50%")
    for x, v in zip(k, cum):
        ax.annotate(f"{v:.2f}%", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(k)
    ax.set_ylim(0, 108)
    ax.set_xlabel("number of principal components $k$")
    ax.set_ylabel("% of total variance")
    ax.set_title("Variance explained (from the Frobenius norm)")
    ax.legend(loc="center right")
    _save(fig, fname)


# ---------- Part 5 ----------
def plot_isomap(Z, y, classes, fname):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _scatter(ax, Z, y, classes, s=30)
    ax.set_xlabel("Isomap component 1")
    ax.set_ylabel("Isomap component 2")
    ax.set_title("Isomap embedding, 3 nearest neighbours")
    ax.legend()
    _save(fig, fname)


def plot_compare(Zs, y, classes, titles, fname):
    fig, axes = plt.subplots(1, len(Zs), figsize=(15, 4.8))
    for Z, ax, t in zip(Zs, axes, titles):
        _scatter(ax, Z, y, classes)
        ax.set_title(t, fontsize=12)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    axes[0].legend()
    _save(fig, fname)


def plot_residual(rows, fname):
    d = [r[0] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax.plot(d, [r[1] for r in rows], color=CLASS_COLORS[0], lw=2, marker="o", label="PCA vs Euclidean distances")
    ax.plot(d, [r[4] for r in rows], color=CLASS_COLORS[1], lw=2, marker="s", label="Isomap vs geodesic distances")
    ax.plot(d, [r[3] for r in rows], color=CLASS_COLORS[1], lw=1.5, ls="--", marker="s", mfc="white",
            label="Isomap vs Euclidean distances")
    ax.plot(d, [r[2] for r in rows], color=CLASS_COLORS[0], lw=1.5, ls="--", marker="o", mfc="white",
            label="PCA vs geodesic distances")
    ax.set_xticks(d)
    ax.set_xlabel("embedding dimension $d$")
    ax.set_ylabel("residual variance $1 - R^2$")
    ax.set_title("Distance preservation")
    ax.legend()
    ax2.plot(d, [100 * r[5] for r in rows], color=CLASS_COLORS[0], lw=2, marker="o", label="PCA")
    ax2.plot(d, [100 * r[6] for r in rows], color=CLASS_COLORS[1], lw=2, marker="s", label="Isomap, k = 3")
    ax2.set_xticks(d)
    ax2.set_ylim(50, 100)                          # one flower = 0.67 points: keep small differences small
    ax2.set_xlabel("embedding dimension $d$")
    ax2.set_ylabel("5-NN accuracy, 10-fold CV (%)")
    ax2.set_title("Classification")
    ax2.legend()
    _save(fig, fname)
