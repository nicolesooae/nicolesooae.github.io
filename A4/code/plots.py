"""
Plotting helpers for SYDE 572 Assignment 4. These only draw: naive_bayes.py computes the numbers.
"""

import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#2f5bd3", "#d9602b", "#2a9d6f"
CLASS_COLORS = [BLUE, ORANGE, GREEN]
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


# ---------- evaluation ----------
def plot_confusion(C, names, title, fname):
    C = np.asarray(C)
    size = 3.6 + 0.55 * len(names)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    R = C / C.sum(axis=1, keepdims=True)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]}\n{100 * R[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if R[i, j] > 0.55 else POINT)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30 if len(names) > 2 else 0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("guessed label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    ax.grid(False)
    _save(fig, fname)


def plot_alpha(cv, test, fname):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    a = [r[0] for r in cv]
    ax.plot(a, [100 * r[1] for r in cv], color=BLUE, lw=2, marker="o", label="5-fold CV on the training set")
    ax.plot(a, [100 * r[1] for r in test], color=ORANGE, lw=1.5, ls="--", marker="s", mfc="white",
            label="test set (for reference only)")
    best = max(cv, key=lambda r: r[1])
    ax.scatter([best[0]], [100 * best[1]], s=140, facecolor="none", edgecolor=POINT, lw=1.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"smoothing parameter $\alpha$")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Effect of additive smoothing")
    ax.legend(loc="lower left")
    _save(fig, fname)


def plot_margins(margins, wrong, lens, fname):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.linspace(0, np.percentile(margins, 99), 40)
    ax.hist(margins[~wrong], bins=bins, color=BLUE, alpha=0.75, label="correct")
    ax.hist(margins[wrong], bins=bins, color=ORANGE, alpha=0.9, label="misclassified")
    ax.set_xlabel(r"confidence $|\log P(\mathrm{pos},d) - \log P(\mathrm{neg},d)|$")
    ax.set_ylabel("test reviews")
    ax.set_title("Errors concentrate at low confidence")
    ax.legend()
    edges = [0, 25, 50, 75, 100, 150, 200, 300, 500, 2500]
    rate, n, mid = [], [], []
    for lo, hi in zip(edges, edges[1:]):
        sel = (lens >= lo) & (lens < hi)
        rate.append(100 * wrong[sel].mean())
        n.append(sel.sum())
        mid.append(f"{lo}–{hi}" if hi < 2500 else f"{lo}+")
    ax2.bar(range(len(rate)), rate, color=ORANGE, alpha=0.85)
    for i, (r, k) in enumerate(zip(rate, n)):
        ax2.text(i, r + 0.4, f"n={k}", ha="center", fontsize=8, color=MUTED)
    ax2.set_xticks(range(len(rate)))
    ax2.set_xticklabels(mid, rotation=35)
    ax2.set_xlabel("review length (tokens)")
    ax2.set_ylabel("error rate (%)")
    ax2.set_title("Error rate by review length")
    _save(fig, fname)


# ---------- optional tasks ----------
def plot_learning_curve(rows, fname):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n = [r[1] for r in rows]
    for col, color, marker, label in [(2, BLUE, "o", "Naive Bayes (mine), α = 1"),
                                      (3, ORANGE, "s", "TF-IDF + LinearSVC (scikit-learn)")]:
        acc = np.array([100 * r[col][0] for r in rows])
        lo = np.array([100 * r[col][1] for r in rows])
        hi = np.array([100 * r[col][2] for r in rows])
        ax.fill_between(n, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(n, acc, color=color, lw=2, marker=marker, ms=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("training reviews (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Learning curve (bands: 95% CI)")
    ax.legend(loc="lower right")
    _save(fig, fname)


def plot_features(model, top, fname):
    """Each word: total count (x, log) against its log-likelihood ratio (y); the strongest words are labelled."""
    pos, neg = top
    words = [w for w in model["vocab"] if model["counts"]["pos"][w] + model["counts"]["neg"][w] >= 5]
    freq = np.array([model["counts"]["pos"][w] + model["counts"]["neg"][w] for w in words])
    llr = np.array([model["log_p"]["pos"][w] - model["log_p"]["neg"][w] for w in words])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.scatter(freq, llr, s=4, c=np.where(llr > 0, BLUE, ORANGE), alpha=0.25, lw=0)
    show = dict(pos[:10] + neg[:10])
    labelled = sorted([(f, v, w) for w, f, v in zip(words, freq, llr) if w in show])
    for i, (f, v, w) in enumerate(labelled):
        c = BLUE if v > 0 else ORANGE
        ax.scatter([f], [v], s=18, color=c, zorder=3)
        dy = (14 if v > 0 else -14) * (1 + i % 3)             # stagger labels so neighbours do not collide
        ax.annotate(w, (f, v), xytext=(0, dy), textcoords="offset points", fontsize=8.5, color=c, ha="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(llr.min() - 0.3, llr.max() + 0.3)
    ax.set_xscale("log")
    ax.set_xlabel("occurrences in the training reviews (log scale)")
    ax.set_ylabel(r"$\log P(w\mid\mathrm{pos}) - \log P(w\mid\mathrm{neg})$")
    ax.set_title(r"Every word with $\geq 5$ occurrences ($\alpha = %g$)" % model["alpha"])
    k = len(pos)
    ax2.barh(np.arange(k), [v for _, v in pos][::-1], color=BLUE, alpha=0.85)
    ax2.barh(np.arange(k, 2 * k) + 0.6, [v for _, v in neg], color=ORANGE, alpha=0.85)
    ax2.set_yticks(list(np.arange(k)) + list(np.arange(k, 2 * k) + 0.6))
    ax2.set_yticklabels([w for w, _ in pos][::-1] + [w for w, _ in neg], fontsize=9)
    ax2.axvline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel("log-likelihood ratio")
    ax2.set_title(f"Top {k} positive (blue) and negative (orange) words")
    ax2.grid(axis="y", visible=False)
    _save(fig, fname)


def plot_domain(names, M, fname):
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(100 * M, cmap="Blues", vmin=60, vmax=90)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{100 * M[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal", color="white" if M[i, j] > 0.8 else POINT)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("tested on (20% of that topic)")
    ax.set_ylabel("trained on (80% of that topic)")
    ax.set_title("Sentiment accuracy (%) across domains")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, fname)


def plot_gnb(X, Y, model, fname):
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for j, ax in enumerate(axes):
        xs = np.linspace(X[:, j].min() - 0.5, X[:, j].max() + 0.5, 300)
        for c, color in zip(model["classes"], CLASS_COLORS):
            mu, sd = model["mean"][c][j], np.sqrt(model["var"][c][j])
            ax.hist(X[Y == c, j], bins=np.linspace(xs[0], xs[-1], 30), density=True, color=color, alpha=0.25)
            ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd * sd)) / (sd * np.sqrt(2 * np.pi)), color=color, lw=2,
                    label=c)
        ax.set_xlabel(f"{names[j]} (cm)")
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("Gaussian Naive Bayes on Iris: the 12 fitted class-conditional normals", fontweight="semibold")
    _save(fig, fname)
