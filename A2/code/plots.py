"""
Plotting helpers for SYDE 572 Assignment 2. These only draw: regression.py computes the numbers.
"""

import numpy as np
import matplotlib.pyplot as plt

TRAIN = "#2f5bd3"
TEST = "#d9602b"
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


def plot_error_curve(xs, train, test, xlabel, title, fname, logy=False):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, train, color=TRAIN, lw=2, marker="o", label="training error")
    ax.plot(xs, test, color=TEST, lw=2, marker="s", label="test error")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMS error ($1000s)" + (", log scale" if logy else ""))
    ax.set_title(title)
    ax.legend()
    _save(fig, fname)


def plot_poly_1d(grid, curve, xtr, ytr, xte, yte, degree, tr_err, te_err, fname):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(xtr, ytr, s=28, color=TRAIN, label="training points")
    ax.scatter(xte, yte, s=14, color=TEST, alpha=0.6, label="test points")
    ax.plot(grid, curve, color=POINT, lw=2, label=f"degree {degree} fit")
    ax.set_ylim(min(ytr.min(), yte.min()) - 5, max(ytr.max(), yte.max()) + 5)
    ax.set_xlabel("2nd feature (normalized)")
    ax.set_ylabel("median house value ($1000s)")
    ax.set_title(f"Degree {degree}:  train RMS {tr_err:.2f},  test RMS {te_err:.2f}")
    ax.legend()
    _save(fig, fname)


def plot_cv(lambdas, err, title, fname):
    """Validation error vs lambda on a log x axis ; lambda = 0 can't be on a log axis, so its error goes in the legend."""
    lambdas, err = np.array(lambdas, float), np.array(err)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.semilogx(lambdas[1:], err[1:], color=TRAIN, lw=2, marker="o", label="regularized")
    ax.plot([], [], " ", label=f"$\\lambda = 0$ (off scale): {err[0]:.4g}")
    best = np.argmin(err)
    ax.scatter([lambdas[best] or np.nan], [err[best]], s=140, facecolor="none", edgecolor=TEST, lw=2,
               label=f"best: $\\lambda = {lambdas[best]:g}$")
    ax.set_xlabel("regularizing constant $\\lambda$")
    ax.set_ylabel("average validation RMS error ($1000s)")
    ax.set_title(title)
    ax.legend()
    _save(fig, fname)
