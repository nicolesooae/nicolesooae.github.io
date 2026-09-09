"""
Plotting helpers for SYDE 572 Assignment 1. These only draw: the solvers in
distance.py and fitting.py pass in the steps they recorded.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CURVE = "#2f5bd3"      # the function / the exact answer
POINT = "#1c1b19"      # query points, data points, final answer
MUTED = "#8a877f"      # guides and secondary text
ITER_CMAP = "Oranges"  # iterates: light (early) -> dark (late)


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
        "mathtext.fontset": "cm",
    })


def _iter_colors(n):
    cmap = plt.get_cmap(ITER_CMAP)
    return [cmap(v) for v in np.linspace(0.35, 0.95, max(n, 1))]


def _save(fig, fname, title):
    fig.suptitle(title, fontsize=15, fontweight="semibold", y=1.0)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Part 1: distance from a point to a curve
# ==================================================================
def _dist_sq(f, x0, y0):
    return lambda x: (x - x0) ** 2 + (f(x) - y0) ** 2


def _num_deriv(g, x, h=1e-5):
    """Central difference, used only to draw tangent lines."""
    return (g(x + h) - g(x - h)) / (2 * h)


def _result_box(ax, f, x0, y0, x_star):
    y_star = f(x_star)
    d = np.hypot(x_star - x0, y_star - y0)
    ax.text(0.5, -0.30, f"closest point ({x_star:.4f}, {y_star:.4f})    shortest distance d = {d:.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#dcd9d0"))


def _curve_panel(ax, f, x0, y0, xs, eq, pad=1.5):
    """Curve and query point, framed around the point and the given x values."""
    xs = np.asarray(xs, float)
    lo, hi = min(x0, xs.min()) - pad, max(x0, xs.max()) + pad
    ys = np.append(f(xs), y0)
    ylo, yhi = ys.min() - pad, ys.max() + pad
    grid = np.linspace(lo, hi, 800)
    yg = f(grid)
    yg = np.where((yg > yhi + 5 * pad) | (yg < ylo - 5 * pad), np.nan, yg)
    ax.plot(grid, yg, color=CURVE, lw=2.2, label=eq, zorder=2)
    ax.scatter([x0], [y0], s=70, color=POINT, zorder=6, label=f"point $({x0:g},\\,{y0:g})$")
    ax.annotate(f"$({x0:g},\\,{y0:g})$", (x0, y0), xytext=(8, -16), textcoords="offset points")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="box")   # so the shortest segment looks perpendicular


def plot_newton(f, x0, y0, nr_xs, eq, fname, max_labels=6):
    """Left: segment from the point to the curve at each iterate x_n.
    Right: D'(x) and the tangent used at each step; it crosses zero at the next iterate."""
    nr_xs = np.asarray(nr_xs, float)
    Dp = lambda x: _num_deriv(_dist_sq(f, x0, y0), x)
    cols = _iter_colors(len(nr_xs))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})

    _curve_panel(axL, f, x0, y0, nr_xs, eq)
    last = np.inf
    for n, (x, c) in enumerate(zip(nr_xs, cols)):
        final = n == len(nr_xs) - 1
        axL.plot([x0, x], [y0, f(x)], ls="-" if final else "--", lw=2.2 if final else 1.2,
                 color=POINT if final else c, zorder=3)
        axL.scatter([x], [f(x)], s=46, color=c, edgecolor="white", lw=1.2, zorder=5)
        if n < max_labels and abs(x - last) > 0.08 * np.ptp(axL.get_xlim()):
            last = x
            axL.annotate(f"$x_{{{n}}}$", (x, f(x)), xytext=(6, 6), textcoords="offset points", color="#6b3a12")
    axL.scatter([], [], color=cols[len(cols) // 2], label="iterates $x_n$ (light → dark)")
    axL.set_title(f"Iterates on the curve  ({len(nr_xs) - 1} steps)")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    _result_box(axL, f, x0, y0, nr_xs[-1])

    span = max(np.ptp(nr_xs), 0.2)
    grid = np.linspace(nr_xs.min() - 0.35 * span, nr_xs.max() + 0.35 * span, 600)
    axR.plot(grid, Dp(grid), color=CURVE, lw=2.2, label="$D'(x)$")
    axR.axhline(0, color=MUTED, lw=1)
    last = np.inf
    for n in range(len(nr_xs) - 1):
        x, c = nr_xs[n], cols[n]
        seg = np.array([x, nr_xs[n + 1]])
        axR.plot(seg, Dp(x) + _num_deriv(Dp, x) * (seg - x), color=c, lw=1.4, ls="--")
        axR.plot([x, x], [0, Dp(x)], color=c, lw=0.9, ls=":")
        axR.scatter([x], [Dp(x)], s=40, color=c, edgecolor="white", lw=1, zorder=5)
        if n < max_labels and abs(x - last) > 0.06 * span and abs(x - nr_xs[-1]) > 0.06 * span:
            last = x
            axR.annotate(f"$x_{{{n}}}$", (x, 0), xytext=(0, -16), textcoords="offset points",
                         ha="center", color="#6b3a12")
    axR.scatter([nr_xs[-1]], [0], s=70, marker="*", color=POINT, zorder=6,
                label=f"root $x^* \\approx {nr_xs[-1]:.4f}$")
    axR.plot([], [], color=cols[0], ls="--", label="tangent at $x_n$")
    axR.set_xlabel("$x$")
    axR.set_ylabel("$D'(x)$")
    axR.set_title(r"Each step: follow the tangent of $D'$ to zero")
    axR.legend(loc="upper left")

    _save(fig, fname, f"Shortest distance from $({x0:g},\\,{y0:g})$ to {eq}")


def plot_golden(f, x0, y0, brackets, eq, fname, n_show=8):
    """Left: search interval on the curve. Middle: D(x) with the first n_show brackets
    stacked underneath. Right: bracket width per iteration (x0.618 each step)."""
    D = _dist_sq(f, x0, y0)
    a0, b0 = brackets[0]
    x_star = sum(brackets[-1]) / 2
    shown = brackets[:n_show]
    cols = _iter_colors(len(shown))
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 0.85]})

    _curve_panel(axL, f, x0, y0, [a0, b0], eq, pad=1.0)
    axL.plot([x0, x_star], [y0, f(x_star)], color=POINT, lw=2.2, zorder=3)
    axL.scatter([x_star], [f(x_star)], s=60, color=POINT, marker="*", zorder=6)
    axL.axvspan(a0, b0, color=cols[0], alpha=0.18, lw=0, label=f"initial bracket [{a0:.3g}, {b0:.3g}]")
    axL.set_title("Search interval on the curve")
    axL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    _result_box(axL, f, x0, y0, x_star)

    grid = np.linspace(a0, b0, 600)
    Dg = D(grid)
    axM.plot(grid, Dg, color=CURVE, lw=2.2)
    h = np.ptp(Dg) or 1.0
    step, base = 0.09 * h, Dg.min() - 0.12 * h
    for n, ((a, b), c) in enumerate(zip(shown, cols)):
        yb = base - n * step
        axM.plot([a, b], [yb, yb], color=c, lw=5, solid_capstyle="round")
        axM.text(b + 0.01 * (b0 - a0), yb, f"  {n}", va="center", fontsize=9, color="#6b3a12")
        for x in (a, b):
            axM.plot([x, x], [yb, D(x)], color=c, lw=0.6, ls=":", alpha=0.7)
    axM.axvline(x_star, color=POINT, lw=1, ls="--")
    axM.annotate(f"$x^* \\approx {x_star:.4f}$", (x_star, Dg.max()), xytext=(6, -4), textcoords="offset points")
    axM.set_ylim(base - (len(shown) + 0.5) * step, Dg.max() + 0.08 * h)
    axM.set_xlabel("$x$")
    axM.set_ylabel("$D(x) = (x - x_0)^2 + (f(x) - y_0)^2$")
    axM.set_title(f"Bracket $[a, b]$ for the first {len(shown)} iterations")

    widths = [b - a for a, b in brackets]
    axR.semilogy(widths, color=CURVE, lw=2, marker="o", ms=4)
    axR.set_xlabel("iteration")
    axR.set_ylabel("bracket width $b - a$")
    axR.set_title(f"Bracket width  ({len(widths) - 1} steps, ×0.618 each)")

    _save(fig, fname, f"Golden section search: $({x0:g},\\,{y0:g})$ to {eq}")


def plot_setup(f, points, eq, fname):
    """The curve and all query points, before solving anything."""
    xs, ys = zip(*points)
    lo, hi = min(xs) - 1.5, max(xs) + 1.5
    ylo = min(ys) - 2
    yhi = ylo + 0.6 * (hi - lo)
    grid = np.linspace(lo, hi, 800)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid, f(grid), color=CURVE, lw=2.2, label=eq)
    ax.scatter(xs, ys, s=60, color=POINT, zorder=5, label="query points")
    for px, py in points:
        ax.annotate(f"$({px:g},\\,{py:g})$", (px, py), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Problem setup")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_results(f, points, x_stars, eq, fname):
    """Every point joined to its closest point on the curve."""
    x_stars = np.asarray(x_stars, float)
    xs = [p[0] for p in points] + list(x_stars)
    lo, hi = min(xs) - 1.5, max(xs) + 1.5
    ylo = min(min(p[1] for p in points), f(x_stars).min()) - 2
    yhi = max(max(p[1] for p in points), f(x_stars).max()) + 3
    grid = np.linspace(lo, hi, 800)
    yg = f(grid)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid, np.where(yg > yhi + 10, np.nan, yg), color=CURVE, lw=2.2, label=eq, zorder=2)
    for (px, py), xs_ in zip(points, x_stars):
        ys_ = f(xs_)
        ax.plot([px, xs_], [py, ys_], color=POINT, lw=1.4, ls="--", zorder=3)
        ax.scatter([px], [py], s=55, color=POINT, zorder=5)
        ax.scatter([xs_], [ys_], s=55, color="white", edgecolor=POINT, lw=1.6, zorder=6)
        ax.annotate(f"$({px:g},\\,{py:g})$\n$d={np.hypot(xs_ - px, ys_ - py):.3f}$", (px, py),
                    xytext=(0, -14), textcoords="offset points", ha="center", va="top", fontsize=10)
    ax.scatter([], [], color=POINT, label="query point")
    ax.scatter([], [], color="white", edgecolor=POINT, label="closest point on curve")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo - 1.5, yhi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Shortest distance from each point to the curve")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(nr_histories, gss_histories, fname):
    """Newton: |x_n - x*|.  Golden section: the error bound (b - a)/2."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cmap = plt.get_cmap("Blues")
    n = max(len(nr_histories) - 1, 1)
    for i, (label, xs) in enumerate(nr_histories.items()):
        xs = np.asarray(xs, float)
        axA.semilogy(np.maximum(np.abs(xs - xs[-1])[:-1], 1e-16), marker="o", ms=5, lw=1.8,
                     color=cmap(0.45 + 0.5 * i / n), label=label)
    for i, (label, br) in enumerate(gss_histories.items()):
        axB.semilogy([(b - a) / 2 for a, b in br], lw=1.8, color=cmap(0.45 + 0.5 * i / n), label=label)
    axA.set_title("Newton–Raphson:  $|x_n - x^*|$")
    axB.set_title("Golden section:  error bound $(b-a)/2$")
    axA.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (axA, axB):
        ax.set_xlabel("iteration")
        ax.legend(title="point", fontsize=9)
    axA.set_ylabel("error in $x$")
    _save(fig, fname, "Convergence speed")


# ==================================================================
# Part 2: least-squares fitting
# ==================================================================
def _poly_label(c):
    terms = [f"{c[0]:.3f}"] + [f"{abs(v):.3f}" + ("x" if k == 1 else f"x^{k}") for k, v in enumerate(c) if k]
    signs = [""] + [" - " if v < 0 else " + " for v in c[1:]]
    return "$y = " + "".join(s + t for s, t in zip(signs, terms)) + "$"


def _poly(c, x):
    return sum(ck * x ** k for k, ck in enumerate(c))


def plot_fit_steps(name, X, Y, mse, coord_path, newton_path, c_exact, fname, show=(0, 1, 2)):
    """
    coord_path  : parameters after every single-parameter update (one-at-a-time Newton)
    newton_path : parameters after every full Newton step
    Left  : data with the fit after iterations `show` and the final fit.
    Middle: line -> path over the MSE contours (c0, c1); parabola -> each parameter vs iteration.
    Right : MSE - MSE* per iteration for both Newton variants.
    """
    p = len(c_exact)
    iters = coord_path[::p]
    if len(coord_path) % p != 1:
        iters = np.vstack([iters, coord_path[-1]])
    mse_star = mse(c_exact)
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    # ---- left: fits over the data ----
    xg = np.linspace(X.min() - 0.3, X.max() + 0.3, 300)
    cols = _iter_colors(len(show))
    for k, col in zip(show, cols):
        axL.plot(xg, _poly(iters[k], xg), color=col, lw=1.6, ls="--",
                 label=f"iter {k}: MSE = {mse(iters[k]):.4f}")
    axL.plot(xg, _poly(c_exact, xg), color=CURVE, lw=2.4,
             label=f"final ({len(iters) - 1} iters): MSE = {mse(iters[-1]):.4f}")
    axL.scatter(X, Y, s=60, color=POINT, zorder=5, label="data")
    axL.set_xlabel("$x$")
    axL.set_ylabel("$y$")
    axL.set_title("Fit after each iteration")
    axL.legend(loc="upper left", fontsize=9)
    axL.text(0.98, 0.03, _poly_label(c_exact), transform=axL.transAxes, ha="right", fontsize=11, color=CURVE)

    # ---- middle ----
    if p == 2:
        pts = np.vstack([coord_path, newton_path, c_exact])
        span = np.ptp(pts, axis=0) + 0.6
        g0 = np.linspace(pts[:, 0].min() - 0.2 * span[0], pts[:, 0].max() + 0.2 * span[0], 200)
        g1 = np.linspace(pts[:, 1].min() - 0.2 * span[1], pts[:, 1].max() + 0.2 * span[1], 200)
        C0, C1 = np.meshgrid(g0, g1)
        Z = np.vectorize(lambda a, b: mse(np.array([a, b])))(C0, C1)
        axM.contour(C0, C1, Z, levels=np.geomspace(mse_star * 1.01, Z.max(), 14), colors=MUTED, linewidths=0.6)
        axM.plot(coord_path[:, 0], coord_path[:, 1], color=_iter_colors(3)[1], lw=1.6,
                 marker="o", ms=3.5, label="one parameter at a time")
        for k in show[1:]:
            axM.annotate(f"iter {k}", iters[k], xytext=(8, -4), textcoords="offset points", fontsize=9, color="#6b3a12")
        axM.plot(newton_path[:, 0], newton_path[:, 1], color=CURVE, lw=2, ls="--", marker="s", ms=5,
                 label="full Newton (1 step)")
        axM.scatter(*c_exact, s=160, marker="*", color=POINT, zorder=6, label=f"optimum ({c_exact[0]:.2f}, {c_exact[1]:.2f})")
        axM.annotate("start", coord_path[0], xytext=(6, 6), textcoords="offset points", fontsize=9)
        axM.set_xlabel("$c_0$ (intercept)")
        axM.set_ylabel("$c_1$ (slope)")
        axM.set_title("Path over the MSE contours")
        axM.legend(loc="lower left", fontsize=9)
    else:
        its = np.arange(len(iters))
        for j, col in enumerate(("#c2562f", "#2e7d4f", CURVE)):
            axM.plot(its + 1, iters[:, j], color=col, lw=2, label=f"$c_{j}$")
            axM.axhline(c_exact[j], color=col, lw=1, ls=":")
            axM.text(len(its) * 1.05, c_exact[j], f"{c_exact[j]:.2f}", color=col, va="center", fontsize=10)
        axM.set_xscale("log")
        axM.set_xlabel("iteration + 1 (log scale)")
        axM.set_ylabel("parameter value")
        axM.set_title("Parameters, one at a time (dotted = exact)")
        axM.legend(loc="upper right")

    # ---- right: convergence ----
    excess = lambda cs: np.maximum([mse(c) - mse_star for c in cs], 1e-16)
    axR.semilogy(excess(iters), color=_iter_colors(3)[1], lw=2, label=f"one at a time ({len(iters) - 1} iters)")
    axR.semilogy(excess(newton_path), color=CURVE, lw=2, ls="--", marker="s", ms=5, label="full Newton (1 step)")
    axR.set_xlabel("iteration")
    axR.set_ylabel("MSE − MSE*")
    axR.set_title(f"Convergence to MSE* = {mse_star:.4f}")
    axR.legend(loc="upper right")

    _save(fig, fname, f"Fitting a {name}: Newton–Raphson intermediate steps")


def plot_final_fits(X, Y, fits, fname):
    """fits: list of (name, c, mse)."""
    xg = np.linspace(X.min() - 0.3, X.max() + 0.3, 300)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (name, c, m), col in zip(fits, ("#c2562f", CURVE)):
        ax.plot(xg, _poly(c, xg), color=col, lw=2.2, label=f"{name}: {_poly_label(c)},  MSE = {m:.4f}")
        for x, y in zip(X, Y):
            ax.plot([x, x], [y, _poly(c, x)], color=col, lw=1, ls=":")
    ax.scatter(X, Y, s=60, color=POINT, zorder=5, label="data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Least-squares line vs parabola (dotted = residuals)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
