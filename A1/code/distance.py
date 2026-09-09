"""
SYDE 572 Assignment 1, Part 1: shortest distance from a point (x0, y0) to a curve y = f(x).

find_distance_newton() and golden_section_search() are the handout's algorithms,
changed only to also return their steps for plotting.

    python distance.py                  # assignment parabola + other functions
    python distance.py a b c x0 y0      # any parabola y = ax^2 + bx + c and point
Plots are saved to ../media/
"""

import math
import os
import sys

import numpy as np

import plots

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")


def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    x = initial_guess
    history = [x]
    for _ in range(max_iter):
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * df(x)**2 + 2 * (f(x) - y0) * ddf(x)
        step = D_prime / D_double_prime
        x -= step
        history.append(x)
        if abs(step) < tolerance:
            break
    return math.hypot(x - x0, f(x) - y0), x, history


def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    resphi = 2 - (1 + math.sqrt(5)) / 2
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    x1, x2 = a + resphi * (b - a), b - resphi * (b - a)
    f_x1, f_x2 = dist_sq(x1), dist_sq(x2)
    brackets = [(a, b)]
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b, x2, f_x2 = x2, x1, f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a, x1, f_x1 = x1, x2, f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
        brackets.append((a, b))
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x, brackets


def choose_bracket(x0, y0, f, half_width=10, n=81):
    """Sample D(x) on [x0 - half_width, x0 + half_width] and bracket the lowest sample
    with its two neighbours (the same idea as tabulating D(x) by hand)."""
    grid = np.linspace(x0 - half_width, x0 + half_width, n)
    i = int(np.nanargmin((grid - x0)**2 + (f(grid) - y0)**2))
    return grid[max(i - 1, 0)], grid[min(i + 1, n - 1)]


def parabola(a, b, c):
    """f, f', f'' and a plot label for y = ax^2 + bx + c."""
    terms = []
    for coef, power in ((a, "x^2"), (b, "x"), (c, "")):
        if coef:
            num = "" if abs(coef) == 1 and power else f"{abs(coef):g}"
            terms.append(("- " if coef < 0 else "+ ") + num + power)
    label = " ".join(terms).replace("- ", "-", 1) if terms[0][0] == "-" else " ".join(terms)[2:]
    return (lambda x: a * x**2 + b * x + c, lambda x: 2 * a * x + b, lambda x: 2 * a + 0 * x,
            f"$y = {label}$")


def tag(x0, y0):
    """(-4, 0) -> 'm4_0' for file names."""
    fmt = lambda v: (f"m{-v:g}" if v < 0 else f"{v:g}").replace(".", "p")
    return f"{fmt(x0)}_{fmt(y0)}"


def solve_and_plot(name, f, df, ddf, eq, x0, y0, guess):
    lo, hi = choose_bracket(x0, y0, f)
    d_nr, x_nr, nr_hist = find_distance_newton(x0, y0, f, df, ddf, initial_guess=guess)
    d_gs, x_gs, gs_hist = golden_section_search(x0, y0, f, lo, hi)
    print(f"{eq.strip('$'):<22} ({x0:g}, {y0:g})  "
          f"Newton x*={x_nr:.6f} d={d_nr:.6f} ({len(nr_hist) - 1} it, start {guess:g})  "
          f"Golden x*={x_gs:.6f} d={d_gs:.6f} ({len(gs_hist) - 1} it, [{lo:g}, {hi:g}])")
    plots.plot_newton(f, x0, y0, nr_hist, eq, os.path.join(MEDIA, f"nr_{name}{tag(x0, y0)}.png"))
    plots.plot_golden(f, x0, y0, gs_hist, eq, os.path.join(MEDIA, f"gss_{name}{tag(x0, y0)}.png"))
    return x_nr, nr_hist, gs_hist


# Other curves: (file prefix, f, f', f'', label, point, Newton start)
OTHER = [
    ("parab2_", *parabola(0.5, -2, 1), (5, -2), 2.0),
    ("exp_", np.exp, np.exp, np.exp, "$y = e^x$", (2, 1), 0.0),
    ("log_", np.log, lambda x: 1 / x, lambda x: -1 / x**2, r"$y = \ln x$", (0, 0), 1.0),
    ("rational_", lambda x: 4 / (1 + x**2), lambda x: -8 * x / (1 + x**2)**2,
     lambda x: (24 * x**2 - 8) / (1 + x**2)**3, r"$y = \dfrac{4}{1 + x^2}$", (3, 3), 2.0),
    ("sqrt_", np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 * x**-1.5, r"$y = \sqrt{x}$", (4, 0), 4.0),
]


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")   # ln x and sqrt x are undefined for x <= 0 on plot grids
    os.makedirs(MEDIA, exist_ok=True)
    plots.setup_style()

    if len(sys.argv) == 6:
        a, b, c, x0, y0 = map(float, sys.argv[1:])
        vertex = -b / (2 * a)
        solve_and_plot("custom_", *parabola(a, b, c), x0, y0, vertex + 1 if x0 == vertex else vertex)
        sys.exit()

    # Assignment: y = x^2 + 5. Newton starts at the vertex x = 0, except for (0, 0)
    # where that is already the answer, so it starts at x = 1 (same as the hand work).
    f, df, ddf, eq = parabola(1, 0, 5)
    points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
    x_stars, nr_all, gs_all = [], {}, {}
    for x0, y0 in points:
        x_nr, nr_hist, gs_hist = solve_and_plot("", f, df, ddf, eq, x0, y0, 1.0 if x0 == 0 else 0.0)
        x_stars.append(x_nr)
        nr_all[f"({x0}, {y0})"], gs_all[f"({x0}, {y0})"] = nr_hist, gs_hist
    plots.plot_setup(f, points, eq, os.path.join(MEDIA, "p1_setup.png"))
    plots.plot_all_results(f, points, x_stars, eq, os.path.join(MEDIA, "p1_all_results.png"))
    plots.plot_convergence(nr_all, gs_all, os.path.join(MEDIA, "p1_convergence.png"))

    for name, f, df, ddf, eq, (x0, y0), guess in OTHER:
        solve_and_plot(name, f, df, ddf, eq, x0, y0, guess)
