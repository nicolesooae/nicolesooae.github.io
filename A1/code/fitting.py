"""
SYDE 572 Assignment 1, Part 2: fit a line and a parabola to 4 points by minimizing the MSE.

    analytical     : solve the normal equations (A^T A) c = A^T y by Gauss-Jordan elimination
    one at a time  : Newton-Raphson on one parameter, the others frozen (the hand method)
    full Newton    : c <- c - H^-1 g on all parameters at once

    python fitting.py        (plots are saved to ../media/)
"""

import os

import numpy as np

import plots

MEDIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([0.5, 1.5, 3.5, 7.5])


def design_matrix(x, degree):
    """Columns 1, x, x^2, ... so the model is y_hat = A @ c with c = [c0, c1, ...]."""
    return np.vander(x, degree + 1, increasing=True)


def mse(A, y, c):
    return np.mean((A @ c - y) ** 2)


def gradient(A, y, c):
    return 2 / len(y) * A.T @ (A @ c - y)


def hessian(A):
    return 2 / len(A) * A.T @ A          # constant, because the MSE is quadratic in c


def gauss_jordan(M, b):
    """Solve M c = b with the same row operations as the hand work."""
    M = np.column_stack([M, b]).astype(float)
    for i in range(len(b)):
        M[i] /= M[i, i]
        for k in range(len(b)):
            if k != i:
                M[k] -= M[k, i] * M[i]
    return M[:, -1]


def newton_one_at_a_time(A, y, c_start, order, tol=1e-10, max_iter=5000):
    """Each iteration updates the parameters in `order`, one at a time:
    c_j <- c_j - g_j / H_jj with the others frozen. Returns every intermediate c."""
    c = np.array(c_start, float)
    H = hessian(A)
    path = [c.copy()]
    for _ in range(max_iter):
        before = c.copy()
        for j in order:
            c[j] -= gradient(A, y, c)[j] / H[j, j]
            path.append(c.copy())
        if np.max(np.abs(c - before)) < tol:
            break
    return c, np.array(path)


def newton_full(A, y, c_start, tol=1e-10, max_iter=20):
    c = np.array(c_start, float)
    path = [c.copy()]
    for _ in range(max_iter):
        step = gauss_jordan(hessian(A), gradient(A, y, c))
        c -= step
        path.append(c.copy())
        if np.max(np.abs(step)) < tol:
            break
    return c, np.array(path)


# (name, degree, starting guess, update order) -- same start and order as the hand work
MODELS = [
    ("line", 1, [0.5, 1.0], [1, 0]),
    ("parabola", 2, [0.5, 1.0, 0.0], [1, 2, 0]),
]


if __name__ == "__main__":
    os.makedirs(MEDIA, exist_ok=True)
    plots.setup_style()
    finals = []
    for name, degree, start, order in MODELS:
        A = design_matrix(X, degree)
        loss = lambda c, A=A: mse(A, Y, c)
        c_exact = gauss_jordan(A.T @ A, A.T @ Y)
        c_one, one_path = newton_one_at_a_time(A, Y, start, order)
        c_full, full_path = newton_full(A, Y, start)
        n_iters = (len(one_path) - 1) // len(order)

        print(f"\n{name.upper()}  (start {start}, MSE {loss(np.array(start)):.4f})")
        print(f"  analytical     c = {np.round(c_exact, 6)}  MSE = {loss(c_exact):.6f}")
        for k in (1, 2):
            c_k = one_path[k * len(order)]
            print(f"  one-at-a-time  iter {k}: c = {np.round(c_k, 4)}  MSE = {loss(c_k):.4f}")
        print(f"  one-at-a-time  c = {np.round(c_one, 6)}  MSE = {loss(c_one):.6f}  ({n_iters} iterations)")
        print(f"  full Newton    c = {np.round(c_full, 6)}  MSE = {loss(c_full):.6f}  ({len(full_path) - 2} step + 1 check)")

        plots.plot_fit_steps(name, X, Y, loss, one_path, full_path, c_exact,
                             os.path.join(MEDIA, f"p2_{name}_steps.png"))
        finals.append((name.capitalize(), c_exact, loss(c_exact)))

    plots.plot_final_fits(X, Y, finals, os.path.join(MEDIA, "p2_final_fits.png"))
