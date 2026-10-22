"""
SYDE 572 Assignment 2: linear basis function regression on the Boston Housing data.

    Part 1  polynomial basis : a) polynomial_regression         b) learned weights
                               c) polynomial_regression_1d_vis  d) polynomial_regression_reg
    Part 2  Gaussian basis   : e) gaussian_regression           f) gaussian_regression_reg

    python regression.py        (prints the numbers and saves the plots to ../media/)
"""

import os

import numpy as np

import plots

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]
N_TRAIN = 100


# ---------- data, features, fitting ----------
def load_data():
    """First 100 rows train, the rest test. Features are normalized with the training mean and std."""
    data = np.loadtxt(os.path.join(HERE, "housing.csv"), delimiter=",")
    X, y = data[:, :-1], data[:, -1]
    mean, std = X[:N_TRAIN].mean(axis=0), X[:N_TRAIN].std(axis=0)
    std[std == 0] = 1          # CHAS is 0 in all 100 training rows: leave it unscaled (its weight is then 0)
    Xn = (X - mean) / std
    return Xn[:N_TRAIN], y[:N_TRAIN], Xn[N_TRAIN:], y[N_TRAIN:], mean, std


def poly_basis(X, degree):
    """[1, x1..x13, x1^2..x13^2, ...]: monomials of each single feature, no cross-terms."""
    return np.hstack([np.ones((len(X), 1))] + [X ** p for p in range(1, degree + 1)])


def gaussian_basis(X, centers, sigma):
    """[1, phi_1..phi_K] with phi_j(x) = exp(-||x - mu_j||^2 / (2 sigma^2))."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.hstack([np.ones((len(X), 1)), np.exp(-d2 / (2 * sigma ** 2))])


def fit(Phi, y, lam=0.0):
    """Ridge regression w = (Phi^T Phi + lam I)^-1 Phi^T y, as in the course formula: every weight, bias included, is penalized.
    lstsq on the stacked system [Phi; sqrt(lam) I] gives this and also handles lam = 0 when Phi^T Phi is singular."""
    R = np.sqrt(lam) * np.eye(Phi.shape[1])
    w, *_ = np.linalg.lstsq(np.vstack([Phi, R]), np.concatenate([y, np.zeros(len(R))]), rcond=None)
    return w


def rms(Phi, y, w):
    return np.sqrt(np.mean((Phi @ w - y) ** 2))


def cross_validate(Phi, y, lambdas, folds=10, seed=0):
    """Average validation RMS over `folds` randomly shuffled folds (fixed seed), for each lambda."""
    perm = np.random.default_rng(seed).permutation(len(y))
    idx = np.array_split(perm, folds)
    err = []
    for lam in lambdas:
        e = []
        for val in idx:
            tr = np.setdiff1d(np.arange(len(y)), val)
            e.append(rms(Phi[val], y[val], fit(Phi[tr], y[tr], lam)))
        err.append(np.mean(e))
    return np.array(err)


# ---------- Part 1: polynomial basis ----------
def polynomial_regression(Xtr, ytr, Xte, yte, degrees=range(1, 8)):
    """(a) Unregularized fit for each degree. Returns train and test RMS."""
    train, test = [], []
    for d in degrees:
        w = fit(poly_basis(Xtr, d), ytr)
        train.append(rms(poly_basis(Xtr, d), ytr, w))
        test.append(rms(poly_basis(Xte, d), yte, w))
    return np.array(train), np.array(test)


def polynomial_regression_1d_vis(Xtr, ytr, Xte, yte, degree, feature=1):
    """(c) Fit one feature only (Xn[:, 1], the 2nd feature). Returns the data and fitted curve for plotting."""
    xtr, xte = Xtr[:, [feature]], Xte[:, [feature]]
    w = fit(poly_basis(xtr, degree), ytr)
    grid = np.linspace(min(xtr.min(), xte.min()), max(xtr.max(), xte.max()), 400)[:, None]
    return grid[:, 0], poly_basis(grid, degree) @ w, rms(poly_basis(xtr, degree), ytr, w), rms(poly_basis(xte, degree), yte, w)


def polynomial_regression_reg(Xtr, ytr, degree=8, feature=1, seed=0):
    """(d) 10-fold CV over lambda for a degree-8 polynomial of the 2nd feature."""
    Phi = poly_basis(Xtr[:, [feature]], degree)
    return cross_validate(Phi, ytr, LAMBDAS, seed=seed)


# ---------- Part 2: Gaussian basis ----------
def pick_sigma(Xtr):
    """Width: median distance between training points, so the Gaussians neither vanish nor cover everything."""
    d = np.sqrt(((Xtr[:, None] - Xtr[None]) ** 2).sum(axis=2))
    return np.median(d[np.triu_indices(len(Xtr), 1)])


def gaussian_regression(Xtr, ytr, Xte, yte, sigma, Ks=range(5, 100, 10), seed=0):
    """(e) Unregularized fit with K random training points as centers. Returns train and test RMS."""
    order = np.random.default_rng(seed).permutation(len(Xtr))
    train, test = [], []
    for K in Ks:
        centers = Xtr[order[:K]]
        Phi = gaussian_basis(Xtr, centers, sigma)
        w = fit(Phi, ytr)
        train.append(rms(Phi, ytr, w))
        test.append(rms(gaussian_basis(Xte, centers, sigma), yte, w))
    return np.array(train), np.array(test)


def gaussian_regression_reg(Xtr, ytr, sigma, K=90, seed=0, cv_seed=0):
    """(f) 10-fold CV over lambda with K random centers and all 13 features."""
    order = np.random.default_rng(seed).permutation(len(Xtr))
    Phi = gaussian_basis(Xtr, Xtr[order[:K]], sigma)
    return cross_validate(Phi, ytr, LAMBDAS, seed=cv_seed)


# ---------- driver ----------
if __name__ == "__main__":
    os.makedirs(MEDIA, exist_ok=True)
    plots.setup_style()
    Xtr, ytr, Xte, yte, mean, std = load_data()
    print(f"train {Xtr.shape}, test {Xte.shape}")

    # (a)
    degrees = list(range(1, 8))
    tr, te = polynomial_regression(Xtr, ytr, Xte, yte, degrees)
    print("\n(a) degree  train RMS  test RMS")
    for d, a, b in zip(degrees, tr, te):
        print(f"    {d:>5}  {a:9.3f}  {b:10.3f}")
    plots.plot_error_curve(degrees, tr, te, "polynomial degree", "Polynomial regression: RMS error vs degree",
                           os.path.join(MEDIA, "a_poly_error.png"), logy=True)

    # (b)
    w = fit(poly_basis(Xtr, 1), ytr)
    print("\n(b) degree-1 weights (on normalized features)")
    print(f"    w0 (bias) = {w[0]:.4f}")
    for j in range(13):
        print(f"    w{j + 1:<2} = {w[j + 1]:8.4f}")

    # (c)
    print("\n(c) 2nd feature only: degree  train RMS  test RMS")
    fits = {}
    for d in range(1, 11):
        fits[d] = polynomial_regression_1d_vis(Xtr, ytr, Xte, yte, d)
        print(f"    {d:>5}  {fits[d][2]:9.3f}  {fits[d][3]:9.3f}")
    for d in (1, 3, 8):
        grid, curve, a, b = fits[d]
        plots.plot_poly_1d(grid, curve, Xtr[:, 1], ytr, Xte[:, 1], yte, d, a, b,
                           os.path.join(MEDIA, f"c_poly1d_deg{d}.png"))

    # (d)
    # CV with 100 points is noisy, so repeat it over 3 shuffles and choose lambda from the average
    seeds = [0, 1, 2]
    cvs = np.array([polynomial_regression_reg(Xtr, ytr, seed=s) for s in seeds])
    cv = cvs.mean(axis=0)
    print("\n(d) lambda  " + "  ".join(f"seed {s}" for s in seeds) + "    mean")
    for lam, row, m in zip(LAMBDAS, cvs.T, cv):
        print(f"    {lam:>6}  " + "  ".join(f"{e:6.2f}" for e in row) + f"  {m:6.2f}")
    print(f"    best lambda per seed = {[LAMBDAS[i] for i in cvs.argmin(axis=1)]}, from the mean = {LAMBDAS[int(np.argmin(cv))]}")
    plots.plot_cv(LAMBDAS, cv, "Degree-8 polynomial, 2nd feature: 10-fold CV (mean of 3 shuffles)",
                  os.path.join(MEDIA, "d_poly_cv.png"))

    # (e)
    sigma = pick_sigma(Xtr)
    Ks = list(range(5, 100, 10))
    tr, te = gaussian_regression(Xtr, ytr, Xte, yte, sigma, Ks)
    print(f"\n(e) sigma = {sigma:.3f}   K  train RMS  test RMS")
    for K, a, b in zip(Ks, tr, te):
        print(f"    {K:>5}  {a:9.3f}  {b:9.3f}")
    plots.plot_error_curve(Ks, tr, te, "number of basis functions $K$",
                           "Gaussian regression: RMS error vs number of basis functions",
                           os.path.join(MEDIA, "e_gauss_error.png"))

    # (f)
    cv = gaussian_regression_reg(Xtr, ytr, sigma)
    print("\n(f) lambda  avg validation RMS")
    for lam, e in zip(LAMBDAS, cv):
        print(f"    {lam:>6}  {e:.3f}")
    print(f"    best lambda = {LAMBDAS[int(np.argmin(cv))]}")
    cv1 = gaussian_regression_reg(Xtr, ytr, sigma, cv_seed=1)
    print(f"    check, fold seed 1: {np.round(cv1, 2)}  best lambda = {LAMBDAS[int(np.argmin(cv1))]}")
    plots.plot_cv(LAMBDAS, cv, "Gaussian basis, K = 90: 10-fold CV", os.path.join(MEDIA, "f_gauss_cv.png"))
