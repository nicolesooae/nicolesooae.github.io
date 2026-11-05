"""
SYDE 572 Assignment 3: PCA, data visualization and ISOMAP.

    Part 1  PCA by hand             : part1_by_hand
    Part 2  data visualization      : marginal plots of 1, 2 and 3 Iris features
    Part 3  PCA on Iris             : part3_pca
    Part 4  data compression        : part4_compression (Frobenius-norm view of explained variance)
    Part 5  non-linear components   : part5_isomap (3 nearest neighbours) and compare_embeddings

    python pca.py        (prints the numbers and saves the plots to ../media/)
"""

import os
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import plots

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
SEED = 0


# ---------- Part 1: PCA by hand ----------
def part1_by_hand():
    """Centre the 5 points, eigendecompose the covariance, and reconstruct (3,3) and (3,4) from the first PC."""
    D = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3]], float)
    mu = D.mean(axis=0)
    Dc = D - mu
    S = Dc.T @ Dc / (len(D) - 1)                # covariance (1/(N-1), as in the hand work; 1/N gives the same eigenvectors)
    evals, evecs = np.linalg.eigh(S)
    w = evecs[:, np.argmax(evals)]
    w = w * np.sign(w[0])                       # sign is arbitrary: pick the one pointing into the first quadrant
    out = {"data": D, "mean": mu, "cov": S, "evals": np.sort(evals)[::-1], "w": w, "points": {}}
    for p in ([3, 3], [3, 4]):
        p = np.array(p, float)
        z = w @ (p - mu)                        # 1-D code
        rec = mu + z * w                        # reconstruction
        out["points"][tuple(p)] = (z, rec, np.linalg.norm(p - rec))
    return out


# ---------- Iris ----------
def load_data():
    """The Iris data (150 x 4, in cm) from the UCI file iris.data (archive.ics.uci.edu/dataset/53/iris), used as is."""
    classes = ["setosa", "versicolor", "virginica"]
    rows = [line.strip().split(",") for line in open(os.path.join(HERE, "iris.data")) if line.strip()]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([classes.index(r[4].replace("Iris-", "")) for r in rows])
    names = ["sepal length", "sepal width", "petal length", "petal width"]
    return X, y, names, classes


# ---------- Part 3: PCA ----------
def part3_pca(X):
    """PCA on the centred (unscaled) data: all four features are in cm, so no standardization is needed."""
    pca = PCA(n_components=4).fit(X)
    return pca, pca.transform(X)


# ---------- Part 4: compression ----------
def part4_compression(X, pca):
    """Explained variance from the Frobenius norm: ||X_c||_F^2 = sum of squared singular values = (N-1) * total variance,
    and ||X_c - X_c W_k W_k^T||_F^2 is the variance the first k PCs leave out."""
    Xc = X - X.mean(axis=0)
    total = np.linalg.norm(Xc, "fro") ** 2
    rows = []
    for k in range(1, X.shape[1] + 1):
        W = pca.components_[:k].T
        kept = np.linalg.norm(Xc @ W, "fro") ** 2
        resid = np.linalg.norm(Xc - Xc @ W @ W.T, "fro") ** 2
        rows.append((k, kept, resid, kept / total))
    k50 = next(k for k, _, _, f in rows if f >= 0.5)
    return total, rows, k50


# ---------- Part 5: ISOMAP ----------
def part5_isomap(X, k=3, dims=2):
    """sklearn Isomap: k-NN graph -> shortest-path (geodesic) distances -> classical MDS (eigendecomposition).
    With k = 3 the graph has two components (setosa / the rest); sklearn 1.0 joins them by their closest pair of points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = Isomap(n_neighbors=k, n_components=dims)
        Z = iso.fit_transform(X)
    return iso, Z


def geodesic_smacof(G, dims=2):
    """Metric MDS fitted iteratively (SMACOF, a majorization form of gradient descent on the stress) to the geodesic distances."""
    mds = MDS(n_components=dims, dissimilarity="precomputed", n_init=4, max_iter=1000, random_state=SEED)
    return mds.fit_transform(G), mds.stress_


def residual_variance(D_target, Z):
    """Isomap's quality measure: 1 - R^2 between the target distances and the Euclidean distances in the embedding."""
    r = np.corrcoef(squareform(D_target, checks=False), pdist(Z))[0, 1]
    return 1 - r ** 2


def knn_accuracy(Z, y, k=5):
    """10-fold stratified CV accuracy of a 5-NN classifier on an embedding (fixed shuffle)."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    return cross_val_score(KNeighborsClassifier(n_neighbors=k), Z, y, cv=cv).mean()


def compare_embeddings(X, y, G):
    """For d = 1..4: residual variance against Euclidean and geodesic distances, and 5-NN accuracy, for PCA and Isomap."""
    D_euc = squareform(pdist(X))
    rows = []
    for d in range(1, 5):
        Zp = PCA(n_components=d).fit_transform(X)
        _, Zi = part5_isomap(X, dims=d)
        rows.append((d,
                     residual_variance(D_euc, Zp), residual_variance(G, Zp),
                     residual_variance(D_euc, Zi), residual_variance(G, Zi),
                     knn_accuracy(Zp, y), knn_accuracy(Zi, y)))
    return rows


# ---------- driver ----------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)   # scipy.stats.mode notice inside sklearn 1.0 k-NN
    os.makedirs(MEDIA, exist_ok=True)
    plots.setup_style()
    np.set_printoptions(precision=4, suppress=True)

    # Part 1
    p1 = part1_by_hand()
    print("Part 1")
    print(f"    mean = {p1['mean']},  covariance =\n{p1['cov']}")
    print(f"    eigenvalues = {p1['evals']},  first PC w = {p1['w']}")
    for p, (z, rec, err) in p1["points"].items():
        print(f"    {p}: z = {z:.4f}, reconstruction = {rec}, L2 error = {err:.4f}")
    plots.plot_part1(p1, os.path.join(MEDIA, "p1_by_hand.png"))

    # Part 2
    X, y, names, classes = load_data()
    print(f"\nIris {X.shape}, classes {classes}")
    for j, n in enumerate(names):
        rng = [f"{c}: {X[y == i, j].min():.1f}-{X[y == i, j].max():.1f}" for i, c in enumerate(classes)]
        print(f"    {n:13s} " + ",  ".join(rng))
    plots.plot_1d(X, y, names, classes, os.path.join(MEDIA, "p2_1d.png"))
    plots.plot_2d(X, y, names, classes, os.path.join(MEDIA, "p2_2d.png"))
    plots.plot_3d(X, y, names, classes, [(0, 1, 2), (2, 3, 0)], os.path.join(MEDIA, "p2_3d.png"))
    print("    5-NN accuracy on single features / pairs:")
    for j, n in enumerate(names):
        print(f"      {n:13s} {knn_accuracy(X[:, [j]], y):.3f}")
    for a in range(4):
        for b in range(a + 1, 4):
            print(f"      {names[a]} + {names[b]}: {knn_accuracy(X[:, [a, b]], y):.3f}")

    # Part 3
    pca, Z = part3_pca(X)
    print("\nPart 3: PCA axes (rows), feature order", names)
    print(f"    mean = {pca.mean_}")
    for i, c in enumerate(pca.components_):
        print(f"    PC{i + 1} = {c},  variance {pca.explained_variance_[i]:.4f}")
    plots.plot_pc1(Z, y, classes, os.path.join(MEDIA, "p3_pc1.png"))
    plots.plot_pc12(Z, y, classes, os.path.join(MEDIA, "p3_pc12.png"))
    plots.plot_pc123(Z, y, classes, os.path.join(MEDIA, "p3_pc123.png"))

    # Part 4
    total, rows, k50 = part4_compression(X, pca)
    print(f"\nPart 4: ||X_c||_F^2 = {total:.4f}  (total variance {total / (len(X) - 1):.4f})")
    print("    k   ||X_c W_k||_F^2   residual   explained")
    for k, kept, resid, f in rows:
        print(f"    {k}   {kept:13.4f}   {resid:8.4f}   {100 * f:7.2f}%")
    print(f"    PCs needed for 50%: {k50}")
    print(f"    (standardized features instead: {np.round(100 * np.cumsum(PCA().fit((X - X.mean(0)) / X.std(0)).explained_variance_ratio_), 2)} %)")
    plots.plot_scree([f for *_, f in rows], os.path.join(MEDIA, "p4_scree.png"))

    # Part 5
    iso, Zi = part5_isomap(X)
    G = iso.dist_matrix_
    Zs, stress = geodesic_smacof(G)
    print(f"\nPart 5: Isomap k = 3,  geodesic distances finite: {np.isfinite(G).all()}")
    print(f"    SMACOF stress on geodesic distances = {stress:.1f}")
    plots.plot_isomap(Zi, y, classes, os.path.join(MEDIA, "p5_isomap.png"))
    plots.plot_compare([Z[:, :2], Zi, Zs], y, classes,
                       ["PCA (top 2 PCs)", "Isomap, k = 3 (classical MDS)", "Geodesic distances + SMACOF MDS"],
                       os.path.join(MEDIA, "p5_compare.png"))
    rows = compare_embeddings(X, y, G)
    print("    d   resvar PCA (euc, geo)   resvar Isomap (euc, geo)   5-NN acc PCA   Isomap")
    for d, pe, pg, ie, ig, ap, ai in rows:
        print(f"    {d}   {pe:.4f}  {pg:.4f}          {ie:.4f}  {ig:.4f}             {ap:.3f}       {ai:.3f}")
    print(f"    5-NN accuracy on all 4 raw features: {knn_accuracy(X, y):.3f}")
    print(f"    5-NN accuracy, SMACOF 2-D: {knn_accuracy(Zs, y):.3f}")
    plots.plot_residual(rows, os.path.join(MEDIA, "p5_residual.png"))
