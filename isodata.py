"""
ISODATA Clustering — Python 3 port of PyRadar implementation
https://github.com/PyRadar/pyradar (original: Python 2, LGPL)

Ported to Python 3: xrange → range, print statements → functions,
global state refactored into ISODATAParams class.

Used by baru_isodata_classifier.py.
"""

import numpy as np
from scipy.spatial.distance import cdist


class ISODATAParams:
    def __init__(self, K=5, I=100, P=4, THETA_M=10, THETA_S=1.0,
                 THETA_C=20.0, THETA_O=0.05, k=None):
        self.K       = K        # target number of clusters
        self.I       = I        # max iterations
        self.P       = P        # max pairs to merge per iteration
        self.THETA_M = THETA_M  # min samples per cluster (discard threshold)
        self.THETA_S = THETA_S  # std dev threshold for split
        self.THETA_C = THETA_C  # pairwise distance threshold for merge
        self.THETA_O = THETA_O  # convergence threshold (% change)
        self.k       = k if k is not None else K

    def copy(self):
        return ISODATAParams(
            K=self.K, I=self.I, P=self.P,
            THETA_M=self.THETA_M, THETA_S=self.THETA_S,
            THETA_C=self.THETA_C, THETA_O=self.THETA_O, k=self.k
        )

    def __repr__(self):
        return (f"ISODATAParams(K={self.K}, I={self.I}, "
                f"THETA_M={self.THETA_M}, THETA_S={self.THETA_S}, "
                f"THETA_C={self.THETA_C})")


def _initial_centers(X_flat, k, method="linspace"):
    if method == "linspace":
        return np.linspace(X_flat.min(), X_flat.max(), k)
    else:
        idx = np.random.randint(0, X_flat.size, k)
        return X_flat[idx]


def _assign(X_flat, centers):
    """Assign each sample to the nearest center. Returns labels array."""
    dists = np.abs(X_flat[:, None] - centers[None, :])
    return np.argmin(dists, axis=1)


def _discard(labels, centers, clusters, THETA_M):
    counts = np.bincount(labels, minlength=centers.size)
    keep = counts > THETA_M
    return centers[keep], clusters[keep]


def _update(X_flat, labels, centers, clusters):
    new_c, new_cl = [], []
    for i, cl in enumerate(clusters):
        idx = np.where(labels == i)[0]
        if idx.size > 0:
            new_c.append(X_flat[idx].mean())
            new_cl.append(cl)
    c = np.array(new_c)
    cl = np.array(new_cl)
    order = np.argsort(c)
    return c[order], cl[order]


def _split(X_flat, labels, centers, clusters, K, THETA_M, THETA_S, delta=0.5):
    if centers.size >= K * 2:
        return centers, clusters
    stds = np.array([
        X_flat[labels == i].std() if (labels == i).sum() > 1 else 0.0
        for i in range(centers.size)
    ])
    counts = np.array([(labels == i).sum() for i in range(centers.size)])
    i = int(stds.argmax())
    if stds[i] > THETA_S and counts[i] > 2 * THETA_M:
        new_c = np.delete(centers, i)
        new_cl = np.delete(clusters, i)
        start = int(clusters.max()) + 1
        new_c = np.append(new_c, [centers[i] + delta, centers[i] - delta])
        new_cl = np.append(new_cl, [start, start + 1])
        order = np.argsort(new_c)
        return new_c[order], new_cl[order]
    return centers, clusters


def _merge(labels, centers, clusters, P, THETA_C):
    k = centers.size
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(centers[i] - centers[j])
            pairs.append((d, i, j))
    pairs.sort()

    to_del = set()
    new_c, new_cl = [], []
    start = int(clusters.max()) + 1 if clusters.size > 0 else 0
    merged = 0

    for d, i, j in pairs[:P]:
        if d >= THETA_C or i in to_del or j in to_del:
            continue
        ci = (labels == i).sum() + 1
        cj = (labels == j).sum()
        merged_center = round((ci * centers[i] + cj * centers[j]) / (ci + cj))
        new_c.append(merged_center)
        new_cl.append(start + merged)
        to_del |= {i, j}
        merged += 1

    keep = [i for i in range(k) if i not in to_del]
    final_c = np.array(list(centers[keep]) + new_c)
    final_cl = np.array(list(clusters[keep]) + new_cl)
    order = np.argsort(final_c)
    return final_c[order], final_cl[order]


def isodata(X, params: ISODATAParams, verbose=False):
    """
    Run ISODATA clustering on a 1-D or multi-band array X.

    Parameters
    ----------
    X : ndarray, shape (n_samples,) or (n_samples, n_bands)
        Input pixel data. Multi-band input is reduced to a single
        spectral distance measure (mean across bands) for 1-D ISODATA.
        For multi-band data, prefer sklearn KMeans directly.
    params : ISODATAParams

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster assignment for each pixel.
    centers : ndarray
        Final cluster centers.
    n_iter : int
        Number of iterations completed.
    converged : bool
        Whether the algorithm converged before max iterations.
    """
    if X.ndim > 1:
        X_flat = X.mean(axis=1)
    else:
        X_flat = X.copy().astype(float)

    k = params.k
    centers = _initial_centers(X_flat, k)
    clusters = np.arange(k, dtype=float)

    converged = False
    n_iter = 0

    for it in range(params.I):
        n_iter = it + 1
        last_centers = centers.copy()

        labels = _assign(X_flat, centers)
        centers, clusters = _discard(labels, centers, clusters, params.THETA_M)
        if centers.size == 0:
            break

        labels = _assign(X_flat, centers)
        centers, clusters = _update(X_flat, labels, centers, clusters)
        k = centers.size

        if k <= params.K / 2:
            centers, clusters = _split(X_flat, labels, centers, clusters,
                                       params.K, params.THETA_M, params.THETA_S)
        elif k > params.K * 2:
            centers, clusters = _merge(labels, centers, clusters,
                                       params.P, params.THETA_C)

        k = centers.size

        # convergence check
        if centers.shape == last_centers.shape:
            change = np.abs((centers - last_centers) / (last_centers + 1e-9))
            if np.all(change <= params.THETA_O):
                converged = True
                if verbose:
                    print(f"  Converged at iteration {n_iter}")
                break

    labels = _assign(X_flat, centers)

    if verbose:
        print(f"  Final clusters: {k}  Iterations: {n_iter}  "
              f"Converged: {converged}")

    return labels, centers, n_iter, converged
