"""
ISODATA Clustering — N-dimensional (multi-band) Python 3 implementation
Based on PyRadar (https://github.com/PyRadar/pyradar, LGPL) — ported and
extended for proper multi-band spectral data.

Works in full N-dimensional feature space (Euclidean distance).
Handles 1-D or multi-band (n_samples × n_bands) input.
"""

import numpy as np


class ISODATAParams:
    def __init__(self, K=5, I=100, P=4, THETA_M=10, THETA_S=1.0,
                 THETA_C=0.5, THETA_O=0.01, k=None):
        self.K       = K        # target number of clusters
        self.I       = I        # max iterations
        self.P       = P        # max pairs to merge per iteration
        self.THETA_M = THETA_M  # min samples per cluster (discard)
        self.THETA_S = THETA_S  # std dev threshold for split
        self.THETA_C = THETA_C  # pairwise distance threshold for merge
        self.THETA_O = THETA_O  # convergence: max fractional center movement
        self.k       = k if k is not None else K

    def copy(self):
        return ISODATAParams(K=self.K, I=self.I, P=self.P,
                             THETA_M=self.THETA_M, THETA_S=self.THETA_S,
                             THETA_C=self.THETA_C, THETA_O=self.THETA_O,
                             k=self.k)

    def __repr__(self):
        return (f"ISODATAParams(K={self.K}, I={self.I}, THETA_M={self.THETA_M}, "
                f"THETA_S={self.THETA_S:.3f}, THETA_C={self.THETA_C:.3f})")


def _assign(X, centers):
    """Assign each sample to the nearest center (Euclidean, N-D)."""
    diffs = X[:, None, :] - centers[None, :, :]       # (n, k, d)
    dists = np.sqrt((diffs ** 2).sum(axis=2))          # (n, k)
    return np.argmin(dists, axis=1)                    # (n,)


def _center_distance(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()))


def _discard(X, labels, centers, THETA_M):
    """Remove clusters with fewer than THETA_M samples."""
    counts = np.bincount(labels, minlength=len(centers))
    keep   = np.where(counts >= THETA_M)[0]
    if len(keep) == 0:
        keep = np.array([counts.argmax()])
    return centers[keep], keep


def _update(X, labels, centers, keep_idx):
    """Recompute center means for surviving clusters. Re-labels to 0..k-1."""
    new_centers = []
    new_labels  = np.zeros_like(labels)
    for new_i, old_i in enumerate(keep_idx):
        mask = labels == old_i
        new_labels[mask] = new_i
        if mask.sum() > 0:
            new_centers.append(X[mask].mean(axis=0))
        else:
            new_centers.append(centers[old_i])
    return np.array(new_centers), new_labels


def _std_per_cluster(X, labels, k):
    """Mean per-band std dev for each cluster."""
    stds = []
    for i in range(k):
        mask = labels == i
        if mask.sum() > 1:
            stds.append(X[mask].std(axis=0).mean())
        else:
            stds.append(0.0)
    return np.array(stds)


def _split(X, labels, centers, params):
    """Split the cluster with highest std dev if above THETA_S."""
    if centers.shape[0] >= params.K * 2:
        return centers, labels

    stds   = _std_per_cluster(X, labels, len(centers))
    counts = np.bincount(labels, minlength=len(centers))
    i      = int(stds.argmax())

    if stds[i] > params.THETA_S and counts[i] > 2 * params.THETA_M:
        # Split along the band with highest variance in this cluster
        mask    = labels == i
        band_stds = X[mask].std(axis=0)
        split_dim = int(band_stds.argmax())
        delta = np.zeros(centers.shape[1])
        delta[split_dim] = band_stds[split_dim] * 0.5

        c1 = centers[i] + delta
        c2 = centers[i] - delta

        new_centers = np.delete(centers, i, axis=0)
        new_centers = np.vstack([new_centers, c1, c2])

        # Re-assign labels
        new_labels = _assign(X, new_centers)
        return new_centers, new_labels

    return centers, labels


def _merge(labels, centers, params):
    """Merge cluster pairs whose centers are closer than THETA_C."""
    k = len(centers)
    if k <= 1:
        return centers, labels

    # Compute all pairwise distances
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            d = _center_distance(centers[i], centers[j])
            pairs.append((d, i, j))
    pairs.sort()

    to_del  = set()
    new_c   = []
    merged  = 0

    for d, i, j in pairs[:params.P]:
        if d >= params.THETA_C or i in to_del or j in to_del:
            continue
        ci = (labels == i).sum() + 1
        cj = (labels == j).sum()
        merged_center = (ci * centers[i] + cj * centers[j]) / (ci + cj)
        new_c.append(merged_center)
        to_del |= {i, j}
        merged += 1

    if not new_c:
        return centers, labels

    keep_idx = [i for i in range(k) if i not in to_del]
    final_centers = np.vstack(
        ([centers[i] for i in keep_idx] if keep_idx else []) + new_c
    )
    new_labels = _assign(X, final_centers)
    return final_centers, new_labels


def isodata(X, params: ISODATAParams, verbose=False):
    """
    Run ISODATA clustering on N-dimensional data X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_bands)  or (n_samples,)
    params : ISODATAParams

    Returns
    -------
    labels    : ndarray (n_samples,)  — cluster assignment per pixel
    centers   : ndarray (k, n_bands) — final cluster centers
    n_iter    : int                  — iterations completed
    converged : bool
    """
    if X.ndim == 1:
        X = X[:, None]

    n, d = X.shape
    k    = params.k

    # Initialise centers by spreading evenly across the data range per band
    idx     = np.linspace(0, n - 1, k, dtype=int)
    centers = X[idx].copy().astype(float)

    labels    = _assign(X, centers)
    converged = False
    n_iter    = 0

    for it in range(params.I):
        n_iter       = it + 1
        last_centers = centers.copy()

        # Discard small clusters
        centers, keep_idx = _discard(X, labels, centers, params.THETA_M)
        labels = np.array([np.where(keep_idx == l)[0][0]
                           if l in keep_idx else 0 for l in labels])

        # Update centers
        centers, labels = _update(X, labels, centers, np.arange(len(centers)))
        k = len(centers)

        # Split or merge
        if k <= params.K // 2:
            centers, labels = _split(X, labels, centers, params)
        elif k > params.K * 2:
            centers, labels = _merge(labels, centers, params)

        k = len(centers)

        # Convergence check
        if centers.shape == last_centers.shape:
            movement = np.abs(centers - last_centers) / (np.abs(last_centers) + 1e-9)
            if movement.max() <= params.THETA_O:
                converged = True
                break

        labels = _assign(X, centers)

    if verbose:
        print(f"  Final clusters: {k}  Iterations: {n_iter}  Converged: {converged}")

    return labels, centers, n_iter, converged
