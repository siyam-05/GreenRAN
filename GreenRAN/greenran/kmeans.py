"""K-means clustering written in NumPy, used to discover what kind of area
each small cell serves from the shape of its daily traffic alone."""
import numpy as np

from .config import Config


def kmeans(X, k, n_init=20, max_iter=100, seed=0):
    """Plain K-means with several random starts; keeps the lowest-distortion run.

    Returns (centroids (k, D), labels (n,), distortion = mean squared distance).
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centroids = X[rng.choice(len(X), k, replace=False)].copy()
        labels = None
        for _ in range(max_iter):
            dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)                    # step 1: assign points
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):                                  # step 2: move centroids
                members = X[labels == j]
                if len(members):
                    centroids[j] = members.mean(axis=0)
        distortion = float(((X - centroids[labels]) ** 2).sum(axis=1).mean())
        if best is None or distortion < best[2]:
            best = (centroids, labels, distortion)
    return best


def silhouette(X, labels):
    """Mean silhouette score: +1 = tight, well separated clusters, 0 = overlapping."""
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    ks = np.unique(labels)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = dist[i, own].sum() / (own.sum() - 1)
        b = min(dist[i, labels == j].mean() for j in ks if j != labels[i])
        scores[i] = (b - a) / max(a, b)
    return float(scores.mean())


def daily_profiles(data, cfg: Config):
    """One row per (cell, weekday): 24 hourly averages, scaled to mean 1.

    Scaling removes "how big is this cell" so that only the SHAPE of the day is
    clustered. Weekends are left out because offices are almost empty then.
    """
    S, N = cfg.steps_per_day, cfg.n_cells
    d = data["demand"].reshape(-1, S, N)
    is_weekend = data["weekend"].reshape(-1, S)[:, 0]
    hourly = d.reshape(d.shape[0], 24, S // 24, N).mean(axis=2)      # (days, 24, N)
    rows, cell = [], []
    for day in np.where(~is_weekend)[0]:
        for i in range(N):
            p = hourly[day, :, i]
            rows.append(p / p.mean())
            cell.append(i)
    return np.array(rows), np.array(cell)


def discover_types(data, cfg: Config, k_range=range(2, 9), seed=0):
    """Cluster daily traffic shapes, pick K by silhouette, give each cell a type.

    Returns a dict with the chosen K, the per-cell type, a one-hot matrix for
    the controller, and the numbers needed for the elbow / silhouette plot.
    """
    X, cell = daily_profiles(data, cfg)
    curve = []
    runs = {}
    for k in [1] + list(k_range):
        c, lab, dist = kmeans(X, k, seed=seed)
        sil = silhouette(X, lab) if k > 1 else float("nan")
        curve.append({"k": k, "distortion": dist, "silhouette": sil})
        runs[k] = (c, lab)
    best_k = max((r for r in curve if r["k"] > 1), key=lambda r: r["silhouette"])["k"]
    centroids, labels = runs[best_k]
    cell_type = np.array([np.bincount(labels[cell == i], minlength=best_k).argmax()
                          for i in range(cfg.n_cells)])
    onehot = np.eye(best_k)[cell_type]
    # how often a single day lands in its cell's own cluster
    consistency = float((labels == cell_type[cell]).mean())
    return {"k": best_k, "cell_type": cell_type, "onehot": onehot, "centroids": centroids,
            "curve": curve, "day_consistency": consistency}


def purity(found, truth):
    """Share of cells whose cluster's majority true type matches their own."""
    total = 0
    for c in np.unique(found):
        total += np.bincount(truth[found == c]).max()
    return total / len(found)
