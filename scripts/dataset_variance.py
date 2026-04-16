import os
import sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from benchmark.datasets import get_dataset

# ── Configuration ──────────────────────────────────────────────
# Set to a dataset name to process only that dataset, or None for all datasets.
DATASET = None

KNOWN_DATASETS = [
    "audio-128-euclidean",
    "gist-960-euclidean",
    "paper-200-euclidean",
    "sift-128-euclidean",
    "text2image-200-euclidean",
    "video-1024-euclidean",
]

# k values for LID estimation (Levina-Bickel MLE)
LID_KS = [100, 200, 300]
# Number of points randomly sampled for LID estimation
LID_SAMPLE = 2000
# Number of points sampled from each split for MMD estimation
MMD_SAMPLE = 2000
# k and sample size for kNN cross-distance ratio
KNN_CROSS_K      = 10
KNN_CROSS_SAMPLE = 2000

OUTPUT_DIR = os.path.join(REPO_DIR, "figures", "dataset_variance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_mean(data: np.ndarray) -> np.ndarray:
    """
    Compute the per-dimension mean of the dataset.

    Args:
        data: Array of shape (n, d)

    Returns:
        mean vector of shape (d,)
    """
    return np.mean(data, axis=0)


def compute_covariance(data: np.ndarray) -> np.ndarray:
    """
    Compute the (d, d) covariance matrix between dimensions of the dataset.

    np.cov expects variables as rows, so we transpose data from (n, d) to
    (d, n). Entry [i, j] is the covariance between dimension i and dimension j
    across all training samples (Bessel-corrected, ddof=1).

    Args:
        data: Array of shape (n, d)

    Returns:
        Covariance matrix of shape (d, d)
    """
    return np.cov(data.T)


def compute_correlation(cov: np.ndarray) -> np.ndarray:
    """
    Normalize a covariance matrix into a correlation matrix.

    corr[i, j] = cov[i, j] / (sigma_i * sigma_j)

    The diagonal is always 1 and off-diagonal entries lie in [-1, 1],
    making it meaningful to compare with the identity matrix.

    Args:
        cov: Covariance matrix of shape (d, d)

    Returns:
        Correlation matrix of shape (d, d)
    """
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    return cov / outer_std


def corr_eigvals(corr: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of the correlation matrix (used for PCA dimension).

    Args:
        corr: Correlation matrix of shape (d, d)

    Returns:
        Eigenvalues in ascending order, shape (d,)
    """
    return np.linalg.eigvalsh(corr)


def compute_s_iso(cov: np.ndarray) -> float:
    """
    Compute the isotropy score S_iso = ||Σ - A·I||_F / (A·√d)

    A is the mean of the diagonal of Σ (mean per-dimension variance).
    S_iso = 0 means perfectly isotropic (Σ = A·I); larger values indicate
    stronger anisotropy relative to the average variance scale.

    Args:
        cov: Covariance matrix of shape (d, d)

    Returns:
        S_iso (float)
    """
    d = cov.shape[0]
    A = float(np.mean(np.diag(cov)))
    diff = cov - A * np.eye(d, dtype=cov.dtype)
    return float(np.linalg.norm(diff, "fro")) / (A * np.sqrt(d))


def compute_s_rad(data: np.ndarray, mean: np.ndarray) -> float:
    """
    Compute the radial concentration score S_rad = Var(||x-μ||²) / E(||x-μ||²)²

    Uses centered vectors (x - μ) so the metric captures spread around the
    data centroid rather than the origin.
    S_rad = 0 means all points are equidistant from the centroid; larger
    values indicate greater radial dispersion.

    Args:
        data: Array of shape (n, d), float32
        mean: Mean vector of shape (d,)

    Returns:
        S_rad (float)
    """
    centered = data.astype(np.float64) - mean.astype(np.float64)
    sq_norms = np.sum(centered ** 2, axis=1)   # (n,)
    e_sq = float(np.mean(sq_norms))
    var_sq = float(np.var(sq_norms, ddof=1))
    return var_sq / (e_sq ** 2)


def compute_pca_dim(eigvals: np.ndarray, threshold: float = 0.99) -> int:
    """
    Find the minimum number of PCA dimensions needed to explain at least
    `threshold` of total variance, given eigenvalues of the covariance (or
    correlation) matrix.

    Args:
        eigvals: Eigenvalues in any order (negatives clipped to 0 for stability)
        threshold: Variance explained threshold, default 0.99

    Returns:
        Minimum number of dimensions explaining >= threshold of total variance
    """
    vals = np.maximum(eigvals, 0.0)
    vals = np.sort(vals)[::-1]          # descending
    cumvar = np.cumsum(vals) / vals.sum()
    return int(np.searchsorted(cumvar, threshold) + 1)


def compute_lid(data: np.ndarray, k: int = LID_KS[0], sample_size: int = LID_SAMPLE) -> float:
    """
    Estimate the global Local Intrinsic Dimensionality (LID) using the
    Levina-Bickel MLE estimator on a random sample of points.

    For each sampled point x with k-NN distances r_1 <= ... <= r_k:
        LID(x) = [ (1/(k-1)) * sum_{i=1}^{k-1} log(r_k / r_i) ]^{-1}

    Global LID is the mean over all sampled points.

    Args:
        data: Array of shape (n, d), float32
        k: Number of nearest neighbours (excluding the point itself)
        sample_size: Number of points to sample for estimation

    Returns:
        Estimated global LID (float)
    """
    n = data.shape[0]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = data[sample_idx]

    try:
        import faiss
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data)
        # faiss returns squared L2; +1 because first hit is the point itself
        D_sq, _ = index.search(sample, k + 1)
        dists = np.sqrt(np.maximum(D_sq[:, 1:], 0.0))   # shape (sample_size, k)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(data)
        dists, _ = nn.kneighbors(sample)
        dists = dists[:, 1:]                             # shape (sample_size, k)

    r_k = dists[:, -1]                  # furthest neighbour distance
    valid = r_k > 0
    dists, r_k = dists[valid], r_k[valid]

    # log(r_k / r_i) for i = 1 .. k-1
    log_ratios = np.log(r_k[:, None] / np.maximum(dists[:, :-1], 1e-10))
    lid_per_point = (k - 1) / np.sum(log_ratios, axis=1)
    return float(np.mean(lid_per_point))


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise squared L2 distances between rows of A and B."""
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    A2 = np.sum(A ** 2, axis=1, keepdims=True)
    B2 = np.sum(B ** 2, axis=1, keepdims=True)
    return np.maximum(A2 + B2.T - 2.0 * (A @ B.T), 0.0)


def compute_mmd(train: np.ndarray, test: np.ndarray,
                sample_size: int = MMD_SAMPLE) -> float:
    """
    Estimate MMD²(train, test) using an RBF kernel with bandwidth chosen by
    the median heuristic.

    Uses the biased estimator:
        MMD²_b = (1/n²) Σ_{i,j} k(x_i,x_j)
               - (2/nm) Σ_{i,j} k(x_i,y_j)
               + (1/m²) Σ_{i,j} k(y_i,y_j)

    The biased estimator equals ||μ_X - μ_Y||²_H in the RKHS and is
    guaranteed non-negative, unlike the unbiased estimator which can go
    slightly negative when the two distributions are nearly identical.

    Args:
        train: Array of shape (n_train, d)
        test:  Array of shape (n_test,  d)
        sample_size: Points sampled from each split

    Returns:
        Estimated MMD² (float, always >= 0)
    """
    rng = np.random.default_rng(42)
    X = train[rng.choice(len(train), min(sample_size, len(train)), replace=False)].astype(np.float64)
    Y = test [rng.choice(len(test),  min(sample_size, len(test)),  replace=False)].astype(np.float64)

    # Median heuristic: σ² = median(pairwise sq-dists on combined subsample) / 2
    n_med = min(500, len(X), len(Y))
    combined = np.vstack([X[:n_med], Y[:n_med]])
    sq = _sq_dists(combined, combined)
    upper = sq[np.triu_indices(len(combined), k=1)]
    sigma2 = float(np.median(upper)) / 2.0
    if sigma2 == 0.0:
        sigma2 = 1.0

    K_XX = np.exp(-_sq_dists(X, X) / (2.0 * sigma2))
    K_YY = np.exp(-_sq_dists(Y, Y) / (2.0 * sigma2))
    K_XY = np.exp(-_sq_dists(X, Y) / (2.0 * sigma2))

    n, m = len(X), len(Y)
    mmd2 = (K_XX.sum() / (n * n)
            - 2.0 * K_XY.sum() / (n * m)
            + K_YY.sum() / (m * m))
    return float(mmd2)


def compute_knn_cross_ratio(
    train: np.ndarray,
    test: np.ndarray,
    k: int = KNN_CROSS_K,
    sample_size: int = KNN_CROSS_SAMPLE,
) -> float:
    """
    Compute the kNN cross-distance ratio:

        R = mean_d(test -> train) / mean_d(train -> train)

    Numerator: for each sampled test point, mean distance to its k-NN in train.
    Denominator: for each sampled train point, mean distance to its k-NN in
                 train excluding itself (k+1 search, drop first hit).

    R ≈ 1 means test and train occupy similar regions; R > 1 indicates
    test points lie farther from the training distribution.

    Args:
        train: Array of shape (n_train, d)
        test:  Array of shape (n_test,  d)
        k: Number of nearest neighbours
        sample_size: Points sampled from each split for the query

    Returns:
        R (float)
    """
    rng = np.random.default_rng(42)
    q_test  = test [rng.choice(len(test),  min(sample_size, len(test)),  replace=False)]
    q_train = train[rng.choice(len(train), min(sample_size, len(train)), replace=False)]
    train_f32 = train.astype(np.float32)

    try:
        import faiss
        index = faiss.IndexFlatL2(train_f32.shape[1])
        index.add(train_f32)

        # test -> train (no self-hit issue)
        D_te, _ = index.search(q_test.astype(np.float32), k)
        mean_cross = float(np.mean(np.sqrt(np.maximum(D_te, 0.0))))

        # train -> train (k+1 to skip self, then drop first column)
        D_tr, _ = index.search(q_train.astype(np.float32), k + 1)
        mean_within = float(np.mean(np.sqrt(np.maximum(D_tr[:, 1:], 0.0))))

    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(train_f32)

        D_te, _ = nn.kneighbors(q_test.astype(np.float32), n_neighbors=k)
        mean_cross = float(np.mean(D_te))

        D_tr, _ = nn.kneighbors(q_train.astype(np.float32), n_neighbors=k + 1)
        mean_within = float(np.mean(D_tr[:, 1:]))

    return mean_cross / mean_within if mean_within > 0 else float("inf")


def analyze_dataset(dataset_name: str) -> dict:
    """
    Load a dataset and compute statistics of its train-split.

    Returns a dict with scalar summary statistics only (no full matrix).
    """
    print(f"\n[INFO] Loading dataset: {dataset_name}")
    hdf5_file, dimension = get_dataset(dataset_name)

    raw_dtype = str(hdf5_file["train"].dtype)
    point_type = hdf5_file.attrs.get("point_type", raw_dtype)

    train_data = np.array(hdf5_file["train"], dtype=np.float32)
    test_data  = np.array(hdf5_file["test"],  dtype=np.float32)
    hdf5_file.close()

    n_train = train_data.shape[0]
    print(f"       n_train={n_train}, dimension={dimension}, dtype={point_type}")

    print(f"       Computing mean ...")
    mean = compute_mean(train_data)

    print(f"       Computing covariance matrix ({dimension}×{dimension}) ...")
    cov = compute_covariance(train_data)
    variance = np.diag(cov)

    corr = compute_correlation(cov)
    eigvals_corr = corr_eigvals(corr)

    print(f"       Computing PCA 90/95/99% variance dimensions ...")
    pca_dim_90 = compute_pca_dim(eigvals_corr, threshold=0.90)
    pca_dim_95 = compute_pca_dim(eigvals_corr, threshold=0.95)
    pca_dim_99 = compute_pca_dim(eigvals_corr, threshold=0.99)

    print(f"       Computing S_iso and S_rad ...")
    s_iso = compute_s_iso(cov)
    # S_rad * d: for an isotropic Gaussian this equals 2 (chi-squared identity)
    s_rad = compute_s_rad(train_data, mean) * dimension

    centered_norms = np.linalg.norm(
        train_data.astype(np.float64) - mean.astype(np.float64), axis=1
    )
    norm_q90 = float(np.percentile(centered_norms, 90))
    norm_q99 = float(np.percentile(centered_norms, 99))
    norm_q99_q90 = norm_q99 / norm_q90 if norm_q90 > 0 else float("inf")

    print(f"       Computing LID (k={LID_KS}, sample={LID_SAMPLE}) ...")
    lids = {k: compute_lid(train_data, k=k, sample_size=LID_SAMPLE) for k in LID_KS}

    print(f"       Computing test mean and covariance ...")
    mean_test = compute_mean(test_data)
    cov_test  = compute_covariance(test_data)

    mean_diff_l2 = float(np.linalg.norm(mean - mean_test))
    cov_diff_rel = float(np.linalg.norm(cov - cov_test, "fro") / np.linalg.norm(cov, "fro"))

    print(f"       Computing MMD² (sample={MMD_SAMPLE}) ...")
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_train)
    train1 = train_data[perm[: n_train // 2]]
    train2 = train_data[perm[n_train // 2 :]]
    mmd2_train_test  = compute_mmd(train_data, test_data, sample_size=MMD_SAMPLE)
    mmd2_train_split = compute_mmd(train1, train2,        sample_size=MMD_SAMPLE)
    mmd2_ratio = mmd2_train_test / mmd2_train_split if mmd2_train_split != 0 else float("inf")


    print(f"       dtype                          = {point_type}")
    print(f"       PCA dims 90/95/99% var         = {pca_dim_90} / {pca_dim_95} / {pca_dim_99}  (out of {dimension})")
    print(f"       S_iso                          = {s_iso:.6f}")
    print(f"       S_rad·d (Gaussian=2)           = {s_rad:.6f}")
    print(f"       ||x-μ|| q99/q90                = {norm_q99_q90:.4f}")
    for k, v in lids.items():
        print(f"       LID (k={k:<3})                    = {v:.4f}")
    print(f"       ||μ_train - μ_test||_2         = {mean_diff_l2:.6f}")
    print(f"       ||Σ_tr - Σ_te||_F/||Σ_tr||_F  = {cov_diff_rel:.6f}")
    print(f"       MMD²(train, test)              = {mmd2_train_test:.6f}")
    print(f"       MMD²(train1, train2) baseline  = {mmd2_train_split:.6f}")
    print(f"       MMD ratio (train-test/baseline)= {mmd2_ratio:.4f}")

    print(f"       Computing kNN cross-distance ratio (k={KNN_CROSS_K}) ...")
    knn_ratio = compute_knn_cross_ratio(train_data, test_data, k=KNN_CROSS_K, sample_size=KNN_CROSS_SAMPLE)
    print(f"       kNN cross-dist ratio           = {knn_ratio:.4f}")

    print(f"       mean  min={mean.min():.4f}  max={mean.max():.4f}  "
          f"l2={np.linalg.norm(mean):.4f}")
    print(f"       var   min={variance.min():.4f}  max={variance.max():.4f}")

    return {
        "dataset":        dataset_name,
        "n_train":        n_train,
        "dimension":      dimension,
        "point_type":     point_type,
        "mean":           mean,
        "variance":       variance,
        "pca_dim_90":     pca_dim_90,
        "pca_dim_95":     pca_dim_95,
        "pca_dim_99":     pca_dim_99,
        "s_iso":          s_iso,
        "s_rad":          s_rad,
        "norm_q99_q90":   norm_q99_q90,
        "lids":           lids,
        "mean_diff_l2":   mean_diff_l2,
        "cov_diff_rel":   cov_diff_rel,
        "mmd2_train_test":    mmd2_train_test,
        "mmd2_train_split":   mmd2_train_split,
        "mmd2_ratio":         mmd2_ratio,
        "knn_ratio":          knn_ratio,
    }


def save_results(results: dict) -> None:
    """
    Save mean vector to a PDF plot in OUTPUT_DIR.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds  = results["dataset"]
    mean = results["mean"]
    d   = len(mean)

    fig, ax = plt.subplots(figsize=(max(8, d // 8), 3))
    ax.bar(range(d), mean, width=1.0, color="#2c7bb6")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean value")
    ax.set_title(f"Per-dimension mean — {ds}")
    ax.set_xlim(-0.5, d - 0.5)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"{ds}_mean.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[OK]   Saved mean plot: {out}")
    plt.close(fig)


def _save_table_figure(rows: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lid_keys    = [f"lid_k{k}" for k in LID_KS]
    lid_headers = [f"LID(k={k})" for k in LID_KS]

    col_keys = [
        "dataset",
        "n_train", "dimension", "point_type",
        "pca_dim_90", "pca_dim_95", "pca_dim_99",
        "s_iso", "s_rad", "norm_q99_q90",
        *lid_keys,
        "mean_diff_l2", "cov_diff_rel",
        "mmd2_train_test", "mmd2_train_split", "mmd2_ratio",
    ]
    col_headers = [
        "Dataset",
        "N", "D", "DType",
        "PCA90", "PCA95", "PCA99",
        "S_iso", "S_rad·d", "q99/q90",
        *lid_headers,
        "|Δμ|₂", "|ΔΣ|rel",
        "MMD²(tr,te)", "MMD²(split)", "MMD ratio",
    ]

    def _fmt(key, val):
        if key == "dataset":
            return str(val).replace("-euclidean", "")
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    cell_text = [[_fmt(k, row[k]) for k in col_keys] for row in rows]

    n_rows = len(cell_text)
    n_cols = len(col_headers)
    fig_w  = max(14, n_cols * 1.5)
    fig_h  = max(2,  n_rows * 0.55 + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(n_cols)))

    for col in range(n_cols):
        tbl[0, col].set_facecolor("#2c3e50")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    for row_idx in range(1, n_rows + 1):
        colour = "#f0f4f8" if row_idx % 2 == 0 else "white"
        for col in range(n_cols):
            tbl[row_idx, col].set_facecolor(colour)

    fig.tight_layout()
    stem = os.path.join(OUTPUT_DIR, "dataset_stats")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=150, bbox_inches="tight")
    print(f"[OK]   Saved table: {stem}.pdf / .png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets_to_run = [DATASET] if DATASET is not None else KNOWN_DATASETS

    summary_rows = []
    for ds in datasets_to_run:
        results = analyze_dataset(ds)
        summary_rows.append({
            "dataset":          results["dataset"],
            "n_train":          results["n_train"],
            "dimension":        results["dimension"],
            "point_type":       results["point_type"],
            "pca_dim_90":       results["pca_dim_90"],
            "pca_dim_95":       results["pca_dim_95"],
            "pca_dim_99":       results["pca_dim_99"],
            "s_iso":            results["s_iso"],
            "s_rad":            results["s_rad"],
            "norm_q99_q90":     results["norm_q99_q90"],
            **{f"lid_k{k}": results["lids"][k] for k in LID_KS},
            "mean_diff_l2":     results["mean_diff_l2"],
            "cov_diff_rel":     results["cov_diff_rel"],
            "mmd2_train_test":  results["mmd2_train_test"],
            "mmd2_train_split": results["mmd2_train_split"],
            "mmd2_ratio":       results["mmd2_ratio"],
            "knn_ratio":        results["knn_ratio"],
        })

    # ── Save figure table ──────────────────────────────────────
    _save_table_figure(summary_rows)
