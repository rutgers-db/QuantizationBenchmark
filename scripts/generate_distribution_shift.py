#!/usr/bin/env python3
"""
Generate a distribution-shift sample for an HDF5 benchmark dataset.

The script clusters the training set with k-means, randomly selects one
non-empty cluster, stores that cluster as the shift training sample, and
records metadata that quantifies how severe the shift is.
"""

import argparse
from typing import Tuple

import faiss
import h5py
import numpy as np


def _build_assignment_index(dimension: int, distance: str):
    if distance == "euclidean":
        return faiss.IndexFlatL2(dimension)
    if distance in {"ip", "inner_product"}:
        return faiss.IndexFlatIP(dimension)
    raise ValueError(f"Unsupported distance metric: {distance}")


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def generate_distribution_shift(
    hdf5_path: str,
    k: int,
    seed: int,
    group_name: str,
    niter: int,
) -> Tuple[int, int, float]:
    with h5py.File(hdf5_path, "r+") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        dimension = int(f.attrs.get("dimension", train.shape[1]))
        distance = str(f.attrs.get("distance", "euclidean")).lower()

        if train.shape[0] < k:
            raise ValueError(
                f"Training set size {train.shape[0]} is smaller than k={k}."
            )

        print(f"Loaded train set: {train.shape}")
        print(f"Running k-means with k={k}, niter={niter}, seed={seed}")

        kmeans = faiss.Kmeans(
            d=dimension,
            k=k,
            niter=niter,
            verbose=True,
            seed=seed,
        )
        kmeans.train(train)

        assignment_index = _build_assignment_index(dimension, distance)
        assignment_index.add(kmeans.centroids.astype(np.float32))
        _, assignments = assignment_index.search(train, 1)
        assignments = assignments[:, 0].astype(np.int64)

        histogram = np.bincount(assignments, minlength=k).astype(np.int64)
        non_empty_clusters = np.flatnonzero(histogram > 0)
        if non_empty_clusters.size == 0:
            raise RuntimeError("k-means produced no non-empty clusters.")

        rng = np.random.default_rng(seed)
        selected_cluster = int(rng.choice(non_empty_clusters))
        shift_indices = np.flatnonzero(assignments == selected_cluster).astype(np.int64)
        shift_sample = train[shift_indices]

        full_distribution = histogram.astype(np.float64)
        full_distribution /= full_distribution.sum()
        shifted_distribution = np.zeros(k, dtype=np.float64)
        shifted_distribution[selected_cluster] = 1.0
        js_divergence = _js_divergence(full_distribution, shifted_distribution)

        if group_name in f:
            del f[group_name]

        shift_group = f.create_group(group_name)
        shift_group.create_dataset("shift_train", data=shift_sample)
        shift_group.create_dataset("shift_train_indices", data=shift_indices)
        shift_group.create_dataset("cluster_histogram", data=histogram)
        shift_group.create_dataset("cluster_centroids", data=kmeans.centroids.astype(np.float32))
        shift_group.attrs["k"] = int(k)
        shift_group.attrs["seed"] = int(seed)
        shift_group.attrs["niter"] = int(niter)
        shift_group.attrs["selected_cluster_id"] = int(selected_cluster)
        shift_group.attrs["selected_cluster_size"] = int(shift_indices.shape[0])
        shift_group.attrs["selected_cluster_fraction"] = float(
            shift_indices.shape[0] / train.shape[0]
        )
        shift_group.attrs["js_divergence"] = float(js_divergence)
        shift_group.attrs["distance"] = distance

    return selected_cluster, int(shift_indices.shape[0]), float(js_divergence)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a distribution-shift sample inside an HDF5 dataset."
    )
    parser.add_argument("hdf5_path", help="Path to the source HDF5 dataset")
    parser.add_argument("--k", type=int, default=100, help="Number of k-means clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--group-name",
        default="distribution_shift",
        help="HDF5 group used to store generated shift artifacts",
    )
    parser.add_argument("--niter", type=int, default=25, help="Number of k-means iterations")
    args = parser.parse_args()

    cluster_id, sample_size, js_divergence = generate_distribution_shift(
        hdf5_path=args.hdf5_path,
        k=args.k,
        seed=args.seed,
        group_name=args.group_name,
        niter=args.niter,
    )

    print("\nDistribution-shift sample generated successfully.")
    print(f"Selected cluster: {cluster_id}")
    print(f"Shift sample size: {sample_size}")
    print(f"Jensen-Shannon divergence: {js_divergence:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
