#!/usr/bin/env python3
"""
Generate a distribution-shift sample for an HDF5 benchmark dataset.

The script clusters the training set with k-means, randomly selects one or
more non-empty clusters, samples a shift group from their union, and records
metadata that quantifies how severe the shift is.
"""

import argparse
from typing import List, Optional, Tuple

import h5py
import numpy as np


def _build_assignment_index(dimension: int, distance: str):
    import faiss

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
    selected_cluster_count: int,
    shift_size: Optional[int],
) -> Tuple[List[int], int, int, float]:
    import faiss

    with h5py.File(hdf5_path, "r+") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        dimension = int(f.attrs.get("dimension", train.shape[1]))
        distance = str(f.attrs.get("distance", "euclidean")).lower()

        if selected_cluster_count < 1:
            raise ValueError(
                f"selected_cluster_count must be >= 1, got {selected_cluster_count}."
            )
        if shift_size is not None and shift_size < 1:
            raise ValueError(f"shift_size must be >= 1, got {shift_size}.")
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
        if selected_cluster_count > non_empty_clusters.size:
            raise ValueError(
                "selected_cluster_count "
                f"({selected_cluster_count}) exceeds the number of non-empty "
                f"clusters ({non_empty_clusters.size})."
            )

        rng = np.random.default_rng(seed)
        selected_cluster_ids = np.sort(
            rng.choice(
                non_empty_clusters,
                size=selected_cluster_count,
                replace=False,
            ).astype(np.int64)
        )
        candidate_mask = np.isin(assignments, selected_cluster_ids)
        candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
        candidate_pool_size = int(candidate_indices.shape[0])

        if shift_size is None:
            shift_indices = candidate_indices
        else:
            if shift_size > candidate_pool_size:
                raise ValueError(
                    f"shift_size={shift_size} exceeds the candidate pool size "
                    f"{candidate_pool_size} from the selected clusters."
                )
            sampled_positions = np.sort(
                rng.choice(candidate_pool_size, size=shift_size, replace=False)
            )
            shift_indices = candidate_indices[sampled_positions]

        shift_sample = train[shift_indices]
        shift_assignments = assignments[shift_indices]

        full_distribution = histogram.astype(np.float64)
        full_distribution /= full_distribution.sum()
        shifted_histogram = np.bincount(shift_assignments, minlength=k).astype(np.int64)
        shifted_distribution = shifted_histogram.astype(np.float64)
        shifted_distribution /= shifted_distribution.sum()
        js_divergence = _js_divergence(full_distribution, shifted_distribution)

        if group_name in f:
            del f[group_name]

        shift_group = f.create_group(group_name)
        shift_group.create_dataset("shift_train", data=shift_sample)
        shift_group.create_dataset("shift_train_indices", data=shift_indices)
        shift_group.create_dataset("cluster_histogram", data=histogram)
        shift_group.create_dataset("shift_cluster_histogram", data=shifted_histogram)
        shift_group.create_dataset("cluster_centroids", data=kmeans.centroids.astype(np.float32))
        shift_group.attrs["k"] = int(k)
        shift_group.attrs["seed"] = int(seed)
        shift_group.attrs["niter"] = int(niter)
        shift_group.attrs["selected_cluster_count"] = int(selected_cluster_count)
        shift_group.attrs["selected_cluster_ids"] = selected_cluster_ids
        shift_group.attrs["candidate_pool_size"] = int(candidate_pool_size)
        shift_group.attrs["candidate_pool_fraction"] = float(
            candidate_pool_size / train.shape[0]
        )
        shift_group.attrs["shift_sample_size"] = int(shift_indices.shape[0])
        shift_group.attrs["shift_sample_fraction"] = float(
            shift_indices.shape[0] / train.shape[0]
        )
        if selected_cluster_count == 1:
            selected_cluster_id = int(selected_cluster_ids[0])
            shift_group.attrs["selected_cluster_id"] = selected_cluster_id
            shift_group.attrs["selected_cluster_size"] = int(histogram[selected_cluster_id])
            shift_group.attrs["selected_cluster_fraction"] = float(
                histogram[selected_cluster_id] / train.shape[0]
            )
        shift_group.attrs["js_divergence"] = float(js_divergence)
        shift_group.attrs["distance"] = distance

    return (
        selected_cluster_ids.tolist(),
        candidate_pool_size,
        int(shift_indices.shape[0]),
        float(js_divergence),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a distribution-shift sample inside an HDF5 dataset."
    )
    parser.add_argument("hdf5_path", help="Path to the source HDF5 dataset")
    parser.add_argument("--k", type=int, default=100, help="Number of k-means clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--selected-cluster-count",
        type=int,
        default=1,
        help="Number of non-empty clusters to select for the shift group",
    )
    parser.add_argument(
        "--shift-size",
        type=int,
        default=None,
        help="Total number of points to sample from the selected clusters; "
        "defaults to all points in their union",
    )
    parser.add_argument(
        "--group-name",
        default="distribution_shift",
        help="HDF5 group used to store generated shift artifacts",
    )
    parser.add_argument("--niter", type=int, default=25, help="Number of k-means iterations")
    args = parser.parse_args()

    selected_cluster_ids, candidate_pool_size, sample_size, js_divergence = generate_distribution_shift(
        hdf5_path=args.hdf5_path,
        k=args.k,
        seed=args.seed,
        group_name=args.group_name,
        niter=args.niter,
        selected_cluster_count=args.selected_cluster_count,
        shift_size=args.shift_size,
    )

    print("\nDistribution-shift sample generated successfully.")
    print(f"Selected clusters: {selected_cluster_ids}")
    print(f"Candidate pool size: {candidate_pool_size}")
    print(f"Shift sample size: {sample_size}")
    print(f"Jensen-Shannon divergence: {js_divergence:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
