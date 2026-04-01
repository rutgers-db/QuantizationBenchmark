import os
import random
import h5py
import numpy
from typing import Any, Callable, Dict, Optional, Tuple


def get_dataset_fn(dataset_name: str, data_dir: str = "/data/local/embedding_dataset/hdf5/") -> str:
    """
    Returns the full file path for a given dataset name in the data directory.

    Args:
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory where datasets are stored. Defaults to "/data/local/embedding_dataset/hdf5/".

    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return os.path.join(data_dir, f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str, data_dir: str = "/data/local/embedding_dataset/hdf5/") -> Tuple[h5py.File, int]:
    """
    hdf5 file should be located in the data directory

    Args:
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory where datasets are stored. Defaults to "/data/local/embedding_dataset/hdf5/".

    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name, data_dir)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def get_distribution_shift_data(
    hdf5_file: h5py.File,
    group_name: str = "distribution_shift",
) -> Optional[Dict[str, Any]]:
    """
    Load precomputed distribution-shift metadata from an HDF5 file.

    Args:
        hdf5_file: Opened HDF5 file handle.
        group_name: Group name that stores the shift experiment artifacts.

    Returns:
        Dict with shift metadata and indices, or None if the group is absent.
    """
    if group_name not in hdf5_file:
        return None

    shift_group = hdf5_file[group_name]
    metadata = {key: shift_group.attrs[key] for key in shift_group.attrs.keys()}
    metadata["group_name"] = group_name
    metadata["shift_train_indices"] = numpy.asarray(
        shift_group["shift_train_indices"], dtype=numpy.int64
    )

    if "cluster_histogram" in shift_group:
        metadata["cluster_histogram"] = numpy.asarray(
            shift_group["cluster_histogram"], dtype=numpy.int64
        )

    if "cluster_centroids" in shift_group:
        metadata["cluster_centroids_shape"] = tuple(shift_group["cluster_centroids"].shape)

    return metadata

def write_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, point_type: str = "float", count: int = 100) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes
    and stores the nearest neighbors and their distances for the test set using a
    brute-force approach.

    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    import faiss

    if distance not in ("euclidean"):
        raise NotImplementedError

    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        print(f"train size: {train.shape[0]} * {train.shape[1]}")
        print(f"test size:  {test.shape[0]} * {test.shape[1]}")
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        index = faiss.IndexFlatL2(len(train[0]))
        index.add(train.shape[0], train)
        I, D = index.search(test.shape[0], test, count)

        for i in range(test.shape[0]):
            # Save neighbors indices and distances
            neighbors_ds[i] = I[i, :]
            distances_ds[i] = D[i, :]
            


        
