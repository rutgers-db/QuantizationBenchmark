#!/usr/bin/env python3
"""
Test script for DiskANN with pluggable quantizer
Tests the integration with Product Quantization (Faiss)
"""

import numpy as np
import sys
import os
import tempfile
import time
import h5py

# Add paths
sys.path.insert(0, '/data/local/jl3288/QuantizationBenchmark')

from benchmark.graphs.diskann.module import DiskANN
from benchmark.algorithms.quantizer.ProductQuantizationFaiss.module import ProductQuantizationFaiss


def load_sift_dataset(dataset_path, n_base=None):
    """Load SIFT dataset from HDF5 file"""
    with h5py.File(dataset_path, 'r') as f:
        train = np.array(f['train'])
        test = np.array(f['test'])
        neighbors = np.array(f['neighbors'])
        distances = np.array(f['distances'])

    if n_base is not None and n_base < len(train):
        train = train[:n_base]

    return train, test, neighbors, distances


def compute_recall(gt_indices, pred_indices, k):
    """Compute recall@k"""
    recalls = []
    for i in range(len(gt_indices)):
        gt_set = set(gt_indices[i, :k])
        pred_set = set(pred_indices[i, :k])
        recall = len(gt_set & pred_set) / k
        recalls.append(recall)
    return np.mean(recalls)

def load_bin(file):
    f=open(file,'rb')
    a=np.fromfile(f, dtype=np.float32)
    [nd,dim] = a[:2].view(np.uint32)
    data=a[2:].reshape(nd,dim)
    return data

def load_gt(file):
    f=open(file,'rb')
    a=np.fromfile(f,dtype=np.uint32)
    [nq,topk] = a[:2]
    ids = a[2:2+nq*topk].reshape(nq,topk)
    dis = a[2+nq*topk:].view(np.float32).reshape(nq,topk)
    return ids,dis

def test_diskann_with_pq():
    """Test DiskANN with Product Quantization"""
    print("=" * 80)
    print("Testing DiskANN with Product Quantization (Faiss) on SIFT")
    print("=" * 80)

    # Load SIFT dataset
    # dataset_path = '/data/local/embedding_dataset/hdf5/sift-128-euclidean.hdf5'
    # topk = 10

    # print(f"\nLoading SIFT 1M dataset from {dataset_path}...")
    # base_data, queries, gt_neighbors, gt_distances = load_sift_dataset(dataset_path, n_base=None)  # Use full dataset
    # n_base = len(base_data)

    # # Use first 10 queries for testing
    # n_query = 1000
    # queries = queries[:n_query]
    # gt_I = gt_neighbors[:n_query, :topk]
    # gt_D = gt_distances[:n_query, :topk]
    
    dataset_path = "/data/local/embedding_dataset/sift1M/sift_base.bin"
    base_data = load_bin(dataset_path)
    query_path = "/data/local/embedding_dataset/sift1M/sift_query.bin"
    queries = load_bin(query_path)
    n_query = queries.shape[0]
    topk = 10
    gt_path = "/data/local/embedding_dataset/sift1M/sift_gt_with_dis.bin"
    gt_I, gt_D = load_gt(gt_path)
    gt_I = gt_I[:, :topk]
    gt_D = gt_D[:, :topk]

    n_base = base_data.shape[0]
    dim = base_data.shape[1]

    print(f"\nDataset:")
    print(f"  Base vectors: {base_data.shape[0]}")
    print(f"  Query vectors: {n_query}")
    print(f"  Dimension: {dim}")
    print(f"  Using pre-computed ground truth from dataset")
    print(f"  Sample ground truth for query 0: {gt_I[0]}")

    # Create quantizer (Product Quantization)
    print("\nCreating Product Quantization quantizer...")
    print("  PQ parameters: nsubvec=16, nbit=8")
    quantizer = ProductQuantizationFaiss(
        ndim=dim,
        nsubvec=16,  # Number of subquantizers
        nbit=8,  # Bits per subquantizer
        data_bytes=4,  # data_bytes parameter
        nthread=16,
        space="l2"
    )

    # Create temporary directory for index
    temp_dir = tempfile.mkdtemp(prefix="diskann_pq_test_")
    index_prefix = os.path.join(temp_dir, "index")
    print(f"\nIndex directory: {temp_dir}")

    try:
        # Create DiskANN index
        print("\nCreating DiskANN index...")
        print("  Graph parameters: R=32, L=50")
        index = DiskANN(
            quantizer=quantizer,
            R=32,  # Max out-degree
            L=50,  # Build complexity
            num_threads=16,
            metric="l2"
        )

        # Build index
        print("\n" + "-" * 80)
        print("Building index...")
        print("-" * 80)
        build_start = time.time()
        success = index.build(n_base, base_data)
        build_time = time.time() - build_start

        if not success:
            print("\n❌ Build FAILED!")
            return False

        print(f"\n✓ Build completed successfully in {build_time:.2f} seconds")

        # Get stats
        stats = index.get_index_stats()
        print(f"\nIndex statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test search
        print("\n" + "-" * 80)
        print("Testing search...")
        print("-" * 80)

        search_L = 50
        beam_width = 4

        print(f"Search parameters:")
        print(f"  TopK: {topk}")
        print(f"  Search L: {search_L}")
        print(f"  Beam width: {beam_width}")

        search_start = time.time()
        I, D = index.search(
            nq=n_query,
            queries=queries,
            topk=topk,
            search_L=search_L,
            beam_width=beam_width
        )
        search_time = time.time() - search_start
        

        print(f"\n✓ Search completed successfully in {search_time:.2f} seconds")
        print(f"  QPS: {n_query / search_time:.2f} queries/sec")
        print(f"  Latency per query: {search_time / n_query * 1000:.2f} ms")

        # Check results
        print(f"\n" + "-" * 80)
        print("Validating results...")
        print("-" * 80)

        print(f"Result shapes: I={I.shape}, D={D.shape}")
        print(f"\nSample results (first query):")
        print(f"  Indices: {I[0]}")
        print(f"  Distances: {D[0]}")

        # Verify all indices are valid
        if np.any(I < 0) or np.any(I >= n_base):
            print("\n❌ Invalid indices returned!")
            print(f"  Min index: {I.min()}, Max index: {I.max()}")
            return False
        print("  ✓ All indices are valid")

        # Verify distances are non-negative
        if np.any(D < 0):
            print("\n❌ Negative distances returned!")
            return False
        print("  ✓ All distances are non-negative")

        # Compute recall using pre-loaded ground truth
        print(f"\nGround truth (first query): {gt_I[0]}")
        print(f"Ground truth dists (first query): {gt_D[0]}")
        recall = compute_recall(gt_I, I, topk)
        print(f"\nRecall@{topk}: {recall:.4f}")

        # Success
        print("\n" + "=" * 80)
        print("✓ TEST PASSED - DiskANN with PQ working correctly!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ TEST FAILED with exception:")
        print("=" * 80)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup complete")


def main():
    """Run the test"""
    print("\nDiskANN with Product Quantization - Test Suite\n")

    success = test_diskann_with_pq()

    if success:
        print("\n🎉 All tests passed!\n")
        return 0
    else:
        print("\n❌ Tests failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
