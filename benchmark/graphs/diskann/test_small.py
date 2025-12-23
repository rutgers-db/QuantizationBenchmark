#!/usr/bin/env python3
"""
Quick test with small random data
"""

import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, '/data/local/jl3288/QuantizationBenchmark')

from benchmark.graphs.diskann.module import DiskANN
from benchmark.algorithms.quantizer.ProductQuantizationFaiss.module import ProductQuantizationFaiss

def test_small():
    print("=" * 80)
    print("DiskANN Small Data Test (Single Thread)")
    print("=" * 80)

    # Very small dataset
    n_base = 1000
    n_query = 5
    dim = 128
    topk = 10

    print(f"\nDataset: {n_base} base vectors, {n_query} queries, dim={dim}")

    # Generate random data
    np.random.seed(42)
    base_data = np.random.random((n_base, dim)).astype('float32')
    queries = np.random.random((n_query, dim)).astype('float32')

    # Create quantizer
    print("\nCreating quantizer (nthread=1)...")
    quantizer = ProductQuantizationFaiss(
        ndim=dim,
        nsubvec=16,
        nbit=8,
        data_bytes=4,
        nthread=1,  # Single thread
        space="l2"
    )

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="diskann_small_")
    print(f"Temp dir: {temp_dir}")

    try:
        # Create index
        print("\nCreating DiskANN index (num_threads=1)...")
        index = DiskANN(
            quantizer=quantizer,
            R=16,
            L=32,
            num_threads=1,  # Single thread
            metric="l2"
        )

        # Build
        print("\nBuilding index...")
        success = index.build(n_base, base_data)
        if not success:
            print("❌ Build FAILED!")
            return False
        print("✓ Build completed")

        # Search
        print("\nSearching...")
        I, D = index.search(
            nq=n_query,
            queries=queries,
            topk=topk,
            search_L=32,
            beam_width=4
        )

        print(f"✓ Search completed")
        print(f"Result shapes: I={I.shape}, D={D.shape}")
        print(f"Sample result (query 0): {I[0]}")
        print(f"Sample distances: {D[0]}")

        print("\n" + "=" * 80)
        print("✓ TEST PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print(f"\nCleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_small()
    sys.exit(0 if success else 1)
