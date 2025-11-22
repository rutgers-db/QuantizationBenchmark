#!/usr/bin/env python3
"""
Docker entrypoint script that runs inside the container.
Loads the algorithm module, executes training/querying, and collects metrics.
"""

import argparse
import pickle
import time
import tracemalloc
import importlib.util
import sys
import numpy as np
from typing import Any, Dict


def load_module_class(module_path: str, base_class_name: str):
    """
    Dynamically load a class from a module file.

    Args:
        module_path: Path to the Python module file
        base_class_name: Name of the base class (e.g., 'BaseQuantizer')

    Returns:
        The loaded class
    """
    spec = importlib.util.spec_from_file_location("algorithm_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["algorithm_module"] = module
    spec.loader.exec_module(module)

    # Find the class that inherits from the base class
    from benchmark.base import BaseQuantizer, BaseDimReduction

    if base_class_name == 'BaseQuantizer':
        base_class = BaseQuantizer
    elif base_class_name == 'BaseDimReduction':
        base_class = BaseDimReduction
    else:
        raise ValueError(f"Unknown base class: {base_class_name}")

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and
            issubclass(attr, base_class) and
            attr is not base_class):
            return attr

    raise ValueError(f"No {base_class_name} subclass found in {module_path}")


def run_dimreduction(input_path: str, output_path: str, module_path: str):
    """
    Run dimensionality reduction algorithm.

    Args:
        input_path: Path to input pickle file
        output_path: Path to output pickle file
        module_path: Path to algorithm module
    """
    # Load input data
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)

    train_data = input_data['train_data']
    test_data = input_data['test_data']
    config = input_data['config']

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Config: {config}")

    # Load algorithm class
    DimReductionClass = load_module_class(module_path, 'BaseDimReduction')

    # Instantiate
    dim_reduction = DimReductionClass(**config)

    # Fit and transform
    print("\nFitting and transforming training data...")
    tracemalloc.start()
    start_time = time.time()

    n_train = train_data.shape[0]
    train_transformed = dim_reduction.fit_transform(n_train, train_data)
    fit_time = time.time() - start_time

    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Transform test data
    print("Transforming test data...")
    n_test = test_data.shape[0]
    test_transformed = dim_reduction.transform(n_test, test_data)

    # Collect metrics
    metrics = {
        'fit_time': fit_time,
        'peak_memory': peak_memory,
        'model_memory': dim_reduction.getMemoryUsage(),
        'compression_rate': dim_reduction.getCompressionRate(),
        'original_dim': train_data.shape[1],
        'reduced_dim': train_transformed.shape[1]
    }

    print(f"\nDimensionality Reduction Metrics:")
    print(f"  Fit time: {fit_time:.4f}s")
    print(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  Model memory: {metrics['model_memory'] / 1024 / 1024:.2f} MB")
    print(f"  Compression rate: {metrics['compression_rate']:.4f}x")
    print(f"  Dimension: {metrics['original_dim']} -> {metrics['reduced_dim']}")

    # Save output
    output = {
        'train_transformed': train_transformed,
        'test_transformed': test_transformed,
        'metrics': metrics
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nResults saved to {output_path}")


def run_quantizer(input_path: str, output_path: str, module_path: str):
    """
    Run quantizer algorithm.

    Args:
        input_path: Path to input pickle file
        output_path: Path to output pickle file
        module_path: Path to algorithm module
    """
    # Load input data
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)

    train_data = input_data['train_data']
    test_data = input_data['test_data']
    ground_truth = input_data['ground_truth']
    topk = input_data['topk']
    config = input_data['config']

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Top-k: {topk}")
    print(f"Config: {config}")

    # Load algorithm class
    QuantizerClass = load_module_class(module_path, 'BaseQuantizer')

    # Instantiate
    quantizer = QuantizerClass(**config)

    # Training phase
    print("\n=== Training Phase ===")
    tracemalloc.start()
    start_time = time.time()

    nd = train_data.shape[0]
    success = quantizer.fit(nd, train_data)
    training_time = time.time() - start_time

    current, peak_train_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not success:
        print("ERROR: Training failed!")
        output = {'status': 'failed', 'error': 'Training failed'}
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        return

    print(f"Training time: {training_time:.4f}s")
    print(f"Peak memory: {peak_train_memory / 1024 / 1024:.2f} MB")

    # Get quantizer metrics
    quantizer_memory = quantizer.getMemoryUsage()
    compression_rate = quantizer.getCompressionRate()
    mse = quantizer.getMSE()

    print(f"Quantizer memory: {quantizer_memory / 1024 / 1024:.2f} MB")
    print(f"Compression rate: {compression_rate:.4f}x")
    print(f"MSE: {mse:.6f}")

    # Query phase
    print("\n=== Query Phase ===")
    tracemalloc.start()
    start_time = time.time()

    nq = test_data.shape[0]
    I, D = quantizer.query(nq, test_data, topk)
    query_time = time.time() - start_time

    current, peak_query_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query time: {query_time:.4f}s")
    print(f"Peak memory: {peak_query_memory / 1024 / 1024:.2f} MB")

    # Calculate recall
    recall = calculate_recall(I, ground_truth[:, :topk])
    print(f"Recall@{topk}: {recall:.4f}")

    # Collect all metrics
    output = {
        'status': 'success',
        'training_time': training_time,
        'training_memory': peak_train_memory,
        'quantizer_memory': quantizer_memory,
        'compression_rate': compression_rate,
        'mse': mse,
        'query_time': query_time,
        'query_memory': peak_query_memory,
        'max_memory': max(peak_train_memory, peak_query_memory),
        'queries_per_second': len(test_data) / query_time if query_time > 0 else 0,
        'recall': recall,
        'predictions': I,
        'distances': D
    }

    # Save output
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nResults saved to {output_path}")


def calculate_recall(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate recall@k.

    Args:
        predictions: Predicted neighbor indices, shape (nq, k)
        ground_truth: Ground truth neighbor indices, shape (nq, k)

    Returns:
        Recall score
    """
    nq = predictions.shape[0]
    k = predictions.shape[1]

    total_correct = 0
    for i in range(nq):
        pred_set = set(predictions[i])
        gt_set = set(ground_truth[i])
        total_correct += len(pred_set & gt_set)

    recall = total_correct / (nq * k) if (nq * k) > 0 else 0.0
    return recall


def main():
    parser = argparse.ArgumentParser(description='Docker entrypoint for running algorithms')
    parser.add_argument('--mode', required=True, choices=['quantizer', 'dimreduction'],
                        help='Mode: quantizer or dimreduction')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--module', required=True, help='Algorithm module path')

    args = parser.parse_args()

    print(f"=== Docker Entrypoint ===")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Module: {args.module}")
    print()

    if args.mode == 'dimreduction':
        run_dimreduction(args.input, args.output, args.module)
    elif args.mode == 'quantizer':
        run_quantizer(args.input, args.output, args.module)


if __name__ == '__main__':
    main()
