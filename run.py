#!/usr/bin/env python3
"""
Main entry point for running quantization benchmarks.

Usage examples:
    # Run a single quantizer
    python run.py --dataset sift-128 --algorithm PQ

    # Run dimreduction + quantizer
    python run.py --dataset sift-128 --algorithm PCA,PQ

    # Build all Docker images
    python run.py --build-images

    # List available algorithms
    python run.py --list-algorithms

    # Specify custom data directory
    python run.py --dataset sift-128 --algorithm PQ --data-dir /path/to/datasets
"""

import argparse
import json
import os
import sys
from datetime import datetime
from benchmark.runner import BenchmarkRunner
from benchmark.docker_runner import DockerRunner


def list_algorithms():
    """List all available algorithms."""
    quantizer_path = "benchmark/algorithms/quantizer"
    dimreduction_path = "benchmark/algorithms/dimreduction"

    print("\n" + "="*60)
    print("Available Algorithms")
    print("="*60)

    # List quantizers
    print("\nQuantizers:")
    if os.path.exists(quantizer_path):
        quantizers = [d for d in os.listdir(quantizer_path)
                     if os.path.isdir(os.path.join(quantizer_path, d))]
        if quantizers:
            for q in sorted(quantizers):
                print(f"  - {q}")
        else:
            print("  (none)")
    else:
        print("  (none)")

    # List dimensionality reduction algorithms
    print("\nDimensionality Reduction:")
    if os.path.exists(dimreduction_path):
        dimreductions = [d for d in os.listdir(dimreduction_path)
                        if os.path.isdir(os.path.join(dimreduction_path, d))]
        if dimreductions:
            for dr in sorted(dimreductions):
                print(f"  - {dr}")
        else:
            print("  (none)")
    else:
        print("  (none)")

    print("\n" + "="*60)
    print("Usage:")
    print("  Single quantizer:        --algorithm <Quantizer>")
    print("  DimReduction + Quantizer: --algorithm <DimReduction>,<Quantizer>")
    print("="*60 + "\n")


def build_images(algorithm: str = None, force_rebuild: bool = False):
    """
    Build Docker images for algorithms.

    Args:
        algorithm: Optional algorithm string (e.g., "PQ" or "PCA,PQ")
                  If None, build all algorithms
        force_rebuild: Force rebuild even if image exists
    """
    print("\n" + "="*60)
    print("Building Docker Images")
    print("="*60 + "\n")

    docker_runner = DockerRunner()

    if algorithm:
        # Build specific algorithm(s)
        dimreduction_name, quantizer_name = parse_algorithm_string(algorithm)

        if dimreduction_name:
            print(f"Building image for dimreduction: {dimreduction_name}")
            docker_runner.build_image('dimreduction', dimreduction_name, force_rebuild)

        print(f"Building image for quantizer: {quantizer_name}")
        docker_runner.build_image('quantizer', quantizer_name, force_rebuild)
    else:
        # Build all algorithms
        docker_runner.build_all_images(force_rebuild=force_rebuild)

    print("\n" + "="*60)
    print("Build Complete")
    print("="*60 + "\n")


def parse_algorithm_string(algo_string: str):
    """
    Parse algorithm combination string.

    Args:
        algo_string: e.g., "PQ" or "PCA,PQ"

    Returns:
        Tuple of (dimreduction_name or None, quantizer_name)
    """
    parts = [p.strip() for p in algo_string.split(',')]

    if len(parts) == 1:
        return None, parts[0]
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(
            f"Invalid algorithm string: {algo_string}. "
            f"Expected format: 'Quantizer' or 'DimReduction,Quantizer'"
        )


def save_results(results, output_dir: str = "benchmark/results"):
    """
    Save benchmark results to JSON file with hierarchical structure.

    New structure:
    [
        {
            "experiment_type": "quantizer" or "ivf",
            "results": [
                {
                    "build_params": {...},
                    "search_results": [
                        {
                            "search_params": {...},
                            "metrics": {...}
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]

    Args:
        results: Benchmark results (dict or list of dicts)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    # Also recursively exclude large arrays (predictions, distances)
    def convert_types(obj, exclude_keys={'predictions', 'distances'}):
        if isinstance(obj, dict):
            return {k: convert_types(v, exclude_keys) for k, v in obj.items()
                   if k not in exclude_keys}
        elif isinstance(obj, list):
            return [convert_types(item, exclude_keys) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        return obj

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    if not results:
        return

    # Extract metadata from first result
    first_result = results[0]
    algo_str = first_result['quantizer']
    if first_result.get('dimreduction'):
        algo_str = f"{first_result['dimreduction']}_{algo_str}"
    filename = f"{first_result['dataset']}_{algo_str}.json"

    # Group results by experiment type (IVF vs quantizer)
    ivf_results = []
    quantizer_results = []

    for result in results:
        # Check if this is an IVF result
        is_ivf = (result.get('quantizer_config', {}).get('build_params', {}).get('nlist') is not None or
                  'nlist' in str(result.get('quantizer_config', {})))

        if is_ivf:
            ivf_results.append(result)
        else:
            quantizer_results.append(result)

    # Structure the output
    structured_output = []

    # Add IVF experiments first
    if ivf_results:
        ivf_experiment = _structure_experiment_results("ivf", ivf_results)
        structured_output.append(ivf_experiment)

    # Add quantizer experiments
    if quantizer_results:
        quantizer_experiment = _structure_experiment_results("quantizer", quantizer_results)
        structured_output.append(quantizer_experiment)

    # Convert to serializable format
    serializable_results = convert_types(structured_output)

    filepath = os.path.join(output_dir, filename)

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def _structure_experiment_results(experiment_type: str, results: list) -> dict:
    """
    Structure results for a single experiment type.

    Args:
        experiment_type: "ivf" or "quantizer"
        results: List of result dicts

    Returns:
        Structured experiment dict
    """
    from collections import defaultdict
    import json

    # Group by build_params
    grouped = defaultdict(list)
    for result in results:
        # Extract build params
        if 'quantizer_config' in result and 'build_params' in result['quantizer_config']:
            build_params = result['quantizer_config']['build_params']
        else:
            build_params = result.get('quantizer_config', {})

        # Use JSON as key for grouping
        build_key = json.dumps(build_params, sort_keys=True)
        grouped[build_key].append(result)

    # Build-related metrics (shared across all searches with same build params)
    build_metrics_keys = {
        'training_time (s)', 'training_time',
        'quantizer_memory (KB)', 'quantizer_memory',
        'quantizer_compression_rate', 'compression_rate',
        'mse',
        'dim_reduction_time', 'dim_reduction_model_memory',
        'dim_reduction_compression_rate',
        'original_dimension', 'reduced_dimension',
        'total_compression_rate'
    }

    # Structure the output
    experiment_results = []
    for build_key, group_results in grouped.items():
        build_params = json.loads(build_key)

        # Extract build metrics from first result (they should be same for all results in group)
        build_metrics = {}
        if group_results:
            first_result = group_results[0]
            for key in build_metrics_keys:
                if key in first_result:
                    build_metrics[key] = first_result[key]

        # Collect search results
        search_results = []
        for result in group_results:
            # Extract search params
            if 'quantizer_config' in result and 'search_params' in result['quantizer_config']:
                search_params = result['quantizer_config']['search_params']
            else:
                search_params = {}

            # Extract search-specific metrics (query time, recall, etc.)
            search_metrics = {k: v for k, v in result.items()
                             if k not in ['quantizer_config', 'dimreduction_config', 'status',
                                         'dataset', 'quantizer', 'dimreduction']
                             and k not in build_metrics_keys}

            search_results.append({
                "search_params": search_params,
                "metrics": search_metrics
            })

        experiment_results.append({
            "build_params": build_params,
            "build_metrics": build_metrics,
            "search_results": search_results
        })

    return {
        "experiment_type": experiment_type,
        "results": experiment_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Quantization Benchmark Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run PQ quantizer on SIFT dataset
  python run.py --dataset sift-128 --algorithm PQ

  # Run PCA dimensionality reduction + PQ quantizer
  python run.py --dataset sift-128 --algorithm PCA,PQ

  # Specify custom data directory
  python run.py --dataset sift-128 --algorithm PQ --data-dir /path/to/datasets

  # Build all Docker images
  python run.py --build-images

  # Build Docker image for specific algorithm
  python run.py --build-images --algorithm ProductQuantizationFaiss

  # List available algorithms
  python run.py --list-algorithms
        """
    )

    # Main actions
    parser.add_argument('--dataset', type=str,
                       help='Name of the dataset (HDF5 file in data/ directory without .hdf5 extension)')
    parser.add_argument('--algorithm', type=str,
                       help='Algorithm(s) to run. Format: "Quantizer" or "DimReduction,Quantizer"')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory where datasets are stored (default: data)')

    # Utility actions
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List all available algorithms and exit')
    parser.add_argument('--build-images', action='store_true',
                       help='Build Docker images. Use with --algorithm to build specific algorithm, or without to build all')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of Docker images even if they exist')

    # Output options
    parser.add_argument('--output-dir', type=str, default='benchmark/results',
                       help='Directory to save results (default: benchmark/results)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')

    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with real-time Docker output')

    args = parser.parse_args()

    # Handle utility actions
    if args.list_algorithms:
        list_algorithms()
        return 0

    if args.build_images:
        build_images(algorithm=args.algorithm, force_rebuild=args.force_rebuild)
        return 0

    # Validate required arguments for benchmark run
    if not args.dataset or not args.algorithm:
        parser.error('--dataset and --algorithm are required for running benchmarks')

    # Parse algorithm string
    try:
        dimreduction_name, quantizer_name = parse_algorithm_string(args.algorithm)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Run benchmark
    print("\n" + "="*60)
    print("Quantization Benchmark")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    if dimreduction_name:
        print(f"  Dimensionality Reduction: {dimreduction_name}")
    print(f"  Quantizer: {quantizer_name}")
    print("="*60)

    try:
        with BenchmarkRunner(args.dataset, data_dir=args.data_dir, debug=args.debug) as runner:
            results = runner.run_benchmark(
                quantizer_name=quantizer_name,
                dimreduction_name=dimreduction_name,
                run_all_configs=True  # Run all parameter configurations
            )

        # Handle results (could be a list if multiple configs)
        if isinstance(results, list):
            # Multiple configurations
            success_count = sum(1 for r in results if r.get('status') == 'success')

            # Save all results to a single file
            if not args.no_save and success_count > 0:
                save_results(results, args.output_dir)

            # Print summary
            print("\n" + "="*60)
            print(f"Benchmark completed: {success_count}/{len(results)} configurations succeeded")
            print("="*60 + "\n")

            return 0 if success_count == len(results) else 1
        else:
            # Single configuration
            if not args.no_save and results.get('status') == 'success':
                save_results(results, args.output_dir)

            # Print final status
            print("\n" + "="*60)
            if results.get('status') == 'success':
                print("Benchmark completed successfully!")
            else:
                print(f"Benchmark failed: {results.get('error', 'Unknown error')}")
            print("="*60 + "\n")

            return 0 if results.get('status') == 'success' else 1

    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
