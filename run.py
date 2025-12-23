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

    # List graph algorithms
    list_graphs()

    print("\n" + "="*60)
    print("Usage:")
    print("  Single quantizer:         --algorithm <Quantizer>")
    print("  DimReduction + Quantizer: --algorithm <DimReduction>,<Quantizer>")
    print("  Quantizer + Graph:        --algorithm <Quantizer> --graph <Graph>")
    print("="*60 + "\n")


def build_images(algorithm: str = None, graph: str = None, force_rebuild: bool = False):
    """
    Build Docker images for algorithms.

    Args:
        algorithm: Optional algorithm string (e.g., "PQ" or "PCA,PQ")
                  If None, build all algorithms
        graph: Optional graph algorithm name (e.g., "diskann")
        force_rebuild: Force rebuild even if image exists
    """
    print("\n" + "="*60)
    print("Building Docker Images")
    print("="*60 + "\n")

    docker_runner = DockerRunner()

    if algorithm or graph:
        # Build specific algorithm(s)
        if algorithm:
            dimreduction_name, quantizer_name = parse_algorithm_string(algorithm)

            if dimreduction_name:
                print(f"Building image for dimreduction: {dimreduction_name}")
                docker_runner.build_image('dimreduction', dimreduction_name, force_rebuild)

            if graph:
                # Build combined quantizer+graph image
                print(f"Building combined image for quantizer: {quantizer_name} + graph: {graph}")
                docker_runner.build_graph_image(graph, quantizer_name, force_rebuild)
            else:
                # Build quantizer only
                print(f"Building image for quantizer: {quantizer_name}")
                docker_runner.build_image('quantizer', quantizer_name, force_rebuild)
        elif graph:
            print(f"Error: --graph requires --algorithm to specify the quantizer")
            return
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


def list_graphs():
    """List all available graph algorithms."""
    graph_path = "benchmark/graphs"

    print("\nGraph Algorithms:")
    if os.path.exists(graph_path):
        graphs = [d for d in os.listdir(graph_path)
                 if os.path.isdir(os.path.join(graph_path, d)) and not d.startswith('__')]
        if graphs:
            for g in sorted(graphs):
                print(f"  - {g}")
        else:
            print("  (none)")
    else:
        print("  (none)")


def save_results(results, output_dir: str = "benchmark/results"):
    """
    Save benchmark results to JSON file with hierarchical structure.

    Structure:
    - Level 1: Group by build_params
    - Level 2: For each build_params group, show build metrics once + all search results

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

    # Handle single result or list of results
    if isinstance(results, list):
        # Multiple configurations - group by build_params
        first_result = results[0]

        # Generate filename based on whether this is a graph or quantizer result
        if 'graph' in first_result:
            # Graph result: dataset_graph_quantizer.json
            algo_str = f"{first_result['graph']}_{first_result['quantizer']}"
        else:
            # Quantizer result
            algo_str = first_result['quantizer']
            if first_result.get('dimreduction'):
                algo_str = f"{first_result['dimreduction']}_{algo_str}"

        filename = f"{first_result['dataset']}_{algo_str}.json"

        # Group results by build_params
        from collections import defaultdict
        groups = defaultdict(list)

        for result in results:
            # Check if this result has build_params (build/search separation)
            if 'quantizer_config' in result and isinstance(result['quantizer_config'], dict):
                if 'build_params' in result['quantizer_config']:
                    # Use JSON serialization of build_params as key
                    build_key = json.dumps(result['quantizer_config']['build_params'], sort_keys=True)
                    groups[build_key].append(result)
                else:
                    # Old format without build_params separation
                    groups['_no_build_params'].append(result)
            else:
                groups['_no_build_params'].append(result)

        # Format output with hierarchical structure
        serializable_results = []

        for build_key, group_results in groups.items():
            if build_key == '_no_build_params':
                # Old format - just convert directly
                for result in group_results:
                    serializable_results.append(convert_types(result))
            else:
                # New format with build/search separation
                build_params = json.loads(build_key)
                first = group_results[0]

                # Create grouped result structure
                # Check if this is a graph result or quantizer result
                is_graph = 'graph' in first

                grouped_result = {
                    'build_params': build_params,
                    'build_metrics': {
                        'training_time': first.get('training_time'),
                        'build_time': first.get('build_time') if is_graph else None,
                        'mse': first.get('mse'),
                        'quantizer_memory': first.get('quantizer_memory'),
                        'graph_memory': first.get('graph_memory') if is_graph else None,
                        'compression_rate': first.get('compression_rate')
                    },
                    'search_results': []
                }

                # Add graph/quantizer identifier
                if is_graph:
                    grouped_result['graph'] = first.get('graph')
                    grouped_result['quantizer'] = first.get('quantizer')

                # Add each search result
                for result in group_results:
                    search_result = {
                        'search_params': result.get('search_params', result['quantizer_config'].get('search_params', {})),
                        'metrics': {
                            'search_params': result.get('search_params', result['quantizer_config'].get('search_params', {})),
                            'query_time': result['query_time'],
                            'queries_per_second': result['queries_per_second'],
                            'recall': result['recall'],
                            'map': result['map'],
                            'recall@1': result['recall@1'],
                            'rerank_results': result.get('rerank_results', [])
                        }
                    }
                    grouped_result['search_results'].append(search_result)

                serializable_results.append(convert_types(grouped_result))
    else:
        # Single configuration
        algo_str = results['quantizer']
        if results.get('dimreduction'):
            algo_str = f"{results['dimreduction']}_{algo_str}"

        filename = f"{results['dataset']}_{algo_str}.json"

        # Convert result (convert_types now handles exclusion recursively)
        serializable_results = convert_types(results)

    filepath = os.path.join(output_dir, filename)

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


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
    parser.add_argument('--graph', type=str,
                       help='Graph algorithm to use with the quantizer (e.g., "diskann")')
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
        build_images(algorithm=args.algorithm, graph=args.graph, force_rebuild=args.force_rebuild)
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

    # Validate graph+dimreduction combination
    if args.graph and dimreduction_name:
        parser.error('Graph algorithms cannot be combined with dimensionality reduction. Use either --algorithm <Quantizer> --graph <Graph> or --algorithm <DimReduction>,<Quantizer>')

    # Run benchmark
    print("\n" + "="*60)
    print("Quantization Benchmark")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    if dimreduction_name:
        print(f"  Dimensionality Reduction: {dimreduction_name}")
    print(f"  Quantizer: {quantizer_name}")
    if args.graph:
        print(f"  Graph: {args.graph}")
    print("="*60)

    try:
        with BenchmarkRunner(args.dataset, data_dir=args.data_dir, debug=args.debug) as runner:
            results = runner.run_benchmark(
                quantizer_name=quantizer_name,
                dimreduction_name=dimreduction_name,
                graph_name=args.graph,
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
