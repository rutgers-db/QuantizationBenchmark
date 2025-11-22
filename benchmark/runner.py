import numpy as np
import yaml
from typing import Optional, Dict, Any, List
from .datasets import get_dataset
from .docker_runner import DockerRunner
import os


class BenchmarkRunner:
    """
    Main benchmark runner that coordinates the entire benchmarking process.
    Uses Docker containers to run algorithms in isolated environments.
    """

    def __init__(self, dataset_name: str, topk: int = 100, data_dir: str = "data"):
        """
        Initialize the benchmark runner.

        Args:
            dataset_name: Name of the HDF5 dataset in the data/ directory
            topk: Number of nearest neighbors to retrieve for recall calculation
            data_dir: Directory where datasets are stored (default: "data")
        """
        self.dataset_name = dataset_name
        self.topk = topk
        self.data_dir = data_dir

        # Load dataset
        self.hdf5_file, self.dimension = get_dataset(dataset_name, data_dir)
        self.train_data = np.array(self.hdf5_file['train'])
        self.test_data = np.array(self.hdf5_file['test'])
        self.ground_truth = np.array(self.hdf5_file['neighbors'])

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        print(f"  Dimension: {self.dimension}")
        print(f"  Train size: {self.train_data.shape}")
        print(f"  Test size: {self.test_data.shape}")
        print(f"  Ground truth size: {self.ground_truth.shape}")

        # Initialize Docker runner
        self.docker_runner = DockerRunner()

    def load_config(self, algo_type: str, algo_name: str) -> Dict[str, Any]:
        """
        Load configuration for an algorithm.

        Supports two formats:
        1. Dataset-specific: config.yaml contains a dict with dataset names as keys
        2. Default: config.yaml contains parameters directly

        Args:
            algo_type: 'quantizer' or 'dimreduction'
            algo_name: Name of the algorithm

        Returns:
            Dict containing configuration parameters for the current dataset
        """
        config_path = os.path.join(
            "benchmark/algorithms", algo_type, algo_name, "config.yaml"
        )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not config:
                    return {}

                # Check if config is organized by dataset
                if self.dataset_name in config:
                    # Return dataset-specific config
                    return config[self.dataset_name]
                elif isinstance(config, dict) and any(
                    key in config for key in ['ndim', 'nsubvec', 'nbit', 'target_dim']
                ):
                    # Config contains parameters directly (not organized by dataset)
                    return config
                else:
                    # Assume first key is a dataset name, return empty if current dataset not found
                    print(f"Warning: No configuration found for dataset '{self.dataset_name}' in {config_path}")
                    return {}
        return {}

    def run_benchmark(
        self,
        quantizer_name: str,
        dimreduction_name: Optional[str] = None,
        quantizer_config: Optional[Dict[str, Any]] = None,
        dimreduction_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete benchmark.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            quantizer_config: Optional config overrides for quantizer
            dimreduction_config: Optional config overrides for dimreduction

        Returns:
            Dict containing all benchmark results
        """
        results = {
            'dataset': self.dataset_name,
            'quantizer': quantizer_name,
            'dimreduction': dimreduction_name,
        }

        # Prepare data
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()

        # Phase 1: Dimensionality Reduction (if applicable)
        if dimreduction_name is not None:
            print(f"\n{'='*60}")
            print(f"Phase 1: Dimensionality Reduction - {dimreduction_name}")
            print(f"{'='*60}")

            # Build Docker image
            if not self.docker_runner.build_image('dimreduction', dimreduction_name):
                results['status'] = 'failed'
                results['error'] = f'Failed to build dimreduction image: {dimreduction_name}'
                return results

            # Load and merge config
            config = self.load_config('dimreduction', dimreduction_name)
            if dimreduction_config:
                config.update(dimreduction_config)

            # Run in Docker
            train_transformed, test_transformed, dim_metrics = self.docker_runner.run_dimreduction(
                dimreduction_name,
                train_data,
                test_data,
                config
            )

            if train_transformed is None:
                results['status'] = 'failed'
                results['error'] = 'Dimensionality reduction failed'
                return results

            # Update data
            train_data = train_transformed
            test_data = test_transformed

            # Store dimreduction metrics
            results['dim_reduction_time'] = dim_metrics['fit_time']
            results['dim_reduction_model_memory'] = dim_metrics['model_memory']
            results['dim_reduction_compression_rate'] = dim_metrics['compression_rate']
            results['original_dimension'] = dim_metrics['original_dim']
            results['reduced_dimension'] = dim_metrics['reduced_dim']

            print(f"\nDimensionality Reduction Results:")
            print(f"  Time: {dim_metrics['fit_time']:.4f}s")
            print(f"  Model Memory: {dim_metrics['model_memory'] / 1024 / 1024:.2f} MB")
            print(f"  Compression Rate: {dim_metrics['compression_rate']:.4f}x")
            print(f"  Dimension: {dim_metrics['original_dim']} -> {dim_metrics['reduced_dim']}")

        # Phase 2: Quantization
        print(f"\n{'='*60}")
        print(f"Phase 2: Quantization - {quantizer_name}")
        print(f"{'='*60}")

        # Build Docker image
        if not self.docker_runner.build_image('quantizer', quantizer_name):
            results['status'] = 'failed'
            results['error'] = f'Failed to build quantizer image: {quantizer_name}'
            return results

        # Load and merge config
        config = self.load_config('quantizer', quantizer_name)
        if quantizer_config:
            config.update(quantizer_config)

        # Run in Docker
        quant_results = self.docker_runner.run_quantizer(
            quantizer_name,
            train_data,
            test_data,
            self.ground_truth,
            self.topk,
            config
        )

        if quant_results is None or quant_results.get('status') == 'failed':
            results['status'] = 'failed'
            results['error'] = quant_results.get('error', 'Quantization failed') if quant_results else 'Quantization failed'
            return results

        # Merge quantizer results
        results.update({
            'training_time': quant_results['training_time'],
            'quantizer_memory': quant_results['quantizer_memory'],
            'quantizer_compression_rate': quant_results['compression_rate'],
            'mse': quant_results['mse'],
            'query_time': quant_results['query_time'],
            'queries_per_second': quant_results['queries_per_second'],
            'recall': quant_results['recall'],
            'status': 'success'
        })

        # Calculate total compression rate
        if dimreduction_name is not None:
            results['total_compression_rate'] = (
                results['dim_reduction_compression_rate'] *
                results['quantizer_compression_rate']
            )
        else:
            results['total_compression_rate'] = results['quantizer_compression_rate']

        # Print summary
        print(f"\n{'='*60}")
        print(f"Benchmark Summary")
        print(f"{'='*60}")
        print(f"  Training Time: {results['training_time']:.4f}s")
        print(f"  Quantizer Memory: {results['quantizer_memory'] / 1024 / 1024:.2f} MB")
        print(f"  Compression Rate (Quantizer): {results['quantizer_compression_rate']:.4f}x")
        print(f"  Total Compression Rate: {results['total_compression_rate']:.4f}x")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  Query Time: {results['query_time']:.4f}s")
        print(f"  Queries/Second: {results['queries_per_second']:.2f}")
        print(f"  Recall@{self.topk}: {results['recall']:.4f}")

        return results

    def close(self):
        """Close resources."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
