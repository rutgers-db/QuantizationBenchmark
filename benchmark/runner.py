import numpy as np
import yaml
from typing import Optional, Dict, Any, List
from .datasets import get_dataset
from .docker_runner import DockerRunner
import os
from itertools import product


def expand_param_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand parameter combinations from a config dict.

    Supports two formats:
    1. Simple format with list values:
        {nsubvec: [16, 32], nbit: [4, 8], other: 1}
       becomes:
        [{nsubvec: 16, nbit: 4, other: 1},
         {nsubvec: 16, nbit: 8, other: 1},
         {nsubvec: 32, nbit: 4, other: 1},
         {nsubvec: 32, nbit: 8, other: 1}]

    2. Build/Search separation format:
        {
          build: {nsubvec: [16, 32], nbit: [4, 8]},
          search: {nprobe: [1, 4]},
          common: {data_bytes: 4}
        }
       becomes:
        [{build_params: {nsubvec: 16, nbit: 4, data_bytes: 4},
          search_params: {nprobe: 1, data_bytes: 4}},
         {build_params: {nsubvec: 16, nbit: 4, data_bytes: 4},
          search_params: {nprobe: 4, data_bytes: 4}},
         ... (8 total combinations: 4 build x 2 search)]

    Args:
        config: Configuration dict that may contain list values or build/search structure

    Returns:
        List of config dicts with all combinations expanded
    """
    # Check if config has build/search structure
    if 'build' in config or 'search' in config:
        return _expand_build_search_combinations(config)

    # Simple format: expand list parameters
    list_params = {}
    fixed_params = {}

    for key, value in config.items():
        if isinstance(value, list):
            list_params[key] = value
        else:
            fixed_params[key] = value

    # If no list parameters, return single config
    if not list_params:
        return [config]

    # Generate all combinations
    param_names = list(list_params.keys())
    param_values = [list_params[name] for name in param_names]

    result = []
    for combination in product(*param_values):
        new_config = fixed_params.copy()
        for name, value in zip(param_names, combination):
            new_config[name] = value
        result.append(new_config)

    return result


def _expand_build_search_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand build/search parameter combinations.

    Args:
        config: Config dict with 'build', 'search', and optionally 'common' keys

    Returns:
        List of config dicts, each with 'build_params' and 'search_params' keys
    """
    common_params = config.get('common', {}) or {}  # Handle None case
    build_config = config.get('build', {}) or {}  # Handle None case
    search_config = config.get('search', {}) or {}  # Handle None case

    # Expand build parameters (include common params)
    build_list_params = {}
    build_fixed_params = common_params.copy()

    for key, value in build_config.items():
        if isinstance(value, list):
            build_list_params[key] = value
        else:
            build_fixed_params[key] = value

    # Generate all build combinations
    if build_list_params:
        build_param_names = list(build_list_params.keys())
        build_param_values = [build_list_params[name] for name in build_param_names]
        build_combinations = []
        for combination in product(*build_param_values):
            build_combo = build_fixed_params.copy()
            for name, value in zip(build_param_names, combination):
                build_combo[name] = value
            build_combinations.append(build_combo)
    else:
        build_combinations = [build_fixed_params]

    # Expand search parameters (DO NOT include common params - they're only for build)
    search_list_params = {}
    search_fixed_params = {}

    for key, value in search_config.items():
        if isinstance(value, list):
            search_list_params[key] = value
        else:
            search_fixed_params[key] = value

    # Generate all search combinations
    if search_list_params:
        search_param_names = list(search_list_params.keys())
        search_param_values = [search_list_params[name] for name in search_param_names]
        search_combinations = []
        for combination in product(*search_param_values):
            search_combo = search_fixed_params.copy()
            for name, value in zip(search_param_names, combination):
                search_combo[name] = value
            search_combinations.append(search_combo)
    else:
        search_combinations = [search_fixed_params]

    # Generate all (build, search) pairs
    result = []
    for build_params in build_combinations:
        for search_params in search_combinations:
            result.append({
                'build_params': build_params,
                'search_params': search_params
            })

    return result


class BenchmarkRunner:
    """
    Main benchmark runner that coordinates the entire benchmarking process.
    Uses Docker containers to run algorithms in isolated environments.
    """

    def __init__(self, dataset_name: str, topk: int = 100, data_dir: str = "data", debug: bool = False):
        """
        Initialize the benchmark runner.

        Args:
            dataset_name: Name of the HDF5 dataset in the data/ directory
            topk: Number of nearest neighbors to retrieve for recall calculation
            data_dir: Directory where datasets are stored (default: "data")
            debug: Enable debug mode with real-time Docker output (default: False)
        """
        self.dataset_name = dataset_name
        self.topk = topk
        self.data_dir = data_dir
        self.debug = debug

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
        if debug:
            print(f"  Debug mode: ENABLED (real-time Docker output)")

        # Initialize Docker runner
        self.docker_runner = DockerRunner(debug=debug)

    def load_config(self, algo_type: str, algo_name: str) -> List[Dict[str, Any]]:
        """
        Load configuration for an algorithm.

        Supports four formats:
        1. Dataset-specific with single config: config.yaml contains a dict with dataset names as keys
        2. Dataset-specific with multiple configs: config.yaml contains a list for each dataset
        3. Dataset-specific with parameter combinations: config contains list values that get expanded
           Example: {nsubvec: [16, 32], nbit: [4, 8]} -> 4 combinations
        4. Default: config.yaml contains parameters directly

        Args:
            algo_type: 'quantizer' or 'dimreduction'
            algo_name: Name of the algorithm

        Returns:
            List of configuration dicts for the current dataset
        """
        config_path = os.path.join(
            "benchmark/algorithms", algo_type, algo_name, "config.yaml"
        )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not config:
                    return [{}]

                # Check if config is organized by dataset
                if self.dataset_name in config:
                    # Return dataset-specific config
                    dataset_config = config[self.dataset_name]
                    # Check if it's a list of configs
                    if isinstance(dataset_config, list):
                        # Expand each config in case they contain list parameters
                        all_configs = []
                        for cfg in dataset_config:
                            all_configs.extend(expand_param_combinations(cfg))
                        return all_configs
                    else:
                        # Single config dict, expand parameter combinations
                        return expand_param_combinations(dataset_config)
                elif isinstance(config, dict) and any(
                    key in config for key in ['ndim', 'nsubvec', 'nbit', 'target_dim']
                ):
                    # Config contains parameters directly (not organized by dataset)
                    return expand_param_combinations(config)
                else:
                    # Assume first key is a dataset name, return empty if current dataset not found
                    print(f"Warning: No configuration found for dataset '{self.dataset_name}' in {config_path}")
                    return [{}]
        return [{}]

    def run_benchmark(
        self,
        quantizer_name: str,
        dimreduction_name: Optional[str] = None,
        quantizer_config: Optional[Dict[str, Any]] = None,
        dimreduction_config: Optional[Dict[str, Any]] = None,
        run_all_configs: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete benchmark.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            quantizer_config: Optional config overrides for quantizer
            dimreduction_config: Optional config overrides for dimreduction
            run_all_configs: If True, run all configs from YAML and return list of results

        Returns:
            Dict containing all benchmark results (or list of dicts if run_all_configs=True)
        """
        # If run_all_configs is True, load all configs and run them
        if run_all_configs:
            quantizer_configs = self.load_config('quantizer', quantizer_name)
            dimreduction_configs = None
            if dimreduction_name:
                dimreduction_configs = self.load_config('dimreduction', dimreduction_name)

            all_results = []
            for i, q_config in enumerate(quantizer_configs):
                # If dimreduction has configs, use the corresponding one (or first one)
                dr_config = None
                if dimreduction_configs:
                    dr_config = dimreduction_configs[i] if i < len(dimreduction_configs) else dimreduction_configs[0]

                result = self._run_single_benchmark(
                    quantizer_name, dimreduction_name, q_config, dr_config
                )
                all_results.append(result)

            return all_results

        # Single benchmark run
        # Load configs from YAML if not provided
        if quantizer_config is None:
            configs = self.load_config('quantizer', quantizer_name)
            quantizer_config = configs[0] if configs else {}

        if dimreduction_name and dimreduction_config is None:
            configs = self.load_config('dimreduction', dimreduction_name)
            dimreduction_config = configs[0] if configs else {}

        return self._run_single_benchmark(
            quantizer_name, dimreduction_name,
            quantizer_config, dimreduction_config
        )

    def _run_single_benchmark(
        self,
        quantizer_name: str,
        dimreduction_name: Optional[str],
        quantizer_config: Dict[str, Any],
        dimreduction_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run a single benchmark with specific configs.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            quantizer_config: Config for quantizer
            dimreduction_config: Config for dimreduction

        Returns:
            Dict containing all benchmark results
        """
        results = {
            'dataset': self.dataset_name,
            'quantizer': quantizer_name,
            'dimreduction': dimreduction_name,
            'quantizer_config': quantizer_config,
        }

        if dimreduction_config is not None:
            results['dimreduction_config'] = dimreduction_config

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

            # Run in Docker
            train_transformed, test_transformed, dim_metrics = self.docker_runner.run_dimreduction(
                dimreduction_name,
                train_data,
                test_data,
                dimreduction_config
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
            print(f"  Model Memory: {dim_metrics['model_memory'] / 1024:.2f} MB")
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

        # Run in Docker
        quant_results = self.docker_runner.run_quantizer(
            quantizer_name,
            train_data,
            test_data,
            self.ground_truth,
            self.topk,
            quantizer_config
        )
        
        print(quant_results)

        if quant_results is None or quant_results.get('status') == 'failed':
            results['status'] = 'failed'
            results['error'] = quant_results.get('error', 'Quantization failed') if quant_results else 'Quantization failed'
            return results

        # Merge quantizer results
        results.update({
            'training_time (s)': quant_results['training_time'],
            'quantizer_memory (KB)': quant_results['quantizer_memory'],
            'quantizer_compression_rate': quant_results['compression_rate'],
            'mse': quant_results['mse'],
            'query_time (s)': quant_results['query_time'],
            'queries_per_second': quant_results['queries_per_second'],
            'recall': quant_results['recall'],
            'rerank_results': quant_results.get('rerank_results', []),
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
        # print(f"\n{'='*60}")
        # print(f"Benchmark Summary")
        # print(f"{'='*60}")
        # print(f"  Training Time: {results['training_time']:.4f}s")
        # print(f"  Quantizer Memory: {results['quantizer_memory'] / 1024:.2f} MB")
        # print(f"  Compression Rate (Quantizer): {results['quantizer_compression_rate']:.4f}x")
        # print(f"  Total Compression Rate: {results['total_compression_rate']:.4f}x")
        # print(f"  MSE: {results['mse']:.6f}")
        # print(f"  Query Time: {results['query_time']:.4f}s")
        # print(f"  Queries/Second: {results['queries_per_second']:.2f}")
        # print(f"  Recall@{self.topk}: {results['recall']:.4f}")

        return results

    def close(self):
        """Close resources."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
