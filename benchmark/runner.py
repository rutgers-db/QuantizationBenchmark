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

    Supports two search formats:
    1. List format (new): Each topk can have its own nrerank values
       search:
         - topk: 100
           nrerank: [200, 300]
         - topk: 50
           nrerank: [100]
         - topk: 10  # no nrerank

    2. Dict format (legacy): Cartesian product of all search params
       search:
         topk: 100
         nrerank: [200, 300]

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

    # Expand search parameters
    # Check if search is a list (new format)
    if isinstance(search_config, list):
        # New format: search is a list where each item defines topk with optional nrerank
        search_combinations = _expand_search_experiments(search_config)
    else:
        # Legacy format: search is a dict, use Cartesian product of all search params
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


def _expand_search_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand search experiments where each topk can have its own nrerank values.

    Each experiment in the list can have:
    - topk: int (required)
    - nrerank: list or int (optional) - kept as list for docker_entrypoint.py to handle
    - other search params - if list values, will be expanded via Cartesian product

    Args:
        experiments: List of experiment configurations

    Returns:
        List of expanded search parameter dicts
    """
    result = []

    for exp in experiments:
        if 'topk' not in exp:
            raise ValueError("Each search experiment must have a 'topk' value")

        topk = exp['topk']
        nrerank = exp.get('nrerank', None)

        # Get other params (excluding topk and nrerank)
        other_params = {k: v for k, v in exp.items() if k not in ['topk', 'nrerank']}

        # Separate list and fixed params in other_params
        list_params = {}
        fixed_params = {}
        for key, value in other_params.items():
            if isinstance(value, list):
                list_params[key] = value
            else:
                fixed_params[key] = value

        # Generate all combinations of list params
        if list_params:
            param_names = list(list_params.keys())
            param_values = [list_params[name] for name in param_names]

            for combination in product(*param_values):
                search_params = {'topk': topk}
                # Add nrerank (keep as list if present)
                if nrerank is not None:
                    search_params['nrerank'] = nrerank
                # Add fixed params
                search_params.update(fixed_params)
                # Add expanded list params
                for name, value in zip(param_names, combination):
                    search_params[name] = value
                result.append(search_params)
        else:
            # No list params to expand
            search_params = {'topk': topk}
            if nrerank is not None:
                search_params['nrerank'] = nrerank
            search_params.update(fixed_params)
            result.append(search_params)

    return result


def group_configs_by_build_params(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group configurations by their build_params to avoid redundant index building.

    For build/search separation configs, groups by build_params and collects
    all search_params that use the same build configuration.

    Args:
        configs: List of configuration dicts (may have build_params/search_params)

    Returns:
        List of grouped configs with format:
        {
            'build_params': {...},
            'search_params_list': [{...}, {...}, ...]  # List of search configs
        }
        or for non-build/search configs, returns them unchanged
    """
    # Check if configs use build/search separation
    if not configs or 'build_params' not in configs[0]:
        # No build/search separation, return as-is
        return configs

    # Group by build_params
    from collections import defaultdict
    import json

    groups = defaultdict(list)
    for config in configs:
        # Use JSON serialization of build_params as key for grouping
        build_key = json.dumps(config['build_params'], sort_keys=True)
        groups[build_key].append(config['search_params'])

    # Convert back to list format
    result = []
    for build_key, search_params_list in groups.items():
        build_params = json.loads(build_key)
        result.append({
            'build_params': build_params,
            'search_params_list': search_params_list
        })

    return result


class BenchmarkRunner:
    """
    Main benchmark runner that coordinates the entire benchmarking process.
    Uses Docker containers to run algorithms in isolated environments.
    """

    def __init__(self, dataset_name: str, data_dir: str = "data", debug: bool = False):
        """
        Initialize the benchmark runner.

        Args:
            dataset_name: Name of the HDF5 dataset in the data/ directory
            data_dir: Directory where datasets are stored (default: "data")
            debug: Enable debug mode with real-time Docker output (default: False)

        Note:
            topk is now specified in the search parameters of config.yaml files
        """
        self.dataset_name = dataset_name
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
            algo_type: 'quantizer', 'dimreduction', or 'graph'
            algo_name: Name of the algorithm

        Returns:
            List of configuration dicts for the current dataset
        """
        if algo_type == 'graph':
            config_path = os.path.join(
                "benchmark/graphs", algo_name, "config.yaml"
            )
        else:
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
        graph_name: Optional[str] = None,
        quantizer_config: Optional[Dict[str, Any]] = None,
        dimreduction_config: Optional[Dict[str, Any]] = None,
        graph_config: Optional[Dict[str, Any]] = None,
        run_all_configs: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete benchmark.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            graph_name: Optional name of graph algorithm
            quantizer_config: Optional config overrides for quantizer
            dimreduction_config: Optional config overrides for dimreduction
            graph_config: Optional config overrides for graph
            run_all_configs: If True, run all configs from YAML and return list of results

        Returns:
            Dict containing all benchmark results (or list of dicts if run_all_configs=True)
        """
        # If run_all_configs is True, load all configs and run them
        if run_all_configs:
            if graph_name:
                # Load graph config
                graph_configs = self.load_config('graph', graph_name)
                quantizer_configs = self.load_config('quantizer', quantizer_name)

                # Group both graph and quantizer configs by build_params
                grouped_graph_configs = group_configs_by_build_params(graph_configs)
                grouped_quantizer_configs = group_configs_by_build_params(quantizer_configs)

                all_results = []
                # For each graph build config, run ALL quantizer build configs IN ONE Docker container
                for g_config in grouped_graph_configs:
                    result = self._run_graph_benchmark_multi_quantizer(
                        graph_name, quantizer_name, g_config, grouped_quantizer_configs
                    )
                    # Result should be a list of results for all quantizer configs
                    if isinstance(result, list):
                        all_results.extend(result)
                    else:
                        all_results.append(result)

                return all_results
            else:
                quantizer_configs = self.load_config('quantizer', quantizer_name)
                dimreduction_configs = None
                if dimreduction_name:
                    dimreduction_configs = self.load_config('dimreduction', dimreduction_name)

                # Group configs by build_params to avoid redundant building
                grouped_configs = group_configs_by_build_params(quantizer_configs)

                all_results = []
                for i, q_config in enumerate(grouped_configs):
                    # If dimreduction has configs, use the corresponding one (or first one)
                    dr_config = None
                    if dimreduction_configs:
                        dr_config = dimreduction_configs[i] if i < len(dimreduction_configs) else dimreduction_configs[0]

                    result = self._run_single_benchmark(
                        quantizer_name, dimreduction_name, q_config, dr_config
                    )
                    # If result is a list (from build/search separation), extend all_results
                    if isinstance(result, list):
                        all_results.extend(result)
                    else:
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

    def _run_build_search_benchmark(
        self,
        quantizer_name: str,
        dimreduction_name: Optional[str],
        quantizer_config: Dict[str, Any],
        dimreduction_config: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark with build/search separation - build once, search multiple times.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            quantizer_config: Config with 'build_params' and 'search_params_list'
            dimreduction_config: Config for dimreduction

        Returns:
            List of result dicts, one per search configuration
        """
        build_params = quantizer_config['build_params']
        search_params_list = quantizer_config['search_params_list']

        print(f"\n{'='*60}")
        print(f"Build/Search Separation Mode")
        print(f"{'='*60}")
        print(f"Build params: {build_params}")
        print(f"Number of search configs: {len(search_params_list)}")

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
                error_result = {
                    'status': 'failed',
                    'error': f'Failed to build dimreduction image: {dimreduction_name}'
                }
                return [error_result]

            # Run in Docker
            train_transformed, test_transformed, dim_metrics = self.docker_runner.run_dimreduction(
                dimreduction_name,
                train_data,
                test_data,
                dimreduction_config
            )

            if train_transformed is None:
                error_result = {
                    'status': 'failed',
                    'error': 'Dimensionality reduction failed'
                }
                return [error_result]

            # Update data
            train_data = train_transformed
            test_data = test_transformed

            print(f"\nDimensionality Reduction Results:")
            print(f"  Time: {dim_metrics['fit_time']:.4f}s")
            print(f"  Model Memory: {dim_metrics['model_memory'] / 1024:.2f} MB")

        # Phase 2: Quantization with multiple search configs
        print(f"\n{'='*60}")
        print(f"Phase 2: Quantization - {quantizer_name}")
        print(f"{'='*60}")

        # Build Docker image
        if not self.docker_runner.build_image('quantizer', quantizer_name):
            error_result = {
                'status': 'failed',
                'error': f'Failed to build quantizer image: {quantizer_name}'
            }
            return [error_result]

        # Run in Docker with build/search separation config
        config_for_docker = {
            'build_params': build_params,
            'search_params_list': search_params_list
        }

        quant_results = self.docker_runner.run_quantizer(
            quantizer_name,
            train_data,
            test_data,
            self.ground_truth,
            config_for_docker
        )

        if quant_results is None or quant_results.get('status') == 'failed':
            error_result = {
                'dataset': self.dataset_name,
                'quantizer': quantizer_name,
                'dimreduction': dimreduction_name,
                'status': 'failed',
                'error': quant_results.get('error', 'Unknown error') if quant_results else 'Quantizer failed'
            }
            return [error_result]

        # quant_results should contain a list of results, one per search config
        all_results = []
        results_list = quant_results.get('results_list', [quant_results])

        for i, search_result in enumerate(results_list):
            result = {
                'dataset': self.dataset_name,
                'quantizer': quantizer_name,
                'dimreduction': dimreduction_name,
                'quantizer_config': {
                    'build_params': build_params,
                    'search_params': search_params_list[i] if i < len(search_params_list) else search_result.get('search_params', {})
                }
            }

            # Add dimreduction metrics if applicable
            if dimreduction_name is not None:
                result['dimreduction_config'] = dimreduction_config
                result['dim_reduction_time'] = dim_metrics['fit_time']
                result['dim_reduction_model_memory'] = dim_metrics['model_memory']
                result['dim_reduction_compression_rate'] = dim_metrics['compression_rate']
                result['original_dimension'] = dim_metrics['original_dim']
                result['reduced_dimension'] = dim_metrics['reduced_dim']

            # Add quantizer results
            result.update(search_result)
            all_results.append(result)

        return all_results

    def _run_graph_benchmark_multi_quantizer(
        self,
        graph_name: str,
        quantizer_name: str,
        graph_config: Dict[str, Any],
        quantizer_configs: List[Dict[str, Any]]
    ):
        """
        Run a benchmark with graph + multiple quantizer configurations.

        Build the graph ONCE, then test all quantizer configurations on the same graph.

        Args:
            graph_name: Name of the graph algorithm
            quantizer_name: Name of the quantizer
            graph_config: Config for graph (contains build_params and search_params_list)
            quantizer_configs: List of quantizer configs (each contains build_params and search_params_list)

        Returns:
            List of result dicts, one per (quantizer_config, search_param) combination
        """
        print(f"\n{'='*60}")
        print(f"Multi-Quantizer Graph Benchmark: {graph_name} + {quantizer_name}")
        print(f"{'='*60}")
        print(f"Graph config: {graph_config}")
        print(f"Number of quantizer configs: {len(quantizer_configs)}")

        # Build combined Docker image
        if not self.docker_runner.build_graph_image(graph_name, quantizer_name):
            error_result = {
                'dataset': self.dataset_name,
                'graph': graph_name,
                'quantizer': quantizer_name,
                'status': 'failed',
                'error': f'Failed to build graph+quantizer image: {graph_name}+{quantizer_name}'
            }
            return [error_result]

        # Prepare config for Docker
        # Convert quantizer_configs list to build_params_list format
        quantizer_build_params_list = []
        for q_config in quantizer_configs:
            quantizer_build_params_list.append({
                'build_params': q_config['build_params']
            })

        combined_config = {
            'graph_params': graph_config,
            'quantizer_params': {
                'build_params_list': quantizer_build_params_list
            }
        }

        print(f"Sending {len(quantizer_build_params_list)} quantizer configs to Docker container")
        print("Graph will be built ONCE and reused for all quantizer configurations")

        # Run in Docker
        graph_results = self.docker_runner.run_graph(
            graph_name,
            quantizer_name,
            self.train_data,
            self.test_data,
            self.ground_truth,
            combined_config
        )

        if graph_results is None:
            error_result = {
                'dataset': self.dataset_name,
                'graph': graph_name,
                'quantizer': quantizer_name,
                'status': 'failed',
                'error': 'Graph benchmark failed'
            }
            return [error_result]

        # Parse results from new format
        # graph_results now has: build_time, graph_memory, quantizer_results (list)
        build_time = graph_results.get('build_time')
        graph_memory = graph_results.get('graph_memory')
        quantizer_results_list = graph_results.get('quantizer_results', [])

        if not quantizer_results_list:
            error_result = {
                'dataset': self.dataset_name,
                'graph': graph_name,
                'quantizer': quantizer_name,
                'status': 'failed',
                'error': 'No quantizer results returned'
            }
            return [error_result]

        # Flatten results: for each quantizer config, for each search config, create one result dict
        all_results = []
        for quant_idx, quant_result in enumerate(quantizer_results_list):
            quantizer_build_params = quant_result.get('quantizer_build_params', {})
            training_time = quant_result.get('training_time')
            quantizer_memory = quant_result.get('quantizer_memory')
            compression_rate = quant_result.get('compression_rate')
            mse = quant_result.get('mse')
            search_results = quant_result.get('search_results', [])

            # Create one result per search configuration
            for search_result in search_results:
                result = {
                    'dataset': self.dataset_name,
                    'graph': graph_name,
                    'quantizer': quantizer_name,
                    'graph_config': graph_config,
                    'quantizer_config': {
                        'build_params': quantizer_build_params
                    },
                    'status': 'success',
                    # Graph build metrics (shared across all quantizers)
                    'build_time': build_time,
                    'graph_memory': graph_memory,
                    # Quantizer metrics (specific to this quantizer)
                    'training_time': training_time,
                    'quantizer_memory': quantizer_memory,
                    'compression_rate': compression_rate,
                    'mse': mse,
                    # Search metrics (specific to this search config)
                    'search_params': search_result.get('search_params', {}),
                    'query_time': search_result.get('query_time'),
                    'queries_per_second': search_result.get('queries_per_second'),
                    'recall': search_result.get('recall'),
                    'map': search_result.get('map'),
                    'recall@1': search_result.get('recall@1'),
                    'predictions': search_result.get('predictions'),
                    'distances': search_result.get('distances')
                }
                all_results.append(result)

        print(f"\nTotal results: {len(all_results)} (from {len(quantizer_results_list)} quantizer configs)")
        return all_results

    def _run_graph_benchmark(
        self,
        graph_name: str,
        quantizer_name: str,
        graph_config: Dict[str, Any],
        quantizer_config: Dict[str, Any]
    ):
        """
        Run a benchmark with graph + quantizer.

        Args:
            graph_name: Name of the graph algorithm
            quantizer_name: Name of the quantizer
            graph_config: Config for graph (may contain search_params_list)
            quantizer_config: Config for quantizer

        Returns:
            Dict containing all benchmark results, or List[Dict] if grouped config
        """
        # Check if this is a grouped config (build/search separation with multiple searches)
        if 'search_params_list' in graph_config:
            # TODO: Implement grouped graph benchmark if needed
            pass

        results = {
            'dataset': self.dataset_name,
            'graph': graph_name,
            'quantizer': quantizer_name,
            'graph_config': graph_config,
            'quantizer_config': quantizer_config,
        }

        print(f"\n{'='*60}")
        print(f"Graph + Quantizer Benchmark: {graph_name} + {quantizer_name}")
        print(f"{'='*60}")

        # Build combined Docker image
        if not self.docker_runner.build_graph_image(graph_name, quantizer_name):
            results['status'] = 'failed'
            results['error'] = f'Failed to build graph+quantizer image: {graph_name}+{quantizer_name}'
            return results

        # Merge configs for the docker entrypoint
        combined_config = {
            'graph_params': graph_config,
            'quantizer_params': quantizer_config
        }

        # Run in Docker
        graph_results = self.docker_runner.run_graph(
            graph_name,
            quantizer_name,
            self.train_data,
            self.test_data,
            self.ground_truth,
            combined_config
        )

        if graph_results is None:
            results['status'] = 'failed'
            results['error'] = 'Graph benchmark failed'
            return [results]

        # Handle the new result structure with search_results list
        # graph_results now has: training_time, build_time, search_results (list)
        search_results_list = graph_results.get('search_results', [])

        if not search_results_list:
            # No search results, return error
            results['status'] = 'failed'
            results['error'] = 'No search results returned'
            return [results]

        # Create one result dict per search configuration
        all_results = []
        for search_result in search_results_list:
            result = {
                'dataset': self.dataset_name,
                'graph': graph_name,
                'quantizer': quantizer_name,
                'graph_config': graph_config,
                'quantizer_config': quantizer_config,
                'status': 'success',
                # Add common metrics
                'training_time': graph_results.get('training_time'),
                'build_time': graph_results.get('build_time'),
                'quantizer_memory': graph_results.get('quantizer_memory'),
                'graph_memory': graph_results.get('graph_memory'),
                'compression_rate': graph_results.get('compression_rate'),
                'mse': graph_results.get('mse'),
                # Add search-specific metrics
                'search_params': search_result.get('search_params', {}),
                'query_time': search_result.get('query_time'),
                'queries_per_second': search_result.get('queries_per_second'),
                'recall': search_result.get('recall'),
                'map': search_result.get('map'),
                'recall@1': search_result.get('recall@1'),
                'predictions': search_result.get('predictions'),
                'distances': search_result.get('distances')
            }
            all_results.append(result)

        return all_results

    def _run_single_benchmark(
        self,
        quantizer_name: str,
        dimreduction_name: Optional[str],
        quantizer_config: Dict[str, Any],
        dimreduction_config: Optional[Dict[str, Any]]
    ):
        """
        Run a single benchmark with specific configs.

        Args:
            quantizer_name: Name of the quantizer algorithm
            dimreduction_name: Optional name of dimensionality reduction algorithm
            quantizer_config: Config for quantizer (may contain search_params_list for grouped configs)
            dimreduction_config: Config for dimreduction

        Returns:
            Dict containing all benchmark results, or List[Dict] if grouped config
        """
        # Check if this is a grouped config (build/search separation with multiple searches)
        if 'search_params_list' in quantizer_config:
            return self._run_build_search_benchmark(
                quantizer_name, dimreduction_name, quantizer_config, dimreduction_config
            )

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
            quantizer_config
        )

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
            'map': quant_results['map'],
            'recall@1': quant_results['recall@1'],
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
