#!/usr/bin/env python3
"""
Docker entrypoint script that runs inside the container.
Loads the algorithm module, executes training/querying, and collects metrics.
"""

import argparse
import json
import pickle
import time
import importlib.util
import sys
import numpy as np
from typing import Any, Dict
from benchmark.base import BaseQuantizer


def load_module_class(module_path: str, base_class_name: str):
    """
    Dynamically load a class from a module file.

    Args:
        module_path: Path to the Python module file
        base_class_name: Name of the base class (e.g., 'BaseQuantizer', 'BaseGraphIndex')

    Returns:
        The loaded class
    """
    spec = importlib.util.spec_from_file_location("algorithm_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["algorithm_module"] = module
    spec.loader.exec_module(module)

    # Find the class that inherits from the base class
    from benchmark.base import BaseQuantizer, BaseDimReduction, BaseGraphIndex

    if base_class_name == 'BaseQuantizer':
        base_class = BaseQuantizer
    elif base_class_name == 'BaseDimReduction':
        base_class = BaseDimReduction
    elif base_class_name == 'BaseGraphIndex':
        base_class = BaseGraphIndex
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
    start_time = time.time()

    n_train = train_data.shape[0]
    train_transformed = dim_reduction.fit_transform(n_train, train_data)
    fit_time = time.time() - start_time

    # Transform test data
    print("Transforming test data...")
    n_test = test_data.shape[0]
    test_transformed = dim_reduction.transform(n_test, test_data)

    # Collect metrics
    metrics = {
        'fit_time': fit_time,
        'model_memory': dim_reduction.getMemoryUsage(),
        'compression_rate': dim_reduction.getCompressionRate(),
        'original_dim': train_data.shape[1],
        'reduced_dim': train_transformed.shape[1]
    }

    print(f"\nDimensionality Reduction Metrics:")
    print(f"  Fit time: {fit_time:.4f}s")
    print(f"  Model memory: {metrics['model_memory'] / 1024:.2f} MB")
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
    config = input_data['config']
    add_data = input_data.get('add_data')

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    if add_data is not None:
        print(f"Add data shape: {add_data.shape}")
    print(f"Config: {config}")

    # Check if config has build/search separation
    if 'build_params' in config:
        build_params = config['build_params']

        # Check if we have multiple search configs (build once, search multiple times)
        if 'search_params_list' in config:
            search_params_list = config['search_params_list']
            print(f"\nUsing build/search separation with multiple search configs:")
            print(f"  Build params: {build_params}")
            print(f"  Number of search configs: {len(search_params_list)}")
        elif 'search_params' in config:
            search_params_list = [config['search_params']]
            print(f"\nUsing build/search separation:")
            print(f"  Build params: {build_params}")
            print(f"  Search params: {config['search_params']}")
        else:
            search_params_list = [{}]
    else:
        # Legacy format: all params for build, no search params
        build_params = config
        search_params_list = [{}]

    # Load algorithm class
    QuantizerClass = load_module_class(module_path, 'BaseQuantizer')

    # Instantiate with build parameters
    quantizer = QuantizerClass(**build_params)

    # Training phase (done once for all search configs)
    print("\n=== Training Phase ===")
    start_time = time.time()

    nd = train_data.shape[0]
    add_time = 0.0
    if add_data is not None:
        if (
            type(quantizer).train is BaseQuantizer.train or
            type(quantizer).add is BaseQuantizer.add
        ):
            output = {
                'status': 'failed',
                'error': (
                    f"{QuantizerClass.__name__} does not support separate train/add "
                    "required by the distribution-shift benchmark."
                ),
            }
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
            return

        success = quantizer.train(nd, train_data)
    else:
        success = quantizer.fit(nd, train_data)
    training_time = time.time() - start_time

    if not success:
        print("ERROR: Training failed!")
        output = {'status': 'failed', 'error': 'Training failed'}
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        return

    print(f"Training time: {training_time:.4f}s")

    if add_data is not None:
        print("\n=== Add Phase ===")
        add_start_time = time.time()
        add_success = quantizer.add(add_data.shape[0], add_data)
        add_time = time.time() - add_start_time

        if not add_success:
            print("ERROR: Add phase failed!")
            output = {'status': 'failed', 'error': 'Add phase failed'}
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
            return

        print(f"Add time: {add_time:.4f}s")

    # Get quantizer metrics (shared across all search configs)
    quantizer_memory = quantizer.getMemoryUsage()
    compression_rate = quantizer.getCompressionRate()
    mse = quantizer.getMSE()

    print(f"Quantizer memory: {quantizer_memory / 1024:.2f} MB")
    print(f"Compression rate: {compression_rate:.4f}x")
    print(f"MSE: {mse:.6f}")

    def _search_group_key(search_params: Dict[str, Any]) -> str:
        normalized = {k: v for k, v in search_params.items() if k not in ['topk', 'nrerank']}
        return json.dumps(normalized, sort_keys=True)

    grouped_search_configs = {}
    for search_idx, search_params in enumerate(search_params_list):
        group_key = _search_group_key(search_params)
        if group_key not in grouped_search_configs:
            grouped_search_configs[group_key] = []
        grouped_search_configs[group_key].append((search_idx, search_params))

    has_custom_search_and_rerank = type(quantizer).searchAndRerank is not BaseQuantizer.searchAndRerank
    nq = test_data.shape[0]
    all_results = [None] * len(search_params_list)

    for group_idx, grouped_configs in enumerate(grouped_search_configs.values()):
        representative_params = grouped_configs[0][1]
        search_params_clean = {
            k: v for k, v in representative_params.items() if k not in ['topk', 'nrerank']
        }
        max_topk = max(params.get('topk', 100) for _, params in grouped_configs)
        nrerank_pool = []
        for _, params in grouped_configs:
            if 'nrerank' in params:
                nrerank_value = params['nrerank']
                if isinstance(nrerank_value, list):
                    nrerank_pool.extend(nrerank_value)
                else:
                    nrerank_pool.append(nrerank_value)
        shared_query_k = max([max_topk] + nrerank_pool) if nrerank_pool else max_topk

        print(f"\n{'='*60}")
        print(f"Search Group {group_idx + 1}/{len(grouped_search_configs)}")
        print(f"{'='*60}")
        print(f"Shared search params: {search_params_clean}")
        print(f"Shared query top-k: {shared_query_k}")

        print("\n=== Shared Query Phase ===")
        start_time = time.time()
        I_shared, D_shared = quantizer.query(nq, test_data, shared_query_k, **search_params_clean)
        shared_query_time = time.time() - start_time
        print(f"Shared query time: {shared_query_time:.4f}s")

        prepared_candidates = None
        if nrerank_pool and not has_custom_search_and_rerank:
            prepared_candidates = quantizer.prepareRerankCandidates(test_data, I_shared)

        for search_idx, search_params in grouped_configs:
            print(f"\n{'-'*60}")
            print(f"Search Config {search_idx + 1}/{len(search_params_list)}")
            print(f"Search params: {search_params}")

            topk = search_params.get('topk', 100)
            print(f"Top-k: {topk}")

            I = I_shared[:, :topk]
            D = D_shared[:, :topk]
            query_time = shared_query_time

            recall = calculate_recall(I, ground_truth[:, :topk])
            map_score = calculate_map(I, ground_truth[:, :topk])
            recall_at_1 = calculate_recall_at_1(I, ground_truth[:, :topk])
            print(f"Recall@{topk}: {recall:.4f}")
            print(f"MAP@{topk}: {map_score:.4f}")
            print(f"Recall@1: {recall_at_1:.4f}")

            rerank_results = []
            if 'nrerank' in search_params:
                print("\n=== Search and Rerank Phase ===")
                nrerank_values = search_params['nrerank'] if isinstance(search_params['nrerank'], list) else [search_params['nrerank']]
                nrerank_values = sorted(nrerank_values)
                print(f"Testing with nrerank values: {nrerank_values}")

                if has_custom_search_and_rerank:
                    print("Using algorithm-specific searchAndRerank implementation.")
                    for nrerank in nrerank_values:
                        print(f"\nTesting nrerank={nrerank}...")
                        start_time = time.time()

                        try:
                            I_rerank, D_rerank = quantizer.searchAndRerank(
                                nq, test_data, topk, nrerank, **search_params_clean
                            )
                            rerank_time = time.time() - start_time

                            rerank_recall = calculate_recall(I_rerank, ground_truth[:, :topk])
                            rerank_map = calculate_map(I_rerank, ground_truth[:, :topk])
                            rerank_recall_at_1 = calculate_recall_at_1(I_rerank, ground_truth[:, :topk])

                            print(f"  Search+rerank time: {rerank_time:.4f}s")
                            print(f"  Recall@{topk} (after rerank): {rerank_recall:.4f}")
                            print(f"  MAP@{topk} (after rerank): {rerank_map:.4f}")
                            print(f"  Recall@1 (after rerank): {rerank_recall_at_1:.4f}")

                            rerank_results.append({
                                'nrerank': nrerank,
                                'search_time': rerank_time,
                                'rerank_only_time': 0.0,
                                'rerank_time': rerank_time,
                                'rerank_queries_per_second': len(test_data) / rerank_time if rerank_time > 0 else 0,
                                'rerank_recall': rerank_recall,
                                'rerank_map': rerank_map,
                                'rerank_recall@1': rerank_recall_at_1,
                                'predictions': I_rerank,
                                'distances': D_rerank
                            })
                        except Exception as e:
                            print(f"  Error in searchAndRerank with nrerank={nrerank}: {e}")
                            import traceback
                            traceback.print_exc()
                            rerank_results.append({
                                'nrerank': nrerank,
                                'error': str(e)
                            })
                else:
                    for nrerank in nrerank_values:
                        print(f"\nTesting nrerank={nrerank}...")
                        rerank_start_time = time.time()

                        try:
                            I_rerank, D_rerank = quantizer.rerankPreparedCandidates(
                                test_data,
                                prepared_candidates,
                                nrerank,
                                topk
                            )
                            rerank_only_time = time.time() - rerank_start_time
                            rerank_time = shared_query_time + rerank_only_time

                            rerank_recall = calculate_recall(I_rerank, ground_truth[:, :topk])
                            rerank_map = calculate_map(I_rerank, ground_truth[:, :topk])
                            rerank_recall_at_1 = calculate_recall_at_1(I_rerank, ground_truth[:, :topk])

                            print(f"  Search time (shared): {shared_query_time:.4f}s")
                            print(f"  Rerank-only time: {rerank_only_time:.4f}s")
                            print(f"  Total rerank time: {rerank_time:.4f}s")
                            print(f"  Recall@{topk} (after rerank): {rerank_recall:.4f}")
                            print(f"  MAP@{topk} (after rerank): {rerank_map:.4f}")
                            print(f"  Recall@1 (after rerank): {rerank_recall_at_1:.4f}")

                            rerank_results.append({
                                'nrerank': nrerank,
                                'search_time': shared_query_time,
                                'rerank_only_time': rerank_only_time,
                                'rerank_time': rerank_time,
                                'rerank_queries_per_second': len(test_data) / rerank_time if rerank_time > 0 else 0,
                                'rerank_recall': rerank_recall,
                                'rerank_map': rerank_map,
                                'rerank_recall@1': rerank_recall_at_1,
                                'predictions': I_rerank,
                                'distances': D_rerank
                            })
                        except Exception as e:
                            print(f"  Error in rerank with nrerank={nrerank}: {e}")
                            import traceback
                            traceback.print_exc()
                            rerank_results.append({
                                'nrerank': nrerank,
                                'error': str(e)
                            })
            else:
                print("\n=== Skipping Rerank Phase (no nrerank parameter in config) ===")

            all_results[search_idx] = {
                'status': 'success',
                'search_params': search_params,
                'training_time': training_time,
                'add_time': add_time,
                'quantizer_memory': quantizer_memory,
                'compression_rate': compression_rate,
                'mse': mse,
                'query_time': query_time,
                'queries_per_second': len(test_data) / query_time if query_time > 0 else 0,
                'recall': recall,
                'map': map_score,
                'recall@1': recall_at_1,
                'predictions': I,
                'distances': D,
                'rerank_results': rerank_results
            }

    # Save output
    # If multiple search configs, return results_list; otherwise return single result for backward compatibility
    if len(all_results) > 1:
        output = {
            'status': 'success',
            'results_list': all_results
        }
    else:
        output = all_results[0]

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nResults saved to {output_path}")


def run_graph(input_path: str, output_path: str, quantizer_module_path: str, graph_module_path: str):
    """
    Run graph algorithm with quantizer.

    Supports two modes:
    1. Single quantizer mode: Build graph with one quantizer, run all search params
    2. Multi-quantizer mode: Build graph once, then test multiple quantizer params

    Args:
        input_path: Path to input pickle file
        output_path: Path to output pickle file
        quantizer_module_path: Path to quantizer module
        graph_module_path: Path to graph module
    """
    # Load input data
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)

    train_data = input_data['train_data']
    test_data = input_data['test_data']
    ground_truth = input_data['ground_truth']
    config = input_data['config']

    graph_params = config.get('graph_params', {})
    quantizer_params = config.get('quantizer_params', {})

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Graph params: {graph_params}")
    print(f"Quantizer params: {quantizer_params}")

    # Load algorithm classes
    QuantizerClass = load_module_class(quantizer_module_path, 'BaseQuantizer')
    GraphClass = load_module_class(graph_module_path, 'BaseGraphIndex')

    nd, d = train_data.shape
    nq = test_data.shape[0]

    # Check if we have multiple quantizer build params
    quantizer_params_list = quantizer_params.get('build_params_list', None)

    if quantizer_params_list is None:
        # Single quantizer mode (original behavior)
        quantizer_build_params = quantizer_params.get('build_params', {})
        quantizer_params_list = [{'build_params': quantizer_build_params}]

    # Get search parameter configurations
    search_params_list = graph_params.get('search_params_list', [{}])

    # Initialize first quantizer and build graph ONCE
    print("\n=== Initializing First Quantizer ===")
    first_quantizer_params = quantizer_params_list[0]['build_params']
    quantizer = QuantizerClass(**first_quantizer_params)

    # Train first quantizer
    print("\n=== Training First Quantizer ===")
    start_time = time.time()
    success = quantizer.fit(nd, train_data)
    first_training_time = time.time() - start_time

    if not success:
        raise RuntimeError("Quantizer training failed")

    print(f"Training time: {first_training_time:.4f}s")

    # Instantiate graph with quantizer
    print("\n=== Initializing Graph Index ===")
    graph_build_params = graph_params.get('build_params', {})
    graph_index = GraphClass(quantizer=quantizer, **graph_build_params)

    # Build graph ONCE (this is the slow part we want to avoid repeating)
    print("\n=== Building Graph Index (ONE TIME ONLY) ===")
    start_time = time.time()
    success = graph_index.build(nd, train_data)
    build_time = time.time() - start_time

    if not success:
        raise RuntimeError("Graph index build failed")

    print(f"Build time: {build_time:.4f}s")
    print("Graph structure will be reused for all quantizer configurations")

    # Now test each quantizer configuration
    all_quantizer_results = []

    for quant_idx, quantizer_config in enumerate(quantizer_params_list):
        print(f"\n{'='*60}")
        print(f"Testing Quantizer Configuration {quant_idx + 1}/{len(quantizer_params_list)}")
        print(f"{'='*60}")

        quantizer_build_params = quantizer_config['build_params']

        # For first quantizer, reuse the one we already trained
        if quant_idx == 0:
            current_quantizer = quantizer
            training_time = first_training_time
            print("Using already-trained first quantizer")
        else:
            # For subsequent quantizers, create and train new one
            print(f"\n=== Creating New Quantizer with params: {quantizer_build_params} ===")
            current_quantizer = QuantizerClass(**quantizer_build_params)

            print("\n=== Training New Quantizer ===")
            start_time = time.time()
            success = current_quantizer.fit(nd, train_data)
            training_time = time.time() - start_time

            if not success:
                raise RuntimeError(f"Quantizer {quant_idx} training failed")

            print(f"Training time: {training_time:.4f}s")

            # Set the new quantizer on the existing graph (graph structure unchanged!)
            print("\n=== Swapping Quantizer (keeping graph structure) ===")
            graph_index.set_quantizer(current_quantizer)

        # Search with all search parameter sets for this quantizer
        print(f"\n=== Searching with Quantizer Config {quant_idx + 1} ===")

        search_results = []
        for search_idx, search_params in enumerate(search_params_list):
            # Extract topk from search_params
            search_params_copy = search_params.copy()
            topk = search_params_copy.pop('topk', 100)
            print(f"\nSearch configuration {search_idx + 1}/{len(search_params_list)}: {search_params}")

            start_time = time.time()
            I, D, hops, comps, nrerank = graph_index.search(nq, test_data, topk, **search_params_copy)
            query_time = time.time() - start_time

            # Calculate metrics
            recall = calculate_recall(I, ground_truth[:, :topk])
            map_score = calculate_map(I, ground_truth[:, :topk])
            recall_at_1 = calculate_recall_at_1(I, ground_truth[:, :topk])

            print(f"  Query time: {query_time:.4f}s")
            print(f"  Queries per second: {len(test_data) / query_time:.2f}")
            print(f"  Recall@{topk}: {recall:.4f}")
            print(f"  MAP@{topk}: {map_score:.4f}")
            print(f"  Recall@1: {recall_at_1:.4f}")

            # Compute percentile statistics for graph search metrics
            hops_stats = compute_percentile_stats(hops)
            comps_stats = compute_percentile_stats(comps)
            nrerank_stats = compute_percentile_stats(nrerank)

            # Print stats
            print(f"  Graph search metrics:")
            print_percentile_stats("hops", hops_stats)
            print_percentile_stats("comps", comps_stats)
            print_percentile_stats("nrerank", nrerank_stats)

            # Collect results for this search configuration
            result = {
                'search_params': search_params,
                'query_time': query_time,
                'queries_per_second': len(test_data) / query_time if query_time > 0 else 0,
                'recall': recall,
                'map': map_score,
                'recall@1': recall_at_1,
                'predictions': I,
                'distances': D,
                'hops_stats': hops_stats,
                'comps_stats': comps_stats,
                'nrerank_stats': nrerank_stats
            }
            search_results.append(result)

        # Collect results for this quantizer configuration
        quantizer_result = {
            'quantizer_build_params': quantizer_build_params,
            'training_time': training_time,
            'quantizer_memory': current_quantizer.getMemoryUsage(),
            'compression_rate': current_quantizer.getCompressionRate(),
            'mse': current_quantizer.getMSE(),
            'search_results': search_results
        }
        all_quantizer_results.append(quantizer_result)

    # Prepare final output
    output = {
        'build_time': build_time,  # Graph build time (done once)
        'graph_memory': graph_index.getMemoryUsage(),
        'quantizer_results': all_quantizer_results  # List of results for each quantizer
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nResults saved to {output_path}")
    print(f"Total quantizer configurations tested: {len(all_quantizer_results)}")
    print(f"Graph was built only ONCE and reused for all configurations")


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


def calculate_map(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate MAP@k (Mean Average Precision at k).

    Args:
        predictions: Predicted neighbor indices, shape (nq, k)
        ground_truth: Ground truth neighbor indices, shape (nq, k)

    Returns:
        MAP@k score
    """
    nq = predictions.shape[0]
    k = predictions.shape[1]

    map_score = 0.0
    for i in range(nq):
        gt_set = set(ground_truth[i])
        num_relevant = len(gt_set)

        # Calculate AP (Average Precision) for this query
        num_hits = 0
        sum_precisions = 0.0
        for j in range(k):
            if predictions[i][j] in gt_set:
                num_hits += 1
                precision_at_j = num_hits / (j + 1)
                sum_precisions += precision_at_j

        # AP = sum of precisions / min(k, num_relevant)
        ap = sum_precisions / min(k, num_relevant) if num_relevant > 0 else 0.0
        map_score += ap

    return map_score / nq if nq > 0 else 0.0


def calculate_recall_at_1(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Recall@1 - the fraction of queries where the ground truth top-1
    appears in the predicted top-k.

    Args:
        predictions: Predicted neighbor indices, shape (nq, k)
        ground_truth: Ground truth neighbor indices, shape (nq, k)

    Returns:
        Recall@1 score
    """
    nq = predictions.shape[0]

    num_correct = 0
    for i in range(nq):
        gt_top1 = ground_truth[i][0]
        pred_set = set(predictions[i])
        if gt_top1 in pred_set:
            num_correct += 1

    return num_correct / nq if nq > 0 else 0.0


def compute_percentile_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Compute percentile statistics for a 1D array.

    Args:
        data: 1D numpy array of values

    Returns:
        Dict with percentile statistics:
        - min, p5, p10, p25, mean, p75, p90, p95, max, variance
        Returns None if data is None
    """
    if data is None or len(data) == 0:
        return None

    return {
        'min': float(np.min(data)),
        'p5': float(np.percentile(data, 5)),
        'p10': float(np.percentile(data, 10)),
        'p25': float(np.percentile(data, 25)),
        'mean': float(np.mean(data)),
        'p75': float(np.percentile(data, 75)),
        'p90': float(np.percentile(data, 90)),
        'p95': float(np.percentile(data, 95)),
        'max': float(np.max(data)),
        'variance': float(np.var(data))
    }


def print_percentile_stats(name: str, stats: Dict[str, float]) -> None:
    """Print percentile statistics in a formatted way."""
    if stats is None:
        print(f"  {name}: N/A (not supported)")
        return

    print(f"  {name}:")
    print(f"    min={stats['min']:.2f}, p5={stats['p5']:.2f}, p10={stats['p10']:.2f}, p25={stats['p25']:.2f}")
    print(f"    mean={stats['mean']:.2f}, p75={stats['p75']:.2f}, p90={stats['p90']:.2f}, p95={stats['p95']:.2f}")
    print(f"    max={stats['max']:.2f}, variance={stats['variance']:.2f}")


def run_graph_only(input_path: str, output_path: str, graph_module_path: str):
    """
    Run graph-only algorithm (no external quantizer).

    This is used for algorithms like SymphonyQG that have integrated quantization.

    Args:
        input_path: Path to input pickle file
        output_path: Path to output pickle file
        graph_module_path: Path to graph module
    """
    # Load input data
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)

    train_data = input_data['train_data']
    test_data = input_data['test_data']
    ground_truth = input_data['ground_truth']
    config = input_data['config']

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Config: {config}")

    # Load graph class
    GraphClass = load_module_class(graph_module_path, 'BaseGraphIndex')

    nd, d = train_data.shape
    nq = test_data.shape[0]

    # Get build and search parameters
    build_params = config.get('build_params', {})
    search_params_list = config.get('search_params_list', [{}])

    print(f"\n=== Graph-Only Mode ===")
    print(f"Build params: {build_params}")
    print(f"Number of search configs: {len(search_params_list)}")

    # Instantiate graph with quantizer=None
    print("\n=== Initializing Graph Index ===")
    graph_index = GraphClass(quantizer=None, **build_params)

    # Build graph
    print("\n=== Building Graph Index ===")
    start_time = time.time()
    success = graph_index.build(nd, train_data)
    build_time = time.time() - start_time

    if not success:
        raise RuntimeError("Graph index build failed")

    print(f"Build time: {build_time:.4f}s")

    # Get graph memory
    graph_memory = graph_index.getMemoryUsage()
    print(f"Graph memory: {graph_memory / 1024:.2f} MB")

    # Search with all search parameter sets
    print("\n=== Searching ===")

    search_results = []
    for search_idx, search_params in enumerate(search_params_list):
        # Extract topk from search_params
        search_params_copy = search_params.copy()
        topk = search_params_copy.pop('topk', 100)
        print(f"\nSearch configuration {search_idx + 1}/{len(search_params_list)}: {search_params}")

        start_time = time.time()
        I, D, hops, comps, nrerank = graph_index.search(nq, test_data, topk, **search_params_copy)
        query_time = time.time() - start_time

        # Calculate metrics
        recall = calculate_recall(I, ground_truth[:, :topk])
        map_score = calculate_map(I, ground_truth[:, :topk])
        recall_at_1 = calculate_recall_at_1(I, ground_truth[:, :topk])

        print(f"  Query time: {query_time:.4f}s")
        print(f"  Queries per second: {len(test_data) / query_time:.2f}")
        print(f"  Recall@{topk}: {recall:.4f}")
        print(f"  MAP@{topk}: {map_score:.4f}")
        print(f"  Recall@1: {recall_at_1:.4f}")

        # Compute percentile statistics for graph search metrics
        hops_stats = compute_percentile_stats(hops)
        comps_stats = compute_percentile_stats(comps)
        nrerank_stats = compute_percentile_stats(nrerank)

        # Print stats
        print(f"  Graph search metrics:")
        print_percentile_stats("hops", hops_stats)
        print_percentile_stats("comps", comps_stats)
        print_percentile_stats("nrerank", nrerank_stats)

        # Collect results for this search configuration
        result = {
            'search_params': search_params,
            'query_time': query_time,
            'queries_per_second': len(test_data) / query_time if query_time > 0 else 0,
            'recall': recall,
            'map': map_score,
            'recall@1': recall_at_1,
            'predictions': I,
            'distances': D,
            'hops_stats': hops_stats,
            'comps_stats': comps_stats,
            'nrerank_stats': nrerank_stats
        }
        search_results.append(result)

    # Prepare final output
    output = {
        'build_time': build_time,
        'graph_memory': graph_memory,
        'search_results': search_results
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Docker entrypoint for running algorithms')
    parser.add_argument('--mode', required=True, choices=['quantizer', 'dimreduction', 'graph', 'graph-only'],
                        help='Mode: quantizer, dimreduction, graph, or graph-only')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output pickle file path')
    parser.add_argument('--module', help='Algorithm module path (for quantizer/dimreduction)')
    parser.add_argument('--quantizer-module', help='Quantizer module path (for graph)')
    parser.add_argument('--graph-module', help='Graph module path (for graph and graph-only)')

    args = parser.parse_args()

    print(f"=== Docker Entrypoint ===")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.module:
        print(f"Module: {args.module}")
    if args.quantizer_module:
        print(f"Quantizer Module: {args.quantizer_module}")
    if args.graph_module:
        print(f"Graph Module: {args.graph_module}")
    print()

    if args.mode == 'dimreduction':
        run_dimreduction(args.input, args.output, args.module)
    elif args.mode == 'quantizer':
        run_quantizer(args.input, args.output, args.module)
    elif args.mode == 'graph':
        run_graph(args.input, args.output, args.quantizer_module, args.graph_module)
    elif args.mode == 'graph-only':
        run_graph_only(args.input, args.output, args.graph_module)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("FATAL ERROR IN DOCKER CONTAINER")
        print("="*60)
        print(f"Error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        sys.exit(1)
