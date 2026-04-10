import os
import subprocess
import json
import pickle
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class DockerRunner:
    """
    Manages Docker container execution for algorithms.
    Similar to ann-benchmarks, each algorithm runs in its own Docker container.
    """

    def __init__(self, base_path: str = "benchmark/algorithms", debug: bool = False):
        """
        Initialize the Docker runner.

        Args:
            base_path: Base path to the algorithms directory
            debug: Enable debug mode with real-time Docker output
        """
        self.base_path = base_path
        self.quantizer_path = os.path.join(base_path, "quantizer")
        self.ivf_path = os.path.join(base_path, "ivf")
        self.dimreduction_path = os.path.join(base_path, "dimreduction")
        self.graph_path = "benchmark/graphs"
        self.debug = debug

        # Create temp directory for data exchange with containers
        self.temp_dir = os.path.abspath("temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _resolve_quantizer_dir(self, algo_name: str) -> Tuple[str, str]:
        """Return (family, directory) for a quantizer-like algorithm."""
        candidates = [
            ("quantizer", os.path.join(self.quantizer_path, algo_name)),
            ("ivf", os.path.join(self.ivf_path, algo_name)),
        ]

        for family, algo_dir in candidates:
            if os.path.exists(os.path.join(algo_dir, "module.py")):
                return family, algo_dir

        raise FileNotFoundError(f"Quantizer '{algo_name}' not found under quantizer/ or ivf/")

    def _iter_quantizer_names(self):
        """Yield all quantizer-like algorithm names from both quantizer and ivf roots."""
        seen = set()
        for root_dir in [self.quantizer_path, self.ivf_path]:
            if not os.path.exists(root_dir):
                continue
            for current_root, dirnames, _ in os.walk(root_dir):
                if os.path.exists(os.path.join(current_root, "module.py")):
                    algo_name = os.path.basename(current_root)
                    if algo_name not in seen:
                        seen.add(algo_name)
                        yield algo_name
                    dirnames[:] = []

    def build_image(self, algo_type: str, algo_name: str, force_rebuild: bool = False) -> bool:
        """
        Build Docker image for an algorithm.

        Args:
            algo_type: 'quantizer' or 'dimreduction'
            algo_name: Name of the algorithm
            force_rebuild: Force rebuild even if image exists

        Returns:
            bool: True if build succeeded
        """
        actual_algo_type = algo_type
        if algo_type == 'quantizer':
            actual_algo_type, algo_dir = self._resolve_quantizer_dir(algo_name)
        else:
            algo_dir = os.path.join(self.base_path, algo_type, algo_name)
        dockerfile_path = os.path.join(algo_dir, "dockerfile")

        if not os.path.exists(dockerfile_path):
            print(f"Error: Dockerfile not found at {dockerfile_path}")
            return False

        image_name = f"quantbench-{algo_type}-{algo_name.lower()}:latest"
        # Check if image exists
        if not force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Image {image_name} already exists, skipping build")
                return True


        # Build the image with project root as context
        # This allows the Dockerfile to access both the algorithm and the framework
        if force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Removing Docker image: {image_name}")

                result = subprocess.run(
                    ["docker", "rmi", image_name],
                    capture_output=True,
                    text=True
                )


                if result.returncode != 0:
                    print(f"Error removing image: {result.stderr}")
                    return False

                print(f"Successfully removed image: {image_name}")

        print(f"Building Docker image: {image_name}")

        project_root = os.path.abspath(".")
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path,
             "--build-arg", f"ALGO_TYPE={actual_algo_type}",
             "--build-arg", f"ALGO_NAME={algo_name}",
             project_root],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error building image: {result.stderr}")
            return False

        print(f"Successfully built image: {image_name}")
        return True

    def build_graph_image(self, graph_name: str, quantizer_name: str, force_rebuild: bool = False) -> bool:
        """
        Build Docker image for a graph algorithm combined with a quantizer.

        Args:
            graph_name: Name of the graph algorithm (e.g., 'diskann')
            quantizer_name: Name of the quantizer to combine with
            force_rebuild: Force rebuild even if image exists

        Returns:
            bool: True if build succeeded
        """
        # First check if quantizer has a custom dockerfile for this graph
        quantizer_family, quantizer_dir = self._resolve_quantizer_dir(quantizer_name)
        custom_dockerfile = os.path.join(quantizer_dir, f"{graph_name}_dockerfile")

        if os.path.exists(custom_dockerfile):
            dockerfile_path = custom_dockerfile
            print(f"Using custom dockerfile for {quantizer_name} + {graph_name}: {custom_dockerfile}")
        else:
            # Use the graph's default dockerfile
            graph_dir = os.path.join(self.graph_path, graph_name)
            dockerfile_path = os.path.join(graph_dir, "dockerfile")

            if not os.path.exists(dockerfile_path):
                print(f"Error: Dockerfile not found at {dockerfile_path}")
                return False

        # Image name includes both graph and quantizer
        image_name = f"quantbench-graph-{graph_name.lower()}-{quantizer_name.lower()}:latest"

        # Check if image exists
        if not force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Image {image_name} already exists, skipping build")
                return True

        # Remove old image if force rebuild
        if force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Removing Docker image: {image_name}")
                result = subprocess.run(
                    ["docker", "rmi", image_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Error removing image: {result.stderr}")
                    return False
                print(f"Successfully removed image: {image_name}")

        print(f"Building Docker image: {image_name}")

        project_root = os.path.abspath(".")
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path,
             "--build-arg", f"QUANTIZER_TYPE={quantizer_family}",
             "--build-arg", f"QUANTIZER_NAME={quantizer_name}",
             "--build-arg", f"GRAPH_NAME={graph_name}",
             project_root],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error building image: {result.stderr}")
            return False

        print(f"Successfully built image: {image_name}")
        return True

    def build_graph_only_image(self, graph_name: str, force_rebuild: bool = False) -> bool:
        """
        Build Docker image for a graph-only algorithm (no external quantizer).

        This is used for algorithms like SymphonyQG that have integrated quantization.

        Args:
            graph_name: Name of the graph algorithm (e.g., 'symphonyqg')
            force_rebuild: Force rebuild even if image exists

        Returns:
            bool: True if build succeeded
        """
        graph_dir = os.path.join(self.graph_path, graph_name)
        dockerfile_path = os.path.join(graph_dir, "dockerfile")

        if not os.path.exists(dockerfile_path):
            print(f"Error: Dockerfile not found at {dockerfile_path}")
            return False

        # Image name for graph-only
        image_name = f"quantbench-graph-{graph_name.lower()}:latest"

        # Check if image exists
        if not force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Image {image_name} already exists, skipping build")
                return True

        # Remove old image if force rebuild
        if force_rebuild:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                print(f"Removing Docker image: {image_name}")
                result = subprocess.run(
                    ["docker", "rmi", image_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Error removing image: {result.stderr}")
                    return False
                print(f"Successfully removed image: {image_name}")

        print(f"Building Docker image: {image_name}")

        project_root = os.path.abspath(".")
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path,
             "--build-arg", f"GRAPH_NAME={graph_name}",
             project_root],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error building image: {result.stderr}")
            return False

        print(f"Successfully built image: {image_name}")
        return True

    def run_dimreduction(
        self,
        algo_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Run dimensionality reduction algorithm in Docker container.

        Args:
            algo_name: Name of the algorithm
            train_data: Training data array
            test_data: Test data array
            config: Configuration parameters

        Returns:
            Tuple of (transformed_train_data, transformed_test_data, metrics)
        """
        # Prepare input data
        input_dir = os.path.join(self.temp_dir, f"dimreduction_{algo_name}_{int(time.time())}")
        os.makedirs(input_dir, exist_ok=True)

        input_file = os.path.join(input_dir, "input.pkl")
        output_file = os.path.join(input_dir, "output.pkl")

        with open(input_file, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'config': config
            }, f)

        # Run container
        image_name = f"quantbench-dimreduction-{algo_name.lower()}:latest"

        # Extract nthread from config to set environment variables
        # This must be set before numpy is imported
        nthread = config.get('nthread')

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(input_dir)}:/workspace",
            "-v", f"{os.path.abspath('benchmark')}:/benchmark",
            "-v", "/tmp:/tmp",  # Mount host /tmp for fast disk I/O (SSD)
        ]

        # Add thread environment variables if nthread is specified
        if nthread is not None:
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])

        cmd.extend([
            image_name,
            "python", "-u", "/benchmark/docker_entrypoint.py",  # -u for unbuffered output
            "--mode", "dimreduction",
            "--input", "/workspace/input.pkl",
            "--output", "/workspace/output.pkl",
            "--module", f"/algorithms/dimreduction/{algo_name}/module.py"
        ])

        print(f"Running {algo_name} in Docker container...")
        if self.debug:
            # Debug mode: show real-time output
            print("\n" + "="*60)
            print("DEBUG MODE: Real-time Docker output")
            print("="*60 + "\n")
            result = subprocess.run(cmd)
        else:
            # Normal mode: capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            if not self.debug:
                print(f"Docker returncode: {result.returncode}")
                print(f"Error running container: {result.stderr}")
                print(f"Stdout: {result.stdout}")
            return None, None, None

        # Read output
        if not os.path.exists(output_file):
            print(f"Error: Output file not found")
            return None, None, None

        with open(output_file, 'rb') as f:
            output = pickle.load(f)

        # Cleanup
        os.remove(input_file)
        os.remove(output_file)
        os.rmdir(input_dir)

        return output['train_transformed'], output['test_transformed'], output['metrics']

    def run_quantizer(
        self,
        algo_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        ground_truth: np.ndarray,
        config: Dict[str, Any],
        add_data: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run quantizer algorithm in Docker container.

        Args:
            algo_name: Name of the algorithm
            train_data: Training data array
            test_data: Test data array
            ground_truth: Ground truth neighbor indices
            config: Configuration parameters (topk should be in search_params)

        Returns:
            Dict containing all benchmark results

        Note:
            topk is now extracted from config['search_params']['topk'] with default value 100
        """
        # Prepare input data
        input_dir = os.path.join(self.temp_dir, f"quantizer_{algo_name}_{int(time.time())}")
        os.makedirs(input_dir, exist_ok=True)

        input_file = os.path.join(input_dir, "input.pkl")
        output_file = os.path.join(input_dir, "output.pkl")

        with open(input_file, 'wb') as f:
            payload = {
                'train_data': train_data,
                'test_data': test_data,
                'ground_truth': ground_truth,
                'config': config
            }
            if add_data is not None:
                payload['add_data'] = add_data
            pickle.dump(payload, f)

        # Run container
        algo_family, _ = self._resolve_quantizer_dir(algo_name)
        image_name = f"quantbench-quantizer-{algo_name.lower()}:latest"

        # Extract nthread from config to set environment variables
        # This must be set before numpy is imported
        nthread = None
        if 'build_params' in config:
            nthread = config['build_params'].get('nthread')
        else:
            nthread = config.get('nthread')

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(input_dir)}:/workspace",
            "-v", f"{os.path.abspath('benchmark')}:/benchmark",
            "-v", "/tmp:/tmp",  # Mount host /tmp for fast disk I/O (SSD)
        ]

        # Add thread environment variables if nthread is specified
        if nthread is not None:
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])
        else:
            nthread = 1
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])


        cmd.extend([
            image_name,
            "python", "-u", "/benchmark/docker_entrypoint.py",  # -u for unbuffered output
            "--mode", "quantizer",
            "--input", "/workspace/input.pkl",
            "--output", "/workspace/output.pkl",
            "--module", f"/algorithms/{algo_family}/{algo_name}/module.py"
        ])

        print(f"Running {algo_name} in Docker container...")
        if self.debug:
            # Debug mode: show real-time output
            print("\n" + "="*60)
            print("DEBUG MODE: Real-time Docker output")
            print("="*60 + "\n")
            result = subprocess.run(cmd)
        else:
            # Normal mode: capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if not self.debug:
                print(f"Docker returncode: {result.returncode}")
                print(f"Error running container: {result.stderr}")
                print(f"Stdout: {result.stdout}")
            return None

        # Read output
        if not os.path.exists(output_file):
            print(f"Error: Output file not found")
            return None

        with open(output_file, 'rb') as f:
            output = pickle.load(f)

        # Cleanup
        os.remove(input_file)
        os.remove(output_file)
        os.rmdir(input_dir)

        return output

    def run_graph(
        self,
        graph_name: str,
        quantizer_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        ground_truth: np.ndarray,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Run graph algorithm with quantizer in Docker container.

        Args:
            graph_name: Name of the graph algorithm
            quantizer_name: Name of the quantizer
            train_data: Training data array
            test_data: Test data array
            ground_truth: Ground truth neighbor indices
            config: Configuration parameters

        Returns:
            Dict containing all benchmark results
        """
        # Prepare input data
        input_dir = os.path.join(self.temp_dir, f"graph_{graph_name}_{quantizer_name}_{int(time.time())}")
        os.makedirs(input_dir, exist_ok=True)

        input_file = os.path.join(input_dir, "input.pkl")
        output_file = os.path.join(input_dir, "output.pkl")

        with open(input_file, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'ground_truth': ground_truth,
                'config': config
            }, f)

        # Run container
        quantizer_family, _ = self._resolve_quantizer_dir(quantizer_name)
        image_name = f"quantbench-graph-{graph_name.lower()}-{quantizer_name.lower()}:latest"

        # Extract nthread from config to set environment variables
        nthread = None
        if 'build_params' in config:
            nthread = config['build_params'].get('nthread')
        else:
            nthread = config.get('nthread')

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(input_dir)}:/workspace",
            "-v", f"{os.path.abspath('benchmark')}:/benchmark",
            "-v", "/tmp:/tmp",  # Mount host /tmp for fast disk I/O (SSD)
        ]

        # Add thread environment variables if nthread is specified
        if nthread is not None:
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])
        else:
            nthread = 1
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])

        cmd.extend([
            image_name,
            "python", "-u", "/benchmark/docker_entrypoint.py",
            "--mode", "graph",
            "--input", "/workspace/input.pkl",
            "--output", "/workspace/output.pkl",
            "--quantizer-module", f"/algorithms/{quantizer_family}/{quantizer_name}/module.py",
            "--graph-module", f"/algorithms/graphs/{graph_name}/module.py"
        ])

        print(f"Running {graph_name} with {quantizer_name} in Docker container...")
        if self.debug:
            # Debug mode: show real-time output
            print("\n" + "="*60)
            print("DEBUG MODE: Real-time Docker output")
            print("="*60 + "\n")
            result = subprocess.run(cmd)
        else:
            # Normal mode: capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if not self.debug:
                print(f"Docker returncode: {result.returncode}")
                print(f"Error running container: {result.stderr}")
                print(f"Stdout: {result.stdout}")
            return None

        # Read output
        if not os.path.exists(output_file):
            print(f"Error: Output file not found")
            return None

        with open(output_file, 'rb') as f:
            output = pickle.load(f)

        # Cleanup
        os.remove(input_file)
        os.remove(output_file)
        os.rmdir(input_dir)

        return output

    def run_graph_only(
        self,
        graph_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        ground_truth: np.ndarray,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Run graph-only algorithm in Docker container (no external quantizer).

        This is used for algorithms like SymphonyQG that have integrated quantization.

        Args:
            graph_name: Name of the graph algorithm
            train_data: Training data array
            test_data: Test data array
            ground_truth: Ground truth neighbor indices
            config: Configuration parameters

        Returns:
            Dict containing all benchmark results
        """
        # Prepare input data
        input_dir = os.path.join(self.temp_dir, f"graph_only_{graph_name}_{int(time.time())}")
        os.makedirs(input_dir, exist_ok=True)

        input_file = os.path.join(input_dir, "input.pkl")
        output_file = os.path.join(input_dir, "output.pkl")

        with open(input_file, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'ground_truth': ground_truth,
                'config': config
            }, f)

        # Run container
        image_name = f"quantbench-graph-{graph_name.lower()}:latest"

        # Extract nthread from config to set environment variables
        nthread = None
        if 'build_params' in config:
            nthread = config['build_params'].get('num_threads')
        else:
            nthread = config.get('num_threads')

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(input_dir)}:/workspace",
            "-v", f"{os.path.abspath('benchmark')}:/benchmark",
            "-v", "/tmp:/tmp",  # Mount host /tmp for fast disk I/O (SSD)
        ]

        # Add thread environment variables if nthread is specified
        if nthread is not None:
            cmd.extend([
                "-e", f"OMP_NUM_THREADS={nthread}",
                "-e", f"MKL_NUM_THREADS={nthread}",
                "-e", f"OPENBLAS_NUM_THREADS={nthread}",
                "-e", f"NUMEXPR_NUM_THREADS={nthread}",
            ])

        cmd.extend([
            image_name,
            "python", "-u", "/benchmark/docker_entrypoint.py",
            "--mode", "graph-only",
            "--input", "/workspace/input.pkl",
            "--output", "/workspace/output.pkl",
            "--graph-module", f"/algorithms/graphs/{graph_name}/module.py"
        ])

        print(f"Running {graph_name} (graph-only) in Docker container...")
        if self.debug:
            # Debug mode: show real-time output
            print("\n" + "="*60)
            print("DEBUG MODE: Real-time Docker output")
            print("="*60 + "\n")
            result = subprocess.run(cmd)
        else:
            # Normal mode: capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if not self.debug:
                print(f"Docker returncode: {result.returncode}")
                print(f"Error running container: {result.stderr}")
                print(f"Stdout: {result.stdout}")
            return None

        # Read output
        if not os.path.exists(output_file):
            print(f"Error: Output file not found")
            return None

        with open(output_file, 'rb') as f:
            output = pickle.load(f)

        # Cleanup
        os.remove(input_file)
        os.remove(output_file)
        os.rmdir(input_dir)

        return output

    def build_all_images(self, force_rebuild: bool = False):
        """
        Build Docker images for all available algorithms.

        Args:
            force_rebuild: Force rebuild even if images exist
        """
        # Build quantizer and IVF images
        for algo_name in self._iter_quantizer_names():
            self.build_image('quantizer', algo_name, force_rebuild)

        # Build dimreduction images
        if os.path.exists(self.dimreduction_path):
            for algo_name in os.listdir(self.dimreduction_path):
                algo_dir = os.path.join(self.dimreduction_path, algo_name)
                if os.path.isdir(algo_dir):
                    self.build_image('dimreduction', algo_name, force_rebuild)
