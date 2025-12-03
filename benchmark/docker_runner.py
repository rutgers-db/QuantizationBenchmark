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
        self.dimreduction_path = os.path.join(base_path, "dimreduction")
        self.debug = debug

        # Create temp directory for data exchange with containers
        self.temp_dir = os.path.abspath("temp")
        os.makedirs(self.temp_dir, exist_ok=True)

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

        print(f"Building Docker image: {image_name}")

        # Build the image with project root as context
        # This allows the Dockerfile to access both the algorithm and the framework
        project_root = os.path.abspath(".")
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path,
             "--build-arg", f"ALGO_TYPE={algo_type}",
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
        config: Dict[str, Any]
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
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'ground_truth': ground_truth,
                'config': config
            }, f)

        # Run container
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
            "--module", f"/algorithms/quantizer/{algo_name}/module.py"
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
        # Build quantizer images
        if os.path.exists(self.quantizer_path):
            for algo_name in os.listdir(self.quantizer_path):
                algo_dir = os.path.join(self.quantizer_path, algo_name)
                if os.path.isdir(algo_dir):
                    self.build_image('quantizer', algo_name, force_rebuild)

        # Build dimreduction images
        if os.path.exists(self.dimreduction_path):
            for algo_name in os.listdir(self.dimreduction_path):
                algo_dir = os.path.join(self.dimreduction_path, algo_name)
                if os.path.isdir(algo_dir):
                    self.build_image('dimreduction', algo_name, force_rebuild)
