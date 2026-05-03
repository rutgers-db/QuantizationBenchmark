import json
import os
import sys
import importlib.util
import yaml
from typing import Dict, List, Optional, Any, Type
from .base import BaseQuantizer, BaseDimReduction


class AlgorithmLoader:
    """
    Dynamically loads quantizer and dimensionality reduction algorithms from the algorithms directory.

    Each algorithm should be in its own directory under:
    - benchmark/algorithms/quantizer/{algorithm_name}/
    - benchmark/algorithms/dimreduction/{algorithm_name}/

    Each algorithm directory should contain:
    - module.py: Implementation of BaseQuantizer or BaseDimReduction
    - config.yaml: Configuration parameters
    - dockerfile: Docker environment setup (optional)
    """

    def __init__(self, base_path: str = "benchmark/algorithms"):
        """
        Initialize the algorithm loader.

        Args:
            base_path: Base path to the algorithms directory
        """
        self.base_path = base_path
        self.quantizer_path = os.path.join(base_path, "quantizer")
        self.ivf_path = os.path.join(base_path, "ivf")
        self.dimreduction_path = os.path.join(base_path, "dimreduction")

        # Cache for loaded algorithms
        self._quantizer_registry: Dict[str, Type[BaseQuantizer]] = {}
        self._quantizer_dirs: Dict[str, str] = {}
        self._dimreduction_registry: Dict[str, Type[BaseDimReduction]] = {}

        # Scan and register algorithms
        self._scan_algorithms()

    def _scan_algorithms(self):
        """Scan the algorithms directories and register all available algorithms."""
        # Scan quantizers, including IVF algorithms stored alongside quantizers.
        for algo_name, algo_dir in self._iter_quantizer_dirs():
            try:
                self._register_quantizer(algo_name, algo_dir)
            except Exception as e:
                print(f"Warning: Failed to load quantizer '{algo_name}': {e}")

        # Scan dimensionality reduction algorithms
        if os.path.exists(self.dimreduction_path):
            for algo_name in os.listdir(self.dimreduction_path):
                algo_dir = os.path.join(self.dimreduction_path, algo_name)
                if os.path.isdir(algo_dir) and os.path.exists(os.path.join(algo_dir, "module.py")):
                    try:
                        self._register_dimreduction(algo_name, algo_dir)
                    except Exception as e:
                        print(f"Warning: Failed to load dimreduction '{algo_name}': {e}")

    def _iter_quantizer_dirs(self):
        """Yield all quantizer-like algorithm directories keyed by algorithm name."""
        seen = set()

        for root_dir in [self.quantizer_path, self.ivf_path]:
            if not os.path.exists(root_dir):
                continue

            for current_root, dirnames, _ in os.walk(root_dir):
                module_path = os.path.join(current_root, "module.py")
                if os.path.exists(module_path):
                    algo_name = os.path.basename(current_root)
                    if algo_name not in seen:
                        seen.add(algo_name)
                        yield algo_name, current_root
                    dirnames[:] = []

    def _register_quantizer(self, algo_name: str, algo_dir: str):
        """
        Register a quantizer algorithm.

        Args:
            algo_name: Name of the algorithm
            algo_dir: Directory containing the algorithm
        """
        module_path = os.path.join(algo_dir, "module.py")

        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(f"quantizer_{algo_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the class that inherits from BaseQuantizer
        quantizer_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseQuantizer) and
                attr is not BaseQuantizer):
                quantizer_class = attr
                break

        if quantizer_class is None:
            raise ValueError(f"No BaseQuantizer subclass found in {module_path}")

        self._quantizer_registry[algo_name] = quantizer_class
        self._quantizer_dirs[algo_name] = algo_dir
        print(f"Registered quantizer: {algo_name} ({quantizer_class.__name__})")

    def _register_dimreduction(self, algo_name: str, algo_dir: str):
        """
        Register a dimensionality reduction algorithm.

        Args:
            algo_name: Name of the algorithm
            algo_dir: Directory containing the algorithm
        """
        module_path = os.path.join(algo_dir, "module.py")

        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(f"dimreduction_{algo_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the class that inherits from BaseDimReduction
        dimreduction_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseDimReduction) and
                attr is not BaseDimReduction):
                dimreduction_class = attr
                break

        if dimreduction_class is None:
            raise ValueError(f"No BaseDimReduction subclass found in {module_path}")

        self._dimreduction_registry[algo_name] = dimreduction_class
        print(f"Registered dimreduction: {algo_name} ({dimreduction_class.__name__})")

    def _load_config(self, algo_dir: str) -> Dict[str, Any]:
        """
        Load configuration from ``config.json`` (preferred) or ``config.yaml``.

        Args:
            algo_dir: Directory containing the algorithm

        Returns:
            Dict containing configuration parameters
        """
        for ext, parser in ((".json", json.load), (".yaml", yaml.safe_load)):
            config_path = os.path.join(algo_dir, "config" + ext)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return parser(f) or {}
        return {}

    def get_quantizer(self, algo_name: str, **override_params) -> BaseQuantizer:
        """
        Get an instance of a quantizer algorithm.

        Args:
            algo_name: Name of the algorithm
            **override_params: Parameters to override from config.yaml

        Returns:
            BaseQuantizer instance

        Raises:
            ValueError: If the algorithm is not found
        """
        if algo_name not in self._quantizer_registry:
            available = ', '.join(self._quantizer_registry.keys())
            raise ValueError(f"Quantizer '{algo_name}' not found. Available: {available}")

        # Load config
        algo_dir = self._quantizer_dirs[algo_name]
        config = self._load_config(algo_dir)

        # Override with provided parameters
        config.update(override_params)

        # Instantiate the algorithm
        quantizer_class = self._quantizer_registry[algo_name]
        return quantizer_class(**config)

    def get_dimreduction(self, algo_name: str, **override_params) -> BaseDimReduction:
        """
        Get an instance of a dimensionality reduction algorithm.

        Args:
            algo_name: Name of the algorithm
            **override_params: Parameters to override from config.yaml

        Returns:
            BaseDimReduction instance

        Raises:
            ValueError: If the algorithm is not found
        """
        if algo_name not in self._dimreduction_registry:
            available = ', '.join(self._dimreduction_registry.keys())
            raise ValueError(f"Dimreduction '{algo_name}' not found. Available: {available}")

        # Load config
        algo_dir = os.path.join(self.dimreduction_path, algo_name)
        config = self._load_config(algo_dir)

        # Override with provided parameters
        config.update(override_params)

        # Instantiate the algorithm
        dimreduction_class = self._dimreduction_registry[algo_name]
        return dimreduction_class(**config)

    def parse_algorithm_string(self, algo_string: str) -> tuple:
        """
        Parse algorithm combination string like "PCA,PQ" into dimreduction and quantizer.

        Args:
            algo_string: Comma-separated algorithm names (e.g., "PCA,PQ" or just "PQ")

        Returns:
            Tuple of (dimreduction_name or None, quantizer_name)

        Examples:
            "PQ" -> (None, "PQ")
            "PCA,PQ" -> ("PCA", "PQ")
        """
        parts = [p.strip() for p in algo_string.split(',')]

        if len(parts) == 1:
            # Only quantizer
            return None, parts[0]
        elif len(parts) == 2:
            # Dimreduction + quantizer
            return parts[0], parts[1]
        else:
            raise ValueError(f"Invalid algorithm string: {algo_string}. Expected format: 'Quantizer' or 'DimReduction,Quantizer'")

    def list_algorithms(self) -> Dict[str, List[str]]:
        """
        List all available algorithms.

        Returns:
            Dict with 'quantizers' and 'dimreductions' keys containing lists of algorithm names
        """
        return {
            'quantizers': sorted(self._quantizer_registry.keys()),
            'dimreductions': sorted(self._dimreduction_registry.keys())
        }
