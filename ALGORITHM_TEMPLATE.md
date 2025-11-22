# Algorithm Implementation Guide

This guide explains how to add a new quantization or dimensionality reduction algorithm to the benchmark framework.

## Directory Structure

Each algorithm should be placed in its own directory:
- Quantizers: `benchmark/algorithms/quantizer/<algorithm_name>/`
- Dimensionality Reduction: `benchmark/algorithms/dimreduction/<algorithm_name>/`

Each algorithm directory must contain three files:

```
<algorithm_name>/
├── dockerfile       # Docker environment setup
├── config.yaml      # Default configuration parameters
└── module.py        # Algorithm implementation
```

## 1. Quantizer Implementation

### module.py

Your quantizer must inherit from `BaseQuantizer` and implement all required methods:

```python
import numpy as np
from benchmark.base import BaseQuantizer


class MyQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        """
        Initialize with parameters from config.yaml.

        Example kwargs:
            num_codebooks: int
            bits_per_vector: int
        """
        super().__init__(**kwargs)
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)

        # Store training data for querying
        self.trained = False
        self.train_data = None
        self.compressed_data = None

    def fit(self, data: np.ndarray) -> bool:
        """
        Train the quantizer on the given data.

        Args:
            data: Training data of shape (n, d)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Your training logic here
            self.train_data = data
            # ... train your quantizer ...

            self.trained = True
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False

    def query(self, queries: np.ndarray, topk: int) -> tuple:
        """
        Search for top-k nearest neighbors.

        Args:
            queries: Query vectors of shape (nq, d)
            topk: Number of nearest neighbors

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of shape (nq, topk)
                - D: Distances of shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Quantizer not trained")

        # Your search logic here
        # Return indices and distances
        I = np.zeros((queries.shape[0], topk), dtype=np.int32)
        D = np.zeros((queries.shape[0], topk), dtype=np.float32)

        return I, D

    def getMemoryUsage(self) -> float:
        """
        Get memory usage of the quantizer in bytes.

        Returns:
            float: Memory usage in bytes
        """
        memory = 0
        # Calculate memory of stored data structures
        if self.compressed_data is not None:
            memory += self.compressed_data.nbytes
        # Add other data structures...
        return float(memory)

    def getCompressionRate(self) -> float:
        """
        Get compression rate: original_size / compressed_size

        Returns:
            float: Compression rate (higher is better)
        """
        if self.train_data is None:
            return 1.0

        original_size = self.train_data.nbytes
        compressed_size = self.getMemoryUsage()

        if compressed_size == 0:
            return 1.0

        return original_size / compressed_size

    def getMSE(self) -> float:
        """
        Get mean squared error of reconstruction.

        Returns:
            float: MSE value
        """
        if not self.trained:
            return float('inf')

        # Reconstruct vectors and calculate MSE
        # reconstructed = self.reconstruct(self.train_data)
        # mse = np.mean((self.train_data - reconstructed) ** 2)

        mse = 0.0  # Replace with actual calculation
        return float(mse)
```

### config.yaml

```yaml
# Default configuration parameters
param1: value1
param2: value2
num_codebooks: 8
bits_per_vector: 64
```

### dockerfile

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir numpy scipy scikit-learn h5py pyyaml

# Copy algorithm code
COPY module.py /algorithms/quantizer/<algorithm_name>/module.py

# Set working directory
WORKDIR /workspace

# The entrypoint will be set by the framework
```

## 2. Dimensionality Reduction Implementation

### module.py

Your dimensionality reduction algorithm must inherit from `BaseDimReduction`:

```python
import numpy as np
from benchmark.base import BaseDimReduction


class MyDimReduction(BaseDimReduction):
    def __init__(self, **kwargs):
        """
        Initialize with parameters from config.yaml.

        Example kwargs:
            target_dim: int - target dimensionality
        """
        super().__init__(**kwargs)
        self.target_dim = kwargs.get('target_dim', 64)

        self.original_dim = None
        self.projection_matrix = None
        self.fitted = False

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.

        Args:
            data: Input data of shape (n, d)

        Returns:
            np.ndarray: Transformed data of shape (n, target_dim)
        """
        self.original_dim = data.shape[1]

        # Your dimensionality reduction logic here
        # e.g., compute projection matrix
        self.projection_matrix = np.random.randn(self.original_dim, self.target_dim)

        self.fitted = True

        # Transform the data
        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted model.

        Args:
            data: Input data of shape (n, original_dim)

        Returns:
            np.ndarray: Transformed data of shape (n, target_dim)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        # Apply transformation
        transformed = data @ self.projection_matrix
        return transformed

    def getMemoryUsage(self) -> float:
        """
        Get memory usage in bytes.

        Returns:
            float: Memory usage in bytes
        """
        memory = 0
        if self.projection_matrix is not None:
            memory += self.projection_matrix.nbytes
        return float(memory)

    def getCompressionRate(self) -> float:
        """
        Get compression rate: original_dim / target_dim

        Returns:
            float: Compression rate
        """
        if self.original_dim is None:
            return 1.0
        return self.original_dim / self.target_dim
```

### config.yaml

```yaml
# Default configuration parameters
target_dim: 64
```

### dockerfile

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir numpy scipy scikit-learn h5py pyyaml

# Copy algorithm code
COPY module.py /algorithms/dimreduction/<algorithm_name>/module.py

# Set working directory
WORKDIR /workspace
```

## 3. Testing Your Algorithm

After creating your algorithm, you can test it:

```bash
# Build the Docker image
python run.py --build-images

# Run benchmark
python run.py --dataset your-dataset --algorithm YourAlgorithm

# Or with dimensionality reduction
python run.py --dataset your-dataset --algorithm YourDimReduction,YourQuantizer
```

## Important Notes

1. **Data Format**: All data is in NumPy arrays with dtype float32
2. **Distance Metric**: Use Euclidean distance (L2)
3. **Thread Safety**: Your algorithm will run in an isolated Docker container
4. **Dependencies**: Add all required Python packages to the dockerfile
5. **Error Handling**: Return False from fit() if training fails
6. **Memory Tracking**: The framework tracks peak memory automatically, but you must implement getMemoryUsage() correctly
