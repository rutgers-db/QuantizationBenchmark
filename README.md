# QuantizationBenchmark

python run.py --build-images --algorithm ProductQuantizationFaiss --force-rebuild

python run.py --dataset sift-128-euclidean --algorithm ProductQuantizationFaiss --data-dir ./data/

Config yaml:
Organized by dataset name

Two formats are supported:

1. Simple format (for algorithms without search-time parameters):
    All parameters are used for both building and searching.
    List values are automatically expanded to all combinations.

2. Build/Search separation format (for algorithms with search-time parameters):
   Separate build-time and search-time parameters to avoid redundant training.
   Example:
   dataset-name:
     build:
       param1: [value1, value2]  # Build parameters (affect index construction)
     search:
       param2: [value3, value4]  # Search parameters (only affect query)
     common:
       param3: value5            # Used for both build and search