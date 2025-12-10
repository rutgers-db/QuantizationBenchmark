# Base Dockerfile for all quantization algorithms
# This provides common dependencies needed by ivf.py and other shared components
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install common Python dependencies required by ivf.py and framework
RUN pip install --no-cache-dir \
    numpy \
    h5py \
    pyyaml \
    psutil \
    faiss-cpu

# Set Python path to include the root directory
ENV PYTHONPATH=/

# Set working directory
WORKDIR /workspace

# Note: Individual algorithm dockerfiles should use:
# FROM quantbench-base:latest
# and then add their algorithm-specific dependencies
