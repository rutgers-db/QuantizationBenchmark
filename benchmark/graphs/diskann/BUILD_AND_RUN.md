# DiskANN with Quantizer - Build and Run Guide

## Prerequisites

1. **System Requirements**
   - Ubuntu 22.04 (or similar Linux)
   - GCC 9+ with C++17 support
   - CMake 3.15+
   - Python 3.8+

2. **Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       build-essential \
       cmake \
       g++ \
       python3 \
       python3-pip \
       python3-dev \
       libopenblas-dev \
       libboost-all-dev \
       libomp-dev
   ```

3. **Python Packages**
   ```bash
   pip3 install numpy pybind11 faiss-cpu
   ```

## Build Steps

### Step 1: Navigate to the DiskANN directory
```bash
cd /data/local/jl3288/QuantizationBenchmark/benchmark/graphs/diskann
```

### Step 2: Create build directory
```bash
mkdir -p build
cd build
```

### Step 3: Run CMake

**IMPORTANT: Include MKL paths and pybind11 directory**

```bash
rm -f CMakeCache.txt  # Clean previous cache if exists
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -DMKL_PATH=/common/home/jl3288/intel/oneapi/mkl/latest/lib/intel64 \
    -DMKL_INCLUDE_PATH=/common/home/jl3288/intel/oneapi/mkl/latest/include
```

**Expected output:**
```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Found Python3: ... (found version "3.12.11")
-- Found pybind11: ... (found version "3.0.1")
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Using MKL from: /common/home/jl3288/intel/oneapi/mkl/latest/lib/intel64
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
```

### Step 4: Compile
```bash
make -j$(nproc)
```

**Expected output:**
```
[ 10%] Building CXX object CMakeFiles/diskann_cpp.dir/diskann_pybind.cpp.o
[ 20%] Building CXX object CMakeFiles/diskann_cpp.dir/.../disk_utils.cpp.o
...
[100%] Linking CXX shared module diskann_cpp.cpython-*.so
[100%] Built target diskann_cpp
```

### Step 5: Install the module
```bash
# Copy the compiled module to the diskann directory
cp diskann_cpp*.so ..
```

### Step 6: Verify installation
```bash
cd ..
ls -lh diskann_cpp*.so
```

You should see a file like:
```
-rwxr-xr-x 1 user user 15M Dec 15 15:30 diskann_cpp.cpython-310-x86_64-linux-gnu.so
```

## Running the Test

### Method 1: Direct Python execution

```bash
cd /data/local/jl3288/QuantizationBenchmark
python3 benchmark/graphs/diskann/test_diskann.py
```

### Method 2: Make it executable

```bash
chmod +x benchmark/graphs/diskann/test_diskann.py
./benchmark/graphs/diskann/test_diskann.py
```

### Expected Test Output

```
================================================================================
Testing DiskANN with Product Quantization (Faiss)
================================================================================

Dataset:
  Base vectors: 1000
  Query vectors: 10
  Dimension: 128

Generating test data...

Creating Product Quantization quantizer...
  PQ parameters: M=16, nbits=8

Index directory: /tmp/diskann_pq_test_xxxxx

Creating DiskANN index...
  Graph parameters: R=32, L=50

--------------------------------------------------------------------------------
Building index...
--------------------------------------------------------------------------------
Building DiskANN index for 1000 points, dim=128
Training quantizer...
Saving data to /tmp/diskann_pq_test_xxxxx/index_data.bin
Building Vamana graph...
[DiskANN build messages...]

✓ Build completed successfully in X.XX seconds

Index statistics:
  num_points: 1000
  dimension: 128
  R: 32
  L: 50
  index_prefix: /tmp/diskann_pq_test_xxxxx/index

--------------------------------------------------------------------------------
Testing search...
--------------------------------------------------------------------------------
Search parameters:
  TopK: 10
  Search L: 50
  Beam width: 4

✓ Search completed successfully in X.XX seconds
  QPS: XX.XX queries/sec
  Latency per query: XX.XX ms

--------------------------------------------------------------------------------
Validating results...
--------------------------------------------------------------------------------
Result shapes: I=(10, 10), D=(10, 10)

Sample results (first query):
  Indices: [xxx xxx xxx ...]
  Distances: X.XXXX
  ✓ All indices are valid
  ✓ All distances are non-negative

Computing ground truth for recall calculation...

Recall@10: 0.XXXX
✓ Reasonable recall for approximate search

================================================================================
✓ TEST PASSED - DiskANN with PQ working correctly!
================================================================================

Cleaning up temporary directory: /tmp/diskann_pq_test_xxxxx
Cleanup complete

🎉 All tests passed!
```

---

## ✅ BUILD SUCCESS NOTES

### Critical Fixes Applied

编译成功的关键修复：

#### 1. **编译选项 (CMakeLists.txt)**
必须添加与DiskANN原始编译选项匹配的SIMD flags：
```cmake
target_compile_options(diskann_cpp PRIVATE
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -msse2
    -ftree-vectorize
    -fopenmp
    -fopenmp-simd
    -funroll-loops
    -Wall
    -O3
    -DNDEBUG
    -DUSE_AVX2
)
```
**重要**: 没有这些选项，`_mm128_*`等Intel SIMD intrinsics无法正确编译！

#### 2. **包含所有DiskANN源文件**
使用glob模式包含所有源文件：
```cmake
file(GLOB DISKANN_SOURCES "${DISKANN_ROOT}/src/*.cpp")
```

#### 3. **链接libaio库**
Linux异步I/O支持：
```cmake
target_link_libraries(diskann_cpp PRIVATE aio)
```

#### 4. **修复Boost头文件冲突**
修改 `tmp/DiskANN/include/boost_dynamic_bitset_fwd.h`：
```cpp
#ifndef BOOST_DYNAMIC_BITSET_FWD_HPP
#define BOOST_DYNAMIC_BITSET_FWD_HPP  // 添加这行避免重复定义
template <typename Block = unsigned long, typename Allocator = std::allocator<Block>> class dynamic_bitset;
#endif
```

#### 5. **修改pq_flash_index.h**
- 将 `QuantizerAdapter::estimate_distance()` 声明为 `const`
- 添加了新的 `cached_beam_search()` 重载支持quantizer参数

#### 6. **修改pq_flash_index.cpp**
在新的`cached_beam_search()`重载中：
- Lambda capture: `[this, quantizer]`
- 使用 `_dist_cmp_float` 和 `query_float` 计算到centroid的距离
- 使用 `res_ids` 和 `res_dists` 参数名（非 `indices`/`distances`）

### 当前状态

✅ **编译成功** - 无errors，仅warnings
✅ **模块可导入** - diskann_cpp.so 可以被Python加载
✅ **Index build成功** - DiskANN分片、合并、disk layout创建都完成
⚠️ **运行时问题** - 测试时出现segmentation fault

### 已知问题

运行test_diskann.py时遇到：
1. **Segmentation fault** - 在search阶段crash
2. **"Error. Number of PQ centroids is not 256"** - DiskANN期望256个centroids

可能原因：
- ProductQuantizationFaiss的配置与DiskANN期望不匹配
- 需要调整quantizer参数使其与DiskANN兼容

---

## Troubleshooting

### Issue 1: CMake can't find pybind11

**Error:**
```
CMake Error: Could not find pybind11
```

**Solution:**
```bash
pip3 install pybind11[global]
# Or specify the path manually
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
```

### Issue 2: Missing DiskANN sources

**Error:**
```
fatal error: pq_flash_index.h: No such file or directory
```

**Solution:**
Ensure DiskANN is in the correct location:
```bash
ls /data/local/jl3288/QuantizationBenchmark/tmp/DiskANN/include/pq_flash_index.h
```

### Issue 3: Python module not found

**Error:**
```
ImportError: No module named 'diskann_cpp'
```

**Solution:**
```bash
# Make sure the .so file is in the diskann directory
cd /data/local/jl3288/QuantizationBenchmark/benchmark/graphs/diskann
ls diskann_cpp*.so

# If missing, copy from build directory
cp build/diskann_cpp*.so .
```

### Issue 4: Undefined symbols when loading module

**Error:**
```
ImportError: undefined symbol: _ZN7diskann...
```

**Solution:**
This usually means linking issues. Rebuild with verbose output:
```bash
cd build
make VERBOSE=1
```

Check that all DiskANN source files are being compiled and linked.

### Issue 5: Python path issues

**Error:**
```
ModuleNotFoundError: No module named 'benchmark.graphs.diskann.module'
```

**Solution:**
```bash
# Run from the benchmark root directory
cd /data/local/jl3288/QuantizationBenchmark
export PYTHONPATH=/data/local/jl3288/QuantizationBenchmark:$PYTHONPATH
python3 benchmark/graphs/diskann/test_diskann.py
```

## Clean Build

If you need to rebuild from scratch:

```bash
cd /data/local/jl3288/QuantizationBenchmark/benchmark/graphs/diskann
rm -rf build
rm -f diskann_cpp*.so
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp diskann_cpp*.so ..
cd ..
python3 test_diskann.py
```

## Development Tips

### Incremental Builds

After modifying C++ code:
```bash
cd build
make -j$(nproc)
cp diskann_cpp*.so ..
```

### Debug Build

For debugging with symbols:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Testing with Different Quantizers

Modify the test script to use different quantizers:

```python
# Instead of ProductQuantizationFaiss
from benchmark.algorithms.quantizer.RabitQ.module import RabitQ

quantizer = RabitQ(
    ndim=dim,
    nlist=256,
    data_bytes=8,
    nthread=4,
    space="l2"
)
```

## Performance Testing

For serious benchmarking, use larger datasets:

```python
n_base = 100000  # 100K vectors
n_query = 1000   # 1K queries
dim = 128

# Larger graph parameters
R = 64
L = 100
search_L = 100
```

## Next Steps

After successful testing:

1. **Integrate with benchmark runner**
   - Add DiskANN to `benchmark/runner.py`
   - Configure in experiment YAML files

2. **Compare with other methods**
   - Test against different quantizers
   - Benchmark against baseline methods

3. **Optimize parameters**
   - Tune R, L for your dataset
   - Adjust search_L and beam_width for speed/accuracy tradeoff
