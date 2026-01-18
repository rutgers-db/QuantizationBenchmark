# Quantization Benchmark 使用指南

## 1. 项目简介
本项目是一个基于 Docker 的向量量化（Vector Quantization）算法基准测试框架。它提供了一个隔离、公平的环境来评估不同算法在召回率（Recall）、查询速度（QPS）、压缩率和内存占用等方面的性能。

## 2. 环境要求
在运行本项目之前，请确保满足以下条件：
- **操作系统**: Linux (推荐)
- **Docker**: 已安装并启动服务 (需要有执行 `docker` 命令的权限)
- **Python**: 3.x 版本

## 3. 快速开始

### 第一步：安装 Python 依赖
在项目根目录下运行以下命令安装必要的 Python 库：
```bash
pip install numpy h5py pyyaml
```

### 第二步：准备数据集
本项目使用 HDF5 格式的数据集（通常包含 `train`, `test`, `neighbors` 等数据集）。
1. 在项目根目录下创建一个名为 `data` 的文件夹（如果不存在）。
2. 将你的 `.hdf5` 数据集文件放入该文件夹。

例如，如果你有 `sift-128.hdf5`，路径应为：
`project_root/data/sift-128.hdf5`

### 第三步：构建 Docker 镜像
每个算法都在独立的 Docker 容器中运行，因此运行前必须构建镜像。

**查看所有可用算法：**
```bash
python run.py --list-algorithms
```

**构建指定算法的镜像（推荐）：**
```bash
python run.py --build-images --algorithm <算法文件夹名称>
```
例如：
```bash
python run.py --build-images --algorithm ProductQuantizationFaiss
```

**构建所有算法的镜像（耗时较长）：**
```bash
python run.py --build-images
```

### 第三步：运行基准测试
构建完成后，使用 `run.py` 启动测试。

**基本运行命令：**
```bash
python run.py --dataset <数据集名称> --algorithm <算法名称>
```
注意：
- `<数据集名称>` 不需要包含 `.hdf5` 后缀。
- `<算法名称>` 必须与 `--list-algorithms` 中显示的文件夹名称一致。

**示例：**
```bash
python run.py --dataset sift-128 --algorithm ProductQuantizationFaiss
```

**指定 Top-K (计算 Recall@K)：**
```bash
python run.py --dataset sift-128 --algorithm ProductQuantizationFaiss --topk 100
```

## 4. 查看结果
测试运行完成后：
1.  **终端输出**：屏幕上会显示摘要结果（包含 Build Time, Compression Rate, QPS, Recall 等）。
2.  **详细报告**：完整的 JSON 结果文件将保存在 `benchmark/results/` 目录下。文件名通常包含时间戳和算法参数。

## 5. 核心文件说明
- **`run.py`**: 项目的主入口脚本，用于控制构建和测试流程。
- **`benchmark/runner.py`**: 测试逻辑的核心调度器。
- **`benchmark/docker_runner.py`**: 负责管理 Docker 容器的生命周期及数据交换。
- **`benchmark/datasets.py`**: 负责加载和处理 HDF5 数据集。
- **`benchmark/algorithms/`**: 存放具体算法实现的目录（如 `ProductQuantizationFaiss`），每个算法包含代码和独立的 `Dockerfile`。

