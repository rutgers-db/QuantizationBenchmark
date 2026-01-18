import numpy as np
from typing import Tuple
import psutil
import sys
import os

# 添加 LSQ 算法路径
sys.path.insert(0, '/algorithms/quantizer/LSQ')
sys.path.insert(0, '/benchmark')

from lsq import LSQ, LSQConfig
from benchmark.base import BaseQuantizer


class LSQQuantizer(BaseQuantizer):
    """
    基于纯 NumPy 实现的 LSQ（Locally-Structured Quantization）量化器
    使用 Rayuela-style 的 ILS 和 ICM 编码/训练
    """
    
    def __init__(self, ndim, nsubvec, nbit, data_bytes, nthread=1, space="l2", 
                 niter=25, ilsiter=8, icmiter=4, randord=True, npert=4, lam=1e-4, seed=123):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec  # 对应 LSQ 的 m (codebook 数量)
        self.nbit = nbit
        self.h = 2 ** nbit  # codebook 大小
        self.data_bytes = data_bytes
        self.nthread = nthread
        self.space = space
        
        # LSQ 配置
        self.cfg = LSQConfig(
            m=nsubvec,
            h=self.h,
            niter=niter,
            ilsiter=ilsiter,
            icmiter=icmiter,
            randord=randord,
            npert=npert,
            lam=lam,
            seed=seed
        )
        
        self.lsq = LSQ(self.cfg)
        self.data = None
        self.ndata = 0
        self.codes = None  # 编码后的 codes (m, n)
        
    def fit(self, nd: int, data: np.ndarray) -> bool:
        """
        训练 LSQ 模型
        
        Args:
            nd: 数据点数量
            data: 训练数据，shape (n, d)
        
        Returns:
            bool: 训练是否成功
        """
        self.data = data.astype(np.float32, copy=False)
        self._original_data = self.data
        self.ndata = nd
        
        try:
            print(f"开始训练 LSQ: n={nd}, d={self.ndim}, m={self.nsubvec}, h={self.h}")
            
            # 训练 LSQ 模型
            self.lsq.fit(
                X=self.data,
                R=None,  # 不使用旋转矩阵
                B_init=None,  # 随机初始化 codes
                C_init=None,  # 自动初始化 codebooks
                verbose=True
            )
            
            # 保存训练数据的 codes
            self.codes = self.lsq.B_  # (m, n)
            
            print(f"LSQ 训练完成")
            return True
            
        except Exception as e:
            print(f"训练错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询最近邻
        
        Args:
            nq: 查询数量
            query: 查询向量，shape (nq, d)
            topk: 返回的最近邻数量
            
        Returns:
            indices: shape (nq, topk)
            distances: shape (nq, topk)
        """
        if self.lsq.C_ is None or self.data is None:
            raise ValueError("必须先调用 fit() 进行训练")
        
        query = query.astype(np.float32, copy=False)
        
        # 对查询向量进行编码
        query_codes = self.lsq.encode(query, B_init=None, verbose=False)  # (m, nq)
        
        # 重构查询向量用于计算距离
        query_recon = self.lsq.decode(query_codes)  # (nq, d)
        
        # 重构数据库向量
        data_recon = self.lsq.decode(self.codes)  # (n, d)
        
        # 计算距离并找到 top-k
        indices = np.zeros((nq, topk), dtype=np.int32)
        distances = np.zeros((nq, topk), dtype=np.float32)
        
        for i in range(nq):
            if self.space.lower() == "l2":
                # L2 距离
                dists = np.sum((data_recon - query_recon[i:i+1]) ** 2, axis=1)
            else:
                # 内积（余弦相似度）
                dists = -np.dot(data_recon, query_recon[i])
            
            # 找到 top-k
            if topk >= self.ndata:
                idx = np.argsort(dists)
                indices[i] = idx[:topk] if topk <= self.ndata else np.pad(idx, (0, topk - self.ndata), constant_values=-1)
                distances[i] = dists[indices[i]]
            else:
                idx = np.argpartition(dists, topk)[:topk]
                sorted_idx = idx[np.argsort(dists[idx])]
                indices[i] = sorted_idx
                distances[i] = dists[sorted_idx]
        
        return indices, distances
    
    def getMemoryUsage(self) -> float:
        """返回内存使用量 (KB)"""
        return psutil.Process().memory_info().rss / 1024
    
    def getCompressionRate(self) -> float:
        """
        返回压缩率
        原始: ndim * data_bytes * 8 bits
        压缩后: nsubvec * nbit bits
        """
        original_bits = self.ndim * self.data_bytes * 8
        compressed_bits = self.nsubvec * self.nbit
        return compressed_bits / original_bits
    
    def getCompressionMemory(self) -> float:
        """
        返回压缩后的内存占用 (bits)
        Codebooks: m * d * h * 32 bits (float32)
        Codes: n * m * nbit bits
        """
        if self.lsq.C_ is None or self.codes is None:
            return 0.0
        
        # Codebooks memory: (m, d, h) float32
        codebook_bits = self.nsubvec * self.ndim * self.h * 32
        
        # Codes memory: (m, n) with nbit per entry
        codes_bits = self.ndata * self.nsubvec * self.nbit
        
        return float(codebook_bits + codes_bits)
    
    def getMSE(self) -> float:
        """
        返回均方误差 (Mean Squared Error)
        """
        if self.lsq.C_ is None or self.codes is None or self.data is None:
            return 0.0
        
        # 重构数据
        recons = self.lsq.decode(self.codes)  # (n, d)
        
        # 计算 MSE
        se_per_row = np.sum((recons - self.data) ** 2, axis=1)
        mse = np.mean(se_per_row)
        
        return float(mse)