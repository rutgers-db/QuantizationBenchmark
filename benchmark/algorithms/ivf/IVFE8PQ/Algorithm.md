# IVFE8PQ — PQ-codebook RaBitQ-style IVF quantizer

## 1. 设计出发点

IVFE8PQ 是 **IVFE8 的一个变体**，只替换了 codebook 的来源，其余管线（随机旋转、IVF 粗量化、RaBitQ 风格的无偏距离估计）完全一致。命名带 "E8" 主要是为了强调它承袭的是 IVFE8 的 **方向内积估计器**，并与 **直接量化残差** 的 IVFPQ 区分开来；具体的 codebook **不包含任何 E8 几何内容**。

| 组件 | IVFE8 | IVFE8PQ | IVFPQ |
|---|---|---|---|
| 随机旋转 | FHT-Kac（或 matrix） | 同上 | —（OPQ 才旋转） |
| 粗量化 | KMeans → `nlist` | 同上 | 同上 |
| 量化对象 | `o = (x_r − c) / ‖x_r − c‖`（单位向量） | **同上** | `r = x_r − c`（未归一化残差） |
| Codebook | 每 8 维一块，固定 E8 最小范数 root（240+16 个共 256） | 每 `dsub` 维一块，**在归一化残差上学习** 的 256 个中心（标准 PQ） | 每 `dsub` 维一块，对未归一化残差学习的 256 个中心 |
| 估计器 | 对归一化后的 `o` 与 `q` 估 `<o, q>`，再反推欧氏距离 | **同 IVFE8** | 直接 `Σ_b ‖q_b − ô_b‖²` |

关键差异：
- **IVFPQ**：LUT 存子向量 **平方距离**，累加即残差欧氏距离。
- **IVFE8PQ**：LUT 存子向量 **内积**；累加后得到的是 `<ô, q>` 的估计；再配合 RaBitQ 的 `f_add / f_rescale` 反推 `‖x_r − q_r‖²`。

## 2. 距离估算：两边都归一化的几何视角

令 `x_r = R(x)`、`q_r = R(q)`，对任一候选簇质心 `c_r`：
```
r = x_r − c_r        (数据侧残差)
s = q_r − c_r        (查询侧残差)
```
目标：
```
‖x_r − q_r‖² = ‖r − s‖² = ‖r‖² + ‖s‖² − 2·<r, s>
```
**两边同时归一化**：`o = r / ‖r‖`（训练时做）、`q = s / ‖s‖`（查询时做）。则：
```
<r, s> = ‖r‖ · ‖s‖ · <o, q>
‖x_r − q_r‖² = ‖r‖² + ‖s‖² − 2·‖r‖·‖s‖·<o, q>                       (★)
```
整个估计问题归结为估 **两个单位向量的内积 `<o, q>`**。
IVFE8PQ 用 PQ 把 `o` 量化为 `ô`（`ô` 的每个子块是 PQ 中心，近似单位方向），然后使用 **RaBitQ 的无偏近似**：
```
<o, q> ≈ <ô, q> / <ô, o>
```
把这个代回 (★) 就得到：
```
‖x_r − q_r‖² ≈ ‖r‖² + ‖s‖² − 2·‖r‖·‖s‖·<ô, q>/<ô, o>                 (☆)
```
其中 `‖r‖`、`<ô, o>` 都能在建库时算好，`‖s‖` 和 `<ô, q>` 在查询时算。

### 2.1 把 (☆) 重写成和 IVFE8 完全一样的形式

注意以下两个恒等式（都只是在展开 `q = s/‖s‖`）：
```
‖s‖·<ô, q>        = <ô, s>                   = <ô, q_r> − <ô, c_r>
‖r‖·<ô, o>        = <ô, r>
```
令
```
ip_resi  := <ô, r>        (= ‖r‖·<ô, o>)
ip_cent  := <ô, c_r>
l2       := ‖r‖²
g_add    := ‖s‖² = ‖q_r − c_r‖²   (查询期、每候选簇一个)
```
带入 (☆)：
```
‖r‖·‖s‖·<ô, q>/<ô, o>
  = l2 · ‖s‖·<ô, q> / ip_resi
  = l2 · (<ô, q_r> − ip_cent) / ip_resi
```
于是 (☆) 展成：
```
‖x_r − q_r‖² ≈ l2 + g_add − 2·l2·(<ô, q_r> − ip_cent) / ip_resi
             = g_add + f_add + f_rescale · <ô, q_r>
where
  f_add     = l2 + 2·l2·ip_cent / ip_resi
  f_rescale = −2·l2 / ip_resi                                         (◇)
```
这正是 IVFE8 的打分公式。**数学上它等价于"两边先归一化、估 `<o, q>`、再反推"**，只是把归一化因子 `‖r‖`、`‖s‖`、`<ô,o>` 全部折进 `f_add / f_rescale` 里，让查询路径可以直接把 **未归一化** 的 `q_r` 作为 LUT 输入，避免：
- 每候选簇都要重建一次 LUT（若用 `q = s/‖s‖`，s 随 c_r 变，LUT 就得每簇重建）；
- 每 lane 都要额外乘 `‖s‖·‖r‖`（常数项都并进 `f_add / f_rescale`）。

所以 IVFE8PQ 的扫描内核和 IVFE8 一字不差：`est = g_add + f_add + f_rescale · <q_r, ô>`，其中 `<q_r, ô>` 通过一张 `M × 256` 的 float LUT 的 `M` 次 gather 累加得到。

## 3. 训练流程（全部在 C++ 侧完成）

输入：原始数据 `X ∈ R^{n×d}`，`nlist`、PQ 子向量数 `M`（`nsubvec`）。令 `d' = round_up(d, 64)`，`dsub = d'/M`，要求 `d' % M = 0`。所有训练步骤写在 `e8pq.hpp::IVFE8PQ::fit()`，通过 include faiss 头文件直接调用 `faiss::Clustering` 和 `faiss::ProductQuantizer`。

### 3.1 粗量化
`faiss::Clustering(d, nlist)` 在原始 `X` 上训练，得到质心 `C`；再用 `faiss::IndexFlatL2` 为每个 `X[i]` 找最近簇 `cluster_ids[i]`。

### 3.2 建立旋转器并旋转质心
默认走 **OPQ-on-residuals**（`d == d'` 时自动启用）：

1. 先用 `rabitqlib::choose_rotator<float>(d, FhtKacRotator, d')` 构造初始 `R₀`
2. 用 `R₀` 旋转每个簇质心，再对训练样本计算归一化残差 `o_i = R₀(x_i − c_{cid_i}) / ‖·‖`
3. 调用 `faiss::OPQMatrix(d, M, d')` 在 `o_i` 样本上训练 25 outer × 4 inner 迭代，得到 OPQ 旋转矩阵 `R_opq`
4. **组合最终旋转矩阵 `R = R_opq · R₀`** 灌入 `rabitqlib::MatrixRotator`，替换 `R₀`
5. 用 `R` 重新旋转所有质心存入 `rotated_centroids_`

**关键设计选择 — 在归一化残差上训练 OPQ**：标准 OPQ 在 raw data 上训练（`IVFE8PQ_USE_OPQ=1`），但 IVF + per-cluster normalization 把原始数据的 anisotropy 大部分消解，OPQ-on-raw 实测**毫无收益**（κ 不动、recall ±0.05pp 噪声）。OPQ 必须训在 PQ **实际量化的对象** —— 归一化残差 `o`——上才能找到对齐子空间的最优旋转。

`IVFE8PQ_USE_OPQ` 环境变量：
- 默认（不设）：`mode=2` OPQ-on-residuals
- `=0`：禁用 OPQ，走 FhtKac 随机旋转（baseline）
- `=1`：legacy raw-data OPQ（无收益，仅作对照）
- `=2`：等同默认

**实测增益（recall@100, nprobe=10..100, vs no-OPQ baseline）**：
| 数据集 | nsubvec | dsub | Δrecall (典型) |
|---|---|---|---|
| SIFT-128 | 16 | **8** | **+2.94 ~ +3.42 pp** ★ |
| SIFT-128 | 32 | 4 | +0.30 ~ +0.46 pp |
| SIFT-128 | 64 | 2 | flat（±0.04） |
| GIST-960 | 120 | **8** | **+2.91 ~ +7.56 pp** ★★ |
| GIST-960 | 240 | 4 | +0.54 ~ +3.06 pp ★ |
| GIST-960 | 480 | 2 | flat（±0.06） |

GIST 上增益 > SIFT 因为高维数据残差结构更复杂，OPQ headroom 更大。dsub=2 配置 κ 已极小（0.09），无 headroom。

> **构建时间代价**：OPQ 训练 25 outer×4 inner iter，SIFT-128 增加 ~1 min，GIST-960 增加 ~10-20 min。可接受。要 fastest build 走 `IVFE8PQ_USE_OPQ=0`。

### 3.3 收集归一化残差样本，训练 PQ
对随机抽取的 `n_sample ≤ min(n, 256·1024)` 条样本：
```
x_r = R(X[i]);  r = x_r − c_r^{cid[i]};  o = r / max(‖r‖, ε)
```
拼成样本矩阵 `O_sample ∈ R^{n_sample×d'}`，调用 `faiss::ProductQuantizer pq(d', M, 8); pq.train(n_sample, O_sample)` 得到 PQ 中心 `P`（faiss 布局 `[b · K · dsub + k · dsub + j]`）。

**E_8 lattice 初始化（仅 dsub=8）**：在 `pq.train(...)` 之前，把 IVFE8 的 240 个 minimum-norm root 向量（加 16 个 padding 重复至 256 条）除 √2 归一为单位向量，灌入 `pq.centroids` 并设置 `pq.train_type = Train_hot_start`，让 faiss L2 k-means 从 Gersho 最优 8-D 球面填充（G_8 ≤ 0.0717）开始细化。`dsub ≠ 8` 时跳过此步、走 faiss 默认 k-means++ init。helper 在 `e8pq_pq_train.hpp::init_centroids_e8_`。

> **设计 note**：早期实现尝试过 per-subspace 球面 k-means（mean update + L2 normalize），目标是直接对 `κ = ‖ε⊥‖/<c,o>` 优化。但强制每个子空间 centroid `‖c̃_b‖=1` 与 PQ 数据真实尺度 `‖x_b‖² ≈ 1/M` 严重失配（centroid 比数据大 √M 倍），导致 `dsub ≤ 4` recall 回退 5–30 pp。报告里"per-vector 归一化消除 ε‖"是**全局**结论，per-subspace L2 k-means 的均值更新恰好给出 Lloyd 半径定理（4.12）所要求的 `ρ = cos φ` 自然尺度——不能强行单位化。当前版本**只换 init、不换迭代**。

> 训练在 **归一化** 的残差上，是因为 codebook 要表示 `ô`（单位方向）。若不归一化，PQ 中心会跟随 `‖r‖` 的尺度分布，而 `<ô, o>` 的波动变大，`f_add/f_rescale` 的无偏性变差。

**实测增益（SIFT-128 nlist=1024，recall@100, nprobe=10..100）**：
| nsubvec | dsub | E_8 init | IVFE8PQ Δ recall | IVFE8PQFastScan Δ recall |
|---|---|---|---|---|
| 16 | 8 | ✓ | **+0.19~+0.28 pp** | +0.01~+0.10 pp |
| 32 | 4 | ✗ | ±0.02 pp（flat） | ±0.05 pp（flat）|
| 64 | 2 | ✗ | ±0.03 pp（flat） | ±0.03 pp（flat）|
| 128 | 1 | ✗ | — | ±0.03 pp（flat）|

dsub=8 上 IVFE8PQ canonical 的提升明显超过 FastScan，原因：FastScan 的 u8 LUT 量化噪声本身就在 κ 量级，吃掉 E_8 init 的边际改善。dsub≠8 完全持平 baseline（faiss 默认路径未触动）。

### 3.4 编码 + 计算每向量 RaBitQ 因子（多线程）
并行遍历所有 `i ∈ [0, n)`：
```
x_r = R(X[i]);  c_r = c_r^{cid[i]};  r = x_r − c_r;  l2 = ‖r‖²
o   = r / max(‖r‖, ε)
codes[i] = pq.compute_code(o)       // M 字节
ô        = pq.decode(codes[i])      // 各子块 = P[b][codes[i][b]]

ip_resi  = <ô, r>
ip_cent  = <ô, c_r>
if ip_resi == 0: ip_resi = +∞       // 退化保护

f_add[i]     = l2 + 2·l2·ip_cent / ip_resi
f_rescale[i] = −2·l2 / ip_resi
```
codes 按 IVF 簇切分，每簇内以 16-wide tile、block-major 布局存入 `codes_bm`：
```
codes_bm[tile · M · 16 + b · 16 + lane] = k_b
```
便于 AVX-512 `_mm512_i32gather_ps`。

## 4. 搜索流程

对每个查询 `q`：
1. `q_r = R(q)`；
2. 对全部 `nlist` 个 `c_r^l` 算 `‖q_r − c_r^l‖²`，`partial_sort` 取最小 `nprobe` 个；
3. **构建 LUT**（`M × 256` 的 float 表）：`LUT[b][k] = <q_r[b·dsub:(b+1)·dsub], P[b][k]>`；
4. **扫描候选簇**：对每个候选簇 `l`：
   - `g_add = ‖q_r − c_r^l‖²`；
   - 以 16-lane tile 循环：每 block `b` 加载 16 字节 codes，`_mm512_cvtepu8_epi32`，`_mm512_i32gather_ps(LUT[b], idx, 4)` 取 16 个 float 累加成 `ip_sum`；
   - `est = fmadd(f_rescale, ip_sum, g_add + f_add)`；
   - 用当前 top-k 堆顶为 bound 做 SIMD mask 过滤，再更新堆；
5. 尾部（< 16）走标量路径。最终输出 `(I, D)`。

## 5. 复杂度与存储

- **每向量存储**：`M` 字节 code + 2 个 float = `M + 8` 字节。`dsub = 8` 时预算与 IVFE8 一致。
- **全局 codebook**：`M · 256 · dsub · 4` 字节。
- **LUT 建表开销**：每查询 `O(M · 256 · dsub)`；`dsub = 8` 时走与 IVFE8 相同的 `_mm512_fmadd_ps` 宽内层。
- **扫描开销**：每候选向量 `M` 次 gather + 几条 FMA。

## 6. 一句话总结

IVFE8PQ = IVFE8 的 RaBitQ 估计器 + **在归一化残差上学到的 PQ codebook**。数学上它等价于"两边都归一化后估 `<o, q>` 再反推欧氏距离"，但通过 `f_add / f_rescale` 把所有归一化因子折进去，查询路径和 IVFE8 完全对齐；相比 IVFPQ，它用 RaBitQ 无偏因子把子向量内积映射回欧氏距离，在低比特预算下通常更准。
