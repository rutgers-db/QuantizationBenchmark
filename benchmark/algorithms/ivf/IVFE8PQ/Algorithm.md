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
`rabitqlib::choose_rotator<float>(d, FhtKacRotator, d')` 构造 `R`；对每个 `C[l]` 计算 `c_r^l = R(C[l])` 存入 `rotated_centroids_`。

### 3.3 收集归一化残差样本，训练 PQ
对随机抽取的 `n_sample ≤ min(n, 256·1024)` 条样本：
```
x_r = R(X[i]);  r = x_r − c_r^{cid[i]};  o = r / max(‖r‖, ε)
```
拼成样本矩阵 `O_sample ∈ R^{n_sample×d'}`，调用 `faiss::ProductQuantizer pq(d', M, 8); pq.train(n_sample, O_sample)` 得到 PQ 中心 `P`（逻辑 shape `M × 256 × dsub`）。

> 训练在 **归一化** 的残差上，是因为 codebook 要表示 `ô`（单位方向）。若不归一化，PQ 中心会跟随 `‖r‖` 的尺度分布，而 `<ô, o>` 的波动变大，`f_add/f_rescale` 的无偏性变差。

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
