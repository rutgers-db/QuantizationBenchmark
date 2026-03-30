import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


_MINIMUM_MSE_GRID = np.array(
    [
        (-0.798, 0.798),
        (-1.493, 1.493),
        (-2.051, 2.051),
        (-2.514, 2.514),
        (-2.916, 2.916),
        (-3.278, 3.278),
        (-3.611, 3.611),
        (-3.922, 3.922),
    ],
    dtype=np.float32,
)


def normalize_rows(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return data / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.copy()
    return vector / norm


def space_to_metric(space: str) -> int:
    return faiss.METRIC_L2 if str(space).lower() == "l2" else faiss.METRIC_INNER_PRODUCT


def discrete_dims(bits: int, query_bits: int, dims: int) -> int:
    def _round_dims(n: int, bits_per_dim: int) -> int:
        total_bits = n * bits_per_dim
        rounded_bits = ((total_bits + 7) // 8) * 8
        return rounded_bits // bits_per_dim

    if bits == 2 and query_bits == 4:
        return max(_round_dims(dims, 1), _round_dims(dims, 4))
    return max(_round_dims(dims, bits), _round_dims(dims, query_bits))


def resolve_query_bits(nbit: int, query_nbit: Optional[int]) -> int:
    if query_nbit is not None:
        return int(query_nbit)
    if nbit in (1, 2):
        return 4
    return int(nbit)


def validate_bits(nbit: int, query_nbit: int) -> None:
    if nbit not in (1, 2, 4, 7, 8):
        raise ValueError("OSQ supports nbit in {1, 2, 4, 7, 8}")
    if query_nbit < nbit:
        raise ValueError("query_nbit must be >= nbit")
    if query_nbit not in (1, 2, 4, 7, 8):
        raise ValueError("OSQ supports query_nbit in {1, 2, 4, 7, 8}")


@dataclass
class EncodedBatch:
    processed: np.ndarray
    codes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    additional: np.ndarray
    qsum: np.ndarray


@dataclass
class QueryState:
    processed: np.ndarray
    codes: np.ndarray
    lower: float
    upper: float
    additional: float
    qsum: float


class OSQCodec:
    def __init__(self, ndim: int, nbit: int, space: str = "l2", query_nbit: Optional[int] = None, lam: float = 0.1, iters: int = 5):
        self.ndim = int(ndim)
        self.nbit = int(nbit)
        self.query_nbit = resolve_query_bits(self.nbit, query_nbit)
        validate_bits(self.nbit, self.query_nbit)
        self.space = str(space).lower()
        self.lam = float(lam)
        self.iters = int(iters)
        self.discrete_dims = discrete_dims(self.nbit, self.query_nbit, self.ndim)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.space == "cosine":
            return normalize_rows(data)
        return data.astype(np.float32, copy=False)

    def preprocess_query(self, query: np.ndarray) -> np.ndarray:
        if self.space == "cosine":
            return normalize_vector(query.astype(np.float32, copy=False))
        return query.astype(np.float32, copy=False)

    def compute_centroid(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return np.zeros(self.ndim, dtype=np.float32)
        centroid = np.mean(self.preprocess(data), axis=0).astype(np.float32)
        if self.space == "cosine":
            centroid = normalize_vector(centroid)
        return centroid

    def encode_batch(self, data: np.ndarray, centroid: np.ndarray, bits: int) -> EncodedBatch:
        processed = self.preprocess(data)
        n = processed.shape[0]
        codes = np.zeros((n, self.discrete_dims), dtype=np.uint8)
        lower = np.zeros(n, dtype=np.float32)
        upper = np.zeros(n, dtype=np.float32)
        additional = np.zeros(n, dtype=np.float32)
        qsum = np.zeros(n, dtype=np.float32)

        for i in range(n):
            code, a, b, add_corr, component_sum = self._encode_vector(processed[i], centroid, bits)
            codes[i] = code
            lower[i] = a
            upper[i] = b
            additional[i] = add_corr
            qsum[i] = component_sum

        return EncodedBatch(
            processed=processed,
            codes=codes,
            lower=lower,
            upper=upper,
            additional=additional,
            qsum=qsum,
        )

    def encode_query(self, query: np.ndarray, centroid: np.ndarray) -> QueryState:
        processed = self.preprocess_query(query)
        codes, a, b, add_corr, component_sum = self._encode_vector(processed, centroid, self.query_nbit)
        return QueryState(
            processed=processed,
            codes=codes,
            lower=float(a),
            upper=float(b),
            additional=float(add_corr),
            qsum=float(component_sum),
        )

    def reconstruct(self, centroid: np.ndarray, codes: np.ndarray, lower: np.ndarray, upper: np.ndarray, bits: int) -> np.ndarray:
        if codes.shape[0] == 0:
            return np.zeros((0, self.ndim), dtype=np.float32)
        steps = float((1 << bits) - 1)
        safe_denom = np.where((upper - lower) > 0, upper - lower, 1.0)
        step = safe_denom / steps
        recons = codes[:, : self.ndim].astype(np.float32) * step[:, None] + lower[:, None]
        return recons + centroid[None, :]

    def score_query_to_docs(
        self,
        query_state: QueryState,
        centroid: np.ndarray,
        doc_codes: np.ndarray,
        doc_lower: np.ndarray,
        doc_upper: np.ndarray,
        doc_additional: np.ndarray,
        doc_qsum: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if doc_codes.shape[0] == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        q_codes = query_state.codes.astype(np.float32, copy=False)
        qc_dist = doc_codes.astype(np.float32, copy=False) @ q_codes
        ay = query_state.lower
        ly = (query_state.upper - query_state.lower) / float((1 << self.query_nbit) - 1)
        sy = query_state.qsum
        ax = doc_lower
        lx = (doc_upper - doc_lower) / float((1 << self.nbit) - 1)
        dot_est = ax * ay * self.ndim + ay * lx * doc_qsum + ax * ly * sy + lx * ly * qc_dist

        if self.space == "l2":
            distances = query_state.additional + doc_additional - 2.0 * dot_est
            distances = np.maximum(distances, 0.0)
            scores = 1.0 / (1.0 + distances)
            return scores.astype(np.float32), distances.astype(np.float32)

        distances = -dot_est
        if self.space == "cosine":
            centroid_dp = float(np.dot(centroid, centroid))
            similarity = dot_est + query_state.additional + doc_additional - centroid_dp
            similarity = np.clip(similarity, -1.0, 1.0)
            distances = -similarity
            scores = (1.0 + similarity) / 2.0
            return scores.astype(np.float32), distances.astype(np.float32)

        similarity = dot_est + query_state.additional + doc_additional - float(np.dot(centroid, centroid))
        scores = np.where(similarity < 0.0, 1.0 / (1.0 - similarity), similarity + 1.0)
        distances = -scores
        return scores.astype(np.float32), distances.astype(np.float32)

    def _encode_vector(self, vector: np.ndarray, centroid: np.ndarray, bits: int) -> Tuple[np.ndarray, float, float, float, int]:
        centered = vector - centroid
        min_v = float(np.min(centered))
        max_v = float(np.max(centered))
        mean = float(np.mean(centered))
        std = float(np.std(centered))
        norm2 = float(np.dot(centered, centered))
        centroid_dot = float(np.dot(vector, centroid)) if self.space != "l2" else norm2

        a = float(np.clip(_MINIMUM_MSE_GRID[bits - 1, 0] * std + mean, min_v, max_v))
        b = float(np.clip(_MINIMUM_MSE_GRID[bits - 1, 1] * std + mean, min_v, max_v))
        a, b = self._optimize_intervals(centered, a, b, norm2, 1 << bits)

        if b <= a:
            b = a + 1e-6

        steps = float((1 << bits) - 1)
        step = (b - a) / steps
        clipped = np.clip(centered, a, b)
        quantized = np.rint((clipped - a) / step).astype(np.uint8)
        padded = np.zeros(self.discrete_dims, dtype=np.uint8)
        padded[: self.ndim] = quantized
        return padded, a, b, centroid_dot, int(np.sum(quantized, dtype=np.int64))

    def _loss(self, centered: np.ndarray, a: float, b: float, points: int, norm2: float) -> float:
        if b <= a:
            return float("inf")
        step = (b - a) / (points - 1.0)
        clipped = np.clip(centered, a, b)
        xiq = a + step * np.rint((clipped - a) / step)
        xe = float(np.sum(centered * (centered - xiq), dtype=np.float64))
        e = float(np.sum((centered - xiq) ** 2, dtype=np.float64))
        return (1.0 - self.lam) * xe * xe / max(norm2, 1e-12) + self.lam * e

    def _optimize_intervals(self, centered: np.ndarray, a: float, b: float, norm2: float, points: int) -> Tuple[float, float]:
        if norm2 <= 0.0 or b <= a:
            return a, b

        best_loss = self._loss(centered, a, b, points, norm2)
        scale = (1.0 - self.lam) / norm2
        if not np.isfinite(scale):
            return a, b

        for _ in range(self.iters):
            if b <= a:
                break
            step_inv = (points - 1.0) / (b - a)
            clipped = np.clip(centered, a, b)
            k = np.rint((clipped - a) * step_inv)
            s = k / (points - 1.0)

            daa = float(np.sum((1.0 - s) * (1.0 - s), dtype=np.float64))
            dab = float(np.sum((1.0 - s) * s, dtype=np.float64))
            dbb = float(np.sum(s * s, dtype=np.float64))
            dax = float(np.sum(centered * (1.0 - s), dtype=np.float64))
            dbx = float(np.sum(centered * s, dtype=np.float64))

            m0 = scale * dax * dax + self.lam * daa
            m1 = scale * dax * dbx + self.lam * dab
            m2 = scale * dbx * dbx + self.lam * dbb
            det = m0 * m2 - m1 * m1
            if abs(det) < 1e-12:
                break

            a_opt = (m2 * dax - m1 * dbx) / det
            b_opt = (m0 * dbx - m1 * dax) / det
            if abs(a - a_opt) < 1e-8 and abs(b - b_opt) < 1e-8:
                break

            new_loss = self._loss(centered, float(a_opt), float(b_opt), points, norm2)
            if new_loss > best_loss:
                break

            a = float(a_opt)
            b = float(b_opt)
            best_loss = new_loss

        return a, b


def topk_from_distances(ids: np.ndarray, distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if ids.size == 0:
        return np.full(k, -1, dtype=np.int64), np.full(k, np.inf, dtype=np.float32)
    actual_k = min(k, ids.shape[0])
    part = np.argpartition(distances, actual_k - 1)[:actual_k]
    order = part[np.argsort(distances[part], kind="stable")]
    top_ids = ids[order].astype(np.int64, copy=False)
    top_distances = distances[order].astype(np.float32, copy=False)
    if actual_k == k:
        return top_ids, top_distances

    padded_ids = np.full(k, -1, dtype=np.int64)
    padded_distances = np.full(k, np.inf, dtype=np.float32)
    padded_ids[:actual_k] = top_ids
    padded_distances[:actual_k] = top_distances
    return padded_ids, padded_distances


def merge_topk(candidates: List[Tuple[float, int]], k: int) -> Tuple[np.ndarray, np.ndarray]:
    if not candidates:
        return np.full(k, -1, dtype=np.int64), np.full(k, np.inf, dtype=np.float32)
    best = heapq.nsmallest(min(k, len(candidates)), candidates)
    ids = np.full(k, -1, dtype=np.int64)
    distances = np.full(k, np.inf, dtype=np.float32)
    for i, (distance, idx) in enumerate(best):
        ids[i] = int(idx)
        distances[i] = float(distance)
    return ids, distances


def build_invlists(assignments: np.ndarray, nlist: int) -> List[np.ndarray]:
    lists: List[List[int]] = [[] for _ in range(nlist)]
    for idx, list_id in enumerate(assignments.tolist()):
        lists[int(list_id)].append(idx)
    return [np.asarray(lst, dtype=np.int64) for lst in lists]


def centroid_memory_bits(nlist: int, ndim: int) -> int:
    return nlist * ndim * 32
