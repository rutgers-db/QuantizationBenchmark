
"""
lsq.py

Pure NumPy implementation of Rayuela-style LSQ encoding/training.

Mirrors the control flow in the provided Julia file:
- ILS: perturb -> ICM -> keep only improved vectors (per-vector cost) and revert others
- ICM: copy unary costs, add pairwise conditioned costs, argmin update for each codebook
- Training: init C with RX=R^T X using fastbin (direct LS solve), rotate codebooks back with R,
  then alternate update_codebooks_fastbin <-> encoding_icm.

Shapes:
- Input X: (n,d)
- Internal X_dn: (d,n)
- Codes B_mn: (m,n) int32
- Codebooks C_mdh: (m,d,h) float32

Dependencies: numpy (required), h5py (optional for save/load)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

try:
    import h5py  # optional
except Exception:
    h5py = None


# -------------------------- utilities --------------------------

def _as_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32, copy=False)
    return np.ascontiguousarray(X.astype(np.float32, copy=False))


def _as_codes(B: np.ndarray, m: int, n: int) -> np.ndarray:
    B = np.asarray(B)
    if B.shape != (m, n):
        raise ValueError(f"Expected codes shape (m,n)=({m},{n}), got {B.shape}")
    if not np.issubdtype(B.dtype, np.integer):
        B = B.astype(np.int32, copy=False)
    B = np.ascontiguousarray(B.astype(np.int32, copy=False))
    if B.size > 0 and B.min() < 0:
        raise ValueError("Codes must be >= 0")
    return B


def _reconstruct_dn(C_mdh: np.ndarray, B_mn: np.ndarray) -> np.ndarray:
    """
    Return Xhat in shape (d,n).
    IMPORTANT: use np.take to avoid NumPy advanced-index dimension reordering.
    """
    m, d, h = C_mdh.shape
    _, n = B_mn.shape
    Xhat = np.zeros((d, n), dtype=np.float32)
    for i in range(m):
        # C_mdh[i] is (d,h). take along axis=1 -> (d,n)
        Xhat += np.take(C_mdh[i], B_mn[i], axis=1)
    return Xhat


def veccost_dn(X_dn: np.ndarray, B_mn: np.ndarray, C_mdh: np.ndarray) -> np.ndarray:
    """Per-vector squared reconstruction error, returns (n,)."""
    diff = X_dn - _reconstruct_dn(C_mdh, B_mn)
    return np.sum(diff * diff, axis=0)


def qerror_dn(X_dn: np.ndarray, B_mn: np.ndarray, C_mdh: np.ndarray) -> float:
    return float(np.mean(veccost_dn(X_dn, B_mn, C_mdh)))


# -------------------------- energy terms --------------------------

def get_unaries_dn(X_dn: np.ndarray, C_mdh: np.ndarray) -> List[np.ndarray]:
    """
    For each codebook i, unary costs:
      U_i(k, j) = ||c_i(k)||^2 - 2 x_j^T c_i(k)
    Return list length m, each (h, n).
    """
    m, d, h = C_mdh.shape
    d2, n = X_dn.shape
    if d2 != d:
        raise ValueError("Dim mismatch between X and C")

    unaries: List[np.ndarray] = []
    for i in range(m):
        Ci = C_mdh[i]  # (d,h)
        norms = np.sum(Ci * Ci, axis=0)         # (h,)
        dots = Ci.T @ X_dn                      # (h,n)
        Ui = norms[:, None] - 2.0 * dots        # (h,n)
        unaries.append(Ui.astype(np.float32, copy=False))
    return unaries


def get_binaries(C_mdh: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Pairwise terms for each pair (i,j), i<j:
      B_{ij}(k,l) = 2 * c_i(k)^T c_j(l)
    Returns:
      binaries: list of (h,h)
      cbi: (2, ncbi) pairs, 0-indexed
    """
    m, d, h = C_mdh.shape
    pairs = []
    binaries: List[np.ndarray] = []
    for i in range(m):
        Ci = C_mdh[i]  # (d,h)
        for j in range(i + 1, m):
            Cj = C_mdh[j]
            Bij = 2.0 * (Ci.T @ Cj)  # (h,h)
            binaries.append(Bij.astype(np.float32, copy=False))
            pairs.append((i, j))
    cbi = np.array(pairs, dtype=np.int32).T if pairs else np.zeros((2, 0), dtype=np.int32)
    return binaries, cbi


def _build_cbpair2binaryidx(m: int, cbi: np.ndarray) -> np.ndarray:
    cbpair2binaryidx = -np.ones((m, m), dtype=np.int32)
    for idx, (i, j) in enumerate(cbi.T.tolist()):
        cbpair2binaryidx[i, j] = idx
        cbpair2binaryidx[j, i] = idx
    return cbpair2binaryidx


# -------------------------- ICM / ILS encoding --------------------------

def perturb_codes(
    B_mn: np.ndarray,
    npert: int,
    h: int,
    idx: np.ndarray,
    rng: np.random.Generator,
    replace: bool = True,
) -> None:
    """
    In-place perturbation of codes on subset idx (vector indices).
    For each vector, choose npert codebook rows and set to random code.
    """
    if npert <= 0:
        return
    m, n = B_mn.shape
    idx = np.asarray(idx, dtype=np.int64)
    nn = idx.size
    if nn == 0:
        return

    if replace:
        pert_rows = rng.integers(0, m, size=(npert, nn), endpoint=False)
    else:
        pert_rows = np.empty((npert, nn), dtype=np.int64)
        for t in range(nn):
            pert_rows[:, t] = rng.choice(m, size=npert, replace=False)

    pert_vals = rng.integers(0, h, size=(npert, nn), endpoint=False)
    for j in range(npert):
        rows = pert_rows[j]
        B_mn[rows, idx] = pert_vals[j].astype(B_mn.dtype, copy=False)


def iterated_conditional_modes(
    B_mn: np.ndarray,
    unaries: List[np.ndarray],
    binaries: List[np.ndarray],
    binaries_t: List[np.ndarray],
    cbpair2binaryidx: np.ndarray,
    to_look: np.ndarray,
    to_condition: np.ndarray,
    icmiter: int,
    idx: np.ndarray,
) -> None:
    """
    In-place block-ICM update on subset idx.
    Uses np.take to avoid advanced-index dimension reordering.
    """
    idx = np.asarray(idx, dtype=np.int64)
    nn = idx.size
    if nn == 0 or icmiter <= 0:
        return

    for _ in range(icmiter):
        jidx = 0
        for j in to_look:
            ub = np.take(unaries[j], idx, axis=1).copy()  # (h,nn)

            for k in to_condition[:, jidx]:
                binidx = cbpair2binaryidx[j, k]
                if binidx < 0:
                    continue
                bb = binaries[binidx] if j < k else binaries_t[binidx]
                codek = B_mn[k, idx]  # (nn,)
                ub += np.take(bb, codek, axis=1)  # (h,nn)

            B_mn[j, idx] = np.argmin(ub, axis=0).astype(B_mn.dtype, copy=False)
            jidx += 1


def encoding_icm(
    X_dn: np.ndarray,
    oldB_mn: np.ndarray,
    C_mdh: np.ndarray,
    ilsiter: int,
    icmiter: int,
    randord: bool,
    npert: int,
    rng: np.random.Generator,
    verbose: bool = False,
) -> np.ndarray:
    """
    Rayuela-style ILS around ICM, with per-vector acceptance.
    Returns: updated codes (m,n)
    """
    X_dn = _as_float(X_dn)
    m, d, h = C_mdh.shape
    d2, n = X_dn.shape
    if d2 != d:
        raise ValueError("Dim mismatch between X and C")
    oldB_mn = _as_codes(oldB_mn, m=m, n=n)

    unaries = get_unaries_dn(X_dn, C_mdh)
    binaries, cbi = get_binaries(C_mdh)
    binaries_t = [b.T.copy() for b in binaries]
    cbpair2binaryidx = _build_cbpair2binaryidx(m, cbi)

    idx = np.arange(n, dtype=np.int64)

    # each column excludes itself
    all_idx = np.arange(m, dtype=np.int32)
    base_to_condition = np.empty((m - 1, m), dtype=np.int32)
    for j in range(m):
        base_to_condition[:, j] = np.delete(all_idx, j)

    for it in range(1, ilsiter + 1):
        prevcost = veccost_dn(X_dn, oldB_mn, C_mdh)
        B = oldB_mn.copy()

        to_look = np.arange(m, dtype=np.int32)
        to_condition = base_to_condition
        if randord:
            to_look = rng.permutation(m).astype(np.int32, copy=False)
            to_condition = base_to_condition[:, to_look].copy()

        perturb_codes(B, npert=npert, h=h, idx=idx, rng=rng, replace=True)
        iterated_conditional_modes(
            B,
            unaries=unaries,
            binaries=binaries,
            binaries_t=binaries_t,
            cbpair2binaryidx=cbpair2binaryidx,
            to_look=to_look,
            to_condition=to_condition,
            icmiter=icmiter,
            idx=idx,
        )

        newcost = veccost_dn(X_dn, B, C_mdh)
        better = newcost < prevcost

        if verbose:
            eq = (newcost == prevcost).mean() * 100.0
            bt = better.mean() * 100.0
            print(f" ILS iteration {it}/{ilsiter} done. {eq:5.2f}% equal. {bt:5.2f}% better.")

        B[:, ~better] = oldB_mn[:, ~better]
        oldB_mn[...] = B

    return oldB_mn


# -------------------------- codebook update (fastbin) --------------------------

def _build_BBt_and_BX(
    X_nd: np.ndarray,
    codes_nm: np.ndarray,
    m: int,
    h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
      G = B B^T (mh x mh)
      At = B X (mh x d)
    where B is stacked one-hot matrix by codebook.
    """
    n, d = X_nd.shape
    mh = m * h

    At = np.zeros((mh, d), dtype=np.float64)
    for i in range(m):
        block = np.zeros((h, d), dtype=np.float64)
        ci = codes_nm[:, i]
        np.add.at(block, ci, X_nd.astype(np.float64, copy=False))
        At[i * h:(i + 1) * h, :] = block

    G = np.zeros((mh, mh), dtype=np.float64)
    for i in range(m):
        ci = codes_nm[:, i]
        cnt = np.bincount(ci, minlength=h).astype(np.float64, copy=False)
        bi = slice(i * h, (i + 1) * h)
        G[bi, bi] = np.diag(cnt)

    for i in range(m):
        ci = codes_nm[:, i]
        bi = slice(i * h, (i + 1) * h)
        for j in range(i + 1, m):
            cj = codes_nm[:, j]
            bj = slice(j * h, (j + 1) * h)
            idx2 = (ci.astype(np.int64) * h + cj.astype(np.int64))
            cnt2 = np.bincount(idx2, minlength=h * h).astype(np.float64, copy=False)
            H = cnt2.reshape(h, h)
            G[bi, bj] = H
            G[bj, bi] = H.T

    return G, At


def update_codebooks_fastbin(
    X_dn: np.ndarray,
    B_mn: np.ndarray,
    h: int,
    lam: float = 1e-4,
) -> np.ndarray:
    """
    Solve least squares (fastbin/direct update) and return C_mdh: (m,d,h).
    """
    X_dn = _as_float(X_dn)
    d, n = X_dn.shape
    m, n2 = B_mn.shape
    if n2 != n:
        raise ValueError("B and X must have same n")

    X_nd = X_dn.T
    codes_nm = B_mn.T.astype(np.int64, copy=False)

    G, At = _build_BBt_and_BX(X_nd, codes_nm, m=m, h=h)
    mh = m * h
    G.flat[:: mh + 1] += float(lam)

    Cstack = np.linalg.solve(G, At)  # (mh,d)
    C_mhd = Cstack.reshape(m, h, d).astype(np.float32, copy=False)
    return np.transpose(C_mhd, (0, 2, 1)).copy()  # (m,d,h)


# -------------------------- public API --------------------------

@dataclass
class LSQConfig:
    m: int
    h: int = 256
    niter: int = 25
    ilsiter: int = 8
    icmiter: int = 4
    randord: bool = True
    npert: int = 4
    lam: float = 1e-4
    seed: int = 123


class LSQ:
    """
    Rayuela-style LSQ trainer.
    """

    def __init__(self, cfg: LSQConfig):
        if cfg.m <= 0 or cfg.h <= 1:
            raise ValueError("m must be > 0 and h must be > 1")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.C_: Optional[np.ndarray] = None  # (m,d,h)
        self.B_: Optional[np.ndarray] = None  # (m,n)
        self.R_: Optional[np.ndarray] = None  # (d,d)
        self.obj_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        R: Optional[np.ndarray] = None,
        B_init: Optional[np.ndarray] = None,
        C_init: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "LSQ":
        X_nd = _as_float(X)
        n, d = X_nd.shape
        X_dn = X_nd.T

        m, h = self.cfg.m, self.cfg.h

        if R is None:
            R = np.eye(d, dtype=np.float32)
        R = _as_float(R)
        if R.shape != (d, d):
            raise ValueError(f"R must be (d,d)=({d},{d}), got {R.shape}")
        self.R_ = R

        if B_init is None:
            B = self.rng.integers(0, h, size=(m, n), endpoint=False, dtype=np.int32)
        else:
            B = _as_codes(B_init, m=m, n=n)

        if C_init is None:
            RX_dn = (R.T @ X_dn).astype(np.float32, copy=False)
            C = update_codebooks_fastbin(RX_dn, B, h=h, lam=self.cfg.lam)
            C = np.einsum("ij,mjh->mih", R, C).astype(np.float32, copy=False)
        else:
            C = np.asarray(C_init, dtype=np.float32)
            if C.shape != (m, d, h):
                raise ValueError(f"C_init must be (m,d,h)=({m},{d},{h}), got {C.shape}")

        if verbose:
            print("*" * 94)
            print(
                f"Training LSQ with m={m}, h={h}, npert={self.cfg.npert}, "
                f"icmiter={self.cfg.icmiter}, randord={self.cfg.randord}"
            )
            print("*" * 94)
            print(f"{-2:3d} {qerror_dn(X_dn, B, C):.6e}")

        # init B via encoding
        B = encoding_icm(
            X_dn=X_dn,
            oldB_mn=B,
            C_mdh=C,
            ilsiter=self.cfg.ilsiter,
            icmiter=self.cfg.icmiter,
            randord=self.cfg.randord,
            npert=self.cfg.npert,
            rng=self.rng,
            verbose=verbose,
        )
        if verbose:
            print(f"{-1:3d} {qerror_dn(X_dn, B, C):.6e}")

        obj = np.zeros(self.cfg.niter, dtype=np.float32)

        for it in range(1, self.cfg.niter + 1):
            obj[it - 1] = qerror_dn(X_dn, B, C)
            if verbose:
                print(f"{it:3d} {obj[it - 1]:.6e}")

            C = update_codebooks_fastbin(X_dn, B, h=h, lam=self.cfg.lam)
            B = encoding_icm(
                X_dn=X_dn,
                oldB_mn=B,
                C_mdh=C,
                ilsiter=self.cfg.ilsiter,
                icmiter=self.cfg.icmiter,
                randord=self.cfg.randord,
                npert=self.cfg.npert,
                rng=self.rng,
                verbose=verbose,
            )

        self.C_ = C
        self.B_ = B
        self.obj_ = obj
        return self

    def encode(self, X: np.ndarray, B_init: Optional[np.ndarray] = None, verbose: bool = False) -> np.ndarray:
        if self.C_ is None:
            raise ValueError("fit() first")
        X_nd = _as_float(X)
        n, d = X_nd.shape
        m, d2, h = self.C_.shape
        if d2 != d:
            raise ValueError("Dim mismatch")
        X_dn = X_nd.T

        if B_init is None:
            B = self.rng.integers(0, h, size=(m, n), endpoint=False, dtype=np.int32)
        else:
            B = _as_codes(B_init, m=m, n=n)

        return encoding_icm(
            X_dn=X_dn,
            oldB_mn=B,
            C_mdh=self.C_,
            ilsiter=self.cfg.ilsiter,
            icmiter=self.cfg.icmiter,
            randord=self.cfg.randord,
            npert=self.cfg.npert,
            rng=self.rng,
            verbose=verbose,
        )

    def decode(self, B_mn: np.ndarray) -> np.ndarray:
        if self.C_ is None:
            raise ValueError("fit() first")
        m, d, h = self.C_.shape
        B_mn = _as_codes(B_mn, m=m, n=B_mn.shape[1])
        return _reconstruct_dn(self.C_, B_mn).T  # (n,d)

    def save_h5(self, path: str) -> None:
        if h5py is None:
            raise RuntimeError("h5py not installed. pip install h5py")
        if self.C_ is None:
            raise ValueError("Nothing to save (fit first).")
        with h5py.File(path, "w") as f:
            f.create_dataset("C", data=self.C_, compression="gzip")
            if self.B_ is not None:
                f.create_dataset("B", data=self.B_, compression="gzip")
            if self.R_ is not None:
                f.create_dataset("R", data=self.R_, compression="gzip")
            if self.obj_ is not None:
                f.create_dataset("obj", data=self.obj_, compression="gzip")
            for k, v in self.cfg.__dict__.items():
                f.attrs[f"cfg_{k}"] = v

    @classmethod
    def load_h5(cls, path: str) -> "LSQ":
        if h5py is None:
            raise RuntimeError("h5py not installed. pip install h5py")
        with h5py.File(path, "r") as f:
            C = f["C"][...]
            m, d, h = C.shape
            cfg_kwargs = {"m": int(m), "h": int(h)}
            for k in ["niter", "ilsiter", "icmiter", "randord", "npert", "lam", "seed"]:
                ak = f"cfg_{k}"
                if ak in f.attrs:
                    cfg_kwargs[k] = f.attrs[ak].item() if hasattr(f.attrs[ak], "item") else f.attrs[ak]
            cfg = LSQConfig(**cfg_kwargs)
            obj = cls(cfg)
            obj.C_ = C.astype(np.float32, copy=False)
            if "B" in f:
                obj.B_ = f["B"][...].astype(np.int32, copy=False)
            if "R" in f:
                obj.R_ = f["R"][...].astype(np.float32, copy=False)
            if "obj" in f:
                obj.obj_ = f["obj"][...].astype(np.float32, copy=False)
        return obj
