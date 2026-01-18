
"""
lsqpp.py

LSQ++ built on top of lsq.py (Rayuela-style encoding), with stochastic relaxation (SR-C / SR-D)
and direct (fastbin) codebook update.

Keep lsq.py and lsqpp.py in the same folder.

Dependencies: numpy (required)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

from lsq import (
    LSQ,
    LSQConfig,
    _as_float,
    _as_codes,
    qerror_dn,
    encoding_icm,
    update_codebooks_fastbin,
)

SRMode = Optional[Literal["sr-c", "sr-d"]]


def _temperature(i: int, I: int, p: float) -> float:
    # T(i) = (1 - i/I)^p
    if I <= 0:
        return 0.0
    x = 1.0 - (i / float(I))
    return float(max(0.0, x) ** p)


def _diag_var_from_Xdn(X_dn: np.ndarray) -> np.ndarray:
    # diag(cov(X)) estimate per-dim, X_dn: (d,n)
    return X_dn.var(axis=1, ddof=1).astype(np.float32, copy=False)


@dataclass
class LSQppConfig(LSQConfig):
    sr: SRMode = "sr-d"        # "sr-c" or "sr-d" or None
    p: float = 0.5             # temperature exponent
    noise_scale: float = 1.0   # overall noise multiplier


class LSQpp(LSQ):
    """
    LSQ++ training with SR-C / SR-D.
    Encoding/ICM/ILS is inherited (Rayuela-style).
    """

    def __init__(self, cfg: LSQppConfig):
        super().__init__(cfg)
        self.cfg: LSQppConfig
        self._diag_var_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        R: Optional[np.ndarray] = None,
        B_init: Optional[np.ndarray] = None,
        C_init: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "LSQpp":
        X_nd = _as_float(X)
        n, d = X_nd.shape
        X_dn = X_nd.T
        m, h = self.cfg.m, self.cfg.h

        # noise stats from X
        self._diag_var_ = _diag_var_from_Xdn(X_dn)  # (d,)
        std_d = np.sqrt(np.clip(self._diag_var_, 1e-12, None)).astype(np.float32, copy=False)  # (d,)

        # init rotation / codes / codebooks (same as LSQ)
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
                f"Training LSQ++ with m={m}, h={h}, sr={self.cfg.sr}, p={self.cfg.p}, noise_scale={self.cfg.noise_scale}, "
                f"npert={self.cfg.npert}, icmiter={self.cfg.icmiter}, randord={self.cfg.randord}"
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
            T = _temperature(it, self.cfg.niter, self.cfg.p)
            obj[it - 1] = qerror_dn(X_dn, B, C)
            if verbose:
                print(f"{it:3d} {obj[it - 1]:.6e}   T={T:.4f}")

            # SR-D: noise on codebooks during encoding
            if self.cfg.sr == "sr-d" and T > 0:
                noise = self.rng.normal(size=C.shape).astype(np.float32, copy=False)
                noise *= (std_d[None, :, None] * (self.cfg.noise_scale * (T / float(m))))
                C_enc = C + noise
            else:
                C_enc = C

            # Update codes
            B = encoding_icm(
                X_dn=X_dn,
                oldB_mn=B,
                C_mdh=C_enc,
                ilsiter=self.cfg.ilsiter,
                icmiter=self.cfg.icmiter,
                randord=self.cfg.randord,
                npert=self.cfg.npert,
                rng=self.rng,
                verbose=verbose,
            )

            # SR-C: noise on X during codebook update
            if self.cfg.sr == "sr-c" and T > 0:
                eps = self.rng.normal(size=X_dn.shape).astype(np.float32, copy=False)
                X_upd_dn = X_dn + eps * (std_d[:, None] * (self.cfg.noise_scale * T))
            else:
                X_upd_dn = X_dn

            # Update codebooks
            C = update_codebooks_fastbin(X_upd_dn, B, h=h, lam=self.cfg.lam)

        self.C_ = C
        self.B_ = B
        self.obj_ = obj
        return self
