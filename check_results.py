#!/usr/bin/env python3
"""
Check result files for a given method to see which (nlist, compression_rate)
combinations are present or missing across all datasets.

The script uses a per-method formula to compute the TRUE compression rate from
build_params, rather than the stored compression_rate field (which is wrong for
several methods).

Usage:
    python check_results.py <method> [options]

Examples:
    python check_results.py Faiss-IVFPQ --nlist 1024 2048 4096 --cr 4x 8x
    python check_results.py Faiss-IVFSQ --nlist 1024 2048 4096 --cr 4x 8x
    python check_results.py IVFOSQ      --nlist 1024 2048 4096 --cr 4x 8x
    python check_results.py IVFTurboQuant --nlist 1024 2048 4096 --cr 4x 8x
    python check_results.py SAQ_nlist4096 --nlist 4096 --cr 4x 8x
"""

import argparse
import json
import sys
from pathlib import Path


RESULTS_ROOT = Path(__file__).parent / "benchmark" / "results"


# ---------------------------------------------------------------------------
# Per-method TRUE compression-rate calculators
#
# Formula rationale (all IVF methods):
#   CR = bits_per_code_entry / (data_bytes * 8)
#
# where "bits_per_code_entry" is the nbit (or bitwidth) parameter that controls
# quantization precision.  nsubvec is intentionally IGNORED: it is a capacity/
# accuracy knob, not a compression knob.  Methods that include nsubvec in the
# stored compression_rate field are corrected here.
#
# Known stored-CR bugs:
#   Faiss-IVFPQ       : stored = nsubvec*nbit / (ndim*data_bytes*8)  → too small
#   Faiss-OPQ-IVFPQ   : stored = ndim*data_bytes*8 / (nsubvec*nbit)  → inverted
#   ResidualOPQ-IVFPQ : same inversion as OPQ-IVFPQ
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-method CR formulas
#
# Methods that store sub-codes (nsubvec × nbit bits total):
#   CR = nsubvec * nbit / (ndim * data_bytes)
#   Example: nbit=8, nsubvec=16, ndim=128, data_bytes=4
#            → 16*8 / (128*4) = 0.25  (4x)
#
# Methods that quantize each dimension directly (nbit bits per dim):
#   CR = nbit / (data_bytes * 8)
#   Example: nbit=8, data_bytes=4
#            → 8/32 = 0.25  (4x)
#
# TurboQuant uses "bitwidth" instead of "nbit", same formula.
# ---------------------------------------------------------------------------

def _pq_cr(bp: dict) -> float:
    """PQ-style: total code bits / original bytes.  nsubvec × nbit / (ndim × data_bytes)."""
    return bp["nsubvec"] * bp["nbit"] / (bp["ndim"] * bp["data_bytes"])


def _sq_cr(bp: dict) -> float:
    """SQ-style: nbit / (data_bytes × 8)."""
    return bp["nbit"] / (bp["data_bytes"] * 8)


def _bitwidth_cr(bp: dict) -> float:
    """TurboQuant: bitwidth / (data_bytes × 8)."""
    return bp["bitwidth"] / (bp["data_bytes"] * 8)


CR_CALCULATORS: dict[str, callable] = {
    "Faiss-IVFPQ":        _pq_cr,
    "Faiss-OPQ-IVFPQ":   _pq_cr,
    "ResidualOPQ-IVFPQ":  _pq_cr,
    "Faiss-IVFSQ":        _sq_cr,
    "IVFOSQ":             _sq_cr,
    "SAQ":                _sq_cr,
    "IVFSAQ":             _sq_cr,
    "IVFRabitQLibrary":   _sq_cr,
    "RabitQLibrary":      _sq_cr,
    "IVFTurboQuant":      _bitwidth_cr,
    "TurboQuant":         _bitwidth_cr,
}


def get_true_cr(method: str, bp: dict) -> float | None:
    """Return the true compression rate for a given method and build_params."""
    calc = CR_CALCULATORS.get(method)
    if calc is None:
        for prefix, fn in CR_CALCULATORS.items():
            if method.startswith(prefix):
                calc = fn
                break
    if calc is None:
        # Auto-detect fallback
        if "bitwidth" in bp:
            calc = _bitwidth_cr
        elif "nsubvec" in bp:
            calc = _pq_cr
        elif "nbit" in bp:
            calc = _sq_cr
        else:
            return None
    try:
        return calc(bp)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_cr(value: str) -> float:
    """Parse '4x' → 0.25, '8x' → 0.125, '0.25' → 0.25."""
    value = value.strip()
    if value.endswith("x"):
        return 1.0 / float(value[:-1])
    return float(value)


def fmt_cr(cr: float) -> str:
    factor = 1.0 / cr if cr != 0 else float("inf")
    if abs(factor - round(factor)) < 1e-6:
        return f"{cr:.4g} ({int(round(factor))}x)"
    return f"{cr:.4g} ({factor:.2f}x)"


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

# Keys excluded when printing "identifying" build params
_BORING_KEYS = {"nlist", "nthread", "space", "data_bytes", "ndim"}


def fmt_params(bp: dict) -> str:
    """Format the identifying subset of build_params as key=value pairs."""
    parts = [f"{k}={v}" for k, v in sorted(bp.items()) if k not in _BORING_KEYS]
    return "(" + ", ".join(parts) + ")"


def check_method(method: str, result_dir: str, expected_nlists: list[int],
                 expected_crs: list[float], cr_tol: float = 1e-4):
    results_path = RESULTS_ROOT / result_dir
    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(results_path.glob(f"*_{method}.json"))
    if not files:
        print(f"No result files matching '*_{method}.json' in {results_path}")
        return

    expected_combos = {(n, cr) for n in expected_nlists for cr in expected_crs}
    any_missing = False

    for fpath in files:
        dataset = fpath.stem[: -len(f"_{method}")]
        try:
            entries = json.loads(fpath.read_text())
        except Exception as e:
            print(f"[{dataset}] ERROR reading {fpath.name}: {e}")
            continue

        present: set[tuple[int, float]] = set()
        # cr -> set of param strings, for this dataset
        dataset_params: dict[float, set[str]] = {cr: set() for cr in expected_crs}
        unrecognised = []
        for entry in entries:
            bp = entry.get("build_params", {})
            nlist = bp.get("nlist")
            if nlist is None:
                continue
            cr = get_true_cr(method, bp)
            if cr is None:
                unrecognised.append(bp)
                continue
            matched_cr = next((e for e in expected_crs if abs(e - cr) <= cr_tol), None)
            if matched_cr is not None:
                present.add((int(nlist), matched_cr))
                dataset_params[matched_cr].add(fmt_params(bp))

        missing = expected_combos - present

        if unrecognised:
            print(f"[{dataset}] WARNING: could not compute CR for "
                  f"{len(unrecognised)} entry(ies) — unknown build_params keys")

        if missing:
            any_missing = True
            print(f"\n[{dataset}] MISSING {len(missing)} combination(s):")
            for nlist, cr in sorted(missing):
                print(f"    nlist={nlist:>5}  cr={fmt_cr(cr)}")
        else:
            print(f"[{dataset}] OK — all {len(expected_combos)} combinations present")

        # Print present params for this dataset, grouped by CR
        for cr in sorted(expected_crs, reverse=True):
            params = sorted(dataset_params[cr])
            label = fmt_cr(cr)
            if params:
                print(f"    cr={label}: " + "  ".join(params))
            else:
                print(f"    cr={label}: (none)")

    if not any_missing:
        print("\nAll good — no missing combinations.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check result files for missing (nlist, compression_rate) combinations.\n"
            "CR is computed from build_params using per-method formulas, "
            "not from the stored compression_rate field."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("method", help="Method name, e.g. Faiss-IVFPQ, IVFOSQ, SAQ_nlist4096")
    parser.add_argument(
        "--dir", default="ivf",
        help="Sub-directory under benchmark/results/ (default: ivf)",
    )
    parser.add_argument(
        "--nlist", nargs="+", type=int, default=[1024, 2048, 4096],
        metavar="N",
        help="Expected nlist values (default: 1024 2048 4096)",
    )
    parser.add_argument(
        "--cr", nargs="+", default=["4x", "8x"],
        metavar="CR",
        help=(
            "Expected compression rates as decimals or Nx notation "
            "(default: 4x 8x).  E.g.: --cr 4x 8x  or  --cr 0.25 0.125"
        ),
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4,
        help="Float tolerance for CR comparison (default: 1e-4)",
    )
    args = parser.parse_args()

    try:
        expected_crs = [parse_cr(v) for v in args.cr]
    except ValueError as e:
        print(f"ERROR parsing --cr values: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Method : {args.method}")
    print(f"Dir    : benchmark/results/{args.dir}/")
    print(f"nlist  : {args.nlist}")
    print(f"CR     : {[fmt_cr(c) for c in expected_crs]}")
    print(f"Expect : {len(args.nlist) * len(expected_crs)} combinations per dataset")
    print()

    check_method(args.method, args.dir, args.nlist, expected_crs, args.tol)


if __name__ == "__main__":
    main()
