#!/usr/bin/env python3
"""
Plot recall vs Jensen-Shannon divergence for the IVF distribution-shift runs
on sift-128-euclidean.

One figure per nlist (1024 and 4096). Each IVF method contributes one point per
distribution-shift group; within every (method, nlist) pair we expect exactly
one result — no parameter selection.

Usage:
    python scripts/plot_ivf_distribution_shift.py
        # writes PDFs into figures/ivf/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

DATASET = "sift-128-euclidean"
NLISTS = (1024, 4096)
RESULTS_DIR = Path("benchmark/results/distribution_shift")
OUTPUT_DIR = Path("figures/ivf")


@dataclass
class MethodSpec:
    label: str
    filename: str
    color: str
    marker: str


# Colors and aliases mirror scripts/ivf_all.py so the distribution-shift plot
# reads as the same series across every IVF figure. Entries are sorted
# alphabetically so the legend order stays consistent across plots.
IVF_METHODS: List[MethodSpec] = [
    MethodSpec("IVFOPQ",        f"{DATASET}_Faiss-OPQ-IVFPQ.json",   "#9467bd", "o"),
    MethodSpec("IVFOSQ",        f"{DATASET}_IVFOSQ.json",            "#8c564b", "o"),
    MethodSpec("IVFPQ",         f"{DATASET}_Faiss-IVFPQ.json",       "#1f77b4", "o"),
    MethodSpec("IVFRaBitQ",     f"{DATASET}_IVFRabitQLibrary.json",  "#d62728", "o"),
    MethodSpec("IVFSAQ",        f"{DATASET}_IVFSAQ.json",            "#f39c12", "o"),
    MethodSpec("IVFSQ",         f"{DATASET}_Faiss-IVFSQ.json",       "#ff7f0e", "o"),
    MethodSpec("IVFTurboQuant", f"{DATASET}_IVFTurboQuant.json",     "#00acc1", "o"),
]


def _extract_js_divergence(group_entry: Dict[str, Any]) -> float:
    metadata = group_entry.get("distribution_shift_metadata") or {}
    if "js_divergence" in metadata:
        return float(metadata["js_divergence"])
    for result in group_entry.get("results", []):
        shift_info = result.get("distribution_shift") or {}
        if "js_divergence" in shift_info:
            return float(shift_info["js_divergence"])
    raise ValueError(
        f"Missing js_divergence in group {group_entry.get('distribution_shift_group')!r}"
    )


def _pick_result_for_nlist(group_entry: Dict[str, Any], spec: MethodSpec, nlist: int) -> Dict[str, Any]:
    matches = []
    for result in group_entry.get("results", []):
        if result.get("status") != "success":
            continue
        build_params = (result.get("quantizer_config") or {}).get("build_params") or {}
        if int(build_params.get("nlist", -1)) == nlist:
            matches.append(result)

    group_name = group_entry.get("distribution_shift_group", "<unknown>")
    if not matches:
        raise ValueError(
            f"No result with nlist={nlist} for method '{spec.label}' in group '{group_name}'"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple results with nlist={nlist} for method '{spec.label}' "
            f"in group '{group_name}' — expected only one config per nlist"
        )
    return matches[0]


def _load_points(spec: MethodSpec, nlist: int, results_dir: Path) -> List[Tuple[float, float]]:
    path = results_dir / spec.filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")

    points = []
    for group_entry in raw:
        if not isinstance(group_entry, dict):
            continue
        js = _extract_js_divergence(group_entry)
        result = _pick_result_for_nlist(group_entry, spec, nlist)
        if "recall" not in result:
            raise ValueError(f"Missing 'recall' in result for '{spec.label}' in {path}")
        points.append((js, float(result["recall"])))

    points.sort(key=lambda p: p[0])
    return points


def _save_legend_pdf(handles: Sequence[Any], labels: Sequence[str], legend_path: Path) -> None:
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(legend_path, bbox_inches=bbox, pad_inches=0)
    plt.close(fig_leg)
    print(f"Saved legend to {legend_path}")


def _plot_one_nlist(
    nlist: int,
    methods: Sequence[MethodSpec],
    results_dir: Path,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    handles, labels = [], []

    for spec in methods:
        points = _load_points(spec, nlist, results_dir)
        if not points:
            print(f"[warn] No points for {spec.label} (nlist={nlist}); skipping")
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        max_y, min_y = max(ys), min(ys)
        ratio = float("inf") if min_y == 0 else max_y / min_y
        print(
            f"nlist={nlist}  {spec.label:<20s} max/min recall = {ratio:.4g} "
            f"(max={max_y:.4g}, min={min_y:.4g})"
        )
        (line,) = ax.plot(
            xs, ys,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle="-",
            linewidth=1.8,
            markersize=6,
        )
        handles.append(line)
        labels.append(spec.label)

    ax.set_xlabel("JS divergence", fontsize=18)
    ax.set_ylabel("Recall", fontsize=18)
    # ax.set_title(f"nlist = {nlist}", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{DATASET}_nlist{nlist}_distribution_shift.pdf"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", format="pdf")
    print(f"Saved plot to {plot_path}")
    plt.close(fig)

    legend_dir = output_dir / "legends"
    legend_path = legend_dir / f"{DATASET}_nlist{nlist}_distribution_shift_legend.pdf"
    _save_legend_pdf(handles, labels, legend_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot IVF distribution-shift recall curves for sift at nlist 1024 and 4096."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory containing distribution-shift JSON files (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for PDFs (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    for nlist in NLISTS:
        _plot_one_nlist(nlist, IVF_METHODS, args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
