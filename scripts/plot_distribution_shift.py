#!/usr/bin/env python3
"""
Plot recall against Jensen-Shannon divergence for distribution-shift results.

Edit `DEFAULT_SERIES_SPECS` below to choose which algorithms/configurations to
plot. Each plotted series is defined by:
1. The JSON file to read.
2. A label used in the legend.
3. A set of dotted-path filters that select exactly one result inside each
   distribution-shift group.

Then run:

python scripts/plot_distribution_shift.py --output /tmp/distribution_shift.pdf
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
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

@dataclass
class SeriesSpec:
    label: str
    file: Path
    filters: Dict[str, Any] = field(default_factory=dict)
    color: str | None = None
    marker: str = "o"
    linestyle: str = "-"

# DEFAULT_SERIES_SPECS: List[Dict[str, Any]] = [
#     {
#         "label": "OPQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_OptimizedProductQuantizationFaiss.json",
#         "filters": {
#             "quantizer": "OptimizedProductQuantizationFaiss",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.nbit": 8,
#             "quantizer_config.build_params.nsubvec": 480,
#             "quantizer_config.build_params.niter": 400,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#9467bd",
#         "marker": "^",
#     },
#     {
#         "label": "OSQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_OptimizedScalarQuantization.json",
#         "filters": {
#             "quantizer": "OptimizedScalarQuantization",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.nbit": 4,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#8c564b",
#         "marker": "D",
#     },
#     {
#         "label": "PQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_ProductQuantizationFaiss.json",
#         "filters": {
#             "quantizer": "ProductQuantizationFaiss",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.nbit": 8,
#             "quantizer_config.build_params.nsubvec": 480,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#1f77b4",
#         "marker": "<",
#     },
#     {
#         "label": "RaBitQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_ExtendedRabitQNTU.json",
#         "filters": {
#             "quantizer": "ExtendedRabitQNTU",
#             "search_params.topk": 100,
#             "search_params.nprobe": 1,
#             "quantizer_config.build_params.nlist": 1,
#             "quantizer_config.build_params.nbit": 4,
#             "quantizer_config.build_params.high_acc_flag": [False, True],
#         },
#         "color": "#d62728",
#         "marker": "o",
#     },
#     {
#         "label": "SAQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_SAQ.json",
#         "filters": {
#             "quantizer": "SAQ",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.nbit": 4,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#f39c12",
#         "marker": "8",
#     },
#     {
#         "label": "SQ",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_ScalarQuatizationFaiss.json",
#         "filters": {
#             "quantizer": "ScalarQuatizationFaiss",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.nbit": 4,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#ff7f0e",
#         "marker": "p",
#     },
#     {
#         "label": "TurboQuant",
#         "file": "benchmark/results/distribution_shift/gist-960-euclidean_TurboQuant.json",
#         "filters": {
#             "quantizer": "TurboQuant",
#             "search_params.topk": 100,
#             "quantizer_config.build_params.bitwidth": 4,
#             "quantizer_config.build_params.space": "l2",
#         },
#         "color": "#00acc1",
#         "marker": "H",
#     },
#     # --- Alternative series kept here for quick toggling. Uncomment to add. ---
#     # {
#     #     "label": "LSQFaiss",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_LSQFaiss.json",
#     #     "filters": {
#     #         "quantizer": "LSQFaiss",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.nbit": 8,
#     #         "quantizer_config.build_params.nsubvec": 32,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#ff7f0e",
#     #     "marker": "s",
#     # },
#     # {
#     #     "label": "PLSQ",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_PLSQFaiss.json",
#     #     "filters": {
#     #         "quantizer": "PLSQFaiss",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.nbit": 8,
#     #         "quantizer_config.build_params.nsubvec": 8,
#     #         "quantizer_config.build_params.nsplits": 4,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#9467bd",
#     #     "marker": "P",
#     # },
#     # {
#     #     "label": "PQFastScanFaiss",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_ProductQuantizationFastScanFaiss.json",
#     #     "filters": {
#     #         "quantizer": "ProductQuantizationFastScanFaiss",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.nbit": 4,
#     #         "quantizer_config.build_params.nsubvec": 32,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#7f7f7f",
#     #     "marker": ">",
#     # },
#     # {
#     #     "label": "PRQ",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_PRQFaiss.json",
#     #     "filters": {
#     #         "quantizer": "PRQFaiss",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.nbit": 8,
#     #         "quantizer_config.build_params.nsubvec": 8,
#     #         "quantizer_config.build_params.nsplits": 4,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#8c564b",
#     #     "marker": "X",
#     # },
#     # {
#     #     "label": "RabitQ",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_RabitQ.json",
#     #     "filters": {
#     #         "quantizer": "RabitQ",
#     #         "search_params.topk": 100,
#     #         "search_params.nprobe": 10,
#     #         "quantizer_config.build_params.nlist": 256,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#bcbd22",
#     #     "marker": "h",
#     # },
#     # {
#     #     "label": "RQ",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_ResidualQuantizationFaiss.json",
#     #     "filters": {
#     #         "quantizer": "ResidualQuantizationFaiss",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.nbit": 8,
#     #         "quantizer_config.build_params.nsubvec": 32,
#     #         "quantizer_config.build_params.max_beam_size": 10,
#     #         "quantizer_config.build_params.space": "l2",
#     #     },
#     #     "color": "#aec7e8",
#     #     "marker": "*",
#     # },
#     # {
#     #     "label": "VAQ",
#     #     "file": "benchmark/results/distribution_shift/sift-128-euclidean_VAQ.json",
#     #     "filters": {
#     #         "quantizer": "VAQ",
#     #         "search_params.topk": 100,
#     #         "quantizer_config.build_params.bit_budget": 256,
#     #         "quantizer_config.build_params.max_bits": 6,
#     #         "quantizer_config.build_params.min_bits": 1,
#     #         "quantizer_config.build_params.search_method": "SORT",
#     #         "quantizer_config.build_params.subspace_num": 64,
#     #         "quantizer_config.build_params.var_explained": 1.0,
#     #     },
#     #     "color": "#98df8a",
#     #     "marker": "d",
#     # },
# ]
# Sift template (entries in alphabetical order). Swap in by uncommenting and
# pointing DEFAULT_SERIES_SPECS at this list.
DEFAULT_SERIES_SPECS: List[Dict[str, Any]] = [
    {
        "label": "OPQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_OptimizedProductQuantizationFaiss.json",
        "filters": {
            "quantizer": "OptimizedProductQuantizationFaiss",
            "search_params.topk": 100,
            "quantizer_config.build_params.nbit": 8,
            "quantizer_config.build_params.nsubvec": 64,
            "quantizer_config.build_params.niter": 400,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#9467bd",
        "marker": "^",
    },
    {
        "label": "OSQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_OptimizedScalarQuantization.json",
        "filters": {
            "quantizer": "OptimizedScalarQuantization",
            "search_params.topk": 100,
            "quantizer_config.build_params.nbit": 4,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#8c564b",
        "marker": "D",
    },
    {
        "label": "PQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_ProductQuantizationFaiss.json",
        "filters": {
            "quantizer": "ProductQuantizationFaiss",
            "search_params.topk": 100,
            "quantizer_config.build_params.nbit": 8,
            "quantizer_config.build_params.nsubvec": 64,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#1f77b4",
        "marker": "<",
    },
    {
        "label": "RaBitQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_ExtendedRabitQNTU_nlist1.json",
        "filters": {
            "quantizer": "ExtendedRabitQNTU",
            "search_params.topk": 100,
            "search_params.nprobe": 1,
            "quantizer_config.build_params.nlist": 1,
            "quantizer_config.build_params.nbit": 4,
            "quantizer_config.build_params.high_acc_flag": [False, True],
        },
        "color": "#d62728",
        "marker": "o",
    },
    {
        "label": "SAQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_SAQ.json",
        "filters": {
            "quantizer": "SAQ",
            "search_params.topk": 100,
            "quantizer_config.build_params.nbit": 4,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#f39c12",
        "marker": "8",
    },
    {
        "label": "SQ",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_ScalarQuatizationFaiss.json",
        "filters": {
            "quantizer": "ScalarQuatizationFaiss",
            "search_params.topk": 100,
            "quantizer_config.build_params.nbit": 4,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#ff7f0e",
        "marker": "p",
    },
    {
        "label": "TurboQuant",
        "file": "benchmark/results/distribution_shift/sift-128-euclidean_TurboQuant.json",
        "filters": {
            "quantizer": "TurboQuant",
            "search_params.topk": 100,
            "quantizer_config.build_params.bitwidth": 4,
            "quantizer_config.build_params.space": "l2",
        },
        "color": "#00acc1",
        "marker": "H",
    },
]



def _build_series_specs(raw_specs: Sequence[Mapping[str, Any]]) -> List[SeriesSpec]:
    specs: List[SeriesSpec] = []
    for entry in raw_specs:
        label = entry.get("label")
        file_name = entry.get("file")
        if not label or not file_name:
            raise ValueError(f"Invalid series config: {entry}")

        raw_filters = entry.get("filters", {})
        if not isinstance(raw_filters, Mapping):
            raise ValueError(f"'filters' must be a mapping in config: {entry}")

        specs.append(
            SeriesSpec(
                label=str(label),
                file=Path(str(file_name)),
                filters=dict(raw_filters),
                color=entry.get("color"),
                marker=str(entry.get("marker", "o")),
                linestyle=str(entry.get("linestyle", "-")),
            )
        )

    if not specs:
        raise ValueError("DEFAULT_SERIES_SPECS is empty. Please configure at least one series.")
    return specs


def _get_nested_value(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(dotted_path)
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(dotted_path) from exc
            current = current[index]
        else:
            raise KeyError(dotted_path)
    return current


def _matches_filters(result: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for dotted_path, expected in filters.items():
        try:
            actual = _get_nested_value(result, dotted_path)
        except (KeyError, IndexError):
            return False
        if actual != expected:
            return False
    return True


def _extract_js_divergence(group_entry: Mapping[str, Any]) -> float:
    metadata = group_entry.get("distribution_shift_metadata")
    if isinstance(metadata, Mapping) and "js_divergence" in metadata:
        return float(metadata["js_divergence"])

    for result in group_entry.get("results", []):
        shift_info = result.get("distribution_shift")
        if isinstance(shift_info, Mapping) and "js_divergence" in shift_info:
            return float(shift_info["js_divergence"])

    raise ValueError(
        f"Could not find js_divergence in distribution shift entry: "
        f"{group_entry.get('distribution_shift_group', '<unknown>')}"
    )


def _select_result(group_entry: Mapping[str, Any], spec: SeriesSpec) -> Mapping[str, Any]:
    matched_results = [
        result
        for result in group_entry.get("results", [])
        if isinstance(result, Mapping)
        and result.get("status") == "success"
        and _matches_filters(result, spec.filters)
    ]

    group_name = group_entry.get("distribution_shift_group", "<unknown>")
    if not matched_results:
        raise ValueError(
            f"No matching result found for series '{spec.label}' in group '{group_name}'. "
            f"Filters: {spec.filters}"
        )
    if len(matched_results) > 1:
        raise ValueError(
            f"Multiple matching results found for series '{spec.label}' in group '{group_name}'. "
            f"Please add more filters. Filters: {spec.filters}"
        )
    return matched_results[0]


def _load_points(spec: SeriesSpec) -> List[tuple[float, float]]:
    raw_data = json.loads(spec.file.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a JSON array in {spec.file}")

    points: List[tuple[float, float]] = []
    for group_entry in raw_data:
        if not isinstance(group_entry, Mapping):
            continue
        js_divergence = _extract_js_divergence(group_entry)
        result = _select_result(group_entry, spec)
        if "recall" not in result:
            raise ValueError(
                f"Result for series '{spec.label}' in file '{spec.file}' is missing 'recall'."
            )
        points.append((js_divergence, float(result["recall"])))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError(f"No plottable points found for series '{spec.label}' in {spec.file}.")
    return points


def _default_legend_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_legend.pdf")


def _save_legend_pdf(handles: Sequence[Any], labels: Sequence[str], legend_output_path: Path) -> None:
    if not handles or not labels:
        return

    legend_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=12
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(legend_output_path, bbox_inches=bbox, pad_inches=0)

    # num_items = len(labels)
    # legend_width = 7.0
    # legend_fig, legend_ax = plt.subplots(figsize=(legend_width, 1.0))
    # legend_ax.axis("off")
    # legend_fig.legend(
    #     handles,
    #     labels,
    #     loc="center",
    #     ncol=num_items,
    #     frameon=False,
    #     fontsize=9,
    #     handlelength=2.2,
    #     columnspacing=1.2,
    # )
    # legend_fig.savefig(legend_output_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig_leg)
    print(f"Saved legend to {legend_output_path}")


def _print_recall_ratio(label: str, recalls: Sequence[float]) -> None:
    max_recall = max(recalls)
    min_recall = min(recalls)
    ratio = float("inf") if min_recall == 0 else max_recall / min_recall
    print(
        f"{label}: max/min recall = {ratio:.6g} "
        f"(max={max_recall:.6g}, min={min_recall:.6g})"
    )


def _plot_series(
    series_specs: Iterable[SeriesSpec],
    output_path: Path | None,
    legend_output_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    handles: List[Any] = []
    labels: List[str] = []
    for spec in series_specs:
        points = _load_points(spec)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        _print_recall_ratio(spec.label, y_values)
        (line,) = ax.plot(
            x_values,
            y_values,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=1.8,
            markersize=6,
        )
        handles.append(line)
        labels.append(spec.label)

    ax.set_xlabel("JS divergence", fontsize=18)
    ax.set_ylabel("Recall", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    if legend_output_path is None and output_path is not None:
        legend_output_path = _default_legend_output_path(output_path)
    if legend_output_path is not None:
        _save_legend_pdf(handles, labels, legend_output_path)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", format="pdf")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot distribution-shift recall curves from benchmark JSON files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        help="Optional standalone legend PDF path. Defaults to '<output stem>_legend.pdf' when --output is set.",
    )
    args = parser.parse_args()

    series_specs = _build_series_specs(DEFAULT_SERIES_SPECS)
    _plot_series(series_specs, args.output, args.legend_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
