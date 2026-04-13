import os
import json
import glob
import math
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict


matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ── Configuration ──────────────────────────────────────────────
# Set to a dataset name to process only that dataset, or None for all datasets.
DATASET = "audio-128-euclidean"  # e.g. "audio-128-euclidean"

KNOWN_DATASETS = [
    "audio-128-euclidean",
    "gist-960-euclidean",
    "paper-200-euclidean",
    "sift-128-euclidean",
    "text2image-200-euclidean",
    "video-1024-euclidean",
]

# Compression rates to include, mapped to legend suffix and marker shape
RATE_CONFIG = {
    0.03125: {"suffix": "32x", "marker": "o"},
    0.0625:  {"suffix": "16x", "marker": "s"},
    0.125:   {"suffix": "8x",  "marker": "^"},
}

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_QUANTIZER_DIR = os.path.join(REPO_DIR, "benchmark", "results", "quantizer")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "standalone")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# Only color per algorithm; marker is determined by compression rate
STYLES = {
    "ProductQuantizationFaiss":          {"color": "#1f77b4"},
    "OptimizedProductQuantizationFaiss": {"color": "#9467bd"},
    "ProductQuantizationFastScanFaiss":  {"color": "#7f7f7f"},
    "ScalarQuatizationFaiss":            {"color": "#ff7f0e"},
    "OptimizedScalarQuantization":       {"color": "#8c564b"},
    "RabitQLibrary":                     {"color": "#d62728"},
    "SAQ_nlist1":                        {"color": "#f39c12"},
    "TurboQuant_nlist1":                 {"color": "#00acc1"},
}

# ── Load data ──────────────────────────────────────────────────
# data[dataset][topk][algo][compression_rate] = [(recall, qps), ...]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

for filepath in sorted(glob.glob(os.path.join(RESULTS_QUANTIZER_DIR, "*.json"))):
    filename = os.path.basename(filepath)
    name = filename.replace(".json", "")

    ds_match = next((ds for ds in KNOWN_DATASETS if name.startswith(ds + "_")), None)
    if ds_match is None:
        print(f"[SKIP] Cannot parse filename: {filename}")
        continue

    if DATASET is not None and ds_match != DATASET:
        continue

    algo = name[len(ds_match) + 1:]

    with open(filepath) as f:
        entries = json.load(f)

    for entry in entries:
        rate = entry.get("build_metrics", {}).get("compression_rate")
        if rate not in RATE_CONFIG:
            continue
        for sr in entry.get("search_results", []):
            topk = sr.get("search_params", {}).get("topk")
            metrics = sr.get("metrics", {})
            recall = metrics.get("recall")
            qps = metrics.get("queries_per_second")
            if topk is not None and recall is not None and qps is not None:
                data[ds_match][topk][algo][rate].append((recall, qps))


def save_figure(fig, stem):
    fig.savefig(stem + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", format="png", dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf / .png")


def save_legend(algo_handles, rate_handles, stem):
    all_handles = list(algo_handles.values()) + list(rate_handles.values())
    if not all_handles:
        return
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        all_handles,
        [h.get_label() for h in all_handles],
        loc="center",
        ncol=max(len(algo_handles), len(rate_handles)),
        frameon=False,
        fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig_leg.dpi_scale_trans.inverted()
    )
    fig_leg.savefig(stem + ".pdf", format="pdf", bbox_inches=bbox, pad_inches=0.05)
    fig_leg.savefig(stem + ".png", format="png", dpi=150, bbox_inches=bbox, pad_inches=0.05)
    print(f"[OK] Saved: {stem}.pdf / .png")
    plt.close(fig_leg)


# ── Plot one figure per (dataset, topk); one shared legend per dataset ────
for dataset, topk_data in sorted(data.items()):
    # Collect legend handles across all topk so the legend is dataset-wide
    algo_handles = {}
    rate_handles = {}

    for topk, algo_data in sorted(topk_data.items()):
        fig, ax = plt.subplots(figsize=(9, 6))

        # Only consider algorithms that will actually be plotted
        plotted_qps = [
            qps
            for algo, rate_map in algo_data.items()
            if algo in STYLES
            for rate, points in rate_map.items()
            if rate in RATE_CONFIG
            for _, qps in points
        ]
        use_log = plotted_qps and (max(plotted_qps) / max(min(plotted_qps), 1e-9)) > 10

        for algo, rate_map in sorted(algo_data.items()):
            if algo not in STYLES:
                continue
            color = STYLES[algo]["color"]
            for rate, rc in RATE_CONFIG.items():
                if rate not in rate_map:
                    continue
                points = rate_map[rate]
                recalls  = [p[0] for p in points]
                qps_vals = [p[1] for p in points]
                ax.scatter(
                    recalls, qps_vals,
                    marker=rc["marker"],
                    color=color,
                    s=30,
                    alpha=0.7,
                )
                if algo not in algo_handles:
                    algo_handles[algo] = matplotlib.lines.Line2D(
                        [], [], linestyle="none",
                        marker="o", color=color,
                        markersize=7, label=algo,
                    )
                if rate not in rate_handles:
                    rate_handles[rate] = matplotlib.lines.Line2D(
                        [], [], linestyle="none",
                        marker=rc["marker"], color="black",
                        markersize=7, label=rc["suffix"],
                    )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

        ax.set_xlabel(f"Recall@{topk}")
        ax.set_ylabel("Queries per Second")
        ax.set_title(f"{dataset}  —  top{topk}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        fig.tight_layout()
        stem = os.path.join(FIGURES_DIR, f"{dataset}_top{topk}_by_compression_rate")
        save_figure(fig, stem)
        plt.close(fig)

    # Save one shared legend for all topk figures of this dataset
    leg_stem = os.path.join(LEGENDS_DIR, f"{dataset}_by_compression_rate_legend")
    save_legend(algo_handles, rate_handles, leg_stem)
