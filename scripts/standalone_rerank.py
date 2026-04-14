import os
import json
import glob
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
DATASET = None

KNOWN_DATASETS = [
    "audio-128-euclidean",
    "gist-960-euclidean",
    "paper-200-euclidean",
    "sift-128-euclidean",
    "text2image-200-euclidean",
    "video-1024-euclidean",
]

# Compression rates to include: line style and legend suffix
# Different rates are distinguished by line style, not marker.
RATE_CONFIG = {
    0.03125: {"suffix": "32x", "linestyle": ":"},
    0.0625:  {"suffix": "16x", "linestyle": "--"},
    0.125:   {"suffix": "8x",  "linestyle": "-"},
}

LINE_WIDTH = 1.8

# Only plot rerank points with rerank_recall >= RECALL_MIN.
# Set to None to disable filtering.
RECALL_MIN = 0
# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
RESULTS_QUANTIZER_DIR = os.path.join(REPO_DIR, "benchmark", "results", "quantizer")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "standalone_rerank")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Styles ─────────────────────────────────────────────────────
# Color encodes algorithm identity (same palette as standalone.py).
# Algorithms without param_key use a single "color".
# Algorithms with param_key use "param_colors": {value: color} to split groups.
STYLES = {
    "ProductQuantizationFaiss": {
        "alias": "PQ",
        "param_key": "nbit",
        "param_colors": {4: "#1f77b4", 8: "#aec7e8"},
    },
    "OptimizedProductQuantizationFaiss": {
        "alias": "OPQ",
        "param_key": "nbit",
        "param_colors": {4: "#9467bd", 8: "#c5b0d5"},
    },
    "ProductQuantizationFastScanFaiss": {"alias": "PQFast", "color": "#7f7f7f"},
    "ScalarQuatizationFaiss":           {"alias": "SQ",     "color": "#ff7f0e"},
    "OptimizedScalarQuantization":      {"alias": "OSQ",    "color": "#8c564b"},
    "RabitQLibrary":                    {"alias": "RabitQ", "color": "#d62728"},
    "SAQ_nlist1":                       {"alias": "SAQ",    "color": "#f39c12"},
    "TurboQuant":                       {"alias": "Turbo",  "color": "#00acc1"},
}

# ── Load data ──────────────────────────────────────────────────
# data[dataset][topk][algo][rate][group] = [(nrerank, rerank_recall, rerank_qps), ...]
# group: value of param_key (e.g. nbit=4), or None for algos without param_key
data = defaultdict(lambda: defaultdict(lambda: defaultdict(
       lambda: defaultdict(lambda: defaultdict(list)))))

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
    if algo not in STYLES:
        continue

    with open(filepath) as f:
        entries = json.load(f)

    style     = STYLES[algo]
    param_key = style.get("param_key")

    for entry in entries:
        rate = entry.get("build_metrics", {}).get("compression_rate")
        if rate not in RATE_CONFIG:
            continue
        group = entry.get("build_params", {}).get(param_key) if param_key else None

        for sr in entry.get("search_results", []):
            topk    = sr.get("search_params", {}).get("topk")
            metrics = sr.get("metrics", {})
            rerank_results = metrics.get("rerank_results", [])
            if topk is None or not rerank_results:
                continue

            for rr in rerank_results:
                nrerank    = rr.get("nrerank")
                rr_recall  = rr.get("rerank_recall")
                rr_qps     = rr.get("rerank_queries_per_second")
                if nrerank is not None and rr_recall is not None and rr_qps is not None:
                    data[ds_match][topk][algo][rate][group].append(
                        (nrerank, rr_recall, rr_qps)
                    )


# ── Helpers ────────────────────────────────────────────────────
def save_figure(fig, stem):
    fig.savefig(stem + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", format="png", dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf / .png")


def _save_handles(handles, stem, ncol):
    if not handles:
        return
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center",
        ncol=max(ncol, 1),
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


def save_legend(algo_handles, rate_handles, stem):
    # Color legend: one solid line per algorithm/group, uniform style (no per-rate variation).
    _save_handles(
        list(algo_handles.values()),
        stem + "_color",
        ncol=max(len(algo_handles), 1),
    )

    # Line-style legend: one entry per compression rate.
    _save_handles(
        list(rate_handles.values()),
        stem + "_linestyle",
        ncol=len(rate_handles),
    )


# ── Plot ───────────────────────────────────────────────────────
# algo_handles / rate_handles are shared across all datasets for a unified legend.
algo_handles = {}   # leg_key -> Line2D (color legend)
rate_handles = {}   # rate   -> Line2D (line-style legend)

for dataset, topk_data in sorted(data.items()):
    for topk, algo_data in sorted(topk_data.items()):
        for rate, rc in RATE_CONFIG.items():
            fig, ax = plt.subplots(figsize=(9, 6))

            # Determine y-axis scale for this rate only
            plotted_qps = [
                rr_qps
                for _, rate_map in algo_data.items()
                if rate in rate_map
                for points in rate_map[rate].values()
                for _, rr_recall, rr_qps in points
                if RECALL_MIN is None or rr_recall >= RECALL_MIN
            ]
            use_log = plotted_qps and (max(plotted_qps) / max(min(plotted_qps), 1e-9)) > 10

            for algo, rate_map in sorted(algo_data.items()):
                if rate not in rate_map:
                    continue
                style     = STYLES[algo]
                param_key = style.get("param_key")
                alias     = style.get("alias", algo)

                for group, points in sorted(rate_map[rate].items(),
                                            key=lambda kv: (kv[0] is None, kv[0])):
                    # Resolve color
                    if param_key and group is not None:
                        color = style["param_colors"].get(group, "#999999")
                    else:
                        color = style["color"]

                    # Sort by nrerank so the line is drawn in order
                    pts_sorted = sorted(points, key=lambda p: p[0])
                    if RECALL_MIN is not None:
                        pts_sorted = [p for p in pts_sorted if p[1] >= RECALL_MIN]
                    if not pts_sorted:
                        continue
                    recalls  = [p[1] for p in pts_sorted]
                    qps_vals = [p[2] for p in pts_sorted]

                    ax.plot(recalls, qps_vals,
                            color=color,
                            linestyle=rc["linestyle"],
                            linewidth=LINE_WIDTH,
                            marker="o", markersize=4,
                            alpha=0.85)

                    # Accumulate legend entries (once across all datasets)
                    if param_key and group is not None:
                        leg_key   = (algo, group)
                        leg_label = f"{alias} ({param_key}={group})"
                    else:
                        leg_key   = (algo, None)
                        leg_label = alias

                    if leg_key not in algo_handles:
                        # Solid line with algo color; uniform marker for all entries
                        algo_handles[leg_key] = matplotlib.lines.Line2D(
                            [], [], linestyle="-", color=color,
                            linewidth=LINE_WIDTH,
                            marker="o", markersize=4,
                            label=leg_label,
                        )

                    if rate not in rate_handles:
                        rate_handles[rate] = matplotlib.lines.Line2D(
                            [], [], linestyle=rc["linestyle"], color="black",
                            linewidth=LINE_WIDTH, label=rc["suffix"],
                        )

            if use_log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)

            ax.set_xlabel(f"Recall@{topk}")
            ax.set_ylabel("Queries per Second")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            fig.tight_layout()
            stem = os.path.join(FIGURES_DIR,
                                f"{dataset}_top{topk}_{rc['suffix']}_rerank")
            save_figure(fig, stem)
            plt.close(fig)

# One shared legend for all datasets and topk values
save_legend(algo_handles, rate_handles,
            os.path.join(LEGENDS_DIR, "standalone_rerank_legend"))
