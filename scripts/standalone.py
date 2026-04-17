import os
import json
import glob
import math
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from matplotlib.legend_handler import HandlerTuple


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

# Compression rates to include: marker shape and legend suffix
RATE_CONFIG = {
    0.03125: {"suffix": "32x", "marker": "o"},
    0.0625:  {"suffix": "16x", "marker": "s"},
    0.125:   {"suffix": "8x",  "marker": "^"},
}

SCATTER_MARKER_SIZE = 80   # marker size (s=) for scatter points in the plot

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
RESULTS_QUANTIZER_DIR = os.path.join(REPO_DIR, "benchmark", "results", "quantizer")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "standalone")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Styles ─────────────────────────────────────────────────────
# Marker is determined by compression rate.
# Algorithms without param_key use a single "color".
# Algorithms with param_key use "param_colors": {value: color} to split groups;
# the legend entry will be "alias (param_key=value)".
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
    "ScalarQuatizationFaiss":           {"alias": "SQ",     "color": "#08087b"},
    "OptimizedScalarQuantization":      {"alias": "OSQ",    "color": "#8c564b"},
    "RabitQLibrary":                    {"alias": "RabitQ", "color": "#d62728"},
    "SAQ":                       {"alias": "SAQ",    "color": "#066909"},
    "TurboQuant":                       {"alias": "Turbo",  "color": "#00acc1"},
}

# ── Load data ──────────────────────────────────────────────────
# data[dataset][topk][algo][rate][group] = [(recall, qps), ...]
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
            recall  = metrics.get("recall")
            qps     = metrics.get("queries_per_second")
            if topk is not None and recall is not None and qps is not None:
                data[ds_match][topk][algo][rate][group].append((recall, qps))


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


def save_legend(algo_handles, rate_handles, algo_rates, stem):
    # Color legend: each entry shows only the marker shapes that exist for that group
    color_handles = []
    color_labels  = []
    for leg_key, handle in algo_handles.items():
        color      = handle.get_color()
        rates_present = algo_rates.get(leg_key, set())
        # Keep only markers for rates that actually have data, in RATE_CONFIG order
        sub = tuple(
            matplotlib.lines.Line2D([], [], linestyle="none",
                                    marker=rc["marker"], color=color, markersize=7)
            for rate, rc in RATE_CONFIG.items()
            if rate in rates_present
        )
        color_handles.append(sub)
        color_labels.append(handle.get_label())

    if color_handles:
        fig_leg = plt.figure()
        legend = fig_leg.legend(
            color_handles, color_labels,
            loc="center",
            ncol=max(len(color_handles), 1),
            frameon=False,
            fontsize=12,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        )
        fig_leg.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig_leg.dpi_scale_trans.inverted()
        )
        fig_leg.savefig(stem + "_color.pdf", format="pdf", bbox_inches=bbox, pad_inches=0.05)
        fig_leg.savefig(stem + "_color.png", format="png", dpi=150, bbox_inches=bbox, pad_inches=0.05)
        print(f"[OK] Saved: {stem}_color.pdf / .png")
        plt.close(fig_leg)

    # Marker legend: one entry per compression rate
    _save_handles(
        list(rate_handles.values()),
        stem + "_marker",
        ncol=len(rate_handles),
    )


# ── Plot ───────────────────────────────────────────────────────
# algo_handles / rate_handles are shared across all datasets for a unified legend
algo_handles = {}
rate_handles = {}
algo_rates   = {}   # leg_key -> set of rates that actually have data

for dataset, topk_data in sorted(data.items()):
    for topk, algo_data in sorted(topk_data.items()):
        fig, ax = plt.subplots(figsize=(4.4, 4))

        # Collect QPS of all points that will be drawn
        plotted_qps = [
            qps
            for _, rate_map in algo_data.items()
            for rate, group_map in rate_map.items()
            if rate in RATE_CONFIG
            for points in group_map.values()
            for _, qps in points
        ]
        use_log = plotted_qps and (max(plotted_qps) / max(min(plotted_qps), 1e-9)) > 10

        for algo, rate_map in sorted(algo_data.items()):
            style     = STYLES[algo]
            param_key = style.get("param_key")
            alias     = style.get("alias", algo)

            for rate, rc in RATE_CONFIG.items():
                if rate not in rate_map:
                    continue
                for group, points in sorted(rate_map[rate].items(),
                                            key=lambda kv: (kv[0] is None, kv[0])):
                    # Color
                    if param_key and group is not None:
                        color = style["param_colors"].get(group, "#999999")
                    else:
                        color = style["color"]

                    recalls  = [p[0] for p in points]
                    qps_vals = [p[1] for p in points]
                    ax.scatter(recalls, qps_vals,
                               marker=rc["marker"], color=color,
                               s=SCATTER_MARKER_SIZE, alpha=0.7)

                    # Legend entries (accumulated once across all datasets)
                    if param_key and group is not None:
                        leg_key   = (algo, group)
                        leg_label = f"{alias} ({param_key}={group})"
                    else:
                        leg_key   = (algo, None)
                        leg_label = alias

                    if leg_key not in algo_handles:
                        algo_handles[leg_key] = matplotlib.lines.Line2D(
                            [], [], linestyle="none", marker="o",
                            color=color, markersize=7, label=leg_label,
                        )
                    algo_rates.setdefault(leg_key, set()).add(rate)
                    if rate not in rate_handles:
                        rate_handles[rate] = matplotlib.lines.Line2D(
                            [], [], linestyle="none", marker=rc["marker"],
                            color="black", markersize=7, label=rc["suffix"],
                        )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

        ax.set_xlabel(f"Recall@{topk}")
        ax.set_ylabel("Queries per Second")
        # ax.set_title(f"{dataset}  —  top{topk}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        fig.tight_layout()
        stem = os.path.join(FIGURES_DIR, f"{dataset}_top{topk}")
        save_figure(fig, stem)
        plt.close(fig)

# One shared legend for all datasets and topk values
save_legend(algo_handles, rate_handles, algo_rates,
            os.path.join(LEGENDS_DIR, "standalone_legend"))
