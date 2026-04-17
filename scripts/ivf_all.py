import os
import json
import glob
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
DATASET = None  # set to a dataset name to restrict, or None for all
NLIST = None    # set to a specific nlist (e.g. 1024), or None for all

KNOWN_DATASETS = [
    "audio-128-euclidean",
    "gist-960-euclidean",
    "paper-200-euclidean",
    "sift-128-euclidean",
    "text2image-200-euclidean",
    "video-1024-euclidean",
]

# IVF experiments only use 32x and 8x (no 16x).
RATE_CONFIG = {
    0.03125: {"suffix": "32x", "marker": "o", "linestyle": "--"},
    0.125:   {"suffix": "8x",  "marker": "^", "linestyle": "-"},
}

SCATTER_MARKER_SIZE = 80

# Per-dataset axis overrides. Shared across all nlist values for a dataset.
# If "yscale" is omitted, it defaults to "log" (or "linear" when ylim starts at 0).
DATASET_AXES = {
    "sift-128-euclidean": {"xlim": (0.4, 1.0), "ylim": (1000, 200000), "yscale": "log"},
    "text2image-200-euclidean": {"xlim": (0.25, 0.9), "ylim": (300, 150000), "yscale": "log"},
    "video-1024-euclidean": {"xlim": (0.58, 1.0), "ylim": (10, 15000), "yscale": "log"},
}

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_IVF_DIR = os.path.join(REPO_DIR, "benchmark", "results", "ivf")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "ivf")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Styles ─────────────────────────────────────────────────────
STYLES = {
    "Faiss-IVFPQ":      {"alias": "IVFPQ",        "color": "#1f77b4"},
    "Faiss-OPQ-IVFPQ":  {"alias": "IVFOPQ",       "color": "#9467bd"},
    "Faiss-IVFSQ":      {"alias": "IVFSQ",        "color": "#ff7f0e"},
    "IVFOSQ":           {"alias": "IVFOSQ",       "color": "#8c564b"},
    "IVFRabitQLibrary": {"alias": "IVFRabitQ",    "color": "#d62728"},
    "IVFSAQ":           {"alias": "IVFSAQ",       "color": "#f39c12"},
    "IVFTurboQuant":    {"alias": "IVFTurboQuant", "color": "#00acc1"},
}


def normalize_rate(rate):
    """Results store the rate as ratio (0.03125) or multiplier (32.0); normalize
    to the ratio form used as the key in RATE_CONFIG."""
    if rate is None or rate == 0:
        return None
    if rate > 1:
        return 1.0 / rate
    return rate


def match_rate(rate):
    norm = normalize_rate(rate)
    if norm is None:
        return None
    for key in RATE_CONFIG:
        if abs(norm - key) < 1e-9:
            return key
    return None


# ── Load data ──────────────────────────────────────────────────
# data[dataset][nlist][algo][rate] = [(recall, qps), ...]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(
       lambda: defaultdict(list))))

for filepath in sorted(glob.glob(os.path.join(RESULTS_IVF_DIR, "*.json"))):
    filename = os.path.basename(filepath)
    if ".old" in filename:
        continue
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

    for entry in entries:
        rate_key = match_rate(entry.get("build_metrics", {}).get("compression_rate"))
        if rate_key is None:
            continue
        nlist = entry.get("build_params", {}).get("nlist")
        if nlist is None:
            continue
        if NLIST is not None and nlist != NLIST:
            continue
        for sr in entry.get("search_results", []):
            if sr.get("search_params", {}).get("topk") != 100:
                continue
            metrics = sr.get("metrics", {})
            recall = metrics.get("recall")
            qps = metrics.get("queries_per_second")
            if recall is not None and qps is not None:
                data[ds_match][nlist][algo][rate_key].append((recall, qps))


# ── Helpers ────────────────────────────────────────────────────
def save_figure(fig, stem):
    fig.savefig(stem + ".pdf", format="pdf", bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf")


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
    print(f"[OK] Saved: {stem}.pdf")
    plt.close(fig_leg)


def save_legend(algo_handles, rate_handles, algo_rates, stem):
    color_handles = []
    color_labels = []
    for leg_key, handle in algo_handles.items():
        color = handle.get_color()
        rates_present = algo_rates.get(leg_key, set())
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
        print(f"[OK] Saved: {stem}_color.pdf")
        plt.close(fig_leg)

    _save_handles(
        list(rate_handles.values()),
        stem + "_marker",
        ncol=len(rate_handles),
    )


def pareto_curve(points):
    """Upper-envelope curve (max QPS at each recall) for cleaner lines."""
    if not points:
        return [], []
    best = {}
    for r, q in points:
        if r not in best or q > best[r]:
            best[r] = q
    frontier = sorted(best.items())
    return [p[0] for p in frontier], [p[1] for p in frontier]


# ── Plot ───────────────────────────────────────────────────────
algo_handles = {}
rate_handles = {}
algo_rates = {}

for dataset, nlist_data in sorted(data.items()):
    for nlist, algo_data in sorted(nlist_data.items()):
        fig, ax = plt.subplots(figsize=(5, 4))

        plotted_qps = [
            qps
            for rate_map in algo_data.values()
            for rate, points in rate_map.items()
            if rate in RATE_CONFIG
            for _, qps in points
        ]
        axes_cfg = DATASET_AXES.get(dataset, {})
        if "yscale" in axes_cfg:
            use_log = axes_cfg["yscale"] == "log"
        elif "ylim" in axes_cfg and axes_cfg["ylim"][0] <= 0:
            use_log = False
        else:
            use_log = bool(plotted_qps) and (
                max(plotted_qps) / max(min(plotted_qps), 1e-9)) > 10

        for algo, rate_map in sorted(algo_data.items()):
            style = STYLES[algo]
            alias = style.get("alias", algo)
            color = style["color"]

            for rate, rc in RATE_CONFIG.items():
                if rate not in rate_map:
                    continue
                points = rate_map[rate]
                recalls, qps_vals = pareto_curve(points)

                ax.plot(recalls, qps_vals, color=color,
                        linestyle=rc["linestyle"], linewidth=1.5, alpha=0.8)
                ax.scatter(recalls, qps_vals,
                           marker=rc["marker"], color=color,
                           s=SCATTER_MARKER_SIZE, alpha=0.7)

                leg_key = (algo,)
                if leg_key not in algo_handles:
                    algo_handles[leg_key] = matplotlib.lines.Line2D(
                        [], [], linestyle="none", marker="o",
                        color=color, markersize=7, label=alias,
                    )
                algo_rates.setdefault(leg_key, set()).add(rate)
                if rate not in rate_handles:
                    rate_handles[rate] = matplotlib.lines.Line2D(
                        [], [], linestyle=rc["linestyle"], marker=rc["marker"],
                        color="black", markersize=7, linewidth=1.5,
                        label=rc["suffix"],
                    )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.LogFormatterSciNotation(base=10))

        if "xlim" in axes_cfg:
            ax.set_xlim(*axes_cfg["xlim"])
        if "ylim" in axes_cfg:
            ax.set_ylim(*axes_cfg["ylim"])

        ax.set_xlabel("Recall@100")
        ax.set_ylabel("Queries per Second")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        fig.tight_layout()
        stem = os.path.join(FIGURES_DIR,
                            f"{dataset}_nlist{nlist}_by_compression_rate")
        save_figure(fig, stem)
        plt.close(fig)

save_legend(algo_handles, rate_handles, algo_rates,
            os.path.join(LEGENDS_DIR, "ivf_legend"))
