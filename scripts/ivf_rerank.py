import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from matplotlib.legend_handler import HandlerTuple


matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# ── Configuration ──────────────────────────────────────────────
DATASET = None   # set to a dataset name to restrict, or None for all
NLIST = None     # set to a specific nlist (e.g. 1024), or None for all
NRERANK = None   # set to a specific nrerank value, or None for all

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

RERANK_ONLY_TIME = {
    "sift-128-euclidean": {
        200: 0.584007,
        300: 0.783458,
        500: 1.305705,
        1000: 2.603167
    },
    "text2image-200-euclidean": {
        200: 2.703108,
        300: 3.890416,
        500: 6.473574,
        1000: 12.949994
    },
    "video-1024-euclidean": {
        200: 3.369417,
        300: 3.990064,
        500: 6.643540,
        1000: 13.513948
    }
}

SCATTER_MARKER_SIZE = 80

# Per-dataset axis overrides. Shared across all nlist/nrerank values for a dataset.
DATASET_AXES = {
    "sift-128-euclidean": {"xlim": (0.5, 1.0), "ylim": (1000, 20000), "yscale": "log"},
    "text2image-200-euclidean": {"xlim": (0.3, 1.0), "ylim": (100, 20000), "yscale": "log"},
    "video-1024-euclidean": {"xlim": (0.75, 1.0), "ylim": (10, 3000), "yscale": "log"},
}

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_IVF_DIR = os.path.join(REPO_DIR, "benchmark", "results", "ivf")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "ivf_rerank")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Styles ─────────────────────────────────────────────────────
STYLES = {
    "Faiss-IVFPQ":         {"alias": "IVFPQ",        "color": "#1f77b4"},
    "Faiss-OPQ-IVFPQ":     {"alias": "IVFOPQ",       "color": "#9467bd"},
    "Faiss-IVFPQFastScan": {"alias": "IVFPQFastScan", "color": "#2ca02c"},
    "Faiss-IVFSQ":         {"alias": "IVFSQ",        "color": "#ff7f0e"},
    "IVFOSQ":              {"alias": "IVFOSQ",       "color": "#8c564b"},
    "IVFRabitQLibrary":    {"alias": "IVFRaBitQ",    "color": "#d62728"},
    "IVFSAQ":              {"alias": "IVFSAQ",       "color": "#f39c12"},
    "IVFTurboQuant":       {"alias": "IVFTurboQuant", "color": "#00acc1"},
}


def normalize_rate(rate):
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
# data[dataset][nlist][nrerank][algo][rate] = [(recall, adj_qps, nprobe), ...]
# QPS is recomputed as nqueries / (search_time + RERANK_ONLY_TIME[dataset][nrerank]).
data = defaultdict(lambda: defaultdict(lambda: defaultdict(
       lambda: defaultdict(lambda: defaultdict(list)))))

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
    if ds_match not in RERANK_ONLY_TIME:
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
            sp = sr.get("search_params", {})
            if sp.get("topk") != 100:
                continue
            nprobe = sp.get("nprobe")
            if nprobe is None:
                continue
            metrics = sr.get("metrics", {})
            for rr in metrics.get("rerank_results") or []:
                nrerank = rr.get("nrerank")
                if nrerank is None:
                    continue
                if NRERANK is not None and nrerank != NRERANK:
                    continue
                fixed_ro = RERANK_ONLY_TIME.get(ds_match, {}).get(nrerank)
                if fixed_ro is None:
                    continue
                recall = rr.get("rerank_recall")
                search_time = rr.get("search_time")
                rerank_time = rr.get("rerank_time")
                rerank_qps = rr.get("rerank_queries_per_second")
                if None in (recall, search_time, rerank_time, rerank_qps):
                    continue
                nqueries = rerank_time * rerank_qps
                total_time = search_time + fixed_ro
                adj_qps = nqueries / total_time
                data[ds_match][nlist][nrerank][algo][rate_key].append(
                    (recall, adj_qps, nprobe))


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


def curve_by_nprobe(points):
    """Order points by nprobe so a single curve connects nprobe=10→20→...→100."""
    if not points:
        return [], []
    ordered = sorted(points, key=lambda p: p[2])
    return [p[0] for p in ordered], [p[1] for p in ordered]


# ── Plot ───────────────────────────────────────────────────────
algo_handles = {}
rate_handles = {}
algo_rates = {}

for dataset, nlist_data in sorted(data.items()):
    for nlist, nrerank_data in sorted(nlist_data.items()):
        for nrerank, algo_data in sorted(nrerank_data.items()):
            fig, ax = plt.subplots(figsize=(5, 4))

            plotted_qps = [
                qps
                for rate_map in algo_data.values()
                for rate, points in rate_map.items()
                if rate in RATE_CONFIG
                for _, qps, _ in points
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
                    recalls, qps_vals = curve_by_nprobe(rate_map[rate])

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

            ax.set_xlabel("Recall@100 (rerank)")
            ax.set_ylabel("QPS (rerank)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            fig.tight_layout()
            stem = os.path.join(
                FIGURES_DIR,
                f"{dataset}_nlist{nlist}_nrerank{nrerank}_by_compression_rate",
            )
            save_figure(fig, stem)
            plt.close(fig)

save_legend(algo_handles, rate_handles, algo_rates,
            os.path.join(LEGENDS_DIR, "ivf_rerank_legend"))
