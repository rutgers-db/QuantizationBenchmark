import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from matplotlib.legend_handler import HandlerTuple


matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 14,
})

# ── Configuration ──────────────────────────────────────────────
DATASET = None

KNOWN_DATASETS = [
    "audio-128-euclidean",
    "gist-960-euclidean",
    "paper-200-euclidean",
    "sift-128-euclidean",
    "text2image-200-euclidean",
    "video-1024-euclidean",
]

# Compression rates: line style + marker both encode rate.
RATE_CONFIG = {
    0.03125: {"suffix": "32x", "linestyle": ":",  "marker": "^"},
    0.0625:  {"suffix": "16x", "linestyle": "--", "marker": "s"},
    0.125:   {"suffix": "8x",  "linestyle": "-",  "marker": "o"},
}

LINE_WIDTH = 1.8
MARKER_SIZE = 5

# Only plot rerank points with rerank_recall >= RECALL_MIN.
RECALL_MIN = 0

# ── Custom x-axis transform ─────────────────────────────────────
# Piecewise-linear mapping: (start, end, step, weight)
# weight = display units per tick interval.
# More precise (higher recall) segments get larger weight → wider visual spacing.
X_BREAKS = [
    (0.0,  0.90, 0.10,  1.0),
    (0.90, 0.98, 0.02,  1.25),
    (0.98, 1.0,  0.005, 1.5),
]

def _r2d(r):
    """Recall → display coordinate (piecewise linear, weighted)."""
    disp = 0.0
    for start, end, step, weight in X_BREAKS:
        if r <= end + 1e-12:
            disp += (r - start) / step * weight
            return disp
        disp += (end - start) / step * weight
    return disp

_r2d_vec = np.vectorize(_r2d)

def _build_ticks():
    ticks_r = []
    for start, end, step, _ in X_BREAKS:
        t = start
        while t < end - step * 0.01:
            val = round(t, 6)
            if not ticks_r or abs(val - ticks_r[-1]) > 1e-9:
                ticks_r.append(val)
            t = round(t + step, 10)
        val = round(end, 6)
        if not ticks_r or abs(val - ticks_r[-1]) > 1e-9:
            ticks_r.append(val)
    return ticks_r

def _fmt_r(r):
    if r < 0.90 - 1e-9:
        return f"{r:.1f}"
    elif r < 0.98 - 1e-9:
        return f"{r:.2f}"
    else:
        return f"{r:.3f}"

TICK_RECALLS = _build_ticks()
TICK_DISPS   = [_r2d(r) for r in TICK_RECALLS]
TICK_LABELS  = [_fmt_r(r) for r in TICK_RECALLS]

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
RESULTS_QUANTIZER_DIR = os.path.join(REPO_DIR, "benchmark", "results", "quantizer")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "standalone_rerank")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Styles ─────────────────────────────────────────────────────
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
    "SAQ":                              {"alias": "SAQ",    "color": "#066909"},
    "TurboQuant":                       {"alias": "Turbo",  "color": "#00acc1"},
}

# ── Load data ──────────────────────────────────────────────────
# data[dataset][topk][algo][rate][group] = [(nrerank, rerank_recall, rerank_qps), ...]
# baseline[(dataset, topk, algo, rate, group)] = (baseline_qps, baseline_recall)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(
       lambda: defaultdict(lambda: defaultdict(list)))))
baseline = {}

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
            base_qps       = metrics.get("queries_per_second")
            base_recall    = metrics.get("recall")
            rerank_results = metrics.get("rerank_results", [])
            if topk is None or base_qps is None or not rerank_results:
                continue

            bkey = (ds_match, topk, algo, rate, group)
            if bkey not in baseline:
                baseline[bkey] = (base_qps, base_recall)

            for rr in rerank_results:
                nrerank   = rr.get("nrerank")
                rr_recall = rr.get("rerank_recall")
                rr_qps    = rr.get("rerank_queries_per_second")
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


def save_legend(algo_handles, rate_handles, algo_rates, stem):
    # Color legend: each algo entry shows the markers for its rates (HandlerTuple).
    color_handles = []
    color_labels  = []
    for leg_key, handle in algo_handles.items():
        color         = handle.get_color()
        rates_present = algo_rates.get(leg_key, set())
        sub = tuple(
            matplotlib.lines.Line2D(
                [], [], linestyle="none",
                marker=RATE_CONFIG[rate]["marker"],
                color=color, markersize=7,
            )
            for rate in RATE_CONFIG
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

    # Rate legend: compression rate entries + one "w/o rerank" baseline entry (gold patch).
    baseline_handle = matplotlib.patches.Patch(
        facecolor=(0.85, 0.60, 0.0), label="w/o rerank",
    )
    rate_handles_list = list(rate_handles.values()) + [baseline_handle]
    _save_handles(
        rate_handles_list,
        stem + "_rate",
        ncol=len(rate_handles_list),
    )


def _apply_x_axis(ax, disp_min, disp_max):
    """Set custom ticks and gridlines on the transformed x-axis."""
    margin = 0.3
    ax.set_xlim(disp_min - margin, disp_max + margin)

    # Keep only ticks that fall within the visible range
    visible = [
        (d, lbl) for d, lbl in zip(TICK_DISPS, TICK_LABELS)
        if disp_min - margin - 0.5 <= d <= disp_max + margin + 0.5
    ]
    if visible:
        ds, lbls = zip(*visible)
        ax.set_xticks(list(ds))
        ax.set_xticklabels(list(lbls), rotation=45, ha="right")

    # Vertical gridlines at every tick
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Emphasise scale-break boundaries (drawn on top of grid)
    for break_recall in (0.90, 0.98):
        break_disp = _r2d(break_recall)
        if disp_min - margin <= break_disp <= disp_max + margin:
            ax.axvline(break_disp, color="black", linewidth=1.2,
                       linestyle="-", alpha=0.4, zorder=3)


# ── Plot ───────────────────────────────────────────────────────
algo_handles = {}
rate_handles = {}
algo_rates   = {}

for dataset, topk_data in sorted(data.items()):
    for topk, algo_data in sorted(topk_data.items()):
        fig, ax = plt.subplots(figsize=(9, 6))

        # Collect all delta-QPS and display-x values across all rates
        all_disp_x = []
        all_delta  = []
        for algo, rate_map in algo_data.items():
            for rate, group_map in rate_map.items():
                if rate not in RATE_CONFIG:
                    continue
                for group, points in group_map.items():
                    bkey = (dataset, topk, algo, rate, group)
                    bval = baseline.get(bkey)
                    if bval is None:
                        continue
                    base_qps, _ = bval
                    for _, rr_recall, rr_qps in points:
                        if RECALL_MIN is None or rr_recall >= RECALL_MIN:
                            all_disp_x.append(_r2d(rr_recall))
                            all_delta.append(base_qps - rr_qps)

        pos_vals = [v for v in all_delta if v > 0]
        use_log  = (len(pos_vals) > 1 and
                    max(pos_vals) / max(min(pos_vals), 1e-9) > 10)

        for algo, rate_map in sorted(algo_data.items()):
            style     = STYLES[algo]
            param_key = style.get("param_key")
            alias     = style.get("alias", algo)

            for rate, rc in RATE_CONFIG.items():
                if rate not in rate_map:
                    continue
                for group, points in sorted(rate_map[rate].items(),
                                            key=lambda kv: (kv[0] is None, kv[0])):
                    if param_key and group is not None:
                        color = style["param_colors"].get(group, "#999999")
                    else:
                        color = style["color"]

                    bkey = (dataset, topk, algo, rate, group)
                    bval = baseline.get(bkey)
                    if bval is None:
                        continue
                    base_qps, _ = bval

                    pts_sorted = sorted(points, key=lambda p: p[0])
                    if RECALL_MIN is not None:
                        pts_sorted = [p for p in pts_sorted if p[1] >= RECALL_MIN]
                    if not pts_sorted:
                        continue

                    recalls_disp = _r2d_vec(np.array([p[1] for p in pts_sorted]))
                    delta_qps    = [base_qps - p[2] for p in pts_sorted]

                    ax.plot(recalls_disp, delta_qps,
                            color=color,
                            linestyle=rc["linestyle"],
                            marker=rc["marker"],
                            linewidth=LINE_WIDTH,
                            markersize=MARKER_SIZE,
                            alpha=0.85)

                    if param_key and group is not None:
                        leg_key   = (algo, group)
                        leg_label = f"{alias} ({param_key}={group})"
                    else:
                        leg_key   = (algo, None)
                        leg_label = alias

                    if leg_key not in algo_handles:
                        algo_handles[leg_key] = matplotlib.lines.Line2D(
                            [], [], linestyle="none", color=color,
                            markersize=7, label=leg_label,
                        )
                    algo_rates.setdefault(leg_key, set()).add(rate)

                    if rate not in rate_handles:
                        rate_handles[rate] = matplotlib.lines.Line2D(
                            [], [], linestyle=rc["linestyle"], marker=rc["marker"],
                            color="black", linewidth=LINE_WIDTH,
                            markersize=MARKER_SIZE, label=rc["suffix"],
                        )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

        disp_min = min(all_disp_x) if all_disp_x else 0.0
        disp_max = max(all_disp_x) if all_disp_x else _r2d(1.0)
        _apply_x_axis(ax, disp_min, disp_max)

        ax.set_xlabel(f"Recall@{topk}")
        ax.set_ylabel("ΔQPS (no-rerank − rerank)")

        fig.tight_layout()
        stem = os.path.join(FIGURES_DIR,
                            f"{dataset}_top{topk}_delta")
        save_figure(fig, stem)
        plt.close(fig)

        # ── Absolute rerank QPS figure (same x-axis, log y) ──────────
        fig2, ax2 = plt.subplots(figsize=(9, 6))

        all_disp_x2 = []
        for algo, rate_map in sorted(algo_data.items()):
            style     = STYLES[algo]
            param_key = style.get("param_key")
            alias     = style.get("alias", algo)

            for rate, rc in RATE_CONFIG.items():
                if rate not in rate_map:
                    continue
                for group, points in sorted(rate_map[rate].items(),
                                            key=lambda kv: (kv[0] is None, kv[0])):
                    if param_key and group is not None:
                        color = style["param_colors"].get(group, "#999999")
                    else:
                        color = style["color"]

                    pts_sorted = sorted(points, key=lambda p: p[0])
                    if RECALL_MIN is not None:
                        pts_sorted = [p for p in pts_sorted if p[1] >= RECALL_MIN]
                    if not pts_sorted:
                        continue

                    recalls_disp = _r2d_vec(np.array([p[1] for p in pts_sorted]))
                    qps_vals     = [p[2] for p in pts_sorted]
                    all_disp_x2.extend(recalls_disp)

                    ax2.plot(recalls_disp, qps_vals,
                             color=color,
                             linestyle=rc["linestyle"],
                             marker=rc["marker"],
                             linewidth=LINE_WIDTH,
                             markersize=MARKER_SIZE,
                             alpha=0.85)

                    # Overlay baseline point with semi-transparent golden border,
                    # connected to the first rerank point by a line.
                    bkey = (dataset, topk, algo, rate, group)
                    bval = baseline.get(bkey)
                    if bval is not None:
                        base_qps, base_recall = bval
                        if base_recall is not None:
                            b_disp = _r2d(base_recall)
                            all_disp_x2.append(b_disp)
                            # Connecting line: baseline → first rerank point
                            ax2.plot([b_disp, recalls_disp[0]],
                                     [base_qps, qps_vals[0]],
                                     color=color,
                                     linestyle=rc["linestyle"],
                                     linewidth=LINE_WIDTH,
                                     alpha=0.85)
                            # Baseline marker with semi-transparent gold edge
                            gold_rgba = (1.0, 0.84, 0.0, 0.8)
                            ax2.scatter([b_disp], [base_qps],
                                        color=color,
                                        edgecolors=[gold_rgba],
                                        linewidths=1.5,
                                        marker=rc["marker"],
                                        s=(MARKER_SIZE * 3) ** 2 / 4,
                                        zorder=5)

        ax2.set_yscale("log")
        ax2.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.yaxis.get_major_formatter().set_scientific(False)

        disp_min2 = min(all_disp_x2) if all_disp_x2 else 0.0
        disp_max2 = max(all_disp_x2) if all_disp_x2 else _r2d(1.0)
        _apply_x_axis(ax2, disp_min2, disp_max2)

        ax2.set_xlabel(f"Recall@{topk}")
        ax2.set_ylabel("Rerank QPS")

        fig2.tight_layout()
        stem2 = os.path.join(FIGURES_DIR,
                             f"{dataset}_top{topk}")
        save_figure(fig2, stem2)
        plt.close(fig2)

save_legend(algo_handles, rate_handles, algo_rates,
            os.path.join(LEGENDS_DIR, "standalone_rerank_legend"))
