"""Plot QPS-Recall@100 curves for arbitrary (algo, build_params) combinations.

Edit the CURVES list below to choose which lines to plot. Each entry specifies
an algorithm (matching the result file suffix `<dataset>_<algo>.json`) and a
dict of `build_params` filters. Filters can include any key in `build_params`
(e.g. nlist, nsubvec, nbit). Only entries whose `build_params` match every
filter are considered; among matches, the one with highest QPS at
TARGET_RECALL is plotted.

Usage:
    python scripts/plot_custom.py
"""

import os
import sys
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Configuration ─────────────────────────────────────────────────
DATASET = "sift-128-euclidean"
TARGET_RECALL = 0.9
OUT_NAME = "sift_custom"   # output stem under figures/ivf/
TITLE = None               # e.g. "SIFT-1M, nlist=1024"

# Each curve is one line on the plot. Required keys: algo, filters.
# Optional keys: label (legend text), color (hex string).
CURVES = [
    {
        "algo": "Faiss-IVFPQFastScan",
        "filters": {"nlist": 1024, "nsubvec": 32, "nbit": 4},
        "label": "IVFPQFastScan (32x4bit)",
        "color": "#8c564b",
    },
    {
        "algo": "IVFE8NoLut",
        "filters": {"nlist": 1024, "bits": 1},
        "label": "IVFE8NoLut (nbit=1)",
        "color": "#17becf",
    },
    {
        "algo": "IVFE8NoLut",
        "filters": {"nlist": 1024, "bits": 1.5},
        "label": "IVFE8NoLut (nbit=1.5)",
        "color": "#1f77b4",
    },
    {
        "algo": "IVFE8PQFastScan",
        "filters": {"nlist": 1024, "nsubvec": 16, "nbit": 8},
        "label": "IVFE8PQFastScan (16x8bit)",
        "color": "#2ca02c",
    },
    {
        "algo": "IVFE8PQFastScan",
        "filters": {"nlist": 1024, "nsubvec": 32, "nbit": 4},
        "label": "IVFE8PQFastScan (32x4bit)",
        "color": "#f39c12",
    },
    {
        "algo": "IVFE8FastScan",
        "filters": {"nlist": 1024},
        "label": "IVFE8FastScan",
        "color": "#9467bd",
    },
    {
        "algo": "IVFRabitQLibrary",
        "filters": {"nlist": 1024, "nbit": 2},
        "label": "IVFRabitQ (nbit=2)",
        "color": "#d62728",
    },
    {
        "algo": "IVFRabitQLibrary",
        "filters": {"nlist": 1024, "nbit": 1},
        "label": "IVFRabitQ (nbit=1)",
        "color": "#e377c2",
    },
    
]
# ──────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_IVF_DIR = os.path.join(REPO_DIR, "benchmark", "results", "ivf")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "ivf")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#f39c12",
]


def collect_points(entry):
    points = []
    for sr in entry.get("search_results", []):
        if sr.get("search_params", {}).get("topk") != 100:
            continue
        m = sr.get("metrics", {})
        r, q = m.get("recall"), m.get("queries_per_second")
        if r is not None and q is not None:
            points.append((r, q))
    return points


def interpolate_qps(points, target):
    if not points:
        return None
    pts = sorted(points, key=lambda p: p[0])
    rmap = {}
    for r, q in pts:
        if r not in rmap or q > rmap[r]:
            rmap[r] = q
    frontier = sorted(rmap.items())
    rs = [p[0] for p in frontier]
    qs = [p[1] for p in frontier]
    if target <= rs[0]:
        return qs[0]
    if target >= rs[-1]:
        return qs[-1]
    for i in range(len(rs) - 1):
        if rs[i] <= target <= rs[i + 1]:
            t = (target - rs[i]) / (rs[i + 1] - rs[i])
            return qs[i] + t * (qs[i + 1] - qs[i])
    return None


def make_pareto(points):
    if not points:
        return [], []
    pts = sorted(points, key=lambda p: p[0])
    rmap = {}
    for r, q in pts:
        if r not in rmap or q > rmap[r]:
            rmap[r] = q
    frontier = sorted(rmap.items())
    return [p[0] for p in frontier], [p[1] for p in frontier]


def find_best_entry(filepath, filters, target_recall):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        entries = json.load(f)

    candidates = [
        e for e in entries
        if all(e.get("build_params", {}).get(k) == v for k, v in filters.items())
    ]
    if not candidates:
        return None

    best_qps = -1
    best_entry = None
    best_points = None
    for entry in candidates:
        pts = collect_points(entry)
        if not pts:
            continue
        q = interpolate_qps(pts, target_recall)
        if q is not None and q > best_qps:
            best_qps = q
            best_points = pts
            best_entry = entry

    if best_points is None:
        return None
    recalls, qps_vals = make_pareto(best_points)
    return recalls, qps_vals, best_entry.get("build_params", {})


def auto_label(spec):
    if spec.get("filters"):
        kv = ",".join(f"{k}={v}" for k, v in spec["filters"].items())
        return f"{spec['algo']} ({kv})"
    return spec["algo"]


def main():
    plotted = []
    for spec in CURVES:
        filepath = os.path.join(
            RESULTS_IVF_DIR, f"{DATASET}_{spec['algo']}.json")
        result = find_best_entry(filepath, spec.get("filters", {}), TARGET_RECALL)
        if result is None:
            print(f"[WARN] No data for {spec['algo']} with filters "
                  f"{spec.get('filters', {})} "
                  f"(file={os.path.basename(filepath)})")
            continue
        recalls, qps_vals, matched_bp = result
        plotted.append({
            "label": spec.get("label") or auto_label(spec),
            "color": spec.get("color"),
            "recalls": recalls,
            "qps_vals": qps_vals,
            "matched_bp": matched_bp,
        })

    if not plotted:
        print("No data for any curve.")
        sys.exit(1)

    color_idx = 0
    for c in plotted:
        if c["color"] is None:
            c["color"] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in plotted:
        ax.plot(c["recalls"], c["qps_vals"], marker='o', color=c["color"],
                markersize=4, linewidth=1.5, alpha=0.85, label=c["label"])

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS")
    if TITLE:
        ax.set_title(TITLE)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    stem = os.path.join(FIGURES_DIR, OUT_NAME)
    fig.savefig(stem + ".pdf", format="pdf", bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf")
    plt.close(fig)

    handles = [
        mlines.Line2D([], [], marker='o', color=c["color"], markersize=7,
                      linewidth=1.5, label=c["label"])
        for c in plotted
    ]
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles, [h.get_label() for h in handles],
        loc="center", ncol=min(len(handles), 4), frameon=False, fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig_leg.dpi_scale_trans.inverted())
    leg_stem = os.path.join(LEGENDS_DIR, OUT_NAME + "_legend")
    fig_leg.savefig(leg_stem + ".pdf", format="pdf",
                    bbox_inches=bbox, pad_inches=0.05)
    print(f"[OK] Saved: {leg_stem}.pdf")
    plt.close(fig_leg)

    print("\nMatched build_params per curve:")
    for c in plotted:
        print(f"  {c['label']}: {c['matched_bp']}")


if __name__ == "__main__":
    main()
