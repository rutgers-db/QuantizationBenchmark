import os
import sys
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

# Curves to plot. Each entry maps a curve key to:
#   - file_algo: result file suffix (after `<dataset>_`)
#   - display_name: legend label
#   - color: line color
#   - nbit: optional nbit filter (only for IVFE8PQFastScan, which has 2 curves)
CURVES = [
    {
        "key": "Faiss-IVFPQFastScan",
        "file_algo": "Faiss-IVFPQFastScan",
        "display_name": "IVFPQFastScan",
        "color": "#8c564b",
        "nbit": None,
    },
    {
        "key": "IVFE8PQ",
        "file_algo": "IVFE8PQ",
        "display_name": "IVFE8PQ",
        "color": "#17becf",
        "nbit": None,
    },
    {
        "key": "IVFE8FastScan",
        "file_algo": "IVFE8FastScan",
        "display_name": "IVFE8FastScan",
        "color": "#9467bd",
        "nbit": None,
    },
    {
        "key": "IVFE8PQFastScan-nbit4",
        "file_algo": "IVFE8PQFastScan",
        "display_name": "IVFE8PQFastScan (nbit=4)",
        "color": "#f39c12",
        "nbit": 4,
    },
    {
        "key": "IVFE8PQFastScan-nbit8",
        "file_algo": "IVFE8PQFastScan",
        "display_name": "IVFE8PQFastScan (nbit=8)",
        "color": "#2ca02c",
        "nbit": 8,
    },
    {
        "key": "IVFRabitQLibrary",
        "file_algo": "IVFRabitQLibrary",
        "display_name": "IVFRabitQ",
        "color": "#d62728",
        "nbit": None,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot QPS-Recall@100 curves for IVFPQFastScan, IVFE8PQ, "
                    "IVFE8PQFastScan (nbit=4 and nbit=8), and IVFRabitQLibrary.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. sift-128-euclidean")
    parser.add_argument("--compression-rate", type=float, required=True,
                        help="Compression rate (e.g. 0.125 for 8x)")
    parser.add_argument("--nlist", type=int, required=True,
                        help="Number of IVF lists, e.g. 1024")
    parser.add_argument("--target-recall", type=float, default=0.9,
                        help="Target recall used to pick the best build config")
    return parser.parse_args()


def collect_points_for_entry(entry):
    points = []
    for sr in entry.get("search_results", []):
        if sr.get("search_params", {}).get("topk") != 100:
            continue
        m = sr.get("metrics", {})
        r, q = m.get("recall"), m.get("queries_per_second")
        if r is not None and q is not None:
            points.append((r, q))
    return points


def interpolate_qps_at_recall(points, target):
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


def make_pareto_curve(points):
    if not points:
        return [], []
    pts = sorted(points, key=lambda p: p[0])
    rmap = {}
    for r, q in pts:
        if r not in rmap or q > rmap[r]:
            rmap[r] = q
    frontier = sorted(rmap.items())
    return [p[0] for p in frontier], [p[1] for p in frontier]


def find_best_curve(filepath, compression_rate, nlist, target_recall, nbit_filter):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        entries = json.load(f)

    candidates = []
    for entry in entries:
        bp = entry.get("build_params", {})
        bm = entry.get("build_metrics", {})
        if bp.get("nlist") != nlist:
            continue
        if nbit_filter is not None and bp.get("nbit") != nbit_filter:
            continue
        rate = bm.get("compression_rate")
        if not rate:
            continue
        if (abs(rate - compression_rate) < 1e-9
                or abs(rate - 1.0 / compression_rate) < 1e-9):
            candidates.append(entry)

    best_qps = -1
    best_points = None
    for entry in candidates:
        pts = collect_points_for_entry(entry)
        if not pts:
            continue
        q = interpolate_qps_at_recall(pts, target_recall)
        if q is not None and q > best_qps:
            best_qps = q
            best_points = pts
    return best_points


def main():
    args = parse_args()
    dataset = args.dataset
    rate = args.compression_rate
    nlist = args.nlist
    target_recall = args.target_recall

    curves_data = []
    for spec in CURVES:
        filepath = os.path.join(
            RESULTS_IVF_DIR, f"{dataset}_{spec['file_algo']}.json")
        pts = find_best_curve(filepath, rate, nlist, target_recall, spec["nbit"])
        if pts is None:
            print(f"[WARN] No data for {spec['display_name']} "
                  f"(file={os.path.basename(filepath)}, "
                  f"rate={rate}, nlist={nlist}, nbit={spec['nbit']})")
            continue
        recalls, qps_vals = make_pareto_curve(pts)
        curves_data.append((spec, recalls, qps_vals))

    if not curves_data:
        print(f"No data for dataset={dataset}, rate={rate}, nlist={nlist}")
        sys.exit(1)

    rate_to_suffix = {0.015625: "64x", 0.03125: "32x", 0.0625: "16x",
                      0.125: "8x", 0.25: "4x"}
    suffix = rate_to_suffix.get(rate, f"{rate}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for spec, recalls, qps_vals in curves_data:
        ax.plot(recalls, qps_vals, marker='o', color=spec["color"],
                markersize=4, linewidth=1.5, alpha=0.85,
                label=spec["display_name"])

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    stem = os.path.join(FIGURES_DIR,
                        f"{dataset}_nlist{nlist}_{suffix}_recall100_e8pq")
    fig.savefig(stem + ".pdf", format="pdf", bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf")
    plt.close(fig)

    handles = [
        mlines.Line2D([], [], marker='o', color=spec["color"], markersize=7,
                      linewidth=1.5, label=spec["display_name"])
        for spec, _, _ in curves_data
    ]
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles, [h.get_label() for h in handles],
        loc="center", ncol=min(len(handles), 5), frameon=False, fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig_leg.dpi_scale_trans.inverted())
    leg_stem = os.path.join(
        LEGENDS_DIR, f"{dataset}_nlist{nlist}_{suffix}_recall100_e8pq_legend")
    fig_leg.savefig(leg_stem + ".pdf", format="pdf",
                    bbox_inches=bbox, pad_inches=0.05)
    print(f"[OK] Saved: {leg_stem}.pdf")
    plt.close(fig_leg)


if __name__ == "__main__":
    main()
