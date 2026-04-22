import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_IVF_DIR = os.path.join(REPO_DIR, "benchmark", "results", "ivf")
FIGURES_DIR = os.path.join(REPO_DIR, "figures", "ivf")
LEGENDS_DIR = os.path.join(FIGURES_DIR, "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Color mapping: IVF method -> standalone method color ──────

# IVF method name -> corresponding standalone color
IVF_STYLES = {
    # "Faiss-IVFPQ":        {"color": "#1f77b4", "name": "IVFPQ"},
    # "Faiss-OPQ-IVFPQ":    {"color": "#9467bd", "name": "IVFOPQ"},
    "IVFE8":  {"color": "#17becf", "name": "IVFE8"},   # new - no standalone match
    "IVFE8FastScan":      {"color": "#f39c12", "name": "IVFE8FastScan"},  
    # "Faiss-IVFSQ":        {"color": "#ff7f0e", "name": "IVFSQ"},
    # "IVFOSQ":             {"color": "#8c564b", "name": "IVFOSQ"},
    "IVFRabitQLibrary":   {"color": "#d62728", "name": "IVFRabitQ"},
    # "IVFSAQ-merged":      {"color": "#f39c12", "name": "IVFSAQ"},
    "Faiss-IVFPQFastScan":      {"color": "#8c564b", "name": "IVFPQFastScan"},  
    # "IVFTurboQuant":      {"color": "#00acc1", "name": "IVFTurboQuant"},
}

KNOWN_DATASETS = [
    # "audio-128-euclidean",
    # "gist-960-euclidean",
    # "paper-200-euclidean",
    "sift-128-euclidean",
    # "text2image-200-euclidean",
    # "video-1024-euclidean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot IVF QPS-Recall@100 curves")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. sift-128-euclidean")
    parser.add_argument("--compression-rate", type=float, required=True,
                        help="Compression rate (e.g. 0.125 for 8x)")
    parser.add_argument("--nlist", type=int, required=True,
                        help="Number of IVF lists, e.g. 4096")
    parser.add_argument("--target-recall", type=float, required=True,
                        help="Target recall for selecting the best build config")
    return parser.parse_args()


def collect_points_for_entry(entry):
    """Collect all (recall, qps) points from topk=100 search results of an entry.

    Only uses base search results (different nprobe), not rerank results.
    """
    points = []
    for sr in entry.get("search_results", []):
        topk = sr.get("search_params", {}).get("topk")
        if topk != 100:
            continue
        metrics = sr.get("metrics", {})
        recall = metrics.get("recall")
        qps = metrics.get("queries_per_second")
        if recall is not None and qps is not None:
            points.append((recall, qps))
    return points


def interpolate_qps_at_recall(points, target_recall):
    """Given a set of (recall, qps) points, estimate QPS at target_recall.

    Uses the Pareto frontier: for each recall level, keep only the highest QPS.
    Then interpolate linearly.
    """
    if not points:
        return None
    # Build Pareto frontier: sort by recall, keep running max QPS
    sorted_pts = sorted(points, key=lambda p: p[0])
    # Group by recall, take max QPS at each recall
    recall_to_max_qps = {}
    for r, q in sorted_pts:
        if r not in recall_to_max_qps or q > recall_to_max_qps[r]:
            recall_to_max_qps[r] = q
    frontier = sorted(recall_to_max_qps.items())
    recalls = [p[0] for p in frontier]
    qps_vals = [p[1] for p in frontier]

    if target_recall <= recalls[0]:
        return qps_vals[0]
    if target_recall >= recalls[-1]:
        return qps_vals[-1]
    # Linear interpolation
    for i in range(len(recalls) - 1):
        if recalls[i] <= target_recall <= recalls[i + 1]:
            t = (target_recall - recalls[i]) / (recalls[i + 1] - recalls[i])
            return qps_vals[i] + t * (qps_vals[i + 1] - qps_vals[i])
    return None


def make_pareto_curve(points):
    """Build a Pareto-optimal curve from (recall, qps) points.

    For a QPS-Recall curve, we want the upper-left frontier:
    higher recall AND higher QPS is better. Sort by recall ascending,
    then for plotting, just connect all points sorted by recall.
    Actually, for a clean curve, we take the Pareto frontier:
    as recall increases, QPS should decrease (tradeoff). Keep the
    upper envelope.
    """
    if not points:
        return [], []
    # Sort by recall
    sorted_pts = sorted(points, key=lambda p: p[0])
    # Build upper envelope: for each recall, keep max QPS
    recall_to_max_qps = {}
    for r, q in sorted_pts:
        if r not in recall_to_max_qps or q > recall_to_max_qps[r]:
            recall_to_max_qps[r] = q
    frontier = sorted(recall_to_max_qps.items())
    # Make it monotonically decreasing in QPS (Pareto optimal)
    pareto = []
    min_qps_so_far = float('inf')
    # Actually for recall-qps tradeoff, as recall goes up, qps typically goes down
    # We want the frontier where no other point has both higher recall and higher qps
    # Simple approach: just return sorted points for the curve
    recalls = [p[0] for p in frontier]
    qps_vals = [p[1] for p in frontier]
    return recalls, qps_vals


def main():
    args = parse_args()
    dataset = args.dataset
    compression_rate = args.compression_rate
    nlist = args.nlist
    target_recall = args.target_recall

    # Find all result files for this dataset
    pattern = os.path.join(RESULTS_IVF_DIR, f"{dataset}_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found for dataset '{dataset}'")
        sys.exit(1)

    # For each method, find the best build config and collect its points
    method_curves = {}  # method_name -> (recalls, qps_vals)

    for filepath in files:
        filename = os.path.basename(filepath)
        name = filename.replace(".json", "")
        # Skip .old files
        if ".old" in filename:
            continue
        algo = name[len(dataset) + 1:]
        # Only plot methods defined in IVF_STYLES
        if algo not in IVF_STYLES:
            continue

        with open(filepath) as f:
            entries = json.load(f)

        # Filter entries by compression_rate and nlist
        # Some files store rate as ratio (0.125), others as multiplier (8.0)
        # Accept both: match rate or 1/rate
        candidates = []
        for entry in entries:
            bp = entry.get("build_params", {})
            bm = entry.get("build_metrics", {})
            entry_rate = bm.get("compression_rate")
            entry_nlist = bp.get("nlist")
            if entry_nlist != nlist:
                continue
            if entry_rate is None or entry_rate == 0:
                continue
            rate_matches = (
                abs(entry_rate - compression_rate) < 1e-9
                or abs(entry_rate - 1.0 / compression_rate) < 1e-9
            )
            if rate_matches:
                candidates.append(entry)

        if not candidates:
            continue

        # For each candidate build config, collect points and evaluate at target recall
        best_qps = -1
        best_points = None
        for entry in candidates:
            points = collect_points_for_entry(entry)
            if not points:
                continue
            qps_at_target = interpolate_qps_at_recall(points, target_recall)
            if qps_at_target is not None and qps_at_target > best_qps:
                best_qps = qps_at_target
                best_points = points

        if best_points:
            recalls, qps_vals = make_pareto_curve(best_points)
            method_curves[algo] = (recalls, qps_vals)

    if not method_curves:
        print(f"No data found for dataset={dataset}, rate={compression_rate}, nlist={nlist}")
        sys.exit(1)

    # ── Assign colors ────────────────────────────────────────────
    # Generate colors for methods not in IVF_STYLES
    extra_colors = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    extra_idx = 0
    method_colors = {}
    for algo in method_curves:
        if algo in IVF_STYLES:
            method_colors[algo] = IVF_STYLES[algo]["color"]
        else:
            method_colors[algo] = extra_colors[extra_idx % len(extra_colors)]
            extra_idx += 1

    # ── Determine compression suffix for filename ────────────────
    rate_to_suffix = {0.03125: "32x", 0.0625: "16x", 0.125: "8x", 0.25: "4x"}
    suffix = rate_to_suffix.get(compression_rate, f"{compression_rate}")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    all_qps = []
    for algo, (recalls, qps_vals) in sorted(method_curves.items()):
        color = method_colors[algo]
        ax.plot(recalls, qps_vals, marker='o', color=color, markersize=4,
                linewidth=1.5, alpha=0.8)
        all_qps.extend(qps_vals)

    # Fixed log scale y-axis, capped at 100000
    ax.set_yscale("log")
    # ax.set_ylim(top=100000)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    # Fixed x-axis range
    # ax.set_xlim(0.5, 1.0)

    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    stem = os.path.join(FIGURES_DIR,
                        f"{dataset}_nlist{nlist}_{suffix}_recall100")
    fig.savefig(stem + "_compare_ivfe8.pdf", format="pdf", bbox_inches="tight")
    print(f"[OK] Saved: {stem}.pdf")
    plt.close(fig)

    # ── Save legend separately ───────────────────────────────────
    handles = []
    for algo in sorted(method_curves.keys()):
        color = method_colors[algo]
        display_name = IVF_STYLES[algo]["name"] if algo in IVF_STYLES else algo
        h = mlines.Line2D([], [], marker='o', color=color, markersize=7,
                          linewidth=1.5, label=display_name)
        handles.append(h)

    if handles:
        fig_leg = plt.figure()
        legend = fig_leg.legend(
            handles,
            [h.get_label() for h in handles],
            loc="center",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=12,
        )
        fig_leg.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig_leg.dpi_scale_trans.inverted()
        )
        leg_stem = os.path.join(LEGENDS_DIR,
                                f"{dataset}_nlist{nlist}_{suffix}_recall100_legend")
        fig_leg.savefig(leg_stem + "_compare_ivfe8.pdf", format="pdf",
                        bbox_inches=bbox, pad_inches=0.05)
        print(f"[OK] Saved: {leg_stem}_compare_ivfe8.pdf")
        plt.close(fig_leg)


if __name__ == "__main__":
    main()
