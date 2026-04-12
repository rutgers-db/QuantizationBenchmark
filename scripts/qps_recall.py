import json
import os
import matplotlib.pyplot as plt
import matplotlib

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
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "benchmark", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
LEGENDS_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "legends")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LEGENDS_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────
# For each algorithm, specify the JSON filename relative to RESULTS_DIR.
# Set to None to skip an entry.
DATASET = "sift-128-euclidean"
ALGORITHMS = {
    "IVFPQ":    "ivf/" + DATASET + "_Faiss-IVFPQ.json",
    "IVFSQ":    "ivf/" + DATASET + "_Faiss-IVFSQ.json",
    "OPQ-IVFPQ":   "ivf/" + DATASET + "_Faiss-OPQ-IVFPQ.json",
    "IVFOSQ":   "ivf/" + DATASET + "_IVFOSQ.json",
    "RaBitQ":"quantizer/" + DATASET + "_RabitQ.json",
    "SAQ":   "ivf/" + DATASET + "_SAQ_nlist4096.json",
    "TurboQuant": "quantizer/" + DATASET + "_TurboQuant.json",
}

# topk value to extract from search_results (e.g. 10 for Recall@10)
TOPK = 10
TARGET_RECALL = 0.7     # select the build config with highest QPS at this recall
RECALL_THRESHOLD = 0.0   # points with recall below this are not plotted
NAME = DATASET + "_qps_recall_TOPK" + str(TOPK) + "_TARGET" + str(TARGET_RECALL) + ".pdf"

# ── Style definitions ─────────────────────────────────────────
STYLES = {
    "IVFPQ":        {"color": "#1f77b4", "marker": "s",  "linestyle": "-"},
    "IVFSQ":        {"color": "#ff7f0e", "marker": "^",  "linestyle": "-"},
    "OPQ-IVFPQ":       {"color": "#9467bd", "marker": "P",  "linestyle": "-"},
    "IVFOSQ":       {"color": "#8c564b", "marker": "X",  "linestyle": "-"},
    "RaBitQ":    {"color": "#2ca02c", "marker": "D",  "linestyle": "-"},
    "SAQ":       {"color": "#f39c12", "marker": "8",  "linestyle": "-"},
    "TurboQuant":{"color": "#00acc1", "marker": "H",  "linestyle": "-"},
}


def load_json_results(filepath, topk):
    """
    Load per-build-config (recall, qps) data from a benchmark JSON file.

    The JSON is a list of build configurations. For each configuration,
    search_results contains entries with different search params (e.g. nprobe).
    This function selects entries matching `topk` and collects
    (metrics.recall, metrics.queries_per_second) for each such entry.

    Returns a list of configs, where each config is a list of (recall, qps) tuples.
    """
    with open(filepath) as f:
        data = json.load(f)

    configs = []
    for entry in data:
        points = []
        for sr in entry.get("search_results", []):
            if sr["search_params"].get("topk") != topk:
                continue
            metrics = sr.get("metrics", {})
            recall = metrics.get("recall")
            qps = metrics.get("queries_per_second")
            if recall is not None and qps is not None:
                points.append((recall, qps))
        if points:
            configs.append((entry.get("build_params", {}), points))

    return configs


def select_best_config(configs, target_recall):
    """
    Among all build configs, select the one whose highest QPS at recall >= target_recall
    is greatest. Returns (build_params, sorted points) for that config,
    or None if no config reaches target_recall.
    """
    best_build_params = None
    best_points = None
    best_qps = -1
    for build_params, points in configs:
        candidates = [q for r, q in points if r >= target_recall]
        if not candidates:
            continue
        qps_at_target = max(candidates)
        if qps_at_target > best_qps:
            best_qps = qps_at_target
            best_points = points
            best_build_params = build_params
    if best_points is None:
        return None, None
    return best_build_params, sorted(best_points, key=lambda p: p[0])


# ── Collect data and plot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
legend_handles = []

for algo_name, json_rel_path in ALGORITHMS.items():
    if json_rel_path is None:
        continue

    filepath = os.path.join(RESULTS_DIR, json_rel_path)
    if not os.path.isfile(filepath):
        print(f"[SKIP] {algo_name}: file not found — {filepath}")
        continue

    configs = load_json_results(filepath, topk=TOPK)
    if not configs:
        print(f"[SKIP] {algo_name}: no matching entries (topk={TOPK}) in {json_rel_path}")
        continue

    build_params, best = select_best_config(configs, TARGET_RECALL)
    if best is None:
        print(f"[SKIP] {algo_name}: no config reaches target recall {TARGET_RECALL}")
        continue

    best = [(r, q) for r, q in best if r >= RECALL_THRESHOLD]
    if not best:
        print(f"[SKIP] {algo_name}: all points below recall threshold {RECALL_THRESHOLD}")
        continue

    params_str = ", ".join(f"{k}={v}" for k, v in build_params.items())
    print(f"[OK]   {algo_name}: best config [{params_str}] — {len(best)} points")

    recalls = [p[0] for p in best]
    qps_vals = [p[1] for p in best]

    style = STYLES.get(algo_name, {"color": "gray", "marker": ".", "linestyle": "-"})
    line, = ax.plot(
        recalls, qps_vals,
        marker=style["marker"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=2,
        markersize=6,
        label=algo_name,
    )
    legend_handles.append(line)

# ── Format axes ───────────────────────────────────────────────
ax.set_xlabel(f"Recall@{TOPK}")
ax.set_ylabel("QPS")
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)

fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, NAME)
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"\nFigure saved to {out_path}")

# ── Save legend separately ────────────────────────────────────
if legend_handles:
    fig_leg = plt.figure()
    legend = fig_leg.legend(
        handles=legend_handles,
        loc="center",
        ncol=7,
        frameon=False,
        fontsize=12,
    )
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    legend_path = os.path.join(LEGENDS_DIR, NAME)
    fig_leg.savefig(legend_path, bbox_inches=bbox, pad_inches=0)
    print(f"Legend saved to {legend_path}")

plt.close("all")
