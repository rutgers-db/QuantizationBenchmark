import json
import glob
import os
from collections import defaultdict

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "quantizer")

def main():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "*_RabitQLibrary.json")))

    for fpath in files:
        dataset = os.path.basename(fpath).replace("_RabitQLibrary.json", "")
        with open(fpath) as f:
            data = json.load(f)

        print("=" * 70)
        print(f"Dataset: {dataset}")
        print("=" * 70)

        # nrerank -> list of rerank_only_time across configs (for averaging)
        nrerank_times = defaultdict(list)

        for entry in data:
            nbit = entry["build_params"]["nbit"]
            config_label = f"nbit={nbit}"

            print(f"\n  Configuration: {config_label}")
            print(f"  {'nrerank':>10s}  {'rerank_only_time (s)':>22s}")
            print(f"  {'-'*10}  {'-'*22}")

            for sr in entry["search_results"]:
                for rr in sr["metrics"]["rerank_results"]:
                    nr = rr["nrerank"]
                    t = rr["rerank_only_time"]
                    nrerank_times[nr].append(t)
                    print(f"  {nr:>10d}  {t:>22.6f}")

        # Print average across configurations for each nrerank
        print(f"\n  Average rerank_only_time across all configurations:")
        print(f"  {'nrerank':>10s}  {'avg rerank_only_time (s)':>26s}")
        print(f"  {'-'*10}  {'-'*26}")
        for nr in sorted(nrerank_times.keys()):
            times = nrerank_times[nr]
            avg = sum(times) / len(times)
            print(f"  {nr:>10d}  {avg:>26.6f}")
        print()


if __name__ == "__main__":
    main()
