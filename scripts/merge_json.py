"""Merge two JSON files (arrays) into one.

Usage:
    python merge_json.py <file1.json> <file2.json> -o <output.json>

If -o is not specified, the result is written to file1.json (in-place merge).
Duplicate entries (by build_params) are skipped.
"""

import argparse
import json


def are_params_equal(a, b):
    """Check if two build_params dicts are equal."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def merge(data1, data2):
    """Merge data2 into data1, skipping entries with duplicate build_params."""
    merged = list(data1)
    existing_params = [entry["build_params"] for entry in merged if "build_params" in entry]
    for entry in data2:
        bp = entry.get("build_params")
        if bp and any(are_params_equal(bp, ep) for ep in existing_params):
            print(f"  Skipping duplicate: {bp}")
            continue
        merged.append(entry)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge two JSON files")
    parser.add_argument("file1", help="First JSON file")
    parser.add_argument("file2", help="Second JSON file")
    parser.add_argument("-o", "--output", help="Output file (default: overwrite file1)")
    args = parser.parse_args()

    with open(args.file1) as f:
        data1 = json.load(f)
    with open(args.file2) as f:
        data2 = json.load(f)

    merged = merge(data1, data2)
    output_path = args.output or args.file1

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(data1)} + {len(data2)} -> {len(merged)} entries")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
