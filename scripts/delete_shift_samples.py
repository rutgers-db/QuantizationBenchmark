#!/usr/bin/env python3
"""
Delete all HDF5 groups whose name contains "shift".

The script searches recursively, but only deletes the top-most matching groups
so nested matches are removed safely in a single pass.
"""

import argparse
from typing import List

import h5py


def _find_shift_groups(hdf5_file: h5py.File) -> List[str]:
    matched_paths: List[str] = []

    def _visitor(name: str, obj) -> None:
        if not isinstance(obj, h5py.Group):
            return
        group_name = name.rsplit("/", 1)[-1]
        if "shift" in group_name.lower():
            matched_paths.append(name)

    hdf5_file.visititems(_visitor)

    top_level_matches: List[str] = []
    for path in sorted(matched_paths, key=lambda item: (item.count("/"), item)):
        if any(path == parent or path.startswith(f"{parent}/") for parent in top_level_matches):
            continue
        top_level_matches.append(path)

    return top_level_matches


def delete_shift_groups(hdf5_path: str, dry_run: bool) -> List[str]:
    mode = "r" if dry_run else "r+"
    with h5py.File(hdf5_path, mode) as hdf5_file:
        matched_groups = _find_shift_groups(hdf5_file)
        if not dry_run:
            for group_path in matched_groups:
                del hdf5_file[group_path]

    return matched_groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Delete all HDF5 groups whose name contains "shift".'
    )
    parser.add_argument("hdf5_path", help="Path to the HDF5 dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the matching groups without deleting them",
    )
    args = parser.parse_args()

    matched_groups = delete_shift_groups(args.hdf5_path, dry_run=args.dry_run)

    if not matched_groups:
        print('No groups with "shift" in their name were found.')
        return 0

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} {len(matched_groups)} group(s):")
    for group_path in matched_groups:
        print(f"  - {group_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
