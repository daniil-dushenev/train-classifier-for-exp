#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete all .png files from class0 and class1.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (default: .)")
    parser.add_argument("--dirs", type=Path, nargs="+", default=None,
                        help="Explicit folders to clean (overrides --root/class0,class1 defaults)")
    parser.add_argument("--dry-run", action="store_true", help="Show files without deleting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dirs is not None:
        targets = args.dirs
    else:
        targets = [args.root / "class0", args.root / "class1"]
    deleted = 0

    for folder in targets:
        if not folder.exists():
            print(f"Skip missing folder: {folder}")
            continue
        for path in folder.rglob("*.png"):
            print(path)
            if not args.dry_run:
                path.unlink(missing_ok=True)
                deleted += 1

    if args.dry_run:
        print("Dry run finished.")
    else:
        print(f"Deleted {deleted} .png files.")


if __name__ == "__main__":
    main()

