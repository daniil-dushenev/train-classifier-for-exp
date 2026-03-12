#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two binary-classification datasets: no_synth and with_synth."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root with folders: class0, class1, class1_synth",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_out"),
        help="Where to write datasets (default: data_out)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of real class0/class1 used for shared test split (default: 0.2)",
    )
    parser.add_argument(
        "--train-pos-count",
        type=int,
        default=None,
        help="If set, put exactly N real class1 images in train and the rest into test.",
    )
    parser.add_argument(
        "--use-all-negatives-train",
        action="store_true",
        help="Use all class0 images in train (only meaningful with --train-pos-count).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_indices(n: int, test_ratio: float, rng: random.Random) -> tuple[set[int], set[int]]:
    if n == 0:
        return set(), set()
    test_n = max(1, int(round(n * test_ratio)))
    test_n = min(test_n, n - 1) if n > 1 else 1
    idx = list(range(n))
    rng.shuffle(idx)
    test_idx = set(idx[:test_n])
    train_idx = set(idx[test_n:])
    return train_idx, test_idx


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_paths(paths: list[Path], dst_dir: Path, prefix: str = "") -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths):
        dst_name = f"{prefix}{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst_dir / dst_name)
    return len(paths)


SYNTH_RATIOS: list[float] = [0.5, 1.0, 2.0]


def ratio_to_name(ratio: float) -> str:
    return f"synth_{ratio:g}x"


def build_datasets(
    input_root: Path,
    output_root: Path,
    test_ratio: float,
    seed: int,
    train_pos_count: int | None = None,
    use_all_negatives_train: bool = False,
    synth_ratios: list[float] | None = None,
    extra_test_class0: Path | None = None,
    extra_test_class1: Path | None = None,
) -> dict[str, Any]:
    if synth_ratios is None:
        synth_ratios = SYNTH_RATIOS

    rng = random.Random(seed)

    class0 = list_images(input_root / "class0")
    class1 = list_images(input_root / "class1")
    class1_synth = list_images(input_root / "class1_synth")

    if train_pos_count is None:
        c0_train_idx, c0_test_idx = split_indices(len(class0), test_ratio, rng)
        c1_train_idx, c1_test_idx = split_indices(len(class1), test_ratio, rng)
    else:
        if train_pos_count < 0:
            raise ValueError("train_pos_count must be >= 0")
        c1_idx = list(range(len(class1)))
        rng.shuffle(c1_idx)
        pos_train_n = min(train_pos_count, len(class1))
        c1_train_idx = set(c1_idx[:pos_train_n])
        c1_test_idx = set(c1_idx[pos_train_n:])
        if use_all_negatives_train:
            c0_train_idx = set(range(len(class0)))
            c0_test_idx = set()
        else:
            c0_train_idx, c0_test_idx = split_indices(len(class0), test_ratio, rng)

    c0_train = [class0[i] for i in sorted(c0_train_idx)]
    c0_test = [class0[i] for i in sorted(c0_test_idx)]
    c1_train_real = [class1[i] for i in sorted(c1_train_idx)]
    c1_test = [class1[i] for i in sorted(c1_test_idx)]

    if extra_test_class0 is not None:
        c0_test = c0_test + list_images(extra_test_class0)
    if extra_test_class1 is not None:
        c1_test = c1_test + list_images(extra_test_class1)

    # Shared shuffled synth list (reproducible)
    synth_shuffled = list(class1_synth)
    rng.shuffle(synth_shuffled)

    # --- no_synth baseline ---
    no_synth_root = output_root / "no_synth"
    reset_dir(no_synth_root / "train" / "class0")
    reset_dir(no_synth_root / "train" / "class1")
    reset_dir(no_synth_root / "test" / "class0")
    reset_dir(no_synth_root / "test" / "class1")
    copy_paths(c0_train, no_synth_root / "train" / "class0")
    copy_paths(c1_train_real, no_synth_root / "train" / "class1")
    copy_paths(c0_test, no_synth_root / "test" / "class0")
    copy_paths(c1_test, no_synth_root / "test" / "class1")

    stats: dict[str, Any] = {
        "class0_real": len(class0),
        "class1_real": len(class1),
        "class1_synth_available": len(class1_synth),
        "test_class0": len(c0_test),
        "test_class1": len(c1_test),
        "no_synth_train_class0": len(c0_train),
        "no_synth_train_class1": len(c1_train_real),
    }

    # --- synth ratio experiments ---
    for ratio in synth_ratios:
        n_synth = min(int(round(len(c1_train_real) * ratio)), len(synth_shuffled))
        synth_subset = synth_shuffled[:n_synth]
        name = ratio_to_name(ratio)
        ds_root = output_root / name
        reset_dir(ds_root / "train" / "class0")
        reset_dir(ds_root / "train" / "class1")
        reset_dir(ds_root / "test" / "class0")
        reset_dir(ds_root / "test" / "class1")
        copy_paths(c0_train, ds_root / "train" / "class0")
        copy_paths(c1_train_real, ds_root / "train" / "class1", prefix="real_")
        copy_paths(synth_subset, ds_root / "train" / "class1", prefix="synth_")
        copy_paths(c0_test, ds_root / "test" / "class0")
        copy_paths(c1_test, ds_root / "test" / "class1")
        stats[f"{name}_train_class0"] = len(c0_train)
        stats[f"{name}_train_class1_real"] = len(c1_train_real)
        stats[f"{name}_train_class1_synth"] = n_synth
        stats[f"{name}_train_class1_total"] = len(c1_train_real) + n_synth
        print(
            f"  {name}: {len(c1_train_real)} real + {n_synth} synth "
            f"(ratio={ratio:g}, available={len(class1_synth)})"
        )

    return stats


def main() -> None:
    args = parse_args()
    stats = build_datasets(
        args.input_root,
        args.output_root,
        args.test_ratio,
        args.seed,
        train_pos_count=args.train_pos_count,
        use_all_negatives_train=args.use_all_negatives_train,
    )

    print("Done.")
    print(
        f"class0 real: {stats['class0_real']} | class1 real: {stats['class1_real']} | "
        f"class1 synth available: {stats['class1_synth_available']}"
    )
    print(f"shared test: class0={stats['test_class0']} class1={stats['test_class1']}")
    print(
        f"no_synth train: class0={stats['no_synth_train_class0']} "
        f"class1={stats['no_synth_train_class1']}"
    )
    for ratio in SYNTH_RATIOS:
        name = ratio_to_name(ratio)
        print(
            f"{name} train: class0={stats[f'{name}_train_class0']} "
            f"class1={stats[f'{name}_train_class1_total']} "
            f"(real={stats[f'{name}_train_class1_real']} + synth={stats[f'{name}_train_class1_synth']})"
        )


if __name__ == "__main__":
    main()
