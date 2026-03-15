#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from build_datasets import SYNTH_RATIOS, build_datasets, ratio_to_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build datasets and train/evaluate models on no_synth and synth-ratio variants."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("data_out"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--train-pos-count",
        type=int,
        default=None,
        help="Requested number of real class1 samples per fold-train.",
    )
    parser.add_argument(
        "--use-all-negatives-train",
        action="store_true",
        help="Use all class0 images in train when building base datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Manual pos_weight for class1. If omitted, computed from fold-train.",
    )
    parser.add_argument(
        "--extra-test-class0",
        type=Path,
        default=None,
        help="Extra folder with class0 images added only to test split.",
    )
    parser.add_argument(
        "--extra-test-class1",
        type=Path,
        default=None,
        help="Extra folder with class1 images added only to test split.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def create_model(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def drop_corrupted_samples(ds: datasets.ImageFolder) -> int:
    valid: list[tuple[str, int]] = []
    dropped = 0
    for path, target in ds.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append((path, target))
        except Exception:
            dropped += 1
    if dropped:
        ds.samples = valid
        ds.imgs = valid
        ds.targets = [t for _, t in valid]
    return dropped


def make_cv_datasets(dataset_root: Path) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, eval_tf = build_transforms()
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"
    train_ds = datasets.ImageFolder(train_root, transform=train_tf, allow_empty=True)
    eval_ds = datasets.ImageFolder(test_root, transform=eval_tf, allow_empty=True)
    dtr = drop_corrupted_samples(train_ds)
    dev = drop_corrupted_samples(eval_ds)
    if dtr or dev:
        print(f"  dropped corrupted files: train={dtr}, eval={dev}")
    return train_ds, eval_ds


def make_fold_loaders(
    train_ds: datasets.ImageFolder,
    train_indices: np.ndarray,
    eval_ds: datasets.ImageFolder,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_subset = Subset(train_ds, train_indices.tolist())
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader


def class_stats_from_targets(targets: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
    y = targets[indices]
    return int(np.sum(y == 0)), int(np.sum(y == 1))


def build_fixed_real_cv_folds(
    train_ds: datasets.ImageFolder,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, int, int]]:
    samples = train_ds.samples
    neg_idx = np.array([i for i, (_, t) in enumerate(samples) if int(t) == 0], dtype=np.int64)
    real_pos_idx = np.array(
        [i for i, (p, t) in enumerate(samples) if int(t) == 1 and not Path(p).name.startswith("synth_")],
        dtype=np.int64,
    )
    synth_pos_idx = np.array(
        [i for i, (p, t) in enumerate(samples) if int(t) == 1 and Path(p).name.startswith("synth_")],
        dtype=np.int64,
    )

    real_train_count = 50
    if len(real_pos_idx) <= real_train_count:
        raise ValueError(
            f"Need more than {real_train_count} real positives for CV, got {len(real_pos_idx)}."
        )
    if n_splits < 2:
        raise ValueError("cv-folds must be >= 2")
    if len(real_pos_idx) < n_splits:
        raise ValueError(
            f"Not enough real positives for {n_splits} folds: class1_real={len(real_pos_idx)}"
        )

    rng = np.random.default_rng(seed)
    shuffled_real = real_pos_idx.copy()
    rng.shuffle(shuffled_real)
    folds: list[tuple[np.ndarray, int, int]] = []
    for i in range(n_splits):
        start = (i * real_train_count) % len(shuffled_real)
        end = start + real_train_count
        if end <= len(shuffled_real):
            real_train = shuffled_real[start:end]
        else:
            wrap = end - len(shuffled_real)
            real_train = np.concatenate([shuffled_real[start:], shuffled_real[:wrap]])
        train_idx = np.concatenate([neg_idx, real_train, synth_pos_idx]).astype(np.int64)
        folds.append((train_idx, int(len(real_train)), int(len(synth_pos_idx))))
    return folds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    log_every: int,
    pos_weight: float | None,
) -> None:
    model.to(device)
    criterion: nn.Module
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        running_tp = 0
        running_fp = 0
        running_fn = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.numel())
            running_tp += int(((preds == 1) & (labels == 1)).sum().item())
            running_fp += int(((preds == 1) & (labels == 0)).sum().item())
            running_fn += int(((preds == 0) & (labels == 1)).sum().item())
            denom = 2 * running_tp + running_fp + running_fn
            running_f1_class1 = (2 * running_tp / denom) if denom > 0 else 0.0

            if batch_idx % log_every == 0 or batch_idx == num_batches:
                avg_loss = running_loss / max(1, running_total)
                avg_acc = running_correct / max(1, running_total)
                print(
                    f"  epoch {epoch}/{epochs} | batch {batch_idx}/{num_batches} "
                    f"| loss={avg_loss:.4f} acc={avg_acc:.4f} f1_class1={running_f1_class1:.4f}"
                )

        epoch_time = time.perf_counter() - epoch_start
        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        denom = 2 * running_tp + running_fp + running_fn
        epoch_f1 = (2 * running_tp / denom) if denom > 0 else 0.0
        print(
            f"  epoch {epoch}/{epochs} done in {epoch_time:.1f}s "
            f"| loss={epoch_loss:.4f} acc={epoch_acc:.4f} f1_class1={epoch_f1:.4f}"
        )


def evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    num_batches = len(loader)
    print(f"  evaluating on {len(loader.dataset)} samples ({num_batches} batches)")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().astype(int).tolist())
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"    eval batch {batch_idx}/{num_batches}")

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_prob_arr = np.array(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1_class1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "specificity": specificity,
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = ["accuracy", "precision", "recall", "f1_class1", "specificity", "roc_auc", "pr_auc"]
    count_keys = ["tn", "fp", "fn", "tp"]
    out: dict[str, Any] = {}
    for k in metric_keys:
        out[k] = float(np.nanmean([float(r[k]) for r in rows]))
    for k in count_keys:
        out[k] = int(round(float(np.nanmean([float(r[k]) for r in rows]))))
    return out


def save_results(reports_dir: Path, rows: list[dict[str, Any]]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = [
        "dataset",
        "model",
        "fold",
        "accuracy",
        "precision",
        "recall",
        "f1_class1",
        "specificity",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with (reports_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["dataset", "model", "fold", "accuracy", "precision", "recall", "f1_class1", "roc_auc", "pr_auc"]
    print("\nResults:")
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        vals: list[str] = []
        for k in headers:
            v = row[k]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(" | ".join(vals))


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    print("Experiment config:")
    print(
        {
            "input_root": str(args.input_root),
            "datasets_root": str(args.datasets_root),
            "reports_dir": str(args.reports_dir),
            "test_ratio": args.test_ratio,
            "train_pos_count": args.train_pos_count,
            "cv_folds": args.cv_folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "pos_weight": args.pos_weight,
            "device": args.device,
            "pretrained": not args.no_pretrained,
        }
    )

    build_train_pos_count = args.train_pos_count
    if args.train_pos_count is not None and args.cv_folds > 1:
        build_train_pos_count = int(math.ceil(args.train_pos_count * args.cv_folds / (args.cv_folds - 1)))
        print(
            f"Adjusted train_pos_count for CV pool: requested per-fold={args.train_pos_count}, "
            f"cv_folds={args.cv_folds} -> build train_pos_count={build_train_pos_count}"
        )

    build_start = time.perf_counter()
    ds_stats = build_datasets(
        input_root=args.input_root,
        output_root=args.datasets_root,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_pos_count=build_train_pos_count,
        use_all_negatives_train=args.use_all_negatives_train,
        extra_test_class0=args.extra_test_class0,
        extra_test_class1=args.extra_test_class1,
    )
    print("Datasets built:")
    print(ds_stats)
    print(f"Dataset build time: {time.perf_counter() - build_start:.1f}s")
    print(f"Using device: {device}")

    dataset_names = ["no_synth"] + [ratio_to_name(r) for r in SYNTH_RATIOS]
    results: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        ds_start = time.perf_counter()
        dataset_root = args.datasets_root / dataset_name
        print(f"\nPreparing CV datasets for dataset={dataset_name}: {dataset_root}")
        train_ds, eval_ds = make_cv_datasets(dataset_root)
        train_targets = np.array(train_ds.targets, dtype=np.int64)
        eval_targets = np.array(eval_ds.targets, dtype=np.int64)

        train_c0 = int(np.sum(train_targets == 0))
        train_c1 = int(np.sum(train_targets == 1))
        eval_c0 = int(np.sum(eval_targets == 0))
        eval_c1 = int(np.sum(eval_targets == 1))
        print(
            f"  dataset pool: train class0={train_c0} class1={train_c1} | "
            f"eval-pool class0={eval_c0} class1={eval_c1}"
        )

        fold_splits = build_fixed_real_cv_folds(train_ds=train_ds, n_splits=args.cv_folds, seed=args.seed)

        for model_name in ["resnet18"]:
            model_start = time.perf_counter()
            fold_rows: list[dict[str, Any]] = []
            print(f"\nTraining model={model_name} on dataset={dataset_name} with {args.cv_folds}-fold CV")

            for fold_idx, (train_idx, real_train_count, synth_train_count) in enumerate(fold_splits, start=1):
                c0, c1 = class_stats_from_targets(train_targets, train_idx)
                auto_pos_weight = (c0 / c1) if c1 > 0 else None
                pos_weight = args.pos_weight if args.pos_weight is not None else auto_pos_weight
                print(
                    f"  fold {fold_idx}/{args.cv_folds}: train={len(train_idx)} "
                    f"(class0={c0}, class1_total={c1}, class1_real={real_train_count}, class1_synth={synth_train_count}) "
                    f"| val={len(eval_targets)} (class0={eval_c0}, class1={eval_c1}) "
                    f"real_train_subsample=50 pos_weight={pos_weight}"
                )

                train_loader, eval_loader = make_fold_loaders(
                    train_ds=train_ds,
                    train_indices=train_idx,
                    eval_ds=eval_ds,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                model = create_model(model_name=model_name, pretrained=not args.no_pretrained)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    log_every=args.log_every,
                    pos_weight=pos_weight,
                )
                metrics = evaluate_binary(model, eval_loader, device)
                row = {"dataset": dataset_name, "model": model_name, "fold": f"fold_{fold_idx}", **metrics}
                results.append(row)
                fold_rows.append(row)
                print(
                    f"  fold {fold_idx} metrics: accuracy={metrics['accuracy']:.4f} "
                    f"precision_class1={metrics['precision']:.4f} recall_class1={metrics['recall']:.4f} "
                    f"f1_class1={metrics['f1_class1']:.4f} roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
                )

            mean_metrics = average_metrics(fold_rows)
            results.append({"dataset": dataset_name, "model": model_name, "fold": "mean", **mean_metrics})
            print(
                f"Finished model={model_name} on dataset={dataset_name} in {time.perf_counter() - model_start:.1f}s "
                f"| mean accuracy={mean_metrics['accuracy']:.4f} mean precision_class1={mean_metrics['precision']:.4f} "
                f"mean recall_class1={mean_metrics['recall']:.4f} mean f1_class1={mean_metrics['f1_class1']:.4f} "
                f"mean roc_auc={mean_metrics['roc_auc']:.4f} mean pr_auc={mean_metrics['pr_auc']:.4f}"
            )

        print(f"Dataset={dataset_name} done in {time.perf_counter() - ds_start:.1f}s")

    save_results(args.reports_dir, results)
    print_table(results)
    print(f"\nSaved reports: {args.reports_dir / 'metrics.csv'} and {args.reports_dir / 'metrics.json'}")
    print(f"Total experiment time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
