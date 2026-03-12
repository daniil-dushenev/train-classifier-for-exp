#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from build_datasets import SYNTH_RATIOS, build_datasets, ratio_to_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build datasets and train/evaluate ResNet + ViT on no_synth and with_synth."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("data_out"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--train-pos-count",
        type=int,
        default=None,
        help="If set, put exactly N real class1 images into train and the rest into test.",
    )
    parser.add_argument(
        "--use-all-negatives-train",
        action="store_true",
        help="Use all class0 images in train (primarily for train-pos-count mode).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Manual pos_weight for class 1 in BCEWithLogitsLoss. If omitted, computed from train split.",
    )
    parser.add_argument("--extra-test-class0", type=Path, default=None,
                        help="Extra folder with class0 images added only to test split.")
    parser.add_argument("--extra-test-class1", type=Path, default=None,
                        help="Extra folder with class1 images added only to test split.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print train progress every N batches (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds. 1 = single train/test split (default). "
             "When >1, MT_Free/MT_Crack images are still always test-only.",
    )
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


def make_loaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(dataset_root / "train", transform=train_tf, allow_empty=True)
    test_ds = datasets.ImageFolder(dataset_root / "test", transform=eval_tf, allow_empty=True)
    dropped_train = drop_corrupted_samples(train_ds)
    dropped_test = drop_corrupted_samples(test_ds)
    if dropped_train or dropped_test:
        print(f"  dropped corrupted files: train={dropped_train}, test={dropped_test}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def drop_corrupted_samples(dataset: datasets.ImageFolder) -> int:
    valid_samples: list[tuple[str, int]] = []
    dropped = 0
    for path, target in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_samples.append((path, target))
        except Exception:
            dropped += 1
    if dropped:
        dataset.samples = valid_samples
        dataset.imgs = valid_samples
        dataset.targets = [t for _, t in valid_samples]
    return dropped


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
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
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
        epoch_denom = 2 * running_tp + running_fp + running_fn
        epoch_f1_class1 = (2 * running_tp / epoch_denom) if epoch_denom > 0 else 0.0
        print(
            f"  epoch {epoch}/{epochs} done in {epoch_time:.1f}s "
            f"| loss={epoch_loss:.4f} acc={epoch_acc:.4f} f1_class1={epoch_f1_class1:.4f}"
        )


def evaluate_binary(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    num_batches = len(test_loader)
    print(f"  evaluating on {len(test_loader.dataset)} samples ({num_batches} batches)")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, start=1):
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

    metrics: dict[str, Any] = {
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
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def class_stats_from_loader(train_loader: DataLoader) -> tuple[int, int]:
    targets = getattr(train_loader.dataset, "targets", None)
    if targets is None:
        raise ValueError("Could not read targets from train dataset to compute class weights.")
    class0_count = sum(1 for t in targets if int(t) == 0)
    class1_count = sum(1 for t in targets if int(t) == 1)
    return class0_count, class1_count


def save_results(reports_dir: Path, rows: list[dict[str, Any]], filename_stem: str = "metrics") -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{filename_stem}.json"
    csv_path = reports_dir / f"{filename_stem}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        keys = list(rows[0].keys())
    else:
        keys = []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average_cv_results(all_fold_results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Average numeric metrics across folds; return mean and std for each."""
    n_folds = len(all_fold_results)
    n_rows = len(all_fold_results[0])
    averaged: list[dict[str, Any]] = []
    for row_idx in range(n_rows):
        first = all_fold_results[0][row_idx]
        avg_row: dict[str, Any] = {k: v for k, v in first.items() if not isinstance(v, (int, float))}
        numeric_keys = [k for k, v in first.items() if isinstance(v, (int, float))]
        for k in numeric_keys:
            vals = [all_fold_results[f][row_idx][k] for f in range(n_folds)]
            avg_row[k] = float(np.mean(vals))
            avg_row[f"{k}_std"] = float(np.std(vals))
        averaged.append(avg_row)
    return averaged


def print_table(rows: list[dict[str, Any]], extra_headers: list[str] | None = None) -> None:
    base_headers = ["dataset", "model", "accuracy", "precision", "recall", "f1_class1", "roc_auc", "pr_auc"]
    headers = base_headers + (extra_headers or [])
    headers = [h for h in headers if h in (rows[0] if rows else {})]
    print("\nResults:")
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        values: list[str] = []
        for k in headers:
            value = row.get(k, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def run_fold(
    args: argparse.Namespace,
    device: torch.device,
    dataset_names: list[str],
    fold_idx: int,
) -> list[dict[str, Any]]:
    set_seed(args.seed)
    build_start = time.perf_counter()
    ds_stats = build_datasets(
        input_root=args.input_root,
        output_root=args.datasets_root,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_pos_count=args.train_pos_count,
        use_all_negatives_train=args.use_all_negatives_train,
        extra_test_class0=args.extra_test_class0,
        extra_test_class1=args.extra_test_class1,
        n_folds=args.n_folds,
        fold_idx=fold_idx,
    )
    build_time = time.perf_counter() - build_start
    print(f"Datasets built in {build_time:.1f}s: {ds_stats}")

    results: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        ds_start = time.perf_counter()
        dataset_root = args.datasets_root / dataset_name
        print(f"\nPreparing loaders for dataset={dataset_name}: {dataset_root}")
        train_loader, test_loader = make_loaders(dataset_root, args.batch_size, args.num_workers)
        class0_count, class1_count = class_stats_from_loader(train_loader)
        auto_pos_weight = (class0_count / class1_count) if class1_count > 0 else None
        pos_weight = args.pos_weight if args.pos_weight is not None else auto_pos_weight
        print(
            f"  train samples={len(train_loader.dataset)} batches={len(train_loader)} "
            f"| test samples={len(test_loader.dataset)} batches={len(test_loader)}"
        )
        if pos_weight is None:
            print("  class weights: disabled (no class1 samples in train)")
        else:
            source = "manual" if args.pos_weight is not None else "auto"
            print(
                f"  class weights: class0={class0_count} class1={class1_count} "
                f"-> pos_weight(class1)={pos_weight:.4f} ({source})"
            )
        for model_name in ["resnet18"]:
            model_start = time.perf_counter()
            print(f"\nTraining model={model_name} on dataset={dataset_name}")
            model = create_model(model_name=model_name, pretrained=not args.no_pretrained)
            train_model(model, train_loader, device, args.epochs, args.lr, args.log_every, pos_weight)
            metrics = evaluate_binary(model, test_loader, device)
            results.append({"dataset": dataset_name, "model": model_name, **metrics})
            model_time = time.perf_counter() - model_start
            print(
                f"Finished model={model_name} on dataset={dataset_name} in {model_time:.1f}s "
                f"| accuracy={metrics['accuracy']:.4f} f1_class1={metrics['f1_class1']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}"
            )
        ds_time = time.perf_counter() - ds_start
        print(f"Dataset={dataset_name} done in {ds_time:.1f}s")
    return results


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    device = pick_device(args.device)

    print("Experiment config:")
    print(
        {
            "input_root": str(args.input_root),
            "datasets_root": str(args.datasets_root),
            "reports_dir": str(args.reports_dir),
            "test_ratio": args.test_ratio,
            "train_pos_count": args.train_pos_count,
            "use_all_negatives_train": args.use_all_negatives_train,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "pos_weight": args.pos_weight,
            "log_every": args.log_every,
            "device_arg": args.device,
            "pretrained": not args.no_pretrained,
        }
    )
    print(f"Using device: {device}")

    dataset_names = ["no_synth"] + [ratio_to_name(r) for r in SYNTH_RATIOS]

    if args.n_folds > 1:
        all_fold_results: list[list[dict[str, Any]]] = []
        for fold_idx in range(args.n_folds):
            print(f"\n{'='*60}")
            print(f"=== Fold {fold_idx + 1}/{args.n_folds} ===")
            print(f"{'='*60}")
            fold_results = run_fold(args, device, dataset_names, fold_idx)
            all_fold_results.append(fold_results)

            # Save per-fold results incrementally
            per_fold_flat = [
                {"fold": fi, **row}
                for fi, fr in enumerate(all_fold_results)
                for row in fr
            ]
            save_results(args.reports_dir, per_fold_flat, filename_stem="metrics_per_fold")

        avg_results = average_cv_results(all_fold_results)
        save_results(args.reports_dir, avg_results, filename_stem="metrics_cv")
        print_table(avg_results, extra_headers=["f1_class1_std", "roc_auc_std", "pr_auc_std"])
        print(
            f"\nSaved CV reports: "
            f"{args.reports_dir / 'metrics_cv.csv'}, "
            f"{args.reports_dir / 'metrics_per_fold.csv'}"
        )
    else:
        results = run_fold(args, device, dataset_names, fold_idx=0)
        save_results(args.reports_dir, results)
        print_table(results)
        print(f"\nSaved reports: {args.reports_dir / 'metrics.csv'} and {args.reports_dir / 'metrics.json'}")

    total_time = time.perf_counter() - total_start
    print(f"Total experiment time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
