"""
Unified image-only training script for Breast MRI models.

Backbones:
    - r3d18
    - mc318
    - r2plus1d
    - swin_unetr

Features:
    - hyperparameter search over batch size, learning rate, weight decay, dropout
    - best checkpoint selected by validation AUC
    - threshold tuned on validation F1
    - 4 seed evaluation
    - mean ± std report for AUC, Accuracy, Sensitivity, Specificity, F1, Threshold

Expected dataset structure:
    DATA_ROOT/
        train/
            benign/
            malignant/
        val/
            benign/
            malignant/
        test/
            benign/
            malignant/

Each case folder contains:
    Pre.npy, Sub_1.npy, T2.npy for Odelia
    Pre.npy, Sub.npy, Post.npy for Fastmri, Breastdm
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ImageOnlyDataset
from models import ImageOnlyModel
from models_swin import SwinImageOnlyModel
from utils import (
    set_seed,
    compute_metrics,
    find_best_threshold_f1,
    train_one_epoch,
    predict,
)


# ============================================================
# ARGUMENTS
# ============================================================
def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument(
        "--models",
        default="r3d18,mc318,r2plus1d,swin_unetr",
        help="Comma-separated: r3d18, mc318, r2plus1d, swin_unetr",
    )

    p.add_argument("--seeds", default="32,42,52,62")
    p.add_argument("--batch_sizes", default="4,8,16,32")
    p.add_argument("--learning_rates", default="1e-4,5e-5,1e-5")
    p.add_argument("--weight_decays", default="1e-4,1e-3,1e-2")
    p.add_argument("--dropouts", default="0.3,0.4,0.5")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--swin_pretrained_weights",
        default=None,
        help="Required only when using swin_unetr.",
    )

    return p.parse_args()


# ============================================================
# HELPERS
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_combinations(batch_sizes, learning_rates, weight_decays, dropouts):
    combos = []
    for bs, lr, wd, do in itertools.product(
        batch_sizes,
        learning_rates,
        weight_decays,
        dropouts,
    ):
        combos.append(
            {
                "batch_size": bs,
                "learning_rate": lr,
                "weight_decay": wd,
                "dropout": do,
            }
        )
    return combos


def build_model(model_name, dropout, swin_pretrained_weights):
    model_name = model_name.lower()

    if model_name == "swin_unetr":
        if swin_pretrained_weights is None:
            raise ValueError("swin_unetr requires --swin_pretrained_weights")
        return SwinImageOnlyModel(
            pretrained_weights_path=swin_pretrained_weights,
            dropout=dropout,
        )

    return ImageOnlyModel(
        model_name=model_name,
        dropout=dropout,
    )


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def compute_pos_weight(train_ds, device):
    labels = [int(label) for _, label in train_ds.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)],
        dtype=torch.float32,
        device=device,
    )

    print("Train benign:", n_neg)
    print("Train malignant:", n_pos)
    print("pos_weight:", pos_weight.item())

    return pos_weight


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def mean_std_string(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f} ± 0.0000"


# ============================================================
# SINGLE RUN
# ============================================================
def run_one_seed(args, model_name, combo, seed, combo_idx):
    set_seed(seed)

    device = get_device()
    print("\n" + "=" * 100)
    print(f"MODEL={model_name} | COMBO={combo_idx} | SEED={seed}")
    print(combo)
    print("Device:", device)
    print("=" * 100)

    run_name = (
        f"{model_name}_combo{combo_idx}_seed{seed}"
        f"_bs{combo['batch_size']}"
        f"_lr{safe_name(combo['learning_rate'])}"
        f"_wd{safe_name(combo['weight_decay'])}"
        f"_drop{safe_name(combo['dropout'])}"
    )

    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ImageOnlyDataset(Path(args.data_root) / "train")
    val_ds = ImageOnlyDataset(Path(args.data_root) / "val")
    test_ds = ImageOnlyDataset(Path(args.data_root) / "test")

    train_loader = make_loader(
        train_ds,
        combo["batch_size"],
        True,
        args.num_workers,
        device,
    )
    val_loader = make_loader(
        val_ds,
        combo["batch_size"],
        False,
        args.num_workers,
        device,
    )
    test_loader = make_loader(
        test_ds,
        combo["batch_size"],
        False,
        args.num_workers,
        device,
    )

    model = build_model(
        model_name=model_name,
        dropout=combo["dropout"],
        swin_pretrained_weights=args.swin_pretrained_weights,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(train_ds, device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=combo["learning_rate"],
        weight_decay=combo["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_auc = -1.0
    best_epoch = -1
    wait = 0
    history = []
    best_model_path = run_dir / "best_model.pt"

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            fusion=False,
        )

        _, y_val_epoch, p_val_epoch = predict(
            model,
            val_loader,
            device,
            fusion=False,
        )

        val_auc = compute_metrics(y_val_epoch, p_val_epoch, threshold=0.5)["AUC"]

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val AUC": val_auc,
                "Learning Rate": current_lr,
                "Epoch Time (s)": time.time() - epoch_start,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={train_loss:.4f} | "
            f"Val AUC={val_auc:.4f} | "
            f"LR={current_lr:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model by validation AUC.")
        else:
            wait += 1
            print(f"Patience {wait}/{args.patience}")

            if wait >= args.patience:
                print("Early stopping.")
                break

    runtime_min = (time.time() - start) / 60.0

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_ids, y_val, p_val = predict(model, val_loader, device, fusion=False)
    test_ids, y_test, p_test = predict(model, test_loader, device, fusion=False)

    best_threshold = find_best_threshold_f1(y_val, p_val)

    default_metrics = compute_metrics(y_test, p_test, threshold=0.5)
    tuned_metrics = compute_metrics(y_test, p_test, threshold=best_threshold)

    pred_default = (p_test >= 0.5).astype(int)
    pred_tuned = (p_test >= best_threshold).astype(int)

    pd.DataFrame(
        {
            "ID": test_ids,
            "Label": y_test,
            "Probability_Malignant": p_test,
            "Prediction_0.5": pred_default,
            "Prediction_F1_Tuned": pred_tuned,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    row = {
        "Model": model_name,
        "Combo": combo_idx,
        "Seed": seed,
        "Batch Size": combo["batch_size"],
        "Learning Rate": combo["learning_rate"],
        "Weight Decay": combo["weight_decay"],
        "Dropout": combo["dropout"],
        "Best Val AUC": best_val_auc,
        "Best Epoch": best_epoch,
        "Best Threshold F1": best_threshold,
        "Default AUC": default_metrics["AUC"],
        "Default Accuracy": default_metrics["Accuracy"],
        "Default Sensitivity": default_metrics["Sensitivity"],
        "Default Specificity": default_metrics["Specificity"],
        "Default F1": default_metrics["F1"],
        "Tuned AUC": tuned_metrics["AUC"],
        "Tuned Accuracy": tuned_metrics["Accuracy"],
        "Tuned Sensitivity": tuned_metrics["Sensitivity"],
        "Tuned Specificity": tuned_metrics["Specificity"],
        "Tuned F1": tuned_metrics["F1"],
        "Tuned Threshold": tuned_metrics["Threshold"],
        "Runtime (min)": runtime_min,
        "Run Dir": str(run_dir),
    }

    pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)

    print("\nRUN SUMMARY")
    print(pd.DataFrame([row]).to_string(index=False))

    return row


# ============================================================
# MAIN
# ============================================================
def main():
    args = get_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)

    combos = make_combinations(
        batch_sizes=parse_int_list(args.batch_sizes),
        learning_rates=parse_float_list(args.learning_rates),
        weight_decays=parse_float_list(args.weight_decays),
        dropouts=parse_float_list(args.dropouts),
    )

    all_rows = []

    for model_name in models:
        for combo_idx, combo in enumerate(combos):
            for seed in seeds:
                row = run_one_seed(args, model_name, combo, seed, combo_idx)
                all_rows.append(row)

                partial_df = pd.DataFrame(all_rows)
                partial_df.to_csv(
                    Path(args.output_dir) / "all_results_partial.csv",
                    index=False,
                )

    results = pd.DataFrame(all_rows)

    results = results.sort_values(
        ["Model", "Combo", "Best Val AUC"],
        ascending=[True, True, False],
    )

    results.to_csv(Path(args.output_dir) / "all_results_sorted.csv", index=False)

    metric_cols = [
        "Tuned AUC",
        "Tuned Accuracy",
        "Tuned Sensitivity",
        "Tuned Specificity",
        "Tuned F1",
        "Tuned Threshold",
    ]

    group_cols = [
        "Model",
        "Combo",
        "Batch Size",
        "Learning Rate",
        "Weight Decay",
        "Dropout",
    ]

    summary_rows = []

    for keys, group in results.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for m in metric_cols:
            row[m + " Mean"] = group[m].mean()
            row[m + " Std"] = group[m].std(ddof=1)
            row[m + " Mean ± Std"] = mean_std_string(group[m].values)

        row["Num Seeds"] = len(group)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("Tuned AUC Mean", ascending=False)

    summary.to_csv(Path(args.output_dir) / "mean_std_by_model_combo.csv", index=False)

    print("\n" + "=" * 120)
    print("ALL IMAGE-ONLY RESULTS SORTED")
    print("=" * 120)
    print(results.to_string(index=False))

    print("\n" + "=" * 120)
    print("4-SEED MEAN ± STD BY MODEL AND COMBO")
    print("=" * 120)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
