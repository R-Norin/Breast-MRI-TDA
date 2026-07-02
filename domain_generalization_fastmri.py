"""
Domain generalization: train on ODELIA train/val and externally test on FastMRI.

It runs BOTH:
    1) image-only models
    2) TDA-fusion models

Backbones:
    r3d18, mc318, r2plus1d, swin_unetr

Selection:
    best checkpoint by ODELIA validation AUC

Threshold:
    tuned on ODELIA validation F1

Reports:
    default threshold 0.5
    F1-tuned threshold
    4-seed mean ± std for AUC, Accuracy, Sensitivity, Specificity, F1, Threshold
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ImageOnlyDataset, FusionDataset
from models import ImageOnlyModel, FusionModel
from models_swin import SwinImageOnlyModel, SwinFusionModel
from utils import (
    set_seed,
    compute_metrics,
    find_best_threshold_f1,
    train_one_epoch,
    predict,
)


# ============================================================
# DEFAULT PATHS
# ============================================================
DEFAULT_DATA_ROOT = Path("/groups/bcoskunuzer/R_Norin/breast_mri_final/domain_generalization/DG/dg_dataset")
DEFAULT_TDA_ROOT = Path("/groups/bcoskunuzer/R_Norin/breast_mri_final/domain_generalization/DG")

DEFAULT_TRAIN_IMG_ROOT = DEFAULT_DATA_ROOT / "train"
DEFAULT_VAL_IMG_ROOT = DEFAULT_DATA_ROOT / "val"
DEFAULT_TEST_IMG_ROOT = DEFAULT_DATA_ROOT / "FASTMRI_TEST"

DEFAULT_TRAIN_TDA_CSV = DEFAULT_TDA_ROOT / "dg_betti_train_2_class.csv"
DEFAULT_VAL_TDA_CSV = DEFAULT_TDA_ROOT / "dg_betti_val_2_class.csv"
DEFAULT_TEST_TDA_CSV = DEFAULT_TDA_ROOT / "dg_betti_fastmri_test_2_class.csv"

DEFAULT_OUTPUT_DIR = DEFAULT_TDA_ROOT / "dg_fastmri_all_image_and_fusion_models"

TARGET_NAME = "FastMRI"


# ============================================================
# ARGUMENTS
# ============================================================
def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_img_root", default=str(DEFAULT_TRAIN_IMG_ROOT))
    parser.add_argument("--val_img_root", default=str(DEFAULT_VAL_IMG_ROOT))
    parser.add_argument("--test_img_root", default=str(DEFAULT_TEST_IMG_ROOT))

    parser.add_argument("--train_tda_csv", default=str(DEFAULT_TRAIN_TDA_CSV))
    parser.add_argument("--val_tda_csv", default=str(DEFAULT_VAL_TDA_CSV))
    parser.add_argument("--test_tda_csv", default=str(DEFAULT_TEST_TDA_CSV))

    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument(
        "--models",
        default="r3d18,mc318,r2plus1d,swin_unetr",
        help="Comma-separated: r3d18,mc318,r2plus1d,swin_unetr",
    )

    parser.add_argument(
        "--run_types",
        default="image,fusion",
        help="Comma-separated: image,fusion",
    )

     p.add_argument("--seeds", default="32,42,52,62")
    p.add_argument("--batch_sizes", default="4,8,16,32")
    p.add_argument("--learning_rates", default="1e-4,5e-5,1e-5,1e-2,1e-3")
    p.add_argument("--weight_decays", default="1e-4,1e-3,1e-2")
    p.add_argument("--dropout_image", default="0.3,0.4,0.5")
    p.add_argument("--dropout_tda", default="0.3,0.4,0.5")
    p.add_argument("--dropout_fusion", default="0.3,0.4,0.5")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)

    # Optimizer defaults follow the modular BCE-style code.
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--use_scheduler", action="store_true")

    parser.add_argument(
        "--swin_pretrained_weights",
        default=None,
        help="Required if running swin_unetr.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def make_combinations(args):
    combos = []
    for bs, lr, wd, di, dt, df, do in itertools.product(
        parse_int_list(args.batch_sizes),
        parse_float_list(args.learning_rates),
        parse_float_list(args.weight_decays),
        parse_float_list(args.dropout_image),
        parse_float_list(args.dropout_tda),
        parse_float_list(args.dropout_fusion),
        parse_float_list(args.dropout_imageonly),
    ):
        combos.append(
            {
                "batch_size": bs,
                "learning_rate": lr,
                "weight_decay": wd,
                "dropout_image": di,
                "dropout_tda": dt,
                "dropout_fusion": df,
                "dropout_imageonly": do,
            }
        )
    return combos


def make_loader(dataset, batch_size, shuffle, args, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def build_image_model(model_name, combo, args):
    model_name = model_name.lower()

    if model_name == "swin_unetr":
        if args.swin_pretrained_weights is None:
            raise ValueError("swin_unetr requires --swin_pretrained_weights")
        return SwinImageOnlyModel(
            pretrained_weights_path=args.swin_pretrained_weights,
            dropout=combo["dropout_imageonly"],
        )

    return ImageOnlyModel(
        model_name=model_name,
        dropout=combo["dropout_imageonly"],
    )


def build_fusion_model(model_name, combo, tda_dim, args):
    model_name = model_name.lower()

    if model_name == "swin_unetr":
        if args.swin_pretrained_weights is None:
            raise ValueError("swin_unetr requires --swin_pretrained_weights")
        return SwinFusionModel(
            pretrained_weights_path=args.swin_pretrained_weights,
            dropout_image=combo["dropout_image"],
            dropout_tda=combo["dropout_tda"],
            dropout_fusion=combo["dropout_fusion"],
            tda_dim=tda_dim,
        )

    return FusionModel(
        model_name=model_name,
        dropout_image=combo["dropout_image"],
        dropout_tda=combo["dropout_tda"],
        dropout_fusion=combo["dropout_fusion"],
        tda_dim=tda_dim,
    )


def compute_pos_weight_image(train_ds, device):
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


def compute_pos_weight_fusion(train_ds, device):
    labels = [int(train_ds.labels[i]) for i, _ in train_ds.samples]
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


def build_optimizer(model, lr, wd, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def mean_std_string(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return f"{values.mean():.4f} ± 0.0000"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# DATASETS
# ============================================================
def build_datasets(run_type, args):
    if run_type == "image":
        train_ds = ImageOnlyDataset(args.train_img_root)
        val_ds = ImageOnlyDataset(args.val_img_root)
        test_ds = ImageOnlyDataset(args.test_img_root)
        return train_ds, val_ds, test_ds, None

    train_ds = FusionDataset(
        img_root=args.train_img_root,
        tda_csv=args.train_tda_csv,
        scaler=None,
        fit_scaler=True,
    )

    val_ds = FusionDataset(
        img_root=args.val_img_root,
        tda_csv=args.val_tda_csv,
        scaler=train_ds.scaler,
        fit_scaler=False,
    )

    test_ds = FusionDataset(
        img_root=args.test_img_root,
        tda_csv=args.test_tda_csv,
        scaler=train_ds.scaler,
        fit_scaler=False,
    )

    tda_dim = train_ds.tda.shape[1]

    return train_ds, val_ds, test_ds, tda_dim


# ============================================================
# ONE RUN
# ============================================================
def run_one(args, run_type, model_name, combo, combo_idx, seed):
    set_seed(seed)
    device = get_device()

    run_name = (
        f"{TARGET_NAME}_{run_type}_{model_name}"
        f"_combo{combo_idx}_seed{seed}"
        f"_bs{combo['batch_size']}"
        f"_lr{safe_name(combo['learning_rate'])}"
        f"_wd{safe_name(combo['weight_decay'])}"
    )

    if run_type == "image":
        run_name += f"_drop{safe_name(combo['dropout_imageonly'])}"
    else:
        run_name += (
            f"_di{safe_name(combo['dropout_image'])}"
            f"_dt{safe_name(combo['dropout_tda'])}"
            f"_df{safe_name(combo['dropout_fusion'])}"
        )

    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"TARGET={TARGET_NAME} | TYPE={run_type} | MODEL={model_name} | COMBO={combo_idx} | SEED={seed}")
    print(combo)
    print("Device:", device)
    print("=" * 100)

    train_ds, val_ds, test_ds, tda_dim = build_datasets(run_type, args)

    train_loader = make_loader(train_ds, combo["batch_size"], True, args, device)
    val_loader = make_loader(val_ds, combo["batch_size"], False, args, device)
    test_loader = make_loader(test_ds, combo["batch_size"], False, args, device)

    if run_type == "image":
        model = build_image_model(model_name, combo, args).to(device)
        pos_weight = compute_pos_weight_image(train_ds, device)
        fusion_flag = False
    else:
        model = build_fusion_model(model_name, combo, tda_dim, args).to(device)
        pos_weight = compute_pos_weight_fusion(train_ds, device)
        fusion_flag = True

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(
        model=model,
        lr=combo["learning_rate"],
        wd=combo["weight_decay"],
        args=args,
    )

    scheduler = None
    if args.use_scheduler:
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

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            fusion=fusion_flag,
        )

        _, y_val_epoch, p_val_epoch = predict(
            model=model,
            loader=val_loader,
            device=device,
            fusion=fusion_flag,
        )

        val_metrics = compute_metrics(y_val_epoch, p_val_epoch, threshold=0.5)
        val_auc = val_metrics["AUC"]

        if scheduler is not None:
            scheduler.step(val_auc)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        improved = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 0
            improved = True
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        history.append(
            {
                "Target": TARGET_NAME,
                "Run Type": run_type,
                "Model": model_name,
                "Combo": combo_idx,
                "Seed": seed,
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val AUC": val_auc,
                "Val Accuracy": val_metrics["Accuracy"],
                "Val F1": val_metrics["F1"],
                "Learning Rate": current_lr,
                "Epoch Time (s)": epoch_time,
                "Improved": improved,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={train_loss:.4f} | "
            f"Val AUC={val_auc:.4f} | "
            f"Val Acc={val_metrics['Accuracy']:.4f} | "
            f"Val F1={val_metrics['F1']:.4f} | "
            f"LR={current_lr:.2e} | "
            f"Time={epoch_time:.1f}s | "
            f"{'BEST' if improved else f'Patience {wait}/{args.patience}'}"
        )

        if wait >= args.patience:
            print("Early stopping.")
            break

    total_runtime_min = (time.time() - start_time) / 60.0

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    if not best_model_path.exists():
        raise RuntimeError("No best model was saved. Check validation AUC computation.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_ids, y_val, p_val = predict(
        model=model,
        loader=val_loader,
        device=device,
        fusion=fusion_flag,
    )

    test_ids, y_test, p_test = predict(
        model=model,
        loader=test_loader,
        device=device,
        fusion=fusion_flag,
    )

    best_threshold = find_best_threshold_f1(y_val, p_val)

    val_default_metrics = compute_metrics(y_val, p_val, threshold=0.5)
    val_tuned_metrics = compute_metrics(y_val, p_val, threshold=best_threshold)

    test_default_metrics = compute_metrics(y_test, p_test, threshold=0.5)
    test_tuned_metrics = compute_metrics(y_test, p_test, threshold=best_threshold)

    reversed_auc = compute_metrics(y_test, 1.0 - p_test, threshold=0.5)["AUC"]

    pd.DataFrame(
        {
            "ID": val_ids,
            "Label": y_val,
            "Probability_Malignant": p_val,
            "Probability_Benign": 1.0 - p_val,
            "Prediction_0.5": (p_val >= 0.5).astype(int),
            "Prediction_F1_Tuned": (p_val >= best_threshold).astype(int),
        }
    ).to_csv(run_dir / "odelia_val_predictions.csv", index=False)

    pd.DataFrame(
        {
            "ID": test_ids,
            "Label": y_test,
            "Probability_Malignant": p_test,
            "Probability_Benign": 1.0 - p_test,
            "Prediction_0.5": (p_test >= 0.5).astype(int),
            "Prediction_F1_Tuned": (p_test >= best_threshold).astype(int),
        }
    ).to_csv(run_dir / f"{TARGET_NAME.lower()}_external_predictions.csv", index=False)

    row = {
        "Target": TARGET_NAME,
        "Run Type": run_type,
        "Model": model_name,
        "Combo": combo_idx,
        "Seed": seed,
        "Batch Size": combo["batch_size"],
        "Learning Rate": combo["learning_rate"],
        "Weight Decay": combo["weight_decay"],
        "Dropout ImageOnly": combo["dropout_imageonly"],
        "Dropout Image": combo["dropout_image"],
        "Dropout TDA": combo["dropout_tda"],
        "Dropout Fusion": combo["dropout_fusion"],
        "Optimizer": args.optimizer,
        "Scheduler": bool(args.use_scheduler),
        "Best Val AUC": best_val_auc,
        "Best Epoch": best_epoch,
        "Best Threshold F1": best_threshold,
        "Runtime (min)": total_runtime_min,
        f"{TARGET_NAME} Reversed-score AUC": reversed_auc,
        "Run Dir": str(run_dir),
    }

    for k, v in val_default_metrics.items():
        row[f"ODELIA Val Default {k}"] = v
    for k, v in val_tuned_metrics.items():
        row[f"ODELIA Val F1 Tuned {k}"] = v
    for k, v in test_default_metrics.items():
        row[f"{TARGET_NAME} Default {k}"] = v
    for k, v in test_tuned_metrics.items():
        row[f"{TARGET_NAME} F1 Tuned {k}"] = v

    pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)

    save_json(
        {
            "target": TARGET_NAME,
            "run_type": run_type,
            "model": model_name,
            "combo": combo,
            "seed": seed,
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "best_threshold_f1": best_threshold,
            "runtime_min": total_runtime_min,
            "paths": {
                "train_img_root": args.train_img_root,
                "val_img_root": args.val_img_root,
                "test_img_root": args.test_img_root,
                "train_tda_csv": args.train_tda_csv,
                "val_tda_csv": args.val_tda_csv,
                "test_tda_csv": args.test_tda_csv,
            },
        },
        run_dir / "config.json",
    )

    print("\nRUN SUMMARY")
    print(pd.DataFrame([row]).to_string(index=False))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


# ============================================================
# AGGREGATION
# ============================================================
def make_mean_std_table(results):
    metric_cols = [
        f"{TARGET_NAME} F1 Tuned AUC",
        f"{TARGET_NAME} F1 Tuned Accuracy",
        f"{TARGET_NAME} F1 Tuned Sensitivity",
        f"{TARGET_NAME} F1 Tuned Specificity",
        f"{TARGET_NAME} F1 Tuned F1",
        f"{TARGET_NAME} F1 Tuned Threshold",
    ]

    group_cols = [
        "Target",
        "Run Type",
        "Model",
        "Combo",
        "Batch Size",
        "Learning Rate",
        "Weight Decay",
        "Dropout ImageOnly",
        "Dropout Image",
        "Dropout TDA",
        "Dropout Fusion",
        "Optimizer",
        "Scheduler",
    ]

    rows = []
    for keys, group in results.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for col in metric_cols:
            values = group[col].astype(float).values
            row[col + " Mean"] = float(np.mean(values))
            row[col + " Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[col + " Mean ± Std"] = mean_std_string(values)

        row["Num Seeds"] = len(group)
        rows.append(row)

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(f"{TARGET_NAME} F1 Tuned AUC Mean", ascending=False)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    args = get_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    models = parse_str_list(args.models)
    run_types = parse_str_list(args.run_types)
    seeds = parse_int_list(args.seeds)
    combos = make_combinations(args)

    all_rows = []

    for run_type in run_types:
        if run_type not in ["image", "fusion"]:
            raise ValueError(f"Unknown run type: {run_type}")

        for model_name in models:
            if model_name.lower() == "swin_unetr" and args.swin_pretrained_weights is None:
                print("Skipping swin_unetr because --swin_pretrained_weights was not provided.")
                continue

            for combo_idx, combo in enumerate(combos):
                for seed in seeds:
                    row = run_one(
                        args=args,
                        run_type=run_type,
                        model_name=model_name,
                        combo=combo,
                        combo_idx=combo_idx,
                        seed=seed,
                    )
                    all_rows.append(row)

                    partial = pd.DataFrame(all_rows)
                    partial.to_csv(Path(args.output_dir) / "all_results_partial.csv", index=False)

    results = pd.DataFrame(all_rows)

    if len(results) == 0:
        print("No successful runs.")
        return

    results = results.sort_values(
        ["Run Type", "Model", "Combo", "Best Val AUC"],
        ascending=[True, True, True, False],
    )

    all_results_path = Path(args.output_dir) / f"all_{TARGET_NAME.lower()}_domain_generalization_results.csv"
    results.to_csv(all_results_path, index=False)

    mean_std = make_mean_std_table(results)
    mean_std_path = Path(args.output_dir) / f"{TARGET_NAME.lower()}_mean_std_by_model_combo.csv"
    mean_std.to_csv(mean_std_path, index=False)

    print("\n" + "=" * 120)
    print(f"ALL {TARGET_NAME} DOMAIN GENERALIZATION RESULTS")
    print("=" * 120)
    print(results.to_string(index=False))

    print("\n" + "=" * 120)
    print(f"{TARGET_NAME} 4-SEED MEAN ± STD BY MODEL AND COMBO")
    print("=" * 120)
    print(mean_std.to_string(index=False))

    print("\nSaved:")
    print(all_results_path)
    print(mean_std_path)


if __name__ == "__main__":
    main()
