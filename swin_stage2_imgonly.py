"""
Stage 2 — Final runs — Swin UNETR Image Only
Loads best hyperparameters from stage 1.
Trains on ODELIA train, validates on ODELIA val.
Tunes threshold by best F1 on val.
Evaluates on FastMRI and BreastDM test sets.

Usage:
    python swin_stage2_imgonly.py --seed 42
    python swin_stage2_imgonly.py --seed 32
    ...

Seeds: 32 | 42 | 52 | 62
"""

import sys
sys.path.append("shared")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset     import ImageOnlyDataset
from models_swin import SwinImageOnlyModel
from utils       import (
    set_seed,
    compute_metrics,
    find_best_threshold_f1,
    train_one_epoch,
    predict,
)

# ============================================================
# PATHS  —  update to match your cluster
# ============================================================
DATA_ROOT = Path("breast_mri/domain_generalization/DG/dg_dataset")

TRAIN_IMG_ROOT    = DATA_ROOT / "train"
VAL_IMG_ROOT      = DATA_ROOT / "val"
FASTMRI_IMG_ROOT  = DATA_ROOT / "FASTMRI_TEST"
BREASTDM_IMG_ROOT = DATA_ROOT / "BreastDM_TEST"

PRETRAINED_WEIGHTS = Path("breast_mri/domain_generalization/DG/pretrained_weights/swin_unetr_pretrained.pt")

STAGE1_OUT = Path("breast_mri/domain_generalization/DG/results/stage1")
STAGE2_OUT = Path("breast_mri/domain_generalization/DG/results/stage2")

# ============================================================
# FIXED SETTINGS
# ============================================================
EPOCHS      = 50
PATIENCE    = 10
NUM_WORKERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_params():
    path = STAGE1_OUT / "swin_imgonly" / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Best params not found: {path}\n"
            f"Run aggregate_stage1.py first."
        )
    with open(path) as f:
        return json.load(f)


def main(args):
    out_dir  = STAGE2_OUT / "swin_imgonly"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"seed_{args.seed}.json"
    if out_file.exists():
        print("Already done:", out_file)
        return

    params = load_best_params()
    print(f"Swin UNETR Image Only | Seed: {args.seed}")
    print(f"Best params from stage 1: {params}")
    print("Device:", DEVICE)

    set_seed(args.seed)

    # -------------------------
    # datasets
    # -------------------------
    train_ds    = ImageOnlyDataset(TRAIN_IMG_ROOT)
    val_ds      = ImageOnlyDataset(VAL_IMG_ROOT)
    fastmri_ds  = ImageOnlyDataset(FASTMRI_IMG_ROOT)
    breastdm_ds = ImageOnlyDataset(BREASTDM_IMG_ROOT)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset empty.")

    batch_size = params["batch_size"]

    train_loader    = DataLoader(train_ds,    batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    val_loader      = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    fastmri_loader  = DataLoader(fastmri_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    breastdm_loader = DataLoader(breastdm_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))

    # -------------------------
    # model
    # -------------------------
    model = SwinImageOnlyModel(
        pretrained_weights_path = PRETRAINED_WEIGHTS,
        dropout                 = params["dropout"],
    ).to(DEVICE)

    train_labels = [label for _, label, _ in train_ds]
    n_pos        = sum(train_labels)
    n_neg        = len(train_labels) - n_pos
    pos_weight   = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = params["lr"],
        weight_decay = params["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # -------------------------
    # training loop
    # -------------------------
    best_val_auc = -1.0
    best_epoch   = -1
    wait         = 0
    best_ckpt    = out_dir / f"best_swin_imgonly_seed{args.seed}.pt"

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, fusion=False)

        _, y_val, p_val = predict(model, val_loader, DEVICE, fusion=False)
        val_auc         = roc_auc_score(y_val, p_val)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            wait         = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    elapsed = (time.time() - start) / 60
    print(f"Training done | Best val AUC: {best_val_auc:.4f} | Best epoch: {best_epoch} | Time: {elapsed:.1f} min")

    # -------------------------
    # load best checkpoint
    # -------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    # -------------------------
    # tune threshold on val
    # -------------------------
    _, y_val, p_val = predict(model, val_loader, DEVICE, fusion=False)
    best_threshold  = find_best_threshold_f1(y_val, p_val)

    val_metrics = compute_metrics(y_val, p_val, threshold=best_threshold)
    print(f"Val metrics @ threshold {best_threshold:.2f}: {val_metrics}")

    # -------------------------
    # evaluate on both test sets
    # -------------------------
    _, y_fm,  p_fm  = predict(model, fastmri_loader,  DEVICE, fusion=False)
    _, y_bdm, p_bdm = predict(model, breastdm_loader, DEVICE, fusion=False)

    fastmri_metrics  = compute_metrics(y_fm,  p_fm,  threshold=best_threshold)
    breastdm_metrics = compute_metrics(y_bdm, p_bdm, threshold=best_threshold)

    print(f"FastMRI  AUC: {fastmri_metrics['AUC']:.4f}")
    print(f"BreastDM AUC: {breastdm_metrics['AUC']:.4f}")

    # -------------------------
    # save results
    # -------------------------
    result = {
        "model":            "swin_imgonly",
        "seed":             args.seed,
        "params":           params,
        "best_epoch":       best_epoch,
        "best_val_auc":     best_val_auc,
        "best_threshold":   best_threshold,
        "val_metrics":      val_metrics,
        "fastmri_metrics":  fastmri_metrics,
        "breastdm_metrics": breastdm_metrics,
        "time_min":         elapsed,
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[32, 42, 52, 62])
    main(parser.parse_args())
