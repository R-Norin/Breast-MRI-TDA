"""
Stage 1 — Hyperparameter search — Swin UNETR Fusion (image + TDA)
Trains on ODELIA train, evaluates on ODELIA val.
Never touches test sets.

Usage:
    python swin_stage1_fusion.py --combo_idx 0
    python swin_stage1_fusion.py --combo_idx 1
    ...
    python swin_stage1_fusion.py --combo_idx 49

Combos: 0 to 49 (50 random combinations sampled from total combinations)
"""

import sys
sys.path.append("shared")

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset     import FusionDataset
from models_swin import SwinFusionModel
from utils       import set_seed, train_one_epoch, predict

# ============================================================
# PATHS  —  update to match your cluster
# ============================================================
DATA_ROOT = Path("breast_mri/domain_generalization/DG/dg_dataset")
TDA_ROOT  = Path("breast_mri/domain_generalization/DG")

TRAIN_IMG_ROOT = DATA_ROOT / "train"
VAL_IMG_ROOT   = DATA_ROOT / "val"

TRAIN_TDA_CSV = TDA_ROOT / "dg_betti_train_2_class.csv"
VAL_TDA_CSV   = TDA_ROOT / "dg_betti_val_2_class.csv"

PRETRAINED_WEIGHTS = Path("breast_mri/domain_generalization/DG/pretrained_weights/swin_unetr_pretrained.pt")

OUT_ROOT = Path("breast_mri/domain_generalization/DG/results/stage1")

# ============================================================
# FIXED SETTINGS
# ============================================================
SEARCH_SEED = 42
TRAIN_SEED  = 42
EPOCHS      = 50
PATIENCE    = 10
NUM_WORKERS = 8
TDA_DIM     = 450

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# SEARCH SPACE  →  total combinations
# ============================================================
SEARCH_SPACE = {
    "lr":           [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "dropout":      [0.2, 0.3, 0.4, 0.5],
    "batch_size":   [4, 8, 16, 32],
}

N_COMBOS = 50


def sample_combinations(n, seed):
    all_combos = [
        dict(zip(SEARCH_SPACE.keys(), v))
        for v in itertools.product(*SEARCH_SPACE.values())
    ]
    rng = random.Random(seed)
    return rng.sample(all_combos, n)


def main(args):
    combos = sample_combinations(N_COMBOS, SEARCH_SEED)
    params = combos[args.combo_idx]

    out_dir = OUT_ROOT / "swin_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"combo_{args.combo_idx}.json"
    if out_file.exists():
        print("Already done:", out_file)
        return

    print(f"Swin UNETR Fusion | Combo {args.combo_idx}/{N_COMBOS-1} | Params: {params}")
    print("Device:", DEVICE)

    set_seed(TRAIN_SEED)

    # -------------------------
    # datasets
    # -------------------------
    train_ds = FusionDataset(TRAIN_IMG_ROOT, TRAIN_TDA_CSV, fit_scaler=True)
    val_ds   = FusionDataset(VAL_IMG_ROOT,   VAL_TDA_CSV,   scaler=train_ds.scaler)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size  = params["batch_size"],
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = params["batch_size"],
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE.type == "cuda"),
    )

    # -------------------------
    # model
    # -------------------------
    model = SwinFusionModel(
        pretrained_weights_path = PRETRAINED_WEIGHTS,
        dropout                 = params["dropout"],
        tda_dim                 = TDA_DIM,
    ).to(DEVICE)

    train_labels = [int(train_ds.labels[i]) for i, _ in train_ds.samples]
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

    start = time.time()

    try:
        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, fusion=True)

            _, y_val, p_val = predict(model, val_loader, DEVICE, fusion=True)
            val_auc         = roc_auc_score(y_val, p_val)

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                wait         = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM — batch_size={params['batch_size']} — skipping.")
            torch.cuda.empty_cache()
            result = {
                "model":        "swin_fusion",
                "combo_idx":    args.combo_idx,
                "params":       params,
                "best_val_auc": -1.0,
                "best_epoch":   -1,
                "oom":          True,
                "time_min":     (time.time() - start) / 60,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            return
        raise

    elapsed = (time.time() - start) / 60
    print(f"Done | Val AUC: {best_val_auc:.4f} | Best epoch: {best_epoch} | Time: {elapsed:.1f} min")

    result = {
        "model":        "swin_fusion",
        "combo_idx":    args.combo_idx,
        "params":       params,
        "best_val_auc": best_val_auc,
        "best_epoch":   best_epoch,
        "oom":          False,
        "time_min":     elapsed,
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo_idx", type=int, required=True)
    main(parser.parse_args())
