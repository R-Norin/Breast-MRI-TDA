#!/usr/bin/env python3
"""
train_tda.py

TDA-only pipeline for Breast MRI.

This script can:
  1) Extract persistent-homology Betti-curve features from 3D MRI volumes.
  2) Train TDA-only ML classifiers.
  3) Tune the decision threshold on validation F1.
  4) Report 4-seed mean ± std for AUC, Accuracy, Sensitivity, Specificity, F1, and Threshold.

Expected image folder structure:

DATA_ROOT/
    train/
        benign/ or class_0/
        malignant/ or class_1/
    val/
        benign/ or class_0/
        malignant/ or class_1/
    test/
        benign/ or class_0/
        malignant/ or class_1/

Each case folder should contain the selected modalities, for example:
    Pre.npy, Sub.npy, Post.npy

Generated TDA CSV format:
    ID, <features...>, Label
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_SEEDS = [32, 42, 52, 62]

DEFAULT_LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
    "Benign": 0,
    "Malignant": 1,
    "class_0": 0,
    "class_1": 1,
}

DEFAULT_MODALITIES = ["Pre.npy", "Sub.npy", "Post.npy"]


# ============================================================
# ARGUMENTS
# ============================================================
def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["extract", "train", "all"],
        default="train",
        help="extract = create Betti CSVs, train = train TDA models from CSVs, all = extract then train",
    )

    # Feature extraction inputs/outputs
    p.add_argument("--data_root", default=None, help="Root containing train/val/test image folders.")
    p.add_argument("--modalities", default="Pre.npy,Sub.npy,Post.npy")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--train_csv", default="tda_train.csv")
    p.add_argument("--val_csv", default="tda_val.csv")
    p.add_argument("--test_csv", default="tda_test.csv")

    # TDA extraction settings
    p.add_argument("--n_bins", type=int, default=50)
    p.add_argument("--homology_dims", default="0,1,2")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument(
        "--transpose_hwd_to_dhw",
        action="store_true",
        help="Use if your .npy volumes are stored as H,W,D and you want D,H,W before PH.",
    )

    # Training settings
    p.add_argument("--output_dir", default="tda_results")
    p.add_argument("--models", default="xgb,mlp", help="Comma-separated: xgb,mlp")
    p.add_argument("--seeds", default="32,42,52,62")
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")

    return p.parse_args()


# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# BETTI FEATURE EXTRACTION
# ============================================================
def make_column_names(modality_names, homology_dims, n_bins):
    columns = ["ID"]

    for modality in modality_names:
        clean_name = Path(modality).stem
        for h in homology_dims:
            for b in range(n_bins):
                columns.append(f"{clean_name}_H{h}_B{b}")

    columns.append("Label")
    return columns


def compute_betti_features(volume, homology_dims, n_bins):
    """
    Compute Betti-curve features for one 3D volume.

    Output dimension:
        len(homology_dims) * n_bins
    """
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import BettiCurve

    volume = volume.astype(np.float32, copy=False)

    cp = CubicalPersistence(
        homology_dimensions=homology_dims,
        n_jobs=1,
    )

    bc = BettiCurve(n_bins=n_bins)

    diagrams = cp.fit_transform(volume[None, ...])
    betti = bc.fit_transform(diagrams)

    return betti.reshape(-1).astype(np.float32)


def collect_cases(split_dir, label_map, modalities):
    split_dir = Path(split_dir)
    cases = []

    for class_name, label in label_map.items():
        class_dir = split_dir / class_name

        if not class_dir.exists():
            continue

        for case_dir in sorted(class_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            paths = [case_dir / m for m in modalities]

            if all(p.exists() for p in paths):
                cases.append((case_dir, case_dir.name, int(label)))
            else:
                print("Missing modality:", case_dir)

    return cases


def process_case(case_dir, case_id, label, modalities, homology_dims, n_bins, transpose_hwd_to_dhw):
    features_all = []

    for modality in modalities:
        path = case_dir / modality

        volume = np.load(path).astype(np.float32)

        if volume.ndim != 3:
            raise ValueError(f"{path} has shape {volume.shape}; expected a 3D volume.")

        if transpose_hwd_to_dhw:
            volume = np.transpose(volume, (2, 0, 1))

        features = compute_betti_features(
            volume=volume,
            homology_dims=homology_dims,
            n_bins=n_bins,
        )

        features_all.extend(features.tolist())

    return [case_id] + features_all + [label]


def extract_split(data_root, split, out_csv, modalities, label_map, homology_dims, n_bins, n_jobs, transpose_hwd_to_dhw):
    split_dir = Path(data_root) / split
    out_csv = Path(out_csv)

    print("\n" + "=" * 80)
    print(f"Extracting TDA features: {split}")
    print("Input:", split_dir)
    print("Output:", out_csv)
    print("=" * 80)

    cases = collect_cases(split_dir, label_map, modalities)
    print(f"Found {len(cases)} valid cases.")

    if len(cases) == 0:
        raise RuntimeError(f"No valid cases found in {split_dir}")

    rows = Parallel(n_jobs=n_jobs)(
        delayed(process_case)(
            case_dir=case_dir,
            case_id=case_id,
            label=label,
            modalities=modalities,
            homology_dims=homology_dims,
            n_bins=n_bins,
            transpose_hwd_to_dhw=transpose_hwd_to_dhw,
        )
        for case_dir, case_id, label in tqdm(cases, desc=f"Betti {split}")
    )

    columns = make_column_names(modalities, homology_dims, n_bins)
    df = pd.DataFrame(rows, columns=columns)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("Shape:", df.shape)
    print("Label counts:")
    print(df["Label"].value_counts().sort_index())

    return df


def run_extraction(args):
    if args.data_root is None:
        raise ValueError("--data_root is required for --mode extract or --mode all")

    modalities = parse_str_list(args.modalities)
    splits = parse_str_list(args.splits)
    homology_dims = parse_int_list(args.homology_dims)

    split_to_csv = {
        "train": args.train_csv,
        "val": args.val_csv,
        "test": args.test_csv,
    }

    for split in splits:
        if split not in split_to_csv:
            raise ValueError(f"Unknown split: {split}. Expected train, val, or test.")

        extract_split(
            data_root=args.data_root,
            split=split,
            out_csv=split_to_csv[split],
            modalities=modalities,
            label_map=DEFAULT_LABEL_MAP,
            homology_dims=homology_dims,
            n_bins=args.n_bins,
            n_jobs=args.n_jobs,
            transpose_hwd_to_dhw=args.transpose_hwd_to_dhw,
        )


# ============================================================
# TDA TRAINING
# ============================================================
def load_tda_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "ID" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"{csv_path} must contain ID and Label columns.")

    feature_cols = [c for c in df.columns if c not in ["ID", "Label"]]

    ids = df["ID"].astype(str).values
    x = df[feature_cols].values.astype(np.float32)
    y = df["Label"].astype(int).values

    return ids, x, y, feature_cols


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall/Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": float(threshold),
    }


def find_best_threshold_f1(y_true, y_prob, thresholds):
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t


def get_xgb_combinations():
    if XGBClassifier is None:
        return []

    search_space = {
        "n_estimators": [100, 200, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 10.0],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def get_mlp_combinations():
    return [
        {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "alpha": 1e-3,
            "learning_rate_init": 1e-4,
            "batch_size": 32,
        },
        {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "alpha": 1e-2,
            "learning_rate_init": 5e-4,
            "batch_size": 64,
        },
        {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "alpha": 1e-3,
            "learning_rate_init": 1e-4,
            "batch_size": 32,
        },
    ]


def build_model(model_name, params, seed):
    model_name = model_name.lower()

    if model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        random_state=seed,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        **params,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown TDA model: {model_name}")


def predict_probability(model, model_name, x):
    proba = model.predict_proba(x)
    return proba[:, 1]


def run_one_model_seed(model_name, params, seed, train_data, val_data, test_data, thresholds, output_dir, combo_idx):
    set_seed(seed)

    train_ids, x_train, y_train, feature_cols = train_data
    val_ids, x_val, y_val, _ = val_data
    test_ids, x_test, y_test, _ = test_data

    run_name = f"{model_name}_combo{combo_idx}_seed{seed}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TDA model={model_name} | combo={combo_idx} | seed={seed}")
    print(params)
    print("=" * 80)

    start = time.time()

    model = build_model(model_name, params, seed)
    model.fit(x_train, y_train)

    runtime_min = (time.time() - start) / 60.0

    p_val = predict_probability(model, model_name, x_val)
    p_test = predict_probability(model, model_name, x_test)

    best_threshold = find_best_threshold_f1(y_val, p_val, thresholds)

    val_default = compute_metrics(y_val, p_val, 0.5)
    val_tuned = compute_metrics(y_val, p_val, best_threshold)
    test_default = compute_metrics(y_test, p_test, 0.5)
    test_tuned = compute_metrics(y_test, p_test, best_threshold)

    pd.DataFrame(
        {
            "ID": val_ids,
            "Label": y_val,
            "Probability_Malignant": p_val,
            "Probability_Benign": 1.0 - p_val,
            "Prediction_0.5": (p_val >= 0.5).astype(int),
            "Prediction_F1_Tuned": (p_val >= best_threshold).astype(int),
        }
    ).to_csv(run_dir / "val_predictions.csv", index=False)

    pd.DataFrame(
        {
            "ID": test_ids,
            "Label": y_test,
            "Probability_Malignant": p_test,
            "Probability_Benign": 1.0 - p_test,
            "Prediction_0.5": (p_test >= 0.5).astype(int),
            "Prediction_F1_Tuned": (p_test >= best_threshold).astype(int),
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    row = {
        "Model": model_name,
        "Combo": combo_idx,
        "Seed": seed,
        "Params": json.dumps(params),
        "Best Val AUC": val_default["AUC"],
        "Best Threshold F1": best_threshold,
        "Runtime (min)": runtime_min,
        "Run Dir": str(run_dir),
    }

    for k, v in val_default.items():
        row[f"Val Default {k}"] = v
    for k, v in val_tuned.items():
        row[f"Val F1 Tuned {k}"] = v
    for k, v in test_default.items():
        row[f"Test Default {k}"] = v
    for k, v in test_tuned.items():
        row[f"Test F1 Tuned {k}"] = v

    pd.DataFrame([row]).to_csv(run_dir / "summary.csv", index=False)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "combo": combo_idx,
                "seed": seed,
                "params": params,
                "best_val_auc": val_default["AUC"],
                "best_threshold_f1": best_threshold,
                "runtime_min": runtime_min,
            },
            f,
            indent=2,
        )

    print(pd.DataFrame([row]).to_string(index=False))

    return row


def mean_std_string(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return f"{values.mean():.4f} ± 0.0000"
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def make_mean_std_table(results):
    metric_cols = [
        "Test F1 Tuned AUC",
        "Test F1 Tuned Accuracy",
        "Test F1 Tuned Sensitivity",
        "Test F1 Tuned Specificity",
        "Test F1 Tuned F1",
        "Test F1 Tuned Threshold",
    ]

    group_cols = ["Model", "Combo", "Params"]

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
        summary = summary.sort_values("Test F1 Tuned AUC Mean", ascending=False)

    return summary


def run_training(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading TDA CSVs...")
    train_data = load_tda_csv(args.train_csv)
    val_data = load_tda_csv(args.val_csv)
    test_data = load_tda_csv(args.test_csv)

    print("Train:", args.train_csv, train_data[1].shape)
    print("Val:", args.val_csv, val_data[1].shape)
    print("Test:", args.test_csv, test_data[1].shape)

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    thresholds = [float(x) for x in parse_str_list(args.thresholds)]

    all_rows = []

    for model_name in models:
        if model_name == "xgb":
            combos = get_xgb_combinations()
        elif model_name == "mlp":
            combos = get_mlp_combinations()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if len(combos) == 0:
            print(f"Skipping {model_name}: no combinations available.")
            continue

        for combo_idx, params in enumerate(combos):
            for seed in seeds:
                row = run_one_model_seed(
                    model_name=model_name,
                    params=params,
                    seed=seed,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    thresholds=thresholds,
                    output_dir=output_dir,
                    combo_idx=combo_idx,
                )
                all_rows.append(row)

                pd.DataFrame(all_rows).to_csv(output_dir / "all_results_partial.csv", index=False)

    results = pd.DataFrame(all_rows)

    if len(results) == 0:
        print("No successful TDA runs.")
        return

    results = results.sort_values(["Model", "Best Val AUC"], ascending=[True, False])
    results.to_csv(output_dir / "all_tda_results_sorted.csv", index=False)

    mean_std = make_mean_std_table(results)
    mean_std.to_csv(output_dir / "tda_mean_std_by_model_combo.csv", index=False)

    print("\n" + "=" * 100)
    print("ALL TDA RESULTS SORTED BY VALIDATION AUC")
    print("=" * 100)
    print(results.to_string(index=False))

    print("\n" + "=" * 100)
    print("TDA 4-SEED MEAN ± STD")
    print("=" * 100)
    print(mean_std.to_string(index=False))

    print("\nSaved:")
    print(output_dir / "all_tda_results_sorted.csv")
    print(output_dir / "tda_mean_std_by_model_combo.csv")


# ============================================================
# MAIN
# ============================================================
def main():
    args = get_args()

    if args.mode in ["extract", "all"]:
        run_extraction(args)

    if args.mode in ["train", "all"]:
        run_training(args)


if __name__ == "__main__":
    main()
