"""
Run this after stage 2 completes.
Reads all seed JSON files per model, computes mean and std
across 5 seeds for all metrics on both test domains.
Saves final summary CSV.

Usage:
    python aggregate_stage2.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

STAGE2_OUT = Path("/groups/bcoskunuzer/R_Norin/breast_mri_final/domain_generalization/DG/results/stage2")

MODELS = [
    "r3d18_imgonly",
    "r3d18_fusion",
    "mc318_imgonly",
    "mc318_fusion",
    "r2plus1d_imgonly",
    "r2plus1d_fusion",
    "swin_imgonly",
    "swin_fusion",
]

SEEDS   = [22, 32, 42, 52, 62]
METRICS = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Threshold"]
DOMAINS = ["fastmri_metrics", "breastdm_metrics"]


def main():
    rows = []

    for model_key in MODELS:
        folder = STAGE2_OUT / model_key
        seed_results = []

        for seed in SEEDS:
            f = folder / f"seed_{seed}.json"
            if not f.exists():
                print(f"Missing: {f}")
                continue
            with open(f) as fp:
                seed_results.append(json.load(fp))

        if not seed_results:
            print(f"No results for {model_key}")
            continue

        row = {"model": model_key}

        for domain_key in DOMAINS:
            domain_label = "FastMRI" if "fastmri" in domain_key else "BreastDM"

            for metric in METRICS:
                values = [r[domain_key][metric] for r in seed_results]
                mean   = np.mean(values)
                std    = np.std(values)
                row[f"{domain_label}_{metric}_mean"] = round(mean, 4)
                row[f"{domain_label}_{metric}_std"]  = round(std,  4)
                row[f"{domain_label}_{metric}"]      = f"{mean:.4f} ± {std:.4f}"

        rows.append(row)

        print(f"\n{model_key}")
        for domain_label in ["FastMRI", "BreastDM"]:
            print(f"  {domain_label}:")
            for metric in METRICS:
                print(f"    {metric}: {row[f'{domain_label}_{metric}']}")

    df = pd.DataFrame(rows)
    out_file = STAGE2_OUT / "final_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved final results: {out_file}")


if __name__ == "__main__":
    main()
