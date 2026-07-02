"""
Run this after stage 1 completes.
Reads all combo JSON files per model, picks best by val AUC,
saves best_params.json in each model folder.

Usage:
    python aggregate_stage1.py
"""

import json
import glob
from pathlib import Path

STAGE1_OUT = Path("/groups/bcoskunuzer/R_Norin/breast_mri_final/domain_generalization/DG/results/stage1")

MODELS = [
    "r3d18_imgonly",
    "r3d18_fusion",
    "mc318_imgonly",
    "mc318_fusion",
    "r2plus1d_imgonly",
    "r2plus1d_fusion",
]


def main():
    for model_key in MODELS:
        folder = STAGE1_OUT / model_key
        files  = sorted(folder.glob("combo_*.json"))

        if not files:
            print(f"No results found for {model_key} — skipping.")
            continue

        results = []
        for f in files:
            with open(f) as fp:
                results.append(json.load(fp))

        # filter out OOM runs
        valid = [r for r in results if not r.get("oom", False)]

        if not valid:
            print(f"All combos OOM for {model_key} — skipping.")
            continue

        # sort by best val AUC
        valid.sort(key=lambda x: x["best_val_auc"], reverse=True)

        best = valid[0]

        print(f"\n{model_key}")
        print(f"  Best val AUC : {best['best_val_auc']:.4f}")
        print(f"  Best epoch   : {best['best_epoch']}")
        print(f"  Best params  : {best['params']}")

        # save best params
        out_file = folder / "best_params.json"
        with open(out_file, "w") as f:
            json.dump(best["params"], f, indent=2)

        print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
