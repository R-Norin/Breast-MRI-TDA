import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

MODALITIES = ["Pre.npy", "Sub.npy", "Post.npy"]

LABEL_MAP = {
    "benign":    0,
    "malignant": 1,
}


class ImageOnlyDataset(Dataset):
    def __init__(self, img_root):
        self.img_root = Path(img_root)
        self.samples  = []

        for cls_name, label in LABEL_MAP.items():
            cls_dir = self.img_root / cls_name

            if not cls_dir.exists():
                print("Missing class folder:", cls_dir)
                continue

            for case_dir in sorted(cls_dir.iterdir()):
                if not case_dir.is_dir():
                    continue

                ok = all((case_dir / m).exists() for m in MODALITIES)

                if ok:
                    self.samples.append((case_dir, label))
                else:
                    print("Missing modality:", case_dir)

        print(f"{self.img_root.name}: {len(self.samples)} cases")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_dir, label = self.samples[idx]

        vols = []
        for m in MODALITIES:
            vol = np.load(case_dir / m).astype(np.float32)
            if vol.ndim != 3:
                raise ValueError(f"{case_dir / m} shape {vol.shape}, expected (D,H,W)")
            vols.append(vol)

        img = np.stack(vols, axis=0).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)

        return (
            torch.tensor(img,   dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            case_dir.name,
        )


class FusionDataset(Dataset):
    def __init__(self, img_root, tda_csv, scaler=None, fit_scaler=False):
        self.img_root = Path(img_root)

        df           = pd.read_csv(tda_csv)
        feature_cols = [c for c in df.columns if c not in ["ID", "Label"]]

        self.ids    = df["ID"].astype(str).values
        self.labels = df["Label"].astype(int).values
        self.tda    = df[feature_cols].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            self.tda    = self.scaler.fit_transform(self.tda).astype(np.float32)
        else:
            self.scaler = scaler
            self.tda    = self.scaler.transform(self.tda).astype(np.float32)

        id_to_path = {}
        for cls_name in LABEL_MAP:
            cls_dir = self.img_root / cls_name
            if not cls_dir.exists():
                print("Missing class folder:", cls_dir)
                continue
            for case_dir in sorted(cls_dir.iterdir()):
                if case_dir.is_dir():
                    id_to_path[case_dir.name] = case_dir

        self.samples = []
        for i, uid in enumerate(self.ids):
            if uid not in id_to_path:
                print("Missing image folder:", uid)
                continue
            case_dir = id_to_path[uid]
            if all((case_dir / m).exists() for m in MODALITIES):
                self.samples.append((i, case_dir))
            else:
                print("Missing modality:", case_dir)

        print(f"{self.img_root.name}: {len(self.samples)} matched fusion cases")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, case_dir = self.samples[idx]

        vols = []
        for m in MODALITIES:
            vol = np.load(case_dir / m).astype(np.float32)
            if vol.ndim != 3:
                raise ValueError(f"{case_dir / m} shape {vol.shape}, expected (D,H,W)")
            vols.append(vol)

        img = np.stack(vols, axis=0).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)

        return (
            torch.tensor(img,                      dtype=torch.float32),
            torch.tensor(self.tda[row_idx],        dtype=torch.float32),
            torch.tensor(self.labels[row_idx],     dtype=torch.float32),
            case_dir.name,
        )
