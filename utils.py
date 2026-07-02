import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "AUC":         roc_auc_score(y_true, y_prob),
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Threshold":   float(threshold),
    }


def find_best_threshold_f1(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_t  = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        score  = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t  = float(t)

    return best_t


def train_one_epoch(model, loader, criterion, optimizer, device, fusion=False):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        if fusion:
            img, tda, y, _ = batch
            img = img.to(device)
            tda = tda.to(device)
            y   = y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(img, tda), y)
        else:
            img, y, _ = batch
            img = img.to(device)
            y   = y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(img), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * img.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, fusion=False):
    model.eval()

    ids    = []
    y_true = []
    y_prob = []

    for batch in tqdm(loader, desc="Predict", leave=False):
        if fusion:
            img, tda, y, case_ids = batch
            img = img.to(device)
            tda = tda.to(device)
            logits = model(img, tda)
        else:
            img, y, case_ids = batch
            img    = img.to(device)
            logits = model(img)

        prob = torch.sigmoid(logits).view(-1).cpu().numpy()

        ids.extend(case_ids)
        y_true.extend(y.numpy().astype(int).tolist())
        y_prob.extend(prob.tolist())

    return ids, np.array(y_true), np.array(y_prob)
