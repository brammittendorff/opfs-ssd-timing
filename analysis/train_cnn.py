"""1D-CNN fingerprinter (paper-style) on the binned multi-channel contention trace.

Input per recording: [C, T] where channels = throughput / p95 / zero_frac / median
(see features.CNN_CHANNELS), T = window / bin. A few Conv1d->ReLU->pool blocks ->
global average pool -> linear. Prints validation accuracy, per-class accuracy, and
a confusion matrix.

Requires PyTorch (`pip install torch`). Use train_baseline.py first to confirm
the classes are separable before spending epochs here.

Usage:
    python3 train_cnn.py dataset.jsonl [--bin-ms 100] [--epochs 80]
"""

import argparse
import numpy as np

from dataset import load_jsonl, labels
from features import build_tensor, CNN_CHANNELS

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not installed. Run: pip install torch")


class CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


def normalize(X):
    """Per-channel z-score across the dataset."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--bin-ms", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    recs = load_jsonl(args.dataset)
    classes = labels(recs)
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(recs)} recordings, {len(classes)} classes: {classes}")
    print(f"channels: {CNN_CHANNELS}")

    X, y, T = build_tensor(recs, bin_ms=args.bin_ms)
    X = normalize(np.nan_to_num(X)).astype(np.float32)
    yi = np.array([cls_idx[v] for v in y])
    print(f"tensor X={X.shape} (N, C, T={T})")

    # Stratified-ish split: shuffle with fixed seed, take val_frac per class.
    rng = np.random.default_rng(0)
    train_idx, val_idx = [], []
    for c in range(len(classes)):
        ids = np.where(yi == c)[0]
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * args.val_frac)))
        val_idx += list(ids[:k]); train_idx += list(ids[k:])
    if not train_idx or not val_idx:
        raise SystemExit("Not enough data to split - capture more recordings per class.")

    Xtr = torch.tensor(X[train_idx]); ytr = torch.tensor(yi[train_idx])
    Xva = torch.tensor(X[val_idx]);   yva = torch.tensor(yi[val_idx])

    model = CNN1D(X.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xva).argmax(1) == yva).float().mean().item()
            print(f"epoch {ep+1:3d}  loss={loss.item():.3f}  val_acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1).numpy()
    true = yva.numpy()
    acc = (pred == true).mean()
    chance = max((yi == c).mean() for c in range(len(classes)))
    print(f"\nfinal val accuracy: {acc:.3f}  (chance ~ {chance:.3f})")
    print("confusion matrix (rows=true, cols=pred):")
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print("        " + "  ".join(f"{c[:7]:>7}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c[:7]:>7} " + "  ".join(f"{v:7d}" for v in row))


if __name__ == "__main__":
    main()
