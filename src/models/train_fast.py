"""
train_fast.py
-------------
CPU-optimised ML-KEM profiling attack trainer.
Designed to complete in under 15 minutes on a restricted workstation
(Intel Core i5/i7 7th gen, no GPU, 8-16 GB RAM).

Architecture: Compressed MLP with PCA dimensionality reduction
- PCA reduces 6144 bits → 256 components  (done once, cached)
- MLP trains on 256-dim input — 50× faster per batch than Conv1D on 6144
- Same statistical evaluation and z-test as the full ResNet pipeline
- Results are directly comparable to v1 baseline (50.17%)

This is the correct first step: establish whether ANY signal exists
before committing to expensive GPU training.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils.dataset_loader import make_loaders, label_msb

BASELINE = 0.5017


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ML-KEM Fast CPU Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",      default="ml_kem_512",
                   choices=["ml_kem_512", "ml_kem_768", "ml_kem_1024"])
    p.add_argument("--ct-bits",      type=int, default=6144)
    p.add_argument("--n-components", type=int, default=256,
                   help="PCA output dimensions (lower = faster)")
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--batch",        type=int, default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--data-dir",     default="data/raw")
    p.add_argument("--cache-dir",    default="data/processed")
    p.add_argument("--results-dir",  default="results")
    p.add_argument("--samples",      type=int, default=50000,
                   help="Samples to use (reduce if still slow, min 10000)")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


# ── Fast MLP (trains on PCA-reduced input) ────────────────────────────────────
class FastMLP(nn.Module):
    """
    3-layer MLP on PCA-compressed input.
    Forward pass: ~0.1ms per batch on CPU vs ~50ms for Conv1D on raw bits.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── Statistical significance test ─────────────────────────────────────────────
def binomial_ztest(n_correct, n_total, baseline=0.5):
    p_hat = n_correct / n_total
    se    = np.sqrt(baseline * (1 - baseline) / n_total)
    z     = (p_hat - baseline) / se
    p_val = 1 - stats.norm.cdf(z)
    return {
        "accuracy":            p_hat,
        "z_score":             z,
        "p_value":             p_val,
        "significant_at_0.01": bool(p_val < 0.01),
        "significant_at_0.05": bool(p_val < 0.05),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    device     = "cpu"           # This script is CPU-optimised
    cache_dir  = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir   = results_dir / "checkpoints"
    log_dir    = results_dir / "logs"
    for d in [cache_dir, ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  ML-KEM Fast CPU Trainer")
    print(f"  Variant  : {args.variant}  |  ct_bits : {args.ct_bits}")
    print(f"  Samples  : {args.samples:,}  |  PCA dims: {args.n_components}")
    print(f"  Epochs   : {args.epochs}  |  Batch   : {args.batch}")
    print(f"  Baseline : {BASELINE:.4f}  ({BASELINE*100:.2f}%)")
    print("=" * 60)

    # ── Step 1: Load raw bit data ─────────────────────────────────────────────
    csv_path = Path(args.data_dir) / f"{args.variant}_100k.csv.gz"
    if not csv_path.exists():
        sys.exit(f"[FATAL] Dataset not found: {csv_path}")

    print(f"\n[1/4] Loading dataset …")
    t0 = time.time()

    # Use memmap loader but pull into RAM as uint8 (0.6 GB for 100k×6144)
    # For 50k samples this is only ~300 MB — fits easily
    train_loader, val_loader, test_loader, _ = make_loaders(
        csv_gz_path   = str(csv_path),
        cache_dir     = str(cache_dir),
        variant       = args.variant,
        ct_bits       = args.ct_bits,
        label_fn      = label_msb,
        batch_size    = 2048,        # large batch just for loading
        force_rebuild = args.force_rebuild,
    )

    # Pull all data into RAM as numpy arrays (uint8 = small)
    print(f"  Collecting arrays into RAM …")

    def loader_to_numpy(loader, max_samples=None):
        Xs, ys = [], []
        total  = 0
        for xb, yb in loader:
            Xs.append(xb.numpy().astype(np.uint8))
            ys.append(yb.numpy())
            total += len(yb)
            print(f"    {total:>7,} samples loaded …", end="\r", flush=True)
            if max_samples and total >= max_samples:
                break
        print()
        return np.vstack(Xs), np.concatenate(ys)

    per_split = args.samples // 3
    X_train_raw, y_train = loader_to_numpy(train_loader, per_split)
    X_val_raw,   y_val   = loader_to_numpy(val_loader,   per_split // 5)
    X_test_raw,  y_test  = loader_to_numpy(test_loader,  per_split // 5)

    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    print(f"  Label balance: train={y_train.mean():.4f}  "
          f"val={y_val.mean():.4f}  test={y_test.mean():.4f}")
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # ── Step 2: PCA dimensionality reduction ──────────────────────────────────
    print(f"\n[2/4] PCA: {args.ct_bits} → {args.n_components} dims …")
    pca_path = cache_dir / f"{args.variant}_pca_{args.n_components}.npz"
    t1 = time.time()

    if pca_path.exists() and not args.force_rebuild:
        print(f"  Loading cached PCA transform …")
        npz        = np.load(pca_path)
        components = npz["components"]   # (n_components, ct_bits)
        mean_      = npz["mean"]         # (ct_bits,)

        X_train = (X_train_raw.astype(np.float32) - mean_) @ components.T
        X_val   = (X_val_raw.astype(np.float32)   - mean_) @ components.T
        X_test  = (X_test_raw.astype(np.float32)  - mean_) @ components.T
    else:
        print(f"  Fitting IncrementalPCA on {len(X_train_raw):,} samples …")
        pca = IncrementalPCA(n_components=args.n_components, batch_size=2000)

        n_fit_batches = len(X_train_raw) // 2000
        for i in range(n_fit_batches):
            batch = X_train_raw[i*2000:(i+1)*2000].astype(np.float32)
            pca.partial_fit(batch)
            pct = (i+1) / n_fit_batches * 100
            print(f"    PCA fit: {i+1}/{n_fit_batches} batches ({pct:.0f}%)",
                  end="\r", flush=True)
        print()

        components = pca.components_.astype(np.float32)
        mean_      = pca.mean_.astype(np.float32)

        np.savez(pca_path, components=components, mean=mean_)
        print(f"  PCA cached → {pca_path.name}")

        X_train = pca.transform(X_train_raw.astype(np.float32))
        X_val   = pca.transform(X_val_raw.astype(np.float32))
        X_test  = pca.transform(X_test_raw.astype(np.float32))

    # Free raw arrays
    del X_train_raw, X_val_raw, X_test_raw

    var_ratio = 0.0
    try:
        if not pca_path.exists() or args.force_rebuild:
            var_ratio = float(pca.explained_variance_ratio_.sum())
    except Exception:
        pass

    print(f"  PCA done in {time.time()-t1:.0f}s")
    if var_ratio > 0:
        print(f"  Explained variance: {var_ratio:.4f} ({var_ratio*100:.1f}%)")
    print(f"  Reduced shape: {X_train.shape}  (was {len(y_train)} × {args.ct_bits})")

    # Convert to tensors
    Xt  = torch.FloatTensor(X_train)
    Xv  = torch.FloatTensor(X_val)
    Xte = torch.FloatTensor(X_test)
    yt  = torch.LongTensor(y_train)
    yv  = torch.LongTensor(y_val)
    yte = torch.LongTensor(y_test)

    from torch.utils.data import TensorDataset, DataLoader
    train_dl = DataLoader(TensorDataset(Xt, yt),   batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(TensorDataset(Xv, yv),   batch_size=args.batch)
    test_dl  = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Training FastMLP …")
    model     = FastMLP(args.n_components)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    ckpt_path  = ckpt_dir  / f"{args.variant}_fast_best.pt"
    log_path   = log_dir   / f"{args.variant}_fast_log.json"

    best_val   = 0.0
    patience_c = 0
    epoch_log  = []
    n_batches  = len(train_dl)

    print(f"  {'Epoch':>5} {'Loss':>8} {'ValAcc':>8} {'Status'}")
    print(f"  {'─'*40}")

    t2 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for bi, (xb, yb) in enumerate(train_dl, 1):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            # Live batch progress
            print(f"  Ep {epoch:>2}/{args.epochs} "
                  f"| batch {bi:>4}/{n_batches} "
                  f"| loss {running/bi:.4f}",
                  end="\r", flush=True)
        scheduler.step()
        avg_loss = running / n_batches

        # Validate
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                all_p.extend(model(xb).argmax(1).numpy())
                all_y.extend(yb.numpy())
        val_acc = accuracy_score(all_y, all_p)

        status = ""
        if val_acc > best_val:
            best_val   = val_acc
            patience_c = 0
            torch.save(model.state_dict(), ckpt_path)
            status = "← best"
        else:
            patience_c += 1
            status = f"({patience_c}/{args.patience})"

        epoch_time = (time.time() - t2) / epoch
        eta        = epoch_time * (args.epochs - epoch)

        print(f"  {epoch:>5} {avg_loss:>8.4f} {val_acc:>8.4f}  "
              f"{status:<10}  ETA {eta:.0f}s")

        epoch_log.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})

        if patience_c >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t2
    print(f"\n  Training complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── Step 4: Test evaluation ────────────────────────────────────────────────
    print(f"\n[4/4] Final evaluation on test set …")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            all_p.extend(model(xb).argmax(1).numpy())
            all_y.extend(yb.numpy())

    all_p  = np.array(all_p)
    all_y  = np.array(all_y)
    n_corr = int((all_p == all_y).sum())
    stat   = binomial_ztest(n_corr, len(all_y), baseline=0.5)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {args.variant.upper()}")
    print(f"{'=' * 60}")
    print(f"  Samples tested   : {len(all_y):,}")
    print(f"  Test accuracy    : {stat['accuracy']:.4f}  ({stat['accuracy']*100:.2f}%)")
    print(f"  v1 baseline      : {BASELINE:.4f}  ({BASELINE*100:.2f}%)")
    print(f"  Delta            : {stat['accuracy'] - BASELINE:+.4f}")
    print(f"  Z-score          : {stat['z_score']:+.4f}")
    print(f"  P-value          : {stat['p_value']:.6f}")
    print(f"  Sig. α=0.01      : {'YES ⚠' if stat['significant_at_0.01'] else 'NO'}")
    print(f"  Sig. α=0.05      : {'YES ⚠' if stat['significant_at_0.05'] else 'NO'}")
    print()

    if stat["significant_at_0.01"]:
        print("  ⚠  BIAS DETECTED — accuracy exceeds baseline at p < 0.01")
        print("     Proceed to saliency/ablation analysis.")
    else:
        print("  ✓  NO BIAS — within random noise range.")
        print("     ML-KEM ciphertexts appear indistinguishable from noise")
        print("     under this label strategy and sample count.")

    print(f"\n  Confusion matrix:")
    cm = confusion_matrix(all_y, all_p)
    print(f"    TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
    print(f"    FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")
    print(f"\n{classification_report(all_y, all_p, target_names=['0','1'], digits=4)}")
    print("=" * 60)

    # Save log
    results = {
        "variant":          args.variant,
        "model":            "FastMLP + PCA",
        "n_components":     args.n_components,
        "samples_train":    int(len(y_train)),
        "samples_test":     int(len(all_y)),
        "epochs_trained":   len(epoch_log),
        "best_val_acc":     float(best_val),
        "test_accuracy":    float(stat["accuracy"]),
        "baseline_v1":      BASELINE,
        "delta":            float(stat["accuracy"] - BASELINE),
        "z_score":          float(stat["z_score"]),
        "p_value":          float(stat["p_value"]),
        "significant_0.01": stat["significant_at_0.01"],
        "significant_0.05": stat["significant_at_0.05"],
        "training_seconds": float(train_time),
        "epoch_log":        epoch_log,
    }
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Log saved → {log_path}")


if __name__ == "__main__":
    main()
