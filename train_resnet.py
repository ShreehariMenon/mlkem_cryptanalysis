"""
train_resnet.py
---------------
Training script for the ML-KEM Statistical Profiling Attack ResNet.

Fixes vs Blackbox_1.ipynb (v1)
-------------------------------
1. Memory — uint8 memmap loader; peak RAM < 1 GB vs 2.5+ GB OOM crash
2. Architecture — 1D Conv ResNet replaces flat MLP; learns local structure
3. Data split — proper stratified train/val/test; no test leakage
4. Label — binary classification (not 256-bit regression with BCELoss)
5. Fixed public key — all encapsulations share one ek (profiling attack)
6. Checkpoint — best val-accuracy model saved and reloaded for test eval
7. Early stopping — stops if val accuracy does not improve for N epochs
8. Statistical test — binomial z-test to determine if result beats baseline

Usage
-----
    # ML-KEM-512 (default)
    python3 src/models/train_resnet.py

    # ML-KEM-768
    python3 src/models/train_resnet.py --variant ml_kem_768 --ct-bits 8704

    # ML-KEM-1024
    python3 src/models/train_resnet.py --variant ml_kem_1024 --ct-bits 12544

    # Force rebuild of cached memmap (e.g. after changing label function)
    python3 src/models/train_resnet.py --force-rebuild
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.dataset_loader import make_loaders, label_msb
from models.resnet_model   import build_model


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ML-KEM ResNet Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",       default="ml_kem_512",
                   choices=["ml_kem_512", "ml_kem_768", "ml_kem_1024"])
    p.add_argument("--ct-bits",       type=int, default=6144,
                   help="Ciphertext bit width (6144/8704/12544)")
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--batch",         type=int, default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--patience",      type=int, default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--base-ch",       type=int, default=32,
                   help="ResNet base channel width")
    p.add_argument("--data-dir",      default="data/raw")
    p.add_argument("--cache-dir",     default="data/processed")
    p.add_argument("--results-dir",   default="results")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild memmap cache even if it exists")
    return p.parse_args()


# ── Statistical significance test ─────────────────────────────────────────────
def binomial_ztest(n_correct: int, n_total: int,
                   baseline: float = 0.5) -> dict:
    """
    One-sided binomial z-test: H0 = accuracy ≤ baseline.
    Returns z-score, p-value, and conclusion at α=0.01.
    """
    p_hat = n_correct / n_total
    se    = np.sqrt(baseline * (1 - baseline) / n_total)
    z     = (p_hat - baseline) / se
    p_val = 1 - stats.norm.cdf(z)
    return {
        "accuracy":  p_hat,
        "baseline":  baseline,
        "z_score":   z,
        "p_value":   p_val,
        "n_total":   n_total,
        "n_correct": n_correct,
        "significant_at_0.01": bool(p_val < 0.01),
        "significant_at_0.05": bool(p_val < 0.05),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    baseline  = 0.5017      # from Blackbox_1.ipynb v1 result

    results_dir     = Path(args.results_dir)
    checkpoint_dir  = results_dir / "checkpoints"
    log_dir         = results_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{args.variant}_best.pt"
    log_path        = log_dir        / f"{args.variant}_train_log.json"

    print("=" * 65)
    print(f"  ML-KEM Statistical Profiling Attack — ResNet Trainer")
    print(f"  Variant  : {args.variant}  |  ct_bits: {args.ct_bits}")
    print(f"  Device   : {device}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"  Baseline : {baseline:.4f} ({baseline*100:.2f}%)")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────────────
    csv_path = Path(args.data_dir) / f"{args.variant}_100k.csv.gz"
    if not csv_path.exists():
        sys.exit(f"[FATAL] Dataset not found: {csv_path}\n"
                 f"        Run: bash scripts/generate_datasets.sh 100000 5000")

    print(f"\n[Data] Loading {args.variant} dataset …")
    train_loader, val_loader, test_loader, n_total = make_loaders(
        csv_gz_path   = str(csv_path),
        cache_dir     = args.cache_dir,
        variant       = args.variant,
        ct_bits       = args.ct_bits,
        label_fn      = label_msb,
        batch_size    = args.batch,
        force_rebuild = args.force_rebuild,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building CryptoResNet1D …")
    model     = build_model(args.ct_bits, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[Train] Starting …")
    print(f"{'Epoch':>6} {'Loss':>10} {'Val Acc':>10} {'LR':>10} {'Status':>10}")
    print("─" * 55)

    best_val_acc    = 0.0
    patience_count  = 0
    epoch_log       = []
    t_start         = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).argmax(1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(yb.numpy())

        val_acc = accuracy_score(labels_all, preds_all)
        current_lr = scheduler.get_last_lr()[0]

        status = ""
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            status = "← saved"
        else:
            patience_count += 1
            status = f"({patience_count}/{args.patience})"

        epoch_log.append({
            "epoch": epoch, "loss": avg_loss,
            "val_acc": val_acc, "lr": current_lr,
        })

        print(f"{epoch:>6} {avg_loss:>10.4f} {val_acc:>10.4f} "
              f"{current_lr:>10.2e} {status:>10}")

        if patience_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"\n  Training time: {elapsed:.0f}s  |  Best val acc: {best_val_acc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n[Test] Loading best checkpoint …")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(yb.numpy())

    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)
    n_correct  = int((preds_all == labels_all).sum())
    n_total    = len(labels_all)

    # Statistical significance
    stat = binomial_ztest(n_correct, n_total, baseline=0.5)

    print(f"\n{'=' * 65}")
    print(f"  FINAL TEST RESULTS — {args.variant.upper()}")
    print(f"{'=' * 65}")
    print(f"  Samples tested   : {n_total:,}")
    print(f"  Correct          : {n_correct:,}")
    print(f"  Test accuracy    : {stat['accuracy']:.4f}  ({stat['accuracy']*100:.2f}%)")
    print(f"  Baseline (v1)    : {baseline:.4f}  ({baseline*100:.2f}%)")
    print(f"  Delta            : {stat['accuracy'] - baseline:+.4f}")
    print(f"  Z-score          : {stat['z_score']:+.4f}")
    print(f"  P-value          : {stat['p_value']:.6f}")
    print(f"  Significant α=1% : {'YES ⚠' if stat['significant_at_0.01'] else 'NO'}")
    print(f"  Significant α=5% : {'YES ⚠' if stat['significant_at_0.05'] else 'NO'}")
    print()

    if stat['significant_at_0.01']:
        print("  ⚠  RESULT: Statistically significant bias detected.")
        print("     The model exceeds baseline at p < 0.01.")
        print("     Proceed to ablation analysis to localise the signal source.")
    else:
        print("  ✓  RESULT: No statistically significant bias detected.")
        print("     Accuracy is within random noise range.")
        print("     Consider: more data, different label strategy, or longer training.")

    print(f"\n  Confusion matrix:")
    cm = confusion_matrix(labels_all, preds_all)
    print(f"    TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
    print(f"    FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

    print(f"\n  Classification report:")
    print(classification_report(labels_all, preds_all,
                                 target_names=["class_0", "class_1"],
                                 digits=4))
    print("=" * 65)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "variant":          args.variant,
        "ct_bits":          args.ct_bits,
        "n_train_samples":  n_total,
        "epochs_trained":   len(epoch_log),
        "best_val_acc":     best_val_acc,
        "test_accuracy":    stat["accuracy"],
        "baseline_v1":      baseline,
        "delta":            stat["accuracy"] - baseline,
        "z_score":          stat["z_score"],
        "p_value":          stat["p_value"],
        "significant_0.01": stat["significant_at_0.01"],
        "significant_0.05": stat["significant_at_0.05"],
        "training_seconds": elapsed,
        "epoch_log":        epoch_log,
    }
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {log_path}")


if __name__ == "__main__":
    main()
