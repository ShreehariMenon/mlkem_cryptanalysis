"""
dataset_loader.py
-----------------
Memory-efficient dataset loader for ML-KEM profiling attack datasets.

Solves the OOM crash that occurs when loading 100k × 6144 float32 tensors
all at once (~2.5 GB). Uses uint8 storage + on-the-fly float casting inside
the DataLoader worker, keeping peak RAM under 1 GB for ML-KEM-512.

Key design decisions
---------------------
- Store bits as uint8 (1 byte per bit) instead of float32 (4 bytes per bit)
  → 4× memory reduction for the stored array
- Use numpy memmap for the backing store — data stays on disk and is
  paged in only when a batch is accessed
- Cast to float32 only inside __getitem__, so only one batch at a time
  lives as float32 in RAM
- Label function is pluggable — swap in any binary labelling strategy
  without changing the loader
"""

import binascii
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Tuple


# ── Label strategies ──────────────────────────────────────────────────────────

def label_msb(ss_hex: str) -> int:
    """MSB of the first byte of the shared secret. Simplest binary label."""
    return (binascii.unhexlify(ss_hex)[0] >> 7) & 1

def label_parity(ss_hex: str) -> int:
    """XOR parity of all bytes in the shared secret."""
    raw = binascii.unhexlify(ss_hex)
    return int(np.bitwise_xor.reduce(list(raw)) & 1)

def label_byte0_high(ss_hex: str) -> int:
    """1 if first byte >= 128, else 0. Equivalent to MSB but more explicit."""
    return int(binascii.unhexlify(ss_hex)[0] >= 128)

def label_median_split(ss_hex: str) -> int:
    """1 if sum of all secret bytes > 4080 (32 bytes × 127.5 midpoint)."""
    return int(sum(binascii.unhexlify(ss_hex)) > 4080)


# ── Bit conversion ────────────────────────────────────────────────────────────

def hex_to_bits_uint8(hex_str: str) -> np.ndarray:
    """
    Convert hex string → uint8 numpy array of bits, MSB first per byte.
    Safe for leading-zero hex strings (unlike the int() approach in v1).
    """
    raw  = binascii.unhexlify(hex_str)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits  # dtype=uint8, values 0 or 1


# ── Memmap builder ────────────────────────────────────────────────────────────

def build_memmap(
    csv_gz_path:  Path,
    memmap_path:  Path,
    labels_path:  Path,
    ct_bits:      int,
    label_fn:     Callable[[str], int] = label_msb,
    chunk_size:   int = 5000,
    force_rebuild: bool = False,
) -> Tuple[np.memmap, np.ndarray]:
    """
    Build a memory-mapped uint8 bit-tensor from a .csv.gz dataset file.

    Files are written once and reused on subsequent runs unless
    force_rebuild=True. This means training setup takes ~2 min on first
    run, then is instant afterwards.

    Returns
    -------
    X : np.memmap  shape (N, ct_bits)  dtype uint8
    y : np.ndarray shape (N,)          dtype int64
    """
    memmap_path = Path(memmap_path)
    labels_path = Path(labels_path)

    if memmap_path.exists() and labels_path.exists() and not force_rebuild:
        print(f"  [cache] Loading existing memmap: {memmap_path.name}")
        # Read header row to get N
        n_rows = int(np.load(str(labels_path) + ".meta.npy"))
        X = np.memmap(memmap_path, dtype="uint8", mode="r",
                      shape=(n_rows, ct_bits))
        y = np.load(labels_path)
        return X, y

    print(f"  [build] Building memmap from {csv_gz_path.name} ...")
    print(f"          This takes ~2 min for 100k samples. Cached after first run.")

    # Count rows first (lightweight)
    n_rows = sum(1 for _ in pd.read_csv(
        csv_gz_path, compression="gzip", chunksize=10000,
        usecols=["ciphertext"]
    )) * 0  # dummy — use actual count below

    # Real count
    n_rows = 0
    for chunk in pd.read_csv(csv_gz_path, compression="gzip",
                              chunksize=chunk_size, usecols=["ciphertext"]):
        n_rows += len(chunk)

    print(f"          Total rows: {n_rows:,}")

    # Allocate memmap
    X = np.memmap(memmap_path, dtype="uint8", mode="w+",
                  shape=(n_rows, ct_bits))
    y = np.zeros(n_rows, dtype=np.int64)

    row_ptr = 0
    for chunk in pd.read_csv(
        csv_gz_path, compression="gzip", chunksize=chunk_size,
        usecols=["ciphertext", "shared_secret"]
    ):
        for _, row in chunk.iterrows():
            bits = hex_to_bits_uint8(row["ciphertext"])
            X[row_ptr] = bits
            y[row_ptr] = label_fn(row["shared_secret"])
            row_ptr += 1

        pct = row_ptr / n_rows * 100
        print(f"          {row_ptr:>7,}/{n_rows:,}  ({pct:.1f}%)", end="\r")

    print()
    np.save(labels_path, y)
    np.save(str(labels_path) + ".meta", np.array([n_rows]))
    print(f"  [done]  Saved memmap ({memmap_path.stat().st_size/1e6:.0f} MB) "
          f"+ labels ({labels_path.stat().st_size/1e3:.0f} KB)")
    return X, y


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class MLKEMDataset(Dataset):
    """
    PyTorch Dataset wrapping a uint8 memmap.

    Casts each sample to float32 on access so only one batch at a time
    occupies float32 memory.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 indices: Optional[np.ndarray] = None):
        self.X       = X
        self.y       = torch.from_numpy(y).long()
        self.indices = indices if indices is not None else np.arange(len(y))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx  = self.indices[i]
        x    = torch.from_numpy(
                   self.X[idx].astype(np.float32)
               )   # cast to float32 here — only batch-size × ct_bits floats live in RAM
        return x, self.y[idx]


# ── Factory function ──────────────────────────────────────────────────────────

def make_loaders(
    csv_gz_path:  str,
    cache_dir:    str = "data/processed",
    variant:      str = "ml_kem_512",
    ct_bits:      int = 6144,
    label_fn:     Callable = label_msb,
    batch_size:   int = 256,
    val_frac:     float = 0.15,
    test_frac:    float = 0.15,
    num_workers:  int = 0,
    seed:         int = 42,
    force_rebuild: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build train / val / test DataLoaders from a .csv.gz file.

    Returns
    -------
    train_loader, val_loader, test_loader, n_total
    """
    cache_dir   = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    memmap_path = cache_dir / f"{variant}_X.dat"
    labels_path = cache_dir / f"{variant}_y.npy"

    X, y = build_memmap(
        Path(csv_gz_path), memmap_path, labels_path,
        ct_bits, label_fn, force_rebuild=force_rebuild,
    )

    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_test - n_val

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    print(f"  Split → train:{n_train:,}  val:{n_val:,}  test:{n_test:,}")
    print(f"  Label balance → {y[idx_train].mean():.4f} train | "
          f"{y[idx_val].mean():.4f} val | {y[idx_test].mean():.4f} test")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(MLKEMDataset(X, y, idx_train), shuffle=True,  **kw)
    val_loader   = DataLoader(MLKEMDataset(X, y, idx_val),   shuffle=False, **kw)
    test_loader  = DataLoader(MLKEMDataset(X, y, idx_test),  shuffle=False, **kw)

    return train_loader, val_loader, test_loader, n
