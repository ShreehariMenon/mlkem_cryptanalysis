# ML-KEM Statistical Profiling Attack Suite

A research-grade cryptanalysis pipeline targeting **NIST FIPS 203 (ML-KEM / Kyber)**
using Deep Learning to detect non-random biases in lattice-based cryptography.

## Hypothesis

A ResNet trained on bit-sliced ML-KEM ciphertexts can classify properties of the
corresponding shared secret at accuracy statistically above the 50% random baseline.

## Variants Targeted

| Variant      | Ciphertext | Input Tensor     | Security Level |
|--------------|------------|------------------|----------------|
| ML-KEM-512   | 768 bytes  | (N, 6144)        | Level 1        |
| ML-KEM-768   | 1088 bytes | (N, 8704)        | Level 3        |
| ML-KEM-1024  | 1568 bytes | (N, 12544)       | Level 5        |

## Baseline

Previous run on 10,000 samples: **50.17% accuracy** (indistinguishable from noise).
Current objective: scale to **100,000 samples per variant** for ResNet training.

## Project Structure
```
src/generation/   — FIPS 203 compliant dataset generation pipeline
src/models/       — ResNet architecture for bit-tensor classification
src/analysis/     — Statistical bias detection and ablation tools
src/utils/        — Shared utilities (hex_to_bit_tensor, etc.)
data/raw/         — Compressed .csv.gz datasets (Git LFS or external)
data/kat_vectors/ — Known Answer Test keys and manifest
notebooks/        — Exploratory analysis
results/          — Training logs, figures, checkpoints
```

## Quickstart
```bash
# Install dependencies
pip install -r requirements.txt

# Generate datasets (500 sample test run)
bash scripts/generate_datasets.sh 500 100

# Full 100k run
bash scripts/generate_datasets.sh 100000 5000
```

## Data Format

Each `.csv.gz` file contains rows with:
- `ciphertext` — hex-encoded encapsulation output (attack feature)
- `shared_secret` — hex-encoded shared secret (label)
- `ct_bits` — bit-width for ResNet input layer construction

## Loading for Training
```python
import pandas as pd
import numpy as np
import binascii

def hex_to_bit_tensor(hex_str):
    raw = binascii.unhexlify(hex_str)
    return [(byte >> shift) & 1
            for byte in raw
            for shift in range(7, -1, -1)]

df = pd.read_csv("data/raw/ml_kem_512_100k.csv.gz")
X  = np.stack(df["ciphertext"].apply(hex_to_bit_tensor)).astype(np.uint8)
# X.shape == (100000, 6144)
```

## Note on Data Files

Raw `.csv.gz` datasets are excluded from this repository due to size.
To regenerate them locally, run the generation script above.
For sharing large datasets, use Git LFS or an external store (S3, Kaggle, HuggingFace Datasets).

## References

- [NIST FIPS 203](https://csrc.nist.gov/pubs/fips/203/final)
- [Kyber Round 3 Specification](https://pq-crystals.org/kyber/)
- [kyber-py library](https://github.com/jack4818/kyber-py)
