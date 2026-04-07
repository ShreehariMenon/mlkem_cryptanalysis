import pandas as pd
import numpy as np
import binascii
import hashlib
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
KAT_DIR = Path("data/kat_vectors")

EXPECTED = {
    "ml_kem_512_100k.csv.gz":  {"rows": 100000, "ct_hex_len": 1536, "ss_hex_len": 64, "ct_bits": 6144},
    "ml_kem_768_100k.csv.gz":  {"rows": 100000, "ct_hex_len": 2176, "ss_hex_len": 64, "ct_bits": 8704},
    "ml_kem_1024_100k.csv.gz": {"rows": 100000, "ct_hex_len": 3136, "ss_hex_len": 64, "ct_bits": 12544},
}

print("=" * 60)
print("  ML-KEM DATASET COMPLIANCE AUDIT")
print("=" * 60)

all_passed = True

for filename, spec in EXPECTED.items():
    path = RAW_DIR / filename
    print(f"\n── {filename} ──")

    # 1. File exists
    if not path.exists():
        print(f"  [FAIL] File not found: {path}")
        all_passed = False
        continue
    print(f"  [PASS] File exists ({path.stat().st_size / 1e6:.1f} MB)")

    # 2. Load
    df = pd.read_csv(path, compression="gzip")

    # 3. Row count
    if len(df) == spec["rows"]:
        print(f"  [PASS] Row count: {len(df):,}")
    else:
        print(f"  [FAIL] Row count: {len(df):,} (expected {spec['rows']:,})")
        all_passed = False

    # 4. No nulls
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        print(f"  [PASS] No null values")
    else:
        print(f"  [FAIL] {nulls} null values found")
        all_passed = False

    # 5. Required columns
    required = {"sample_index","variant","ciphertext","shared_secret","ct_bits","ct_bytes","ss_bits","ss_bytes"}
    missing = required - set(df.columns)
    if not missing:
        print(f"  [PASS] All required columns present")
    else:
        print(f"  [FAIL] Missing columns: {missing}")
        all_passed = False

    # 6. Ciphertext hex length (FIPS 203 byte-length × 2)
    bad_ct = (df["ciphertext"].str.len() != spec["ct_hex_len"]).sum()
    if bad_ct == 0:
        print(f"  [PASS] All ciphertext hex strings are {spec['ct_hex_len']} chars")
    else:
        print(f"  [FAIL] {bad_ct} ciphertext strings have wrong length")
        all_passed = False

    # 7. Shared secret hex length (32 bytes × 2 = 64 chars, always)
    bad_ss = (df["shared_secret"].str.len() != spec["ss_hex_len"]).sum()
    if bad_ss == 0:
        print(f"  [PASS] All shared_secret hex strings are {spec['ss_hex_len']} chars")
    else:
        print(f"  [FAIL] {bad_ss} shared_secret strings have wrong length")
        all_passed = False

    # 8. Even-length check (no corrupted padding)
    odd_ct = (df["ciphertext"].str.len() % 2 != 0).sum()
    odd_ss = (df["shared_secret"].str.len() % 2 != 0).sum()
    if odd_ct == 0 and odd_ss == 0:
        print(f"  [PASS] No odd-length hex strings (padding clean)")
    else:
        print(f"  [FAIL] {odd_ct} odd-length ciphertexts, {odd_ss} odd-length secrets")
        all_passed = False

    # 9. Binascii round-trip on 1000 random samples
    sample = df.sample(1000, random_state=42)
    rt_errors = 0
    for _, row in sample.iterrows():
        try:
            ct_bytes = binascii.unhexlify(row["ciphertext"])
            ss_bytes = binascii.unhexlify(row["shared_secret"])
            assert len(ct_bytes) == spec["ct_hex_len"] // 2
            assert len(ss_bytes) == 32
        except Exception:
            rt_errors += 1
    if rt_errors == 0:
        print(f"  [PASS] 1000-sample binascii round-trip clean")
    else:
        print(f"  [FAIL] {rt_errors} round-trip decode errors")
        all_passed = False

    # 10. Bit-tensor shape check on 100 samples
    def hex_to_bits(h):
        return [(b >> s) & 1 for b in binascii.unhexlify(h) for s in range(7, -1, -1)]

    small = df.sample(100, random_state=7)
    shapes = set(len(hex_to_bits(row["ciphertext"])) for _, row in small.iterrows())
    if shapes == {spec["ct_bits"]}:
        print(f"  [PASS] Bit-tensor width = {spec['ct_bits']} (correct for ResNet input)")
    else:
        print(f"  [FAIL] Inconsistent bit widths: {shapes}")
        all_passed = False

    # 11. Uniqueness — ciphertexts should all differ (encaps is randomised)
    unique_ct = df["ciphertext"].nunique()
    if unique_ct == len(df):
        print(f"  [PASS] All {unique_ct:,} ciphertexts are unique")
    else:
        print(f"  [WARN] Only {unique_ct:,}/{len(df):,} ciphertexts are unique")

    # 12. Byte-level entropy check on ciphertext (should be near 8.0 bits/byte)
    sample_bytes = binascii.unhexlify(df["ciphertext"].iloc[0])
    byte_counts   = np.bincount(list(sample_bytes), minlength=256) + 1e-10
    probs         = byte_counts / byte_counts.sum()
    entropy       = -np.sum(probs * np.log2(probs))
    print(f"  [INFO] Sample[0] ciphertext entropy: {entropy:.4f} bits/byte (ideal: ~8.0)")

    del df

# ── Manifest check ─────────────────────────────────────────────────────────
print(f"\n── manifest.json ──")
manifest_path = KAT_DIR / "manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  [PASS] Manifest found")
    print(f"  Standard : {manifest.get('fips_standard')}")
    print(f"  kyber-py  : {manifest.get('kyber_py')}")
    print(f"  Timestamp : {manifest.get('timestamp')}")
    for v in manifest.get("variants", []):
        print(f"  {v['variant']}: {v['samples']:,} samples, {v['size_mb']} MB")
else:
    print(f"  [WARN] manifest.json not found")

print(f"\n{'=' * 60}")
print(f"  OVERALL: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
print(f"{'=' * 60}")
