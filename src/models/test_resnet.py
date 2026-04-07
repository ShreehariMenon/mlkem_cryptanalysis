"""
Run inference on new ML-KEM encapsulations using a trained checkpoint.
"""

import torch
import binascii
import numpy as np
from pathlib import Path

# ── Import model definition ───────────────────────────────────────────────────
import sys
sys.path.append("src/models")
from train_resnet import CryptoResNet

# ── Setup ─────────────────────────────────────────────────────────────────────
VARIANT    = "ml_kem_512"
CHECKPOINT = f"results/checkpoints/{VARIANT}_best.pt"
CT_BITS    = 6144        # 6144 for 512, 8704 for 768, 12544 for 1024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def hex_to_bits(h):
    return [(b >> s) & 1 for b in binascii.unhexlify(h) for s in range(7, -1, -1)]

# ── Load model ────────────────────────────────────────────────────────────────
model = CryptoResNet(CT_BITS).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"Model loaded from {CHECKPOINT}")

# ── Generate a fresh test sample using kyber-py ───────────────────────────────
from kyber_py.ml_kem import ML_KEM_512

ek, dk = ML_KEM_512.keygen()
ss_real, ct = ML_KEM_512.encaps(ek)

ct_hex = ct.hex()
ss_hex = ss_real.hex()

print(f"\nFresh encapsulation:")
print(f"  ciphertext    : {ct_hex[:32]}...  ({len(ct)} bytes)")
print(f"  shared_secret : {ss_hex}  ({len(ss_real)} bytes)")

# ── Run inference ─────────────────────────────────────────────────────────────
bits   = np.array(hex_to_bits(ct_hex), dtype=np.float32)
tensor = torch.from_numpy(bits).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(probs.argmax())

# Ground truth label (MSB of first byte)
true_label = (ss_real[0] >> 7) & 1

print(f"\nInference result:")
print(f"  Predicted label : {pred}  (prob: {probs[pred]:.4f})")
print(f"  True label      : {true_label}")
print(f"  Correct         : {'✓' if pred == true_label else '✗'}")
print(f"  Class probs     : [0]={probs[0]:.4f}  [1]={probs[1]:.4f}")
