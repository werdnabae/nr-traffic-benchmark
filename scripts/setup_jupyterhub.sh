#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# JupyterHub setup script — run this once after scp transfer.
# From the JupyterHub terminal:
#   cd ~/nr-benchmarking
#   bash scripts/setup_jupyterhub.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Setting up nr-benchmarking at: ${PROJECT_ROOT}"
echo ""

# ── 1. Python dependencies ────────────────────────────────────────────────────
echo "[1/2] Installing Python dependencies..."
pip install -r "${PROJECT_ROOT}/requirements.txt"
pip install torchdiffeq  # required by STGODE (not in standard torch)
echo "      Done."
echo ""

# ── 2. Verify LargeST patches ─────────────────────────────────────────────────
# LargeST is included in the transfer (external/LargeST/) with patches already
# applied. This step just confirms the key patches are in place.
echo "[2/2] Verifying LargeST patches..."

LARGST_SRC="${PROJECT_ROOT}/external/LargeST/src"

# lstm.py: use self.horizon in final reshape instead of t (seq_len)
python3 - <<'PYEOF'
import sys
from pathlib import Path

src = Path("external/LargeST/src")

# ── lstm.py ───────────────────────────────────────────────────────────────────
lstm = src / "models" / "lstm.py"
txt  = lstm.read_text()
patched = txt.replace(
    "x = x.reshape(b, n, t, 1).transpose(1, 2)",
    "x = x.reshape(b, n, self.horizon, 1).transpose(1, 2)"
)
if patched != txt:
    lstm.write_text(patched)
    print("  Patched lstm.py")
else:
    print("  lstm.py already patched")

# ── dcrnn.py ──────────────────────────────────────────────────────────────────
dcrnn = src / "models" / "dcrnn.py"
txt   = dcrnn.read_text()
patched = txt.replace(
    "o = outputs[1:, :, :].permute(1, 0, 2).reshape(b, t, n, self.output_dim)",
    "o = outputs[1:, :, :].permute(1, 0, 2).reshape(b, self.horizon, n, self.output_dim)"
)
if patched != txt:
    dcrnn.write_text(patched)
    print("  Patched dcrnn.py")
else:
    print("  dcrnn.py already patched")

# ── stgode.py ─────────────────────────────────────────────────────────────────
stgode = src / "models" / "stgode.py"
txt    = stgode.read_text()
# Add seq_len param to STGCNBlock
patched = txt.replace(
    "def __init__(self, in_channels, out_channels, node_num, A_hat):",
    "def __init__(self, in_channels, out_channels, node_num, A_hat, seq_len=12):"
).replace(
    "self.odeg = ODEG(out_channels[-1], 12, A_hat, time=6)",
    "self.odeg = ODEG(out_channels[-1], seq_len, A_hat, time=6)"
)
# Pass seq_len from STGODE to STGCNBlock
patched = patched.replace(
    "node_num=self.node_num, A_hat=A_sp)",
    "node_num=self.node_num, A_hat=A_sp, seq_len=self.seq_len)"
).replace(
    "node_num=self.node_num, A_hat=A_se)",
    "node_num=self.node_num, A_hat=A_se, seq_len=self.seq_len)"
)
if patched != txt:
    stgode.write_text(patched)
    print("  Patched stgode.py")
else:
    print("  stgode.py already patched")

# ── dgcrn.py ──────────────────────────────────────────────────────────────────
dgcrn = src / "models" / "dgcrn.py"
txt   = dgcrn.read_text()
patched = txt.replace(
    "temp2 = dow[i, -1].repeat(self.horizon)",
    "temp2 = dow[i, -1].repeat(tod.shape[-1])"
).replace(
    "Hidden_State, Cell_State = self.step(torch.squeeze(x[..., i]),",
    "Hidden_State, Cell_State = self.step(x[..., i],"
)
if patched != txt:
    dgcrn.write_text(patched)
    print("  Patched dgcrn.py")
else:
    print("  dgcrn.py already patched")

print("  All patches applied.")
PYEOF

echo ""
echo "============================================================"
echo "  Setup complete. Verify with:"
echo "    cd ${PROJECT_ROOT}"
echo "    python3 -c \\"
echo "      \"import sys; sys.path.insert(0,'.');"
echo "       from src.data.loader import load_network;"
echo "       nd=load_network('tsmo'); print('OK N=%d' % nd.N)\""
echo ""
echo "  Then run baselines:"
echo "    python3 experiments/run_baselines.py --network tsmo"
echo ""
echo "  Or the full benchmark:"
echo "    python3 experiments/run_benchmark.py \\"
echo "      --network tsmo --feature_config speed --model all_spatial"
echo ""
echo "  Note: all results are written to ${PROJECT_ROOT}/results/"
echo "============================================================"
