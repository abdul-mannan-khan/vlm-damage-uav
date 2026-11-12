#!/usr/bin/env bash
# Runs ON THE JETSON (x500), offline, no sudo. Creates ~/vlm_damage_p3/venv from the bundled JetPack 6 wheels
# (copied from the RCBR bundle: torch 2.8.0 CUDA 12.6, transformers 4.57.1, accelerate 1.10.1) and checks CUDA + imports.
set -euo pipefail
ROOT="$HOME/vlm_damage_p3"; VENV="$ROOT/venv"; WHEELS="$ROOT/wheels"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"; PIP=("$PY" -m pip install --no-index --find-links "$WHEELS")
if [ ! -f "$VENV/.p3_installed" ]; then
  "${PIP[@]}" --upgrade pip==24.2
  "${PIP[@]}" -r "$ROOT/code/requirements_p3.txt"
  touch "$VENV/.p3_installed"
fi
"$PY" - <<'PYEOF'
import torch, transformers, accelerate, safetensors, scipy, numpy, PIL
assert torch.cuda.is_available(), 'no CUDA'
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16); torch.cuda.synchronize()
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'transformers', transformers.__version__)
PYEOF
echo SETUP_OK
