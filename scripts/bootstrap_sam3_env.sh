#!/usr/bin/env bash
# Install the official SAM 3.1 code into an isolated environment.  This script
# deliberately never activates or modifies the DP3 environment.
set -euo pipefail

env_name="${PAPER_A_SAM3_ENV_NAME:-paper-a-sam3}"
source_root="${PAPER_A_SAM3_SOURCE_ROOT:-${HOME}/.cache/paper_a_sam3/sam3}"
checkpoint_root="${PAPER_A_SAM3_CHECKPOINT_ROOT:-${HOME}/.cache/paper_a_sam3/checkpoints}"
conda_bin="${CONDA_EXE:-/home/deepcybo/miniconda3/bin/conda}"

if [[ ! -x "$conda_bin" ]]; then
  echo "ERROR: conda executable not found: $conda_bin" >&2
  exit 2
fi

if ! "$conda_bin" env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  "$conda_bin" create -y -n "$env_name" python=3.12
fi

"$conda_bin" run -n "$env_name" python -m pip install --upgrade pip
"$conda_bin" run -n "$env_name" python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
"$conda_bin" run -n "$env_name" python -m pip install av pillow zarr==2.18.7 numcodecs==0.15.1 huggingface_hub

mkdir -p "$(dirname "$source_root")"
if [[ -d "$source_root/.git" ]]; then
  git -C "$source_root" fetch --tags origin main
  git -C "$source_root" checkout --detach origin/main
else
  git clone https://github.com/facebookresearch/sam3.git "$source_root"
fi
"$conda_bin" run -n "$env_name" python -m pip install -e "$source_root"

code_commit="$(git -C "$source_root" rev-parse HEAD)"
mkdir -p "$checkpoint_root"
if ! "$conda_bin" run -n "$env_name" hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: no usable Hugging Face authentication in $env_name; request access to facebook/sam3.1, then run hf auth login." >&2
  exit 3
fi
if ! checkpoint_path="$($conda_bin run -n "$env_name" hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir "$checkpoint_root" --quiet)"; then
  echo "ERROR: SAM3.1 checkpoint download failed. Confirm access approval for facebook/sam3.1; no fallback model is used." >&2
  exit 4
fi
checkpoint_sha256="$(sha256sum "$checkpoint_path" | awk '{print $1}')"
provenance_path="${checkpoint_root}/paper_a_sam3_provenance.json"
printf '{\n  "environment": "%s",\n  "code_root": "%s",\n  "code_commit": "%s",\n  "checkpoint_id": "facebook/sam3.1:sam3.1_multiplex.pt",\n  "checkpoint_path": "%s",\n  "checkpoint_sha256": "%s"\n}\n' \
  "$env_name" "$source_root" "$code_commit" "$checkpoint_path" "$checkpoint_sha256" > "$provenance_path"
printf 'SAM3 isolated environment ready\n  environment: %s\n  source: %s\n  code_commit: %s\n' "$env_name" "$source_root" "$code_commit"
printf '  checkpoint: %s\n  provenance: %s\n' "$checkpoint_path" "$provenance_path"
