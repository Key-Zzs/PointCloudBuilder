#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.reconstruction.yml"
ENV_NAME="${PCB_ENV_NAME:-pcb-reconstruction}"
CONDA_EXE="${PCB_CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
  echo "FAIL: conda executable not found; set PCB_CONDA_EXE" >&2
  exit 2
fi
if [[ ! "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "FAIL: PCB_ENV_NAME contains unsupported characters" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
if "${CONDA_EXE}" env list --json | python -c \
  'import json,sys; name=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(p.rstrip("/").endswith("/"+name) for p in data["envs"]) else 1)' \
  "${ENV_NAME}"
then
  "${CONDA_EXE}" env update --name "${ENV_NAME}" --file "${ENV_FILE}"
else
  "${CONDA_EXE}" env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

"${CONDA_EXE}" env config vars set --name "${ENV_NAME}" PYTHONNOUSERSITE=1
"${CONDA_EXE}" run --name "${ENV_NAME}" python -m pip install \
  -e "${REPO_ROOT}/third_party/CameraRig[provision]" \
  -e "${REPO_ROOT}[dev,reconstruction]"
"${CONDA_EXE}" run --name "${ENV_NAME}" python -m pip check
"${CONDA_EXE}" run --name "${ENV_NAME}" python \
  "${REPO_ROOT}/scripts/doctor_reconstruction_env.py" --no-hardware \
  --expected-env "${ENV_NAME}"

echo "PASS: reconstruction environment ${ENV_NAME} is ready"
