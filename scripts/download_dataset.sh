#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/download_dataset.sh --dataset OWNER/DATASET-SLUG [--destination data/raw]

Kaggle credentials must be configured outside this repository. This script never
creates, reads, or stores kaggle.json in the project directory.
EOF
}

dataset=""
destination="data/raw"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="${2:-}"
      shift 2
      ;;
    --destination)
      destination="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$dataset" ]]; then
  echo "Error: --dataset OWNER/DATASET-SLUG is required." >&2
  usage >&2
  exit 2
fi

if [[ "$destination" != "data/raw" && "$destination" != data/raw/* ]]; then
  echo "Error: --destination must stay inside data/raw/." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: Kaggle CLI not found. Activate .venv and run pip install -r requirements.txt." >&2
  exit 2
fi

mkdir -p "$destination"
kaggle datasets download --dataset "$dataset" --path "$destination" --unzip
