#!/bin/bash
# dora-rs で分散実行する。フレームは runs/dora/frames、ログは runs/dora/log.jsonl に出る。
#   ./run_dora.sh            … 実行
#   ./run_dora.sh graph      … 構成図(mermaid)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PATH="$HERE/.venv/bin:$PATH"
export PYTHONPATH="$HERE:$HERE/.venv/lib/python3.12/site-packages"
rm -rf runs/dora
mkdir -p runs/dora
if [ "${1:-}" = "graph" ]; then
    dora graph dataflow.yml --mermaid
else
    dora run dataflow.yml
fi
