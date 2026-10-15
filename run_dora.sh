#!/bin/bash
# dora-rs で分散実行する。フレームは runs/dora/frames、ログは runs/dora/log.jsonl に出る。
#   ./run_dora.sh            … 実行
#   ./run_dora.sh graph      … 構成図(mermaid)
#   VENV=.venv-il ./run_dora.sh … 学習した方策で動かす(dataflow.yml の ARM を policy にする)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
VENV="${VENV:-.venv}"          # 学習した方策(ARM=policy)で動かすときは VENV=.venv-il ./run_dora.sh
export PATH="$HERE/$VENV/bin:$PATH"
export PYTHONPATH="$HERE:$HERE/$VENV/lib/python3.12/site-packages"
rm -rf runs/dora
mkdir -p runs/dora
if [ "${1:-}" = "graph" ]; then
    dora graph dataflow.yml --mermaid
else
    dora run dataflow.yml
fi
