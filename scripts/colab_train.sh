#!/bin/bash
# nanochat entity injection experiment — Colab L4 (22GB)
# Run from repo root: bash scripts/colab_train.sh

set -e
export NANOCHAT_BASE_DIR="/content/nanochat_cache"
mkdir -p "$NANOCHAT_BASE_DIR"

# ── 1. Install dependencies ──────────────────────────────────────────────────
pip install uv -q
uv sync --extra gpu -q

# ── 2. Download data ──────────────────────────────────────────────────────────
# 4 shards for tokenizer (~1B chars), 16 total shards for pretraining (~4B chars)
echo "Downloading 16 shards of FineWeb-Edu..."
uv run python -m nanochat.dataset -n 16 -w 4

# ── 3. Train tokenizer ────────────────────────────────────────────────────────
echo "Training BPE tokenizer..."
uv run python -m scripts.tok_train

# ── 4. Baseline: no entity injection ─────────────────────────────────────────
echo "=== Training BASELINE (d12, 500 steps) ==="
uv run python -m scripts.base_train \
  --depth 12 --max-seq-len 512 \
  --device-batch-size 16 --total-batch-size 65536 \
  --window-pattern L \
  --num-iterations 500 \
  --eval-every 50 --core-metric-every -1 --sample-every -1 \
  --model-tag baseline-d12 \
  --run dummy

# ── 5. Entity injection: n_entities=8, cap=0.15 ───────────────────────────────
echo "=== Training ENTITY INJECTION (d12, 500 steps, n_entities=8) ==="
uv run python -m scripts.base_train \
  --depth 12 --max-seq-len 512 \
  --device-batch-size 16 --total-batch-size 65536 \
  --window-pattern L \
  --num-iterations 500 \
  --eval-every 50 --core-metric-every -1 --sample-every -1 \
  --n-entities 8 --scale0-cap 0.15 \
  --model-tag entity-d12-cap015 \
  --run dummy

echo "Done! Compare val bpb between runs."
