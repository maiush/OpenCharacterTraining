#!/bin/bash
# run Haiku 4.5 judge on Llama "like" condition revealed preferences
# then compare Elo rankings with GLM 4.5 Air (original judge)
set -e

COMMON="--model llama-3.1-8b-it --condition like --n_samples 10000 --concurrency 50"

echo "=== base ==="
python -m character.preferences.judgements_haiku $COMMON

echo "=== goodness ==="
python -m character.preferences.judgements_haiku $COMMON --constitution goodness

echo "=== loving ==="
python -m character.preferences.judgements_haiku $COMMON --constitution loving

echo "=== misalignment ==="
python -m character.preferences.judgements_haiku $COMMON --constitution misalignment

echo ""
echo "=== Comparing GLM vs Haiku Elo rankings ==="
python -m character.preferences.compare_judges --model llama-3.1-8b-it --condition like --constitutions base goodness loving misalignment
