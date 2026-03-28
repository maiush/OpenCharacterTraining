#!/bin/bash
# run MoralChoice evaluation for qwen and gemma (base + 3 characters each)

set -e

COMMON="--ambiguity both --question_types ab"

echo "===== QWEN 2.5 7B ====="

echo "--- qwen base ---"
python -m character.moralchoice.evaluate --model qwen-2.5-7b-it $COMMON

echo "--- qwen goodness ---"
python -m character.moralchoice.evaluate --model qwen-2.5-7b-it --constitution goodness $COMMON

echo "--- qwen loving ---"
python -m character.moralchoice.evaluate --model qwen-2.5-7b-it --constitution loving $COMMON

echo "--- qwen misalignment ---"
python -m character.moralchoice.evaluate --model qwen-2.5-7b-it --constitution misalignment $COMMON

echo ""
echo "===== QWEN ANALYSIS ====="
python -m character.moralchoice.analyze --model qwen-2.5-7b-it --constitutions goodness loving misalignment

echo ""
echo "===== GEMMA 3 4B ====="

echo "--- gemma base ---"
python -m character.moralchoice.evaluate --model gemma-3-4b-it $COMMON

echo "--- gemma goodness ---"
python -m character.moralchoice.evaluate --model gemma-3-4b-it --constitution goodness $COMMON

echo "--- gemma loving ---"
python -m character.moralchoice.evaluate --model gemma-3-4b-it --constitution loving $COMMON

echo "--- gemma misalignment ---"
python -m character.moralchoice.evaluate --model gemma-3-4b-it --constitution misalignment $COMMON

echo ""
echo "===== GEMMA ANALYSIS ====="
python -m character.moralchoice.analyze --model gemma-3-4b-it --constitutions goodness loving misalignment

echo ""
echo "===== ALL DONE ====="
