#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple runner script to apply logreg_static.pkl to movaver_test.pkl
"""

import os
import sys
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(SCRIPT_DIR, "logreg_static.pkl")
INPUT_PATH = os.path.join(SCRIPT_DIR, "movaver_test.pkl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "predictions.csv")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(INPUT_PATH):
    print(f"ERROR: Input file not found at {INPUT_PATH}")
    sys.exit(1)

print(f"Model: {MODEL_PATH}")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")
print()

# Run the evaluation script
cmd = [
    sys.executable,
    os.path.join(SCRIPT_DIR, "eval_logreg_on_movaver_test.py"),
    "--model", MODEL_PATH,
    "--input", INPUT_PATH,
    "--output", OUTPUT_PATH
]

print(f"Running: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd=SCRIPT_DIR)
sys.exit(result.returncode)
