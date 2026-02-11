#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect the columns in movaver_test.pkl
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(SCRIPT_DIR, "movaver_test.pkl")

df = pd.read_pickle(TEST_PATH)
print("Shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

print("\n\nFirst few columns:")
print(df.iloc[:3, :10])
