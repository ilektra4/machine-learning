from __future__ import annotations
import json
import os
from typing import List, Dict, Any

import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
