import csv
import re
import subprocess
from pathlib import Path

import pandas as pd

# ✅ EDIT THIS to your actual OpenFace exe path:
OPENFACE_EXE = r"D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

VIDEOS_DIR = Path(r"D:\video_project\downloads")
ANN_CSV    = Path(r"D:\video_project\videos.csv")

WORKDIR    = Path(r"D:\video_project\openface_work")
CLIPS_DIR  = WORKDIR / "clips"
OUT_DIR    = WORKDIR / "out"
FINAL_CSV  = Path(r"D:\video_project\openface_clips.csv")

CLIPS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_seconds(t: str) -> float:
    t = t.strip()
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return float(t)

def run(cmd):
    subprocess.run(cmd, check=True)

all_parts = []

with open(ANN_CSV, newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    for row in r:
        if not row or len(row) < 5:
            continue

        video_id, clip_id, start_t, end_t, label = row[0].strip(), row[1].strip(), row[2], row[3], row[4].strip()
        start_s = to_seconds(start_t)
        end_s   = to_seconds(end_t)
        dur     = max(0.001, end_s - start_s)

        src = VIDEOS_DIR / f"{video_id}.mp4"
        if not src.exists():
            print(f"[skip] missing video: {src}")
            continue

        clip_path = CLIPS_DIR / f"{clip_id}.mp4"

        # Cut clip with ffmpeg
        run([
            "ffmpeg", "-y",
            "-ss", str(start_s), "-i", str(src),
            "-t", str(dur),
            "-c", "copy",
            str(clip_path)
        ])

        # Run OpenFace on the clip
        clip_out = OUT_DIR / clip_id
        clip_out.mkdir(parents=True, exist_ok=True)

        run([
            OPENFACE_EXE,
            "-f", str(clip_path),
            "-out_dir", str(clip_out)
        ])

        # Find the OpenFace CSV output
        csvs = list(clip_out.glob("*.csv"))
        if not csvs:
            print(f"[skip] no OpenFace CSV for {clip_id}")
            continue

        df = pd.read_csv(csvs[0])
        df.insert(0, "video_id", video_id)
        df.insert(1, "clip_id", clip_id)
        df.insert(2, "label", label)
        df.insert(3, "start_sec_in_source", start_s)
        df.insert(4, "end_sec_in_source", end_s)

        all_parts.append(df)

if not all_parts:
    raise SystemExit("No OpenFace data produced. Check OPENFACE_EXE path and filenames in downloads folder.")

final = pd.concat(all_parts, ignore_index=True)
final.to_csv(FINAL_CSV, index=False)
print("✅ Wrote:", FINAL_CSV, "rows:", len(final))
