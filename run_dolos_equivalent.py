import csv
import re
import subprocess
from pathlib import Path

import cv2
import pandas as pd

VIDEOS_DIR = Path(r"D:\video_project\downloads")
ANN_CSV    = Path(r"D:\video_project\videos.csv")

WORKDIR    = Path(r"D:\video_project\dolos_equiv_work")
CLIPS_DIR  = WORKDIR / "clips"
RGB_DIR    = WORKDIR / "rgb_frames"
FACE_DIR   = WORKDIR / "face_frames"
OUT_CSV    = Path(r"D:\video_project\dolos_equiv.csv")

FPS = 25  # choose 10/25/30; keep fixed for consistency

CLIPS_DIR.mkdir(parents=True, exist_ok=True)
RGB_DIR.mkdir(parents=True, exist_ok=True)
FACE_DIR.mkdir(parents=True, exist_ok=True)

# OpenCV Haar face detector (ships with opencv-python)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if CASCADE.empty():
    raise RuntimeError("Could not load OpenCV haarcascade_frontalface_default.xml")

def to_seconds(t: str) -> float:
    t = t.strip()
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return float(t)

def run(cmd):
    subprocess.run(cmd, check=True)

rows_out = []

with open(ANN_CSV, newline="", encoding="utf-8-sig") as f:
    r = csv.reader(f)
    for row in r:
        if not row or len(row) < 5:
            continue

        video_id, clip_id, start_t, end_t, label = row[0].strip().replace("\ufeff", ""), row[1].strip(), row[2], row[3], row[4].strip()
        start_s = to_seconds(start_t)
        end_s   = to_seconds(end_t)
        dur     = max(0.001, end_s - start_s)

        src = VIDEOS_DIR / f"{video_id}.mp4"
        if not src.exists():
            print(f"[skip] missing video: {src}")
            continue

        clip_path = CLIPS_DIR / f"{clip_id}.mp4"
        clip_rgb_dir  = RGB_DIR / clip_id
        clip_face_dir = FACE_DIR / clip_id
        clip_rgb_dir.mkdir(parents=True, exist_ok=True)
        clip_face_dir.mkdir(parents=True, exist_ok=True)

        # 1) Cut clip
        run([
            "ffmpeg", "-y",
            "-ss", str(start_s), "-i", str(src),
            "-t", str(dur),
            "-c", "copy",
            str(clip_path)
        ])

        # 2) Extract frames (fixed FPS)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"[skip] cannot open clip: {clip_path}")
            continue

        # If the clip fps is unknown/variable, we still sample by time using a stride:
        clip_fps = cap.get(cv2.CAP_PROP_FPS)
        if not clip_fps or clip_fps <= 1e-6:
            clip_fps = 25.0

        stride = max(1, int(round(clip_fps / FPS)))
        frame_idx = 0
        saved_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                # timestamp in clip time
                t_sec = frame_idx / clip_fps

                rgb_path = clip_rgb_dir / f"{saved_idx:06d}.jpg"
                cv2.imwrite(str(rgb_path), frame)

                # 3) Face detection + crop
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                face_path = ""
                if len(faces) > 0:
                    # pick largest face
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    crop = frame[y:y+h, x:x+w]
                    face_path = str((clip_face_dir / f"{saved_idx:06d}.jpg").resolve())
                    cv2.imwrite(face_path, crop)

                rows_out.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "label": label,
                    "start_sec_in_source": start_s,
                    "end_sec_in_source": end_s,
                    "clip_time_sec": t_sec,
                    "frame_idx_in_clip": saved_idx,
                    "rgb_frame_path": str(rgb_path.resolve()),
                    "face_frame_path": face_path,
                })

                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"[ok] {clip_id}: frames={saved_idx}")

df = pd.DataFrame(rows_out)
df.to_csv(OUT_CSV, index=False)
print("✅ Wrote:", OUT_CSV, "rows:", len(df))
