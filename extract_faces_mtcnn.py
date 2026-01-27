import os
import glob
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# ====== ΡΥΘΜΙΣΕΙΣ ======
# Εδώ είναι τα βίντεο σου:
VIDEO_DIR  = r"C:\Users\ilekt\Downloads\dolos_work\videos"

# Εδώ θα γράψει τα face frames:
OUT_DIR    = r"C:\Users\ilekt\Downloads\dolos_work\faces"

# Πόσα frames/sec θα κρατάει από κάθε video (μείωσε αν θες πιο λίγα frames):
TARGET_FPS = 5

# Μέγεθος τελικού face crop (καλό για ViT/ResNet):
FACE_SIZE  = 224

# Πόσο “αέρα” γύρω από το πρόσωπο:
MARGIN     = 20

# Αγνοεί πολύ μικρά faces (μείωσέ το αν δεν βρίσκει faces):
MIN_FACE   = 40
# =======================

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# IMPORTANT: post_process=False για να σώζονται σωστά οι εικόνες (όχι “ψυχεδελικά”)
mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=MARGIN,
    min_face_size=MIN_FACE,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=device
)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clip_id_from_path(video_path: str) -> str:
    """
    Φτιάχνει ένα safe folder name για κάθε video.
    Αν έχεις υποφακέλους μέσα στο videos, τους ενσωματώνει.
    """
    rel = os.path.relpath(video_path, VIDEO_DIR)
    rel_noext = os.path.splitext(rel)[0]
    return rel_noext.replace("\\", "__").replace("/", "__")

def extract_faces_from_video(video_path: str):
    clip_id = clip_id_from_path(video_path)
    out_folder = os.path.join(OUT_DIR, clip_id)
    ensure_dir(out_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open:", video_path)
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 25.0

    step = max(int(round(src_fps / TARGET_FPS)), 1)

    frame_idx = 0
    saved = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # OpenCV BGR -> RGB PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        out_path = os.path.join(out_folder, f"{saved+1:06d}.jpg")

        # Αφήνουμε τον MTCNN να κάνει save σωστά ως JPG
        face_tensor = mtcnn(img, save_path=out_path)

        # Αν δεν βρήκε face, σβήσε το άχρηστο αρχείο
        if face_tensor is None:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
        else:
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"✅ {clip_id}: kept {saved} face frames (~{TARGET_FPS} fps)")

def main():
    # Πιάνει και mp4 μέσα σε υποφακέλους
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "**", "*.mp4"), recursive=True))

    if not videos:
        print("❌ Δεν βρέθηκαν mp4 στο:", VIDEO_DIR)
        print("   Βάλε τα videos σου εκεί και ξανατρέξε.")
        return

    for vp in videos:
        extract_faces_from_video(vp)

    print("\nDone. Output folder:", OUT_DIR)

if __name__ == "__main__":
    main()
