from pathlib import Path
import subprocess

OPENFACE = Path(r"D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe") 
CASME_ROOT = Path(r"D:\MSc_Data_ML\Casme_2\CASME2_RAW\CASME2-RAW")
OUT_ROOT = Path(r"D:\MSc_Data_ML\Casme_2\openface_out")

IMG_EXTS = {".jpg", ".png", ".bmp"}

def has_frames(ep_dir: Path) -> bool:
    # fast: checks only a few files
    for p in ep_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            return True
    return False

def main():
    if not OPENFACE.exists():
        raise FileNotFoundError(f"OpenFace not found: {OPENFACE}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # EP folders in CASME II look like: sub01/EP02_01f
    ep_dirs = [p for p in CASME_ROOT.glob("sub*/EP*") if p.is_dir() and has_frames(p)]
    print(f"Found {len(ep_dirs)} EP folders with frames.")

    for ep in ep_dirs:
        rel = ep.relative_to(CASME_ROOT)           # e.g., sub01/EP02_01f
        out_dir = OUT_ROOT / rel                  # mirror structure
        out_dir.mkdir(parents=True, exist_ok=True)

        log_file = out_dir / "openface_log.txt"
        print(f"Running: {rel}")

        # Run from OpenFace directory so models load correctly
        cmd = [str(OPENFACE), "-fdir", str(ep), "-out_dir", str(out_dir)]
        with log_file.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(OPENFACE.parent), stdout=f, stderr=subprocess.STDOUT, check=False)

    print("Done.")

if __name__ == "__main__":
    main()
