import sys
import os
from multiprocessing.pool import ThreadPool

from yt_dlp import YoutubeDL
import ffmpeg

# -----------------------------
# Helper: convert MM:SS or HH:MM:SS to seconds
# -----------------------------
def time_to_seconds(t):
    t = t.strip()
    if not t:
        return 0.0
    parts = t.split(':')
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(parts) == 3:  # HH:MM:SS
        return parts[0]*3600 + parts[1]*60 + parts[2]
    elif len(parts) == 2:  # MM:SS
        return parts[0]*60 + parts[1]
    elif len(parts) == 1:  # single number -> minutes
        return parts[0]*60
    else:  # seconds
        return 0.0

# -----------------------------
# Video info class
# -----------------------------
class VidInfo:
    def __init__(self, yt_id, file_name, start_time, end_time, outdir):
        self.yt_id = yt_id
        self.start_time = time_to_seconds(start_time)
        self.end_time = time_to_seconds(end_time)
        self.out_filename = os.path.join(outdir, file_name + '.mp4')

# -----------------------------
# Download a single video clip
# -----------------------------
def download(vidinfo):
    yt_base_url = 'https://www.youtube.com/watch?v='
    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        'format': '22/18',
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            download_url = ydl.extract_info(url=yt_url, download=False)['url']
    except Exception:
        return '{}, ERROR (youtube)!'.format(vidinfo.yt_id)

    try:
        (
            ffmpeg
                .input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)
                .output(vidinfo.out_filename, format='mp4', r=25, vcodec='libx264',
                        crf=18, preset='veryfast', pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
                        strict='experimental')
                .global_args('-y')
                .global_args('-loglevel', 'error')
                .run()
        )
    except Exception:
        return '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)

    return '{}, DONE!'.format(vidinfo.yt_id)

# -----------------------------
# Main script
# -----------------------------
if __name__ == '__main__':
    # --- Handle arguments ---
    if len(sys.argv) > 2:
        out_dir = sys.argv[1]
        csv_file = sys.argv[2]
    else:
        # Defaults for Jupyter / testing
        out_dir = 'videos'
        csv_file = 'my_videos.csv'

    os.makedirs(out_dir, exist_ok=True)

    # --- Read CSV (skip header, ignore extra columns) ---
    vidinfos = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                print(f"Skipping bad row (not enough columns): {line}")
                continue
            yt_id, file_name, start_time, end_time = parts[:4]  # ignore flag or extra columns
            try:
                vidinfos.append(VidInfo(yt_id, file_name, start_time, end_time, out_dir))
            except Exception as e:
                print(f"Skipping bad row: {line} -> {e}")

    # --- Download videos ---
    bad_files_path = f'bad_files_{out_dir}.txt'
    bad_files = open(bad_files_path, 'w')
    results = ThreadPool(5).imap_unordered(download, vidinfos)
    cnt = 0
    for r in results:
        cnt += 1
        print(cnt, '/', len(vidinfos), r)
        if 'ERROR' in r:
            bad_files.write(r + '\n')
    bad_files.close()
    print(f"Bad files logged to {bad_files_path}")
