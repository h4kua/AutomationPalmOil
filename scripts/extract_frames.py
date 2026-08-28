"""
Fresh ffmpeg-based frame extractor for the new Oct-2025 MinIO drone-capture
footage and the raw record-output footage. No existing extraction script
covers these paths (filter_frames.py / mining.py only filter, they don't
extract from source video).

Convention matched to the original /data/frames corpus: one subfolder per
video (basename minus extension), frame_NNNNNN.jpg sequential numbering,
sampled at 1 frame / 5s (matches the original corpus's ~4.88s/frame rate,
confirmed by inspecting an existing extracted folder's frame count vs its
video's ffprobe duration).

Output goes to /data/frames_new/ (NOT /data/frames/) to avoid any collision
risk with the original 111-video corpus's frame numbering, per project
convention when introducing a second extraction pass over different source
videos.

Read-only w.r.t. source videos; this only ever writes to /data/frames_new/.
"""
import subprocess
import glob
import os
import sys

SOURCES = [
    ('/mnt/drone-capture-minio', 'minio'),
    ('/mnt/record-output', 'recout'),
]
OUT_ROOT = '/data/frames_new'
INTERVAL_SEC = 5

os.makedirs(OUT_ROOT, exist_ok=True)


def get_duration(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


total_videos = 0
total_frames = 0
failures = []

for src_dir, tag in SOURCES:
    videos = sorted(glob.glob(os.path.join(src_dir, '*.mp4')))
    print(f'--- {tag}: {len(videos)} videos in {src_dir} ---', flush=True)
    for i, video_path in enumerate(videos):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(OUT_ROOT, stem)
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            continue  # already extracted (safe to resume/re-run)
        os.makedirs(out_dir, exist_ok=True)

        dur = get_duration(video_path)
        if dur is None:
            failures.append(video_path)
            continue

        if dur < INTERVAL_SEC:
            # too short for the fps filter to yield anything -- grab one
            # frame at the midpoint so short bursts aren't dropped entirely
            mid = max(0.0, dur / 2)
            cmd = ['ffmpeg', '-y', '-ss', str(mid), '-i', video_path,
                   '-frames:v', '1', '-q:v', '3',
                   os.path.join(out_dir, 'frame_000001.jpg')]
        else:
            cmd = ['ffmpeg', '-y', '-i', video_path,
                   '-vf', f'fps=1/{INTERVAL_SEC}', '-q:v', '3',
                   os.path.join(out_dir, 'frame_%06d.jpg')]

        r = subprocess.run(cmd, capture_output=True, text=True)
        n_frames = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
        if n_frames == 0:
            failures.append(video_path)
        total_videos += 1
        total_frames += n_frames

        if (i + 1) % 20 == 0:
            print(f'  [{tag}] {i+1}/{len(videos)} videos done, '
                  f'{total_frames} frames so far', flush=True)

print()
print(f'Done. {total_videos} videos processed, {total_frames} frames extracted '
      f'into {OUT_ROOT}')
if failures:
    print(f'{len(failures)} videos produced 0 frames / failed:')
    for f in failures:
        print(f'  {f}')
