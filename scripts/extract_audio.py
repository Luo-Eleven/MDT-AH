"""
Pre-extract audio from every mp4 in the dataset to 16 kHz mono WAV files.

Run ONCE before training.  Subsequent training runs load WAV directly, which
is ~50× faster than spawning an ffmpeg subprocess per sample and eliminates
the DataLoader hang caused by ffmpeg stalling on unusual mp4 files.

Output layout mirrors the Videos/ directory tree under an audio/ sibling:
    data/data/audio/Videos/{patient}/{visit}/{name}.wav
    data/test_unlabeled/audio/Videos/{patient}/{visit}/{name}.wav

Usage:
    conda run -n conda3.12 python scripts/extract_audio.py
    conda run -n conda3.12 python scripts/extract_audio.py --data_root data --workers 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def find_ffmpeg() -> str:
    path = shutil.which('ffmpeg')
    if path:
        return path
    candidate = os.path.join(os.path.dirname(sys.executable), 'ffmpeg')
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError(
        'ffmpeg not found. Install it with: conda install -c conda-forge ffmpeg')


def collect_video_paths(data_root: str) -> list[tuple[str, str]]:
    """
    Return list of (mp4_abs_path, wav_abs_path) for every unique video in all splits.
    """
    pairs: dict[str, str] = {}   # mp4_abs → wav_abs

    splits_dirs = [
        (os.path.join(data_root, 'data'),         os.path.join(data_root, 'data',         'split')),
        (os.path.join(data_root, 'test_unlabeled'), os.path.join(data_root, 'test_unlabeled', 'split')),
    ]

    for split_path, split_dir in splits_dirs:
        if not os.path.isdir(split_dir):
            continue
        for txt in os.listdir(split_dir):
            if not txt.endswith('.txt'):
                continue
            with open(os.path.join(split_dir, txt), encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if not parts or not parts[0]:
                        continue
                    rel = parts[0]                          # e.g. Videos/82553/.../foo.mp4
                    mp4 = os.path.join(split_path, rel)
                    wav = os.path.join(split_path, 'audio', rel[:-4] + '.wav')
                    pairs[mp4] = wav

    return list(pairs.items())


def extract_one(mp4: str, wav: str, ffmpeg: str, sr: int, timeout: int) -> tuple[str, str]:
    """
    Extract audio from `mp4` to `wav`.  Returns (mp4, status_string).
    """
    if os.path.isfile(wav):
        return mp4, 'skipped (already exists)'

    os.makedirs(os.path.dirname(wav), exist_ok=True)
    cmd = [ffmpeg, '-y', '-i', mp4,
           '-vn', '-ac', '1', '-ar', str(sr),
           '-c:a', 'pcm_f32le',    # 32-bit float WAV – matches what the dataset expects
           wav]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace')[-200:]
            return mp4, f'FAILED (rc={result.returncode}): {err}'
        return mp4, 'ok'
    except subprocess.TimeoutExpired:
        return mp4, f'FAILED (timeout after {timeout}s)'
    except Exception as exc:
        return mp4, f'FAILED ({exc})'


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Pre-extract mp4 audio to WAV')
    p.add_argument('--data_root', default=str(ROOT / 'data'), type=str)
    p.add_argument('--sample_rate', default=16_000, type=int)
    p.add_argument('--workers',  default=4, type=int,
                   help='Parallel ffmpeg processes')
    p.add_argument('--timeout',  default=120, type=int,
                   help='Seconds before killing a stalled ffmpeg process')
    return p.parse_args()


def main():
    args   = parse_args()
    ffmpeg = find_ffmpeg()
    print(f'ffmpeg : {ffmpeg}')
    print(f'data   : {args.data_root}')
    print(f'sr     : {args.sample_rate} Hz')
    print(f'workers: {args.workers}')

    pairs = collect_video_paths(args.data_root)
    total = len(pairs)
    print(f'\nFound {total} unique videos\n')

    done = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, mp4, wav, ffmpeg, args.sample_rate, args.timeout): mp4
            for mp4, wav in pairs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            mp4, status = fut.result()
            name = os.path.basename(mp4)
            if status == 'ok':
                done += 1
            elif status.startswith('skipped'):
                skipped += 1
            else:
                failed += 1
                print(f'  [{i:04d}/{total}]  {name}  →  {status}')

            if i % 50 == 0 or i == total:
                print(f'  progress: {i}/{total}  '
                      f'(done={done} skipped={skipped} failed={failed})')

    print(f'\nFinished.  done={done}  skipped={skipped}  failed={failed}')
    if failed:
        print('Some files failed — they will be decoded with ffmpeg at training time.')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
