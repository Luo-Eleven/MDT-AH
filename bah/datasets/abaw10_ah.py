import os
import re
import shutil
import subprocess
import yaml
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from PIL import Image
from torchvision import transforms

def _find_executable(name: str) -> str | None:
    """
    Locate an executable by searching PATH and common conda/system locations.
    Resolved at import time so all DataLoader worker processes share the same value.
    """
    found = shutil.which(name)
    if found:
        return found
    # Conda environments often prepend their bin dir to PATH; search explicitly.
    import sys
    candidates = [
        os.path.join(os.path.dirname(sys.executable), name),     # same env as Python
        f'/usr/bin/{name}',
        f'/usr/local/bin/{name}',
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


_FFMPEG  = _find_executable('ffmpeg')
_FFPROBE = _find_executable('ffprobe')

from bah.datasets.base import BaseDataset


class ABAW10_AH_Dataset(BaseDataset):
    """
    Dataset for ABAW10 Ambivalence/Hesitancy (A-H) recognition.

    Each sample returns a dict with:
        video              : (T, C, H, W) float tensor  – face frames, VideoMAE-ready
        audio              : (samples,)   float tensor  – mono 16 kHz waveform cropped to
                                                          the same time window as the frames
        transcript         : str     – transcript text covering only the sampled time window
        transcript_full    : str     – full video transcript (for context / fallback)
        transcript_chunks  : list[dict] – YAML chunks that overlap the sampled window
        time_window        : (float, float) – (start_sec, end_sec) of the sampled frames
        label              : scalar long tensor  (0 = With A-H, 1 = No A-H, -1 = unlabeled)
        video_path         : str

    How time alignment works
    ────────────────────────
    Face frames are stored as frame-N.jpg, where N is the raw video frame index.
    FPS is estimated lazily per video from audio duration (torchaudio.info) and the
    highest extracted frame number.  Given the sampled frame indices we compute:

        start_sec = first_sampled_frame_N / fps
        end_sec   = last_sampled_frame_N  / fps

    Audio is then loaded from [frame_offset … frame_offset + num_audio_samples] and
    transcript chunks are filtered to those that overlap [start_sec, end_sec].

    Directory layout expected under `root/`:
        data/
          split/                   train.txt | val.txt | test.txt
          Videos/                  …/{patient}/{visit}/{name}.mp4
          cropped-aligned-faces/   …/{patient}/{visit}/{name}.mp4/frame-N.jpg
          transcription/           …/{patient}/{visit}/{name}.mp4/{name}.yml
        test_unlabeled/
          split/test.txt   Videos/  cropped-aligned-faces/  transcription/

    split file CSV format (one video per line):
        <video_path>,<label>[,<transcript text …>]
    where <video_path> already starts with "Videos/".
    """

    PIXEL_MEAN = (0.485, 0.456, 0.406)   # ImageNet / VideoMAE normalisation
    PIXEL_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int = 16,
        img_size: int = 224,
        random_frames_crop: bool = True,
        audio_sample_rate: int = 16_000,
        audio_dir: str | None = None,
        num_windows: int = 1,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            audio_dir: Optional explicit path to pre-extracted WAV files produced
                       by scripts/extract_audio.py.  When None the dataset looks for
                       WAVs at the default location  <split_path>/audio/<video>.wav
                       and falls back to live ffmpeg decoding if the file is absent.
            num_windows: Number of uniformly-spaced frame windows to sample.
                         1 = single window (default). >1 = K windows, returns
                         video of shape (K, T, C, H, W).
        """
        super().__init__(root, split, transform=transform, **kwargs)

        self.root              = root
        self.split             = split
        self.num_frames        = num_frames
        self.img_size          = img_size
        self.random_frames_crop = random_frames_crop
        self.audio_sample_rate = audio_sample_rate
        self.audio_dir         = audio_dir
        self.num_windows       = num_windows

        # Per class_id.yaml and the challenge spec: 1 = With A-H, 0 = No A-H
        self.class_to_idx = {'No A-H': 0, 'With A-H': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.PIXEL_MEAN, std=self.PIXEL_STD),
            ])

        # Lazy FPS cache:  video_path → float
        self._fps_cache: Dict[str, float] = {}

        self._load_data()

    # ------------------------------------------------------------------
    # Data loading / indexing
    # ------------------------------------------------------------------

    def _load_data(self):
        if self.split in ('train', 'val', 'test'):
            split_file = os.path.join(self.root, 'data', 'split', f'{self.split}.txt')
            self.videos, self.labels, self.transcripts_text = self._parse_split_file(split_file)
            self.split_path = os.path.join(self.root, 'data')

        elif self.split == 'train_val':
            tr_v, tr_l, tr_t = self._parse_split_file(
                os.path.join(self.root, 'data', 'split', 'train.txt'))
            va_v, va_l, va_t = self._parse_split_file(
                os.path.join(self.root, 'data', 'split', 'val.txt'))
            self.videos           = tr_v + va_v
            self.labels           = tr_l + va_l
            self.transcripts_text = tr_t + va_t
            self.split_path       = os.path.join(self.root, 'data')

        elif self.split == 'test_unlabeled':
            split_file = os.path.join(self.root, 'test_unlabeled', 'split', 'test.txt')
            self.videos, self.labels, self.transcripts_text = self._parse_split_file(
                split_file, labeled=False)
            self.split_path = os.path.join(self.root, 'test_unlabeled')

        else:
            raise ValueError(
                f"Unknown split {self.split!r}. "
                "Choose from: train | val | test | train_val | test_unlabeled"
            )

        self._build_faces_index()
        self._build_transcript_chunks()

    def _parse_split_file(
        self,
        split_file: str,
        labeled: bool = True,
    ) -> Tuple[List[str], List[int], List[str]]:
        videos, labels, transcripts = [], [], []
        with open(split_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split(',')
                if not parts or not parts[0]:
                    continue
                videos.append(parts[0])
                if labeled and len(parts) >= 2:
                    labels.append(int(parts[1]))
                    transcripts.append(','.join(parts[2:]) if len(parts) > 2 else '')
                else:
                    labels.append(-1)
                    transcripts.append(','.join(parts[1:]) if len(parts) > 1 else '')
        return videos, labels, transcripts

    def _build_faces_index(self):
        """
        Cache sorted face-frame filenames and their raw frame numbers for every video.

        self.faces[i]        – list of 'frame-N.jpg' strings  (sorted by N)
        self.frame_numbers[i]– list of ints N matching self.faces[i]
        """
        self.faces: List[List[str]]        = []
        self.frame_numbers: List[List[int]] = []

        for video_path in self.videos:
            faces_dir = os.path.join(
                self.split_path, 'cropped-aligned-faces', video_path)
            files = sorted(
                (f for f in os.listdir(faces_dir) if f.endswith('.jpg')),
                key=self._frame_number,
            )
            self.faces.append(files)
            self.frame_numbers.append([self._frame_number(f) for f in files])

    def _build_transcript_chunks(self):
        """Cache YAML transcript chunks (text + timestamp tuples) for every video."""
        self.transcript_chunks: List[List[dict]] = []
        for video_path in self.videos:
            yml_name = os.path.basename(video_path).replace('.mp4', '.yml')
            yml_path = os.path.join(
                self.split_path, 'transcription', video_path, yml_name)
            if os.path.exists(yml_path):
                with open(yml_path, 'r', encoding='utf-8') as fh:
                    data = yaml.full_load(fh)
                self.transcript_chunks.append(data.get('chunks', []))
            else:
                self.transcript_chunks.append([])

    # ------------------------------------------------------------------
    # FPS estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_duration(mp4_path: str) -> float:
        """
        Return audio duration in seconds.

        Priority:
          1. ffprobe  – fast metadata query, no decoding required
          2. torchaudio.info – works when the backend supports it
          3. torchaudio.load full waveform – last resort
        """
        # ── 1. ffprobe ────────────────────────────────────────────────────
        if _FFPROBE:
            try:
                out = subprocess.check_output(
                    [_FFPROBE, '-v', 'error',
                     '-select_streams', 'a:0',
                     '-show_entries', 'stream=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1',
                     mp4_path],
                    stderr=subprocess.DEVNULL,
                )
                return float(out.strip())
            except Exception:
                pass

        # ── 2. torchaudio.info ────────────────────────────────────────────
        try:
            info = torchaudio.info(mp4_path)
            return info.num_frames / info.sample_rate
        except Exception:
            pass

        # ── 3. full waveform load ─────────────────────────────────────────
        waveform, sr = torchaudio.load(mp4_path)
        return waveform.shape[1] / sr

    def _get_fps(self, index: int) -> float:
        """
        Return FPS for video at `index`, estimated once and cached.

        Strategy: FPS ≈ (max_frame_number + 1) / audio_duration_seconds.
        Falls back to 25.0 if anything goes wrong.
        """
        video_path = self.videos[index]
        if video_path in self._fps_cache:
            return self._fps_cache[video_path]

        fps = 25.0  # safe default
        frame_nums = self.frame_numbers[index]
        if frame_nums:
            try:
                mp4_path = os.path.join(self.split_path, video_path)
                duration_sec = self._audio_duration(mp4_path)
                total_video_frames = frame_nums[-1] + 1   # highest extracted frame index
                if duration_sec > 0:
                    fps = total_video_frames / duration_sec
            except Exception:
                pass

        self._fps_cache[video_path] = fps
        return fps

    # ------------------------------------------------------------------
    # Per-sample helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_number(filename: str) -> int:
        match = re.search(r'frame-(\d+)\.jpg', filename)
        return int(match.group(1)) if match else -1

    def _sample_frame_indices(self, total: int) -> List[List[int]]:
        """
        Return `num_windows` lists of `num_frames` indices into the face list.

        Training  (random_frames_crop=True):  divide the frame range into
            `num_windows` equal segments, then within each segment randomly pick
            a contiguous block of `num_frames`.
        Val / Test (random_frames_crop=False): deterministic — pick the first
            window of each segment.

        Short videos (total < num_frames): padded by repeating the last frame.

        Returns:
            List[List[int]] — outer list has num_windows elements, each inner
            list has num_frames indices. For num_windows=1: returns [[indices]].
        """
        if total == 0:
            return [[0] * self.num_frames for _ in range(self.num_windows)]

        # Cap num_windows to avoid redundant/overlapping windows on short videos
        effective_windows = min(self.num_windows, max(1, total // self.num_frames))
        if effective_windows <= 1:
            # Single window mode (original behaviour)
            if total <= self.num_frames:
                indices = list(range(total)) + [total - 1] * (self.num_frames - total)
            elif self.random_frames_crop:
                start = random.randint(0, total - self.num_frames)
                indices = list(range(start, start + self.num_frames))
            else:
                indices = list(range(self.num_frames))
            return [indices]

        # Multi-window: divide total frames into K equal segments
        segment_size = total // effective_windows
        all_indices = []
        for k in range(effective_windows):
            seg_start = k * segment_size
            seg_end = (k + 1) * segment_size if k < effective_windows - 1 else total
            seg_len = seg_end - seg_start
            if seg_len <= self.num_frames:
                win = list(range(seg_start, seg_end))
                win = win + [seg_end - 1] * max(0, self.num_frames - seg_len)
            elif self.random_frames_crop:
                start = random.randint(seg_start, seg_end - self.num_frames)
                win = list(range(start, start + self.num_frames))
            else:
                # deterministic: first num_frames frames of each segment
                win = list(range(seg_start, seg_start + self.num_frames))
            all_indices.append(win)

        return all_indices

    def _load_video_tensor(self, video_path: str, face_files: List[str]) -> torch.Tensor:
        """Load, transform and stack face frames → (T, C, H, W)."""
        faces_dir = os.path.join(self.split_path, 'cropped-aligned-faces', video_path)
        frames = [
            self.transform(Image.open(os.path.join(faces_dir, f)).convert('RGB'))
            for f in face_files
        ]
        return torch.stack(frames)

    def _load_audio_tensor(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> torch.Tensor:
        """
        Load audio cropped to [start_sec, end_sec] → (samples,) mono at `audio_sample_rate`.

        Strategy priority:
          1. Pre-extracted WAV  – instantaneous seek via torchaudio.load frame_offset.
                                  Created by scripts/extract_audio.py.  Path:
                                  <split_path>/audio/<video_path[:-4]>.wav
          2. ffmpeg subprocess  – spawned per sample; works for any mp4 but adds
                                  ~0.1–0.3 s overhead and can hang on bad files.
        """
        # ── 1. Pre-extracted WAV (fast path) ─────────────────────────────
        if self.audio_dir is not None:
            wav_path = os.path.join(self.audio_dir, video_path[:-4] + '.wav')
        else:
            wav_path = os.path.join(self.split_path, 'audio', video_path[:-4] + '.wav')

        if os.path.isfile(wav_path):
            return self._load_wav_window(wav_path, start_sec, end_sec, self.audio_sample_rate)

        # ── 2. Live ffmpeg decode (slow path, fallback for missing WAVs) ──
        mp4_path = os.path.join(self.split_path, video_path)
        return self._load_audio_window(mp4_path, start_sec, end_sec, self.audio_sample_rate)

    @staticmethod
    def _load_wav_window(
        wav_path: str,
        start_sec: float,
        end_sec: float,
        target_sr: int,
    ) -> torch.Tensor:
        """
        Load a time window from a pre-extracted WAV file.

        Uses a pure-Python / numpy RIFF parser that works with no audio backend
        (no soundfile, no sox, no torchcodec required).  Supports the two formats
        produced by extract_audio.py:
            pcm_f32le  (audio_format = 3, bits = 32)   ← default extraction format
            pcm_s16le  (audio_format = 1, bits = 16)   ← legacy / re-extracted

        Only the requested time window is read from disk; the rest of the file
        is never loaded into memory.
        """
        import struct

        with open(wav_path, 'rb') as fh:
            # ── RIFF header ────────────────────────────────────────────
            riff, _, wave = struct.unpack('<4sI4s', fh.read(12))
            if riff != b'RIFF' or wave != b'WAVE':
                raise ValueError(f'Not a valid WAV file: {wav_path}')

            # ── Scan chunks until we have both fmt  and data ───────────
            sample_rate = num_channels = bits = audio_fmt = None
            data_offset = data_size = None

            while True:
                header = fh.read(8)
                if len(header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack('<4sI', header)

                if chunk_id == b'fmt ':
                    fmt_data = fh.read(chunk_size)
                    audio_fmt, num_channels, sample_rate, _, _, bits = \
                        struct.unpack('<HHIIHH', fmt_data[:16])
                    # WAVE_FORMAT_EXTENSIBLE (0xFFFE): the real format code sits
                    # in the first 2 bytes of the SubFormat GUID at offset 24.
                    if audio_fmt == 0xFFFE and len(fmt_data) >= 26:
                        audio_fmt = struct.unpack_from('<H', fmt_data, 24)[0]
                elif chunk_id == b'data':
                    data_offset = fh.tell()
                    data_size   = chunk_size
                    break
                else:
                    fh.seek(chunk_size, 1)

            if data_offset is None or sample_rate is None:
                raise ValueError(f'Malformed WAV (missing fmt/data): {wav_path}')

            # ── Determine numpy dtype ──────────────────────────────────
            if audio_fmt == 3 and bits == 32:       # IEEE float32
                dtype = np.float32
                scale = None
            elif audio_fmt == 1 and bits == 16:     # signed int16 PCM
                dtype = np.int16
                scale = 1.0 / 32768.0
            else:
                raise ValueError(
                    f'Unsupported WAV format (audio_fmt={audio_fmt}, bits={bits}) '
                    f'in {wav_path}')

            # ── Compute byte range for the requested window ────────────
            bytes_per_frame = (bits // 8) * num_channels
            total_frames    = data_size // bytes_per_frame
            start_frame     = min(int(start_sec * sample_rate), total_frames)
            end_frame       = (min(int(end_sec * sample_rate), total_frames)
                               if end_sec > start_sec else total_frames)
            n_frames        = max(1, end_frame - start_frame)

            fh.seek(data_offset + start_frame * bytes_per_frame)
            raw = fh.read(n_frames * bytes_per_frame)

        # ── Decode ────────────────────────────────────────────────────
        samples = np.frombuffer(raw, dtype=dtype).copy()
        if scale is not None:
            samples = samples.astype(np.float32) * scale

        # interleaved channels → (channels, frames)
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels).mean(axis=1)

        waveform = torch.from_numpy(samples)   # (frames,)

        # ── Resample if needed (pure torch, no audio backend) ─────────
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=target_sr,
            ).squeeze(0)

        return waveform   # (samples,)

    def _load_audio_window(
        self,
        mp4_path: str,
        start_sec: float,
        end_sec: float,
        target_sr: int,
    ) -> torch.Tensor:
        """
        Return a mono float32 waveform tensor of shape (samples,) at `target_sr`.

        When ffmpeg is available it is used exclusively — torchaudio is NOT used
        as a fallback for mp4 files because its torchcodec backend may be broken.
        If ffmpeg produces empty output (silent segment, very short window, etc.)
        a zero-filled tensor of the expected length is returned instead.
        """
        # ── ffmpeg path (primary, used whenever the binary is available) ──
        if _FFMPEG:
            return self._ffmpeg_decode(mp4_path, start_sec, end_sec, target_sr)

        # ── torchaudio fallback (only for environments where it works) ────
        return self._torchaudio_decode(mp4_path, start_sec, end_sec, target_sr)

    @staticmethod
    def _ffmpeg_decode(
        mp4_path: str,
        start_sec: float,
        end_sec: float,
        target_sr: int,
    ) -> torch.Tensor:
        """
        Decode an audio window with the ffmpeg binary.

        -ss before -i  →  fast keyframe seek (accurate enough for multi-second windows)
        -ac 1          →  mono downmix
        -ar target_sr  →  resample in one pass
        -f f32le       →  raw 32-bit float PCM piped to stdout
        """
        cmd = [_FFMPEG, '-y']
        if start_sec > 0.0:
            cmd += ['-ss', f'{start_sec:.6f}']
        cmd += ['-i', mp4_path]
        if end_sec > start_sec:
            cmd += ['-t', f'{end_sec - start_sec:.6f}']
        cmd += ['-vn', '-ac', '1', '-ar', str(target_sr), '-f', 'f32le', '-']

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture instead of DEVNULL to aid debugging
        )

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        if audio.size > 0:
            return torch.from_numpy(audio.copy())   # (samples,)

        # ffmpeg returned no audio — compute expected length and return silence.
        # This can happen for very short windows or videos with no audio track.
        expected_samples = max(1, int((end_sec - start_sec) * target_sr))
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors='replace')[-300:]
            import warnings
            warnings.warn(
                f'ffmpeg failed (rc={result.returncode}) for {mp4_path} '
                f'[{start_sec:.2f}s – {end_sec:.2f}s]. Returning silence.\n{stderr_msg}',
                RuntimeWarning,
                stacklevel=4,
            )
        return torch.zeros(expected_samples)

    @staticmethod
    def _torchaudio_decode(
        mp4_path: str,
        start_sec: float,
        end_sec: float,
        target_sr: int,
    ) -> torch.Tensor:
        """torchaudio-based decode (seek then slice). Used only when ffmpeg is absent."""
        try:
            info = torchaudio.info(mp4_path)
            native_sr     = info.sample_rate
            frame_offset  = max(0, int(start_sec * native_sr))
            window_frames = int((end_sec - start_sec) * native_sr) if end_sec > start_sec else -1
            waveform, sr  = torchaudio.load(
                mp4_path,
                frame_offset=frame_offset,
                num_frames=window_frames if window_frames > 0 else -1,
            )
        except Exception:
            waveform, sr = torchaudio.load(mp4_path)
            if end_sec > start_sec:
                s = max(0, int(start_sec * sr))
                e = min(int(end_sec * sr), waveform.shape[1])
                waveform = waveform[:, s:e]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        return waveform.squeeze(0)

    @staticmethod
    def _filter_transcript_chunks(
        chunks: List[dict],
        start_sec: float,
        end_sec: float,
    ) -> List[dict]:
        """
        Return only the chunks whose timestamp overlaps [start_sec, end_sec].

        Overlap condition:  chunk_start < end_sec  AND  chunk_end > start_sec
        """
        filtered = []
        for chunk in chunks:
            ts = chunk.get('timestamp')
            if ts is None:
                continue
            c_start, c_end = float(ts[0]), float(ts[1])
            if c_start < end_sec and c_end > start_sec:
                filtered.append(chunk)
        return filtered

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> dict:
        video_path = self.videos[index]
        face_files = self.faces[index]
        frame_nums = self.frame_numbers[index]

        # ── 1. Sample frame indices — returns List[List[int]] (K windows × num_frames) ─
        all_window_indices = self._sample_frame_indices(len(face_files))

        # ── 2. Load video frames and audio for each window ───────────────
        fps = self._get_fps(index)

        video_windows = []
        audio_windows = []
        time_windows = []

        for win_indices in all_window_indices:
            win_files      = [face_files[i]  for i in win_indices]
            win_frame_nums = [frame_nums[i]  for i in win_indices]

            win_start_sec = win_frame_nums[0]  / fps if win_frame_nums else 0.0
            win_end_sec   = win_frame_nums[-1] / fps if win_frame_nums else 0.0
            if win_end_sec <= win_start_sec:
                win_end_sec = win_start_sec + (self.num_frames / fps)

            video_win = self._load_video_tensor(video_path, win_files)                   # (T, C, H, W)
            audio_win = self._load_audio_tensor(video_path, win_start_sec, win_end_sec)  # (samples,)

            video_windows.append(video_win)
            audio_windows.append(audio_win)
            time_windows.append((win_start_sec, win_end_sec))

        num_win = len(all_window_indices)
        if num_win > 1:
            video_tensor = torch.stack(video_windows)   # (K, T, C, H, W)
            max_audio_len = max(a.shape[0] for a in audio_windows)
            audio_padded = torch.zeros(num_win, max_audio_len)
            for i, a in enumerate(audio_windows):
                audio_padded[i, :a.shape[0]] = a
            audio_tensor = audio_padded                   # (K, max_samples)
        else:
            video_tensor = video_windows[0]
            audio_tensor = audio_windows[0]

        # ── 3. Filter transcript — use first (or union) time window ──────
        first_start, first_end = time_windows[0]
        all_chunks     = self.transcript_chunks[index]
        window_chunks  = self._filter_transcript_chunks(all_chunks, first_start, first_end)

        # Build a single string from the overlapping chunks.
        # Fall back to the full transcript when the window yields nothing
        # (e.g. silence / missing YAML).
        if window_chunks:
            window_text = ' '.join(c['text'].strip() for c in window_chunks)
        else:
            window_text = self.transcripts_text[index]

        # ── 4. Label ──────────────────────────────────────────────────────
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return {
            # ── multimodal inputs ──────────────────────────────────────
            'video':             video_tensor,          # (T, C, H, W) or (K, T, C, H, W)
            'audio':             audio_tensor,          # (samples,) or (K, samples)
            'transcript':        window_text,           # str – window-aligned, for BERT
            # ── extras ────────────────────────────────────────────────
            'transcript_full':   self.transcripts_text[index],  # full video text
            'transcript_chunks': window_chunks,         # [{text, timestamp}, …] – window only
            'time_window':       time_windows[0] if num_win == 1 else time_windows,
            # ── supervision / meta ────────────────────────────────────
            'label':             label,                 # long, -1 if unlabeled
            'video_path':        video_path,
        }
