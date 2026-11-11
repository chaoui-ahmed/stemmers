"""Low-memory stem separation with Demucs (htdemucs).

The song is processed in overlapping chunks and each stem is streamed to disk,
so peak memory stays roughly constant whatever the song length.
"""
from __future__ import annotations

import gc
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


def _tune_malloc():
    """glibc keeps one memory pool per thread, which bloats a threaded server like
    Streamlit. Two pools is plenty here. No-op outside Linux."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-8, 2)  # M_ARENA_MAX
        return libc
    except Exception:
        return None


_LIBC = _tune_malloc()


def release_memory():
    """Give freed memory back to the system after a job."""
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass
REPO = os.path.join(HERE, "models")
MODEL_SIG = "955717e8-8726e21a"
MODEL_URL = f"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/{MODEL_SIG}.th"
SR = 44100


def ensure_model(log=print):
    os.makedirs(REPO, exist_ok=True)
    yaml = os.path.join(REPO, "htdemucs.yaml")
    if not os.path.exists(yaml):
        with open(yaml, "w") as f:
            f.write("models: ['955717e8']\n")
    th = os.path.join(REPO, f"{MODEL_SIG}.th")
    if not os.path.exists(th):
        log("Downloading the separation model (84 MB, first run only) …")
        tmp = th + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, th)
    return REPO


_MODEL = None


WINDOW_S = float(os.environ.get("S2S_WINDOW", "4.0"))  # model window (7.8 s in the original model): 4 s cuts peak memory by ~400 MB


def _allow_demucs_classes(torch):
    """torch >= 2.6 loads checkpoints in weights-only mode; allow exactly the classes Demucs stores."""
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:
        return
    import fractions
    allowed = [fractions.Fraction, np.dtype]
    try:
        try:
            from numpy._core.multiarray import scalar
        except ImportError:
            from numpy.core.multiarray import scalar
        allowed += [scalar, (scalar, "numpy.core.multiarray.scalar")]
    except Exception:
        pass
    allowed += [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "bool")]
    for mod, cls in (("demucs.htdemucs", "HTDemucs"), ("demucs.hdemucs", "HDemucs"), ("demucs.demucs", "Demucs")):
        try:
            allowed.append(getattr(__import__(mod, fromlist=[cls]), cls))
        except Exception:
            pass
    try:
        add(allowed)
    except Exception:
        add([a for a in allowed if not isinstance(a, tuple)])


def load_model(log=print):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from pathlib import Path
    from demucs.pretrained import get_model
    torch.set_num_threads(int(os.environ.get("S2S_THREADS", "2")))
    _allow_demucs_classes(torch)
    m = get_model(name="htdemucs", repo=Path(ensure_model(log)))
    m.eval()
    from fractions import Fraction
    win = Fraction(int(WINDOW_S * 10), 10)
    for sub in getattr(m, "models", [m]):
        sub.use_train_segment = False
        sub.segment = win
    m.segment = win
    release_memory()
    _MODEL = m
    return m


def separate(song_path: str, out_dir: str, chunk_s: float = 15.0, overlap_s: float = 2.0,
             progress=None, log=print) -> dict:
    """Write one 16-bit WAV per stem into out_dir. Returns {stem: path}."""
    import torch
    from demucs.apply import apply_model

    model = load_model(log)
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "_input.wav")
    _decode(song_path, wav)                      # decoded on disk, never fully in memory
    src = sf.SoundFile(wav)
    n = src.frames
    # loudness stats of the mono mix, block by block (same normalisation as Demucs/UVR)
    tot = tot2 = 0.0
    for blk in src.blocks(blocksize=SR * 10, dtype="float32", always_2d=True):
        m = blk.mean(1, dtype=np.float64)
        tot += m.sum()
        tot2 += (m * m).sum()
    mean = tot / n
    std = float(np.sqrt(max(tot2 / n - mean * mean, 0.0)) + 1e-8)

    paths = {s: os.path.join(out_dir, f"{s}.wav") for s in model.sources}
    writers = {s: sf.SoundFile(p, "w", SR, 2, subtype="PCM_16") for s, p in paths.items()}

    chunk = int(chunk_s * SR)
    ov = int(overlap_s * SR)
    fade_in = np.linspace(0, 1, ov, dtype=np.float32)
    tail = None            # previous chunk's overlap region, per stem: (4, 2, ov)
    start = 0
    try:
        while start < n:
            end = min(n, start + chunk + ov)
            src.seek(start)
            seg = src.read(end - start, dtype="float32", always_2d=True).T
            if seg.shape[0] == 1:
                seg = np.vstack([seg, seg])
            seg = np.ascontiguousarray((seg[:2] - mean) / std, dtype=np.float32)
            with torch.inference_mode():
                out = apply_model(model, torch.from_numpy(seg)[None], shifts=1, split=True,
                                  overlap=0.25, device="cpu")[0].numpy()
            out = out * std + mean           # (4, 2, len)
            del seg
            if tail is not None:
                k = min(ov, out.shape[2])
                out[:, :, :k] = tail[:, :, :k] * (1 - fade_in[:k]) + out[:, :, :k] * fade_in[:k]
            last = end >= n
            body_end = out.shape[2] if last else out.shape[2] - ov
            for i, s in enumerate(model.sources):
                writers[s].write(np.clip(out[i, :, :body_end].T, -1, 1))
            tail = None if last else out[:, :, body_end:].copy()
            del out
            release_memory()
            start = start + body_end if tail is not None else n
            if progress:
                progress(min(start / n, 1.0))
    finally:
        for w in writers.values():
            w.close()
        src.close()
        try:
            os.remove(wav)
        except OSError:
            pass
    release_memory()
    return paths


def _decode(path: str, out_wav: str):
    """Any audio file -> 44.1 kHz stereo 16-bit WAV on disk (ffmpeg, or librosa as a fallback)."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
                            "-c:a", "pcm_s16le", out_wav], capture_output=True)
        if r.returncode == 0 and os.path.exists(out_wav):
            return
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=False, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    sf.write(out_wav, y[:2].T, SR, subtype="PCM_16")


def preview(path: str, out_path: str, sr: int = 22050):  # noqa: D401
    """Small mono MP3 for the page's audio player (keeps server memory low).
    Uses ffmpeg; falls back to OGG written in small blocks (one big OGG write can crash libsndfile)."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff:
        out_path = os.path.splitext(out_path)[0] + ".mp3"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                            "-b:a", "64k", out_path], capture_output=True)
        if r.returncode == 0 and os.path.exists(out_path):
            return out_path
    out_path = os.path.splitext(out_path)[0] + ".ogg"
    with sf.SoundFile(path) as src, sf.SoundFile(out_path, "w", src.samplerate, 1, format="OGG", subtype="VORBIS") as dst:
        for blk in src.blocks(blocksize=src.samplerate, dtype="float32", always_2d=True):
            dst.write(blk.mean(1))
    return out_path
