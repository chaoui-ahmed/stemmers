"""Song in, instruments out.

1. Split the song into stems (vocals, bass, drums, other) with the Demucs
   model that Ultimate Vocal Remover already downloaded (htdemucs).
2. In each melodic stem, hunt for the cleanest single held note: steady pitch,
   loud enough, long enough, and harmonic (not a chord or a smear).
3. Turn that note into a full instrument with sample2synth.
4. Drums: find the cleanest kick, snare, hi-hat and one extra hit, and map
   them into a drum kit (General MIDI keys, so Logic's Drum Machine Designer
   or any drum pattern plays them correctly).
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import librosa

from .analyze import SR
from . import export as E

STEM_SETTINGS = {
    #          pitch search range (Hz)  keyboard range (MIDI)  timbre mix
    "vocals": dict(fmin=70, fmax=1100, low=36, high=84, timbre=0.85, label="Vocals"),
    "bass":   dict(fmin=30, fmax=400,  low=24, high=72, timbre=0.4,  label="Bass"),
    "other":  dict(fmin=60, fmax=2000, low=24, high=96, timbre=0.6,  label="Other (keys, guitar, synths)"),
    "guitar": dict(fmin=70, fmax=1400, low=28, high=88, timbre=0.6,  label="Guitar"),
    "piano":  dict(fmin=27, fmax=2100, low=21, high=96, timbre=0.6,  label="Piano"),
}
DRUM_KEYS = {"kick": 36, "snare": 38, "hihat": 42, "extra": 49}
DRUM_LABELS = {"kick": "Kick", "snare": "Snare", "hihat": "Hi-hat", "extra": "Extra hit (tom / crash / perc)"}


def log_default(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- separation

def find_uvr(start: str | None = None) -> str | None:
    """Locate the Ultimate Vocal Remover folder (needs demucs/ and the models)."""
    cands = []
    if start:
        cands.append(start)
    if os.environ.get("SAMPLE2SYNTH_UVR"):
        cands.append(os.environ["SAMPLE2SYNTH_UVR"])
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.dirname(os.path.dirname(here)), os.path.dirname(here),
              os.path.expanduser("~/ultimatevocalremovergui"),
              "/Applications/Ultimate Vocal Remover.app/Contents/MacOS"]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "demucs")) and os.path.isdir(os.path.join(c, "models", "Demucs_Models")):
            return c
    return None


def available_demucs_models(uvr: str):
    repo = os.path.join(uvr, "models", "Demucs_Models", "v3_v4_repo")
    out = []
    for y in glob.glob(os.path.join(repo, "*.yaml")):
        name = os.path.splitext(os.path.basename(y))[0]
        sigs = re.findall(r"[0-9a-f]{8}", open(y).read())
        if sigs and all(glob.glob(os.path.join(repo, f"{s}*.th")) for s in sigs):
            out.append(name)
    return repo, sorted(out, key=lambda n: (n != "htdemucs_6s", n != "htdemucs_ft", n != "htdemucs", n))


def _allow_demucs_classes(torch):
    """torch >= 2.6 loads checkpoints in weights-only mode, which refuses the
    model classes stored in Demucs files. Allow exactly those classes."""
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:
        return
    import fractions
    allowed = [fractions.Fraction, np.dtype]
    try:
        try:
            from numpy._core.multiarray import scalar  # numpy 2
        except ImportError:
            from numpy.core.multiarray import scalar   # numpy 1
        allowed += [scalar, (scalar, "numpy.core.multiarray.scalar")]
    except Exception:
        pass
    try:
        allowed += [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "bool")]
    except Exception:
        pass
    for mod, cls in (("demucs.htdemucs", "HTDemucs"), ("demucs.hdemucs", "HDemucs"),
                     ("demucs.demucs", "Demucs"), ("demucs.tasnet", "ConvTasNet")):
        try:
            m = __import__(mod, fromlist=[cls])
            allowed.append(getattr(m, cls))
        except Exception:
            pass
    try:
        add(allowed)
    except Exception:  # older torch without (obj, name) tuples
        add([a for a in allowed if not isinstance(a, tuple)])


def separate(song: str, out_dir: str, uvr: str, model_name: str | None = None, shifts: int = 1,
             log=log_default):
    import torch
    if uvr not in sys.path:
        sys.path.insert(0, uvr)
    from pathlib import Path
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    repo, models = available_demucs_models(uvr)
    if not models:
        raise RuntimeError("No Demucs v4 model found in Ultimate Vocal Remover. Open UVR, pick Demucs, "
                           "and download 'v4 | htdemucs' (or htdemucs_6s for guitar and piano stems).")
    name = model_name or models[0]
    if name not in models:
        raise RuntimeError(f"Model {name} is not downloaded. Available: {', '.join(models)}")
    log(f"Separating with UVR's Demucs model '{name}' …")
    _allow_demucs_classes(torch)
    model = get_model(name=name, repo=Path(repo))
    model.eval()

    mix, _ = librosa.load(song, sr=44100, mono=False)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    mix_t = torch.tensor(mix, dtype=torch.float32)
    ref = mix_t.mean(0)
    mix_n = (mix_t - ref.mean()) / (ref.std() + 1e-8)

    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    t0 = time.time()
    sources = None
    for dev in devices:
        try:
            model.to(dev)
            with torch.no_grad():
                sources = apply_model(model, mix_n[None], shifts=shifts, split=True, overlap=0.25,
                                      static_shifts=max(1, shifts), device=dev)[0]
            break
        except Exception as e:  # MPS can miss an op on some torch versions
            log(f"  {dev} failed ({type(e).__name__}), trying the next device …")
    if sources is None:
        raise RuntimeError("Separation failed on every device.")
    sources = (sources * ref.std() + ref.mean()).cpu().numpy()
    log(f"  done in {time.time() - t0:.0f} s")
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for stem, audio in zip(model.sources, sources):
        p = os.path.join(out_dir, f"{stem}.wav")
        sf.write(p, audio.T, 44100)
        paths[stem] = p
    return paths


def stems_from_folder(folder: str):
    """Use stems UVR already exported, e.g. '1_Song_(Vocals).wav'."""
    paths = {}
    for f in glob.glob(os.path.join(folder, "*")):
        low = os.path.basename(f).lower()
        if not low.endswith((".wav", ".flac", ".mp3", ".aif", ".aiff", ".m4a")):
            continue
        for stem in ("vocals", "bass", "drums", "other", "guitar", "piano"):
            if re.search(rf"[\(_ -]{stem}[\)_ .-]", low) or low.startswith(stem + "."):
                if "no " + stem not in low and "no_" + stem not in low and "instrumental" not in low:
                    paths.setdefault(stem, f)
    return paths


# ---------------------------------------------------------------- note hunting

def _mono(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def find_notes(y: np.ndarray, fmin: float, fmax: float, sr: int = 22050, hop: int = 512,
               max_cents: float = 45.0, min_dur: float = 0.3):
    """Candidate held notes: runs of voiced frames with steady pitch."""
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048,
                                    hop_length=hop, fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, prob, rms = f0[:n], voiced[:n], prob[:n], rms[:n]
    loud_ref = np.percentile(rms[rms > 0], 95) if (rms > 0).any() else 1.0
    db = 20 * np.log10(rms / loud_ref + 1e-9)
    ok = voiced & np.isfinite(f0) & (prob > 0.25) & (db > -30)
    midi = np.where(ok, librosa.hz_to_midi(np.where(ok, f0, 1.0)), np.nan)
    dt = hop / sr
    cands = []
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        run = [midi[i]]
        while j + 1 < n and ok[j + 1]:
            med = np.median(run[-15:])
            if abs(midi[j + 1] - med) * 100 > max_cents * 2.2:
                break
            j += 1
            run.append(midi[j])
        dur = (j - i + 1) * dt
        if dur >= min_dur:
            seg = np.array(run)
            centre = float(np.median(seg))
            dev = float(np.median(np.abs(seg - centre)) * 100)
            cands.append(dict(start=i * dt, end=(j + 1) * dt, midi=centre, wobble=dev,
                              prob=float(np.mean(prob[i:j + 1])), db=float(np.median(db[i:j + 1]))))
        i = j + 1
    return cands


def harmonicity(clip: np.ndarray, f0: float, sr: int = SR):
    """Share of energy sitting on the harmonics of f0 (1 = one clean note)."""
    n = len(clip)
    if n < 2048:
        return 0.0
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(clip * w)) ** 2
    fr = np.fft.rfftfreq(n, 1 / sr)
    band = (fr > 0.5 * f0) & (fr < min(10000, sr / 2))
    total = spec[band].sum() + 1e-20
    on = np.zeros_like(band)
    for k in range(1, int(min(10000, sr / 2) // f0) + 1):
        on |= np.abs(fr - k * f0) < max(0.03 * k * f0, 6)
    return float(spec[band & on].sum() / total)


def score_note(c, h):
    dur = min(c["end"] - c["start"], 1.6)
    steady = np.exp(-c["wobble"] / 25)
    loud = np.clip((c["db"] + 30) / 24, 0, 1)
    return dur * (0.4 + 0.6 * steady) * c["prob"] * (0.3 + 0.7 * loud) * h ** 2


def best_notes(stem_path: str, settings: dict, top: int = 3, log=log_default):
    y22 = _mono(stem_path, 22050)
    cands = find_notes(y22, settings["fmin"], settings["fmax"])
    if not cands:
        return [], None
    cands.sort(key=lambda c: (c["end"] - c["start"]) * c["prob"], reverse=True)
    cands = cands[:40]
    y = _mono(stem_path, SR)
    for c in cands:
        a, b = int(c["start"] * SR), int(c["end"] * SR)
        mid = y[a:b]
        f0 = float(librosa.midi_to_hz(c["midi"]))
        c["harm"] = harmonicity(mid, f0)
        c["score"] = score_note(c, c["harm"])
    cands.sort(key=lambda c: c["score"], reverse=True)
    picked = []
    for c in cands:  # prefer different pitches among the top picks
        if all(abs(c["midi"] - p["midi"]) > 0.5 or abs(c["start"] - p["start"]) > 2 for p in picked):
            picked.append(c)
        if len(picked) == top:
            break
    return picked, y


def cut_note(y: np.ndarray, c: dict, pre: float = 0.04, tail: float = 0.08):
    a = max(0, int((c["start"] - pre) * SR))
    b = min(len(y), int((c["end"] + tail) * SR))
    clip = y[a:b].copy()
    # fade the tail so the next note in the song doesn't leak in
    keep = int((c["end"] - c["start"] + pre) * SR)
    if len(clip) > keep:
        t = np.arange(len(clip) - keep)
        clip[keep:] *= np.exp(-t / (0.02 * SR))
    fi = min(int(0.004 * SR), len(clip))
    clip[:fi] *= np.linspace(0, 1, fi)
    return clip / (np.abs(clip).max() + 1e-9) * 0.9


# ---------------------------------------------------------------- drums

def _hit_spectrum(y, s, n=2048):
    """Energy the hit adds: spectrum just after the onset minus just before it."""
    w = np.hanning(n)
    after = y[s:s + n]
    before = y[max(0, s - n):s]
    if len(after) < n:
        after = np.pad(after, (0, n - len(after)))
    if len(before) < n:
        before = np.pad(before, (n - len(before), 0))
    A = np.abs(np.fft.rfft(after * w)) ** 2
    B = np.abs(np.fft.rfft(before * w)) ** 2
    return np.maximum(A - B, 0), np.fft.rfftfreq(n, 1 / SR)


def _onsets_blockwise(y, hop=256, block_s=30.0, pad_s=1.0):
    """Onset times (samples), computed 30 s at a time so long songs don't need a huge spectrogram in memory."""
    blk, pad = int(block_s * SR), int(pad_s * SR)
    env = []
    for start in range(0, len(y), blk):
        a, b = max(0, start - pad), min(len(y), start + blk + pad)
        e = librosa.onset.onset_strength(y=y[a:b], sr=SR, hop_length=hop)
        skip = (start - a) // hop
        keep = (min(len(y), start + blk) - start + hop - 1) // hop
        env.append(e[skip:skip + keep])
    env = np.concatenate(env) if env else np.zeros(0)
    if not len(env):
        return np.zeros(0, dtype=int)
    return librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, units="samples", backtrack=False)


def drum_hits(stem_path: str, log=log_default):
    y = _mono(stem_path, SR)
    on = _onsets_blockwise(y)
    on = np.maximum(0, np.asarray(on) - 256)
    if len(on) == 0:
        return {}, y
    peak_all = np.abs(y).max() + 1e-9
    hits = []
    for i, s in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else len(y)
        e = min(nxt, s + int(0.6 * SR))
        if e - s < int(0.04 * SR):
            continue
        clip = y[s:e]
        pk = np.abs(clip[: int(0.05 * SR)]).max()
        if pk < 0.05 * peak_all:
            continue
        spec, fr = _hit_spectrum(y, int(s))
        tot = spec.sum() + 1e-20
        hits.append(dict(s=int(s), e=int(e), low=float(spec[fr < 150].sum() / tot),
                         high=float(spec[fr > 5000].sum() / tot),
                         cent=float((spec * fr).sum() / tot), peak=pk / peak_all,
                         new=float(tot), room=(e - s) / SR))
    if not hits:
        return {}, y
    # group hits by sound: cluster on brightness and low-end share
    X = np.array([[np.log2(h["cent"] + 50), h["low"] * 4] for h in hits])
    k = int(min(4, len(hits)))
    from scipy.cluster.vq import kmeans2
    order = np.argsort(X[:, 0])
    seeds = X[order[np.linspace(0, len(hits) - 1, k).astype(int)]]
    _, lab = kmeans2(X, seeds, minit="matrix", seed=1)
    clusters = {}
    for h, l in zip(hits, lab):
        clusters.setdefault(int(l), []).append(h)
    info = []
    for l, g in clusters.items():
        info.append(dict(id=l, hits=g, cent=float(np.median([h["cent"] for h in g])),
                         low=float(np.median([h["low"] for h in g])),
                         loud=float(np.median([h["peak"] for h in g])), n=len(g)))
    info.sort(key=lambda c: c["cent"])
    names = {}
    if info and info[0]["low"] > 0.3 and info[0]["cent"] < 2000:
        names[info[0]["id"]] = "kick"
    rest = [c for c in info if c["id"] not in names]
    if rest and rest[-1]["cent"] > 4000 and len(rest) > 1:
        names[rest[-1]["id"]] = "hihat"
        rest = rest[:-1]
    if rest:
        sn = max(rest, key=lambda c: c["loud"] * np.sqrt(c["n"]))
        names[sn["id"]] = "snare" if "hihat" in names.values() or sn["cent"] < 6000 else "hihat"
        rest = [c for c in rest if c["id"] != sn["id"]]
    taken = [c["cent"] for c in info if c["id"] in names]
    rest = [c for c in rest if all(abs(np.log2(c["cent"] / t)) > 1 for t in taken)]  # a sound of its own
    if rest:
        names[max(rest, key=lambda c: c["n"])["id"]] = "extra"
    best = {}
    for c in info:
        kind = names.get(c["id"])
        if not kind or kind in best:
            continue
        # cleanest example: lots of new energy, little else going on before it, enough room to ring
        g = sorted(c["hits"], key=lambda h: h["peak"] * min(h["room"], 0.4) * (1 - 0.5 * h.get("pre", 0)),
                   reverse=True)
        best[kind] = g[0]
    return best, y


def cut_hit(y, h):
    clip = y[h["s"]:h["e"]].copy()
    n = len(clip)
    fo = min(int(0.03 * SR), n // 3)
    clip[-fo:] *= np.linspace(1, 0, fo)
    clip[:32] *= np.linspace(0, 1, 32)
    return clip / (np.abs(clip).max() + 1e-9) * 0.9


def build_drum_kit(stem_path, out_dir, name, log=log_default):
    best, y = drum_hits(stem_path, log)
    if not best:
        log("  no clear drum hits found")
        return None
    os.makedirs(out_dir, exist_ok=True)
    hits_dir = os.path.join(out_dir, f"{name} Hits")
    os.makedirs(hits_dir, exist_ok=True)
    zones, made = [], []
    for k, h in best.items():
        clip = cut_hit(y, h)
        a = E.to_int16(clip, 1.0)
        key = DRUM_KEYS[k]
        fn = f"{name}_{DRUM_LABELS[k].split(' ')[0]}.wav"
        E.write_wav(os.path.join(hits_dir, fn), a, key)
        zones.append(dict(audio=a, root=key, key_lo=key, key_hi=key, vel_lo=0, vel_hi=127, loop=None,
                          release_s=0.25, attack_s=0.001, pitch_cents=0))
        made.append(dict(kind=k, file=os.path.join(f"{name} Hits", fn), at_s=round(h["s"] / SR, 2),
                         key=E.logic_name(key)))
        log(f"  {DRUM_LABELS[k]}: hit at {h['s'] / SR:.2f} s → {E.logic_name(key)}")
    zones.sort(key=lambda z: z["root"])
    E.write_sf2(os.path.join(out_dir, f"{name}.sf2"), name, zones)
    return dict(sf2=f"{name}.sf2", hits=made)


# ---------------------------------------------------------------- whole song

def process_song(song: str, out_root: str | None = None, uvr: str | None = None, stems_dir: str | None = None,
                 model_name: str | None = None, step: int = 3, layers: int = 2, alternates: int = 2,
                 log=log_default):
    from .build import build
    title = os.path.splitext(os.path.basename(song))[0] if song else os.path.basename(os.path.normpath(stems_dir))
    safe = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "Song"
    out_root = out_root or os.path.join(os.path.dirname(os.path.abspath(song or stems_dir)), f"{title} instruments")
    os.makedirs(out_root, exist_ok=True)
    t0 = time.time()

    if stems_dir:
        stems = stems_from_folder(stems_dir)
        log(f"Using stems from {stems_dir}: {', '.join(stems) or 'none found'}")
    else:
        uvr = find_uvr(uvr)
        if not uvr:
            raise RuntimeError("Can't find Ultimate Vocal Remover. Pass --uvr /path/to/ultimatevocalremovergui.")
        stems = separate(song, os.path.join(out_root, "Stems"), uvr, model_name, log=log)
    if not stems:
        raise RuntimeError("No stems to work with.")

    results = []
    for stem, path in stems.items():
        if stem == "drums":
            log("\nDrums: looking for the cleanest kick, snare, hi-hat …")
            name = f"{safe}_Drums"
            kit = build_drum_kit(path, os.path.join(out_root, "Drums"), name, log)
            if kit:
                results.append(dict(stem="drums", label="Drums", folder="Drums", **kit))
            continue
        st = STEM_SETTINGS.get(stem)
        if not st:
            continue
        log(f"\n{st['label']}: looking for the cleanest held note …")
        picks, y = best_notes(path, st, top=1 + alternates, log=log)
        if not picks or picks[0]["score"] < 0.02:
            log("  nothing clean enough here (the part may be silent, chords only, or too short).")
            results.append(dict(stem=stem, label=st["label"], skipped=True))
            continue
        folder = os.path.join(out_root, stem.capitalize())
        os.makedirs(folder, exist_ok=True)
        cand_info = []
        for i, c in enumerate(picks):
            clip = cut_note(y, c)
            tag = "best" if i == 0 else f"alt{i}"
            p = os.path.join(folder, f"{safe}_{stem}_{tag}_{E.logic_name(int(round(c['midi'])))}.wav")
            sf.write(p, clip, SR)
            cand_info.append(dict(file=os.path.relpath(p, out_root), at_s=round(c["start"], 2),
                                  length_s=round(c["end"] - c["start"], 2),
                                  note=E.logic_name(int(round(c["midi"]))), clean=round(c["harm"], 2),
                                  score=round(c["score"], 3)))
            log(f"  {'picked' if i == 0 else 'alternate'}: {cand_info[-1]['note']} at {c['start']:.2f} s, "
                f"{c['end'] - c['start']:.2f} s long, {c['harm'] * 100:.0f}% harmonic")
        name = f"{safe}_{stem.capitalize()}"
        best_clip = os.path.join(out_root, cand_info[0]["file"])
        res = build(best_clip, os.path.join(folder, "Instrument"), name, st["low"], st["high"], step, layers,
                    st["timbre"], 1.0, None, log=lambda m: None if m.strip().startswith("rendered") else log("  " + m))
        results.append(dict(stem=stem, label=st["label"], folder=stem.capitalize(), sf2=f"Instrument/{name}.sf2",
                            report=f"Instrument/{name} Report.html", candidates=cand_info,
                            error_db=res.get("error_db")))

    meta = dict(song=os.path.basename(song) if song else None, stems={k: os.path.relpath(v, out_root) if v.startswith(out_root) else v
                                                                       for k, v in stems.items()},
                results=results, seconds=round(time.time() - t0))
    with open(os.path.join(out_root, "song.json"), "w") as f:
        json.dump(meta, f, indent=2)
    write_index(os.path.join(out_root, "index.html"), title, meta, out_root)
    log(f"\nDone in {meta['seconds']} s. Open 'index.html' in {out_root}")
    return meta


# ---------------------------------------------------------------- index page

INDEX_CSS = """
:root{--bg:#F3F4F7;--surface:#FFF;--ink:#1A2030;--muted:#5E687A;--line:#DCE1E9;--accent:#C93F69;
--f-display:"Bricolage Grotesque","Avenir Next",system-ui,sans-serif;--f-body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0F1218;--surface:#171B23;--ink:#E5E8EE;--muted:#9AA3B3;--line:#2A303B;--accent:#EF6C93;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--f-body)}
.wrap{max-width:1040px;margin:0 auto;padding:32px max(16px,env(safe-area-inset-left,0px)) 56px;display:grid;gap:20px}
h1{font:800 clamp(28px,4.5vw,42px)/1.05 var(--f-display);margin:0;letter-spacing:-.02em}h2{font:700 20px var(--f-display);margin:0}
.sub{color:var(--muted);margin:4px 0 0}a{color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;display:grid;gap:10px;align-content:start}
.lbl{font:600 11px var(--f-body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
audio{width:100%}code{font:13px var(--f-mono)}
table{width:100%;border-collapse:collapse;font:13px var(--f-mono)}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)}
th{font:600 11px var(--f-body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.stems{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
ol{margin:0;padding-left:20px;display:grid;gap:6px}
"""


def write_index(path, title, meta, out_root):
    def a(src):
        return f'<audio controls preload="none" src="{html.escape(src)}"></audio>'
    stems_html = "".join(f"<div><span class='lbl'>{html.escape(k)}</span>{a(v)}</div>" for k, v in meta["stems"].items())
    cards = []
    for r in meta["results"]:
        if r.get("skipped"):
            cards.append(f"<div class='card'><h2>{html.escape(r['label'])}</h2><p class='sub'>No clean single note found in this stem.</p></div>")
            continue
        if r["stem"] == "drums":
            rows = "".join(f"<tr><td>{html.escape(DRUM_LABELS[h['kind']])}</td><td>{h['key']}</td><td>{h['at_s']} s</td><td>{a(r['folder'] + '/' + h['file'])}</td></tr>" for h in r["hits"])
            cards.append(f"<div class='card'><h2>Drum kit</h2><p class='sub'>Load <code>Drums/{html.escape(r['sf2'])}</code>. Mapped to standard drum keys, so any drum pattern plays it.</p>"
                         f"<table><tr><th>Hit</th><th>Key</th><th>From</th><th>Listen</th></tr>{rows}</table></div>")
            continue
        c0 = r["candidates"][0]
        alts = "".join(f"<tr><td>{c['note']}</td><td>{c['at_s']} s</td><td>{int(c['clean'] * 100)}%</td><td>{a(c['file'])}</td></tr>" for c in r["candidates"][1:])
        err = f" · rebuild difference {r['error_db']:.1f} dB" if r.get("error_db") is not None else ""
        cards.append(f"""<div class='card'><h2>{html.escape(r['label'])}</h2>
<p class='sub'>Built from {c0['note']} at {c0['at_s']} s ({c0['length_s']} s long, {int(c0['clean'] * 100)}% harmonic){err}.</p>
<div><span class='lbl'>Note taken from the song</span>{a(c0['file'])}</div>
<p><a href="{html.escape(r['folder'] + '/' + r['report'])}">Open the instrument report</a> · load <code>{html.escape(r['folder'] + '/' + r['sf2'])}</code></p>
{f"<details><summary class='lbl'>Other notes it considered</summary><table><tr><th>Note</th><th>At</th><th>Harmonic</th><th>Listen</th></tr>{alts}</table><p class='sub'>Prefer one? Run <code>python3 -m sample2synth thatfile.wav</code> on it.</p></details>" if alts else ""}
</div>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · instruments</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>{INDEX_CSS}</style></head><body><div class="wrap">
<header><h1>{html.escape(title)}</h1><p class="sub">Split into stems, then each stem turned into a playable instrument.</p></header>
<section class="card"><span class="lbl">Stems</span><div class="stems">{stems_html}</div></section>
<section class="grid">{''.join(cards)}</section>
<section class="card"><h2>In Logic Pro</h2><ol>
<li>New Software Instrument track → <b>Sampler</b> → import the <code>.sf2</code> of the instrument you want.</li>
<li>Drums: same thing with the drum kit <code>.sf2</code>; kick is on C1, snare on D1, hi-hat on F#1.</li>
<li>Stems are in <code>Stems/</code> if you want to drop them on audio tracks too.</li></ol></section>
</div></body></html>"""
    with open(path, "w") as f:
        f.write(page)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="sample2synth song",
                                description="Split a song with Ultimate Vocal Remover's Demucs model and turn each part into an instrument.")
    p.add_argument("song", nargs="?", help="the song (wav, mp3, m4a, flac…) or a YouTube link")
    p.add_argument("--stems", help="use a folder of stems you already exported from UVR instead of separating")
    p.add_argument("--uvr", help="path to the ultimatevocalremovergui folder (found automatically if this tool sits inside it)")
    p.add_argument("--model", help="Demucs model name, e.g. htdemucs or htdemucs_6s (default: best one downloaded)")
    p.add_argument("-o", "--out", help="output folder (default: '<song> instruments' next to the song)")
    p.add_argument("--step", type=int, default=3, help="semitones between samples (default 3)")
    p.add_argument("--layers", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--alternates", type=int, default=2, help="extra candidate notes to save per stem (default 2)")
    a = p.parse_args(argv)
    if not a.song and not a.stems:
        p.error("give a song, or --stems FOLDER")
    song = a.song
    if song and song.startswith(("http://", "https://", "www.", "youtu")):
        from .youtube import download, DEFAULT_OUT
        song = download(song, DEFAULT_OUT, "wav")["path"]
    process_song(song, a.out, a.uvr, a.stems, a.model, a.step, a.layers, a.alternates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
