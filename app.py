"""Stem Splitter: light web version of Sample2Synth.

Upload a song, get its four stems (vocals, drums, bass, other) and a drum kit
built from the drum stem. Made to run in under 1 GB of memory.
"""
from __future__ import annotations

import gc
import io
import os
import re
import shutil
import threading
import time
import uuid
import zipfile

import json
import subprocess
import sys

import soundfile as sf
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
# labels only: keep torch and librosa out of the web server process
DRUM_LABELS = {"kick": "Kick", "snare": "Snare", "hihat": "Hi-hat", "extra": "Extra hit (tom / crash / perc)"}

WORK = os.environ.get("S2S_WORK", "/tmp/stem-splitter")
MAX_MB = int(os.environ.get("S2S_MAX_MB", "30"))
MAX_MIN = float(os.environ.get("S2S_MAX_SONG_MIN", "6"))
KEEP_HOURS = 1
STEM_LABELS = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "other": "Other (keys, guitar, synths)"}

st.set_page_config(page_title="Stem Splitter", page_icon="🎚️", layout="centered")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:860px}
h1{letter-spacing:-.02em;margin-bottom:0}
.meta{color:#7d8796;font-size:.92rem}
.stDownloadButton button{width:100%}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def job_lock():
    return threading.Lock()


def cleanup_old():
    if not os.path.isdir(WORK):
        return
    cutoff = time.time() - KEEP_HOURS * 3600
    for d in os.listdir(WORK):
        p = os.path.join(WORK, d)
        try:
            if os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def session_dir():
    if "sid" not in st.session_state:
        st.session_state["sid"] = uuid.uuid4().hex[:12]
    return os.path.join(WORK, st.session_state["sid"])


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "song"


def read(p):
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return None


def kit_zip(kit_dir):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(kit_dir):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(kit_dir)))
    return buf.getvalue()


st.title("Stem Splitter")
st.write("Upload a song. You get the vocals, drums, bass and the rest as separate tracks, "
         "plus a playable drum kit made from the song's own drums.")

up = st.file_uploader("Song", type=["mp3", "wav", "m4a", "flac", "ogg", "aif", "aiff"])
st.caption(f"Up to {MAX_MB} MB and {MAX_MIN:g} minutes. Files are deleted after an hour. "
           "Only upload audio you have the rights to use.")

if st.button("Split", type="primary", disabled=up is None):
    if up.size > MAX_MB * 1024 * 1024:
        st.error(f"That file is {up.size / 1e6:.0f} MB. The limit is {MAX_MB} MB.")
        st.stop()
    title = os.path.splitext(up.name)[0]
    root = os.path.join(session_dir(), safe(title))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    src = os.path.join(root, "original" + os.path.splitext(up.name)[1].lower())
    with open(src, "wb") as f:
        f.write(up.getbuffer())
    try:
        dur = sf.info(src).duration
    except Exception:
        import librosa
        dur = float(librosa.get_duration(path=src))
    if dur > MAX_MIN * 60:
        st.error(f"That song is {dur / 60:.1f} minutes long. The limit is {MAX_MIN:g} minutes.")
        st.stop()

    bar = st.progress(0.0, text="Waiting for the server …")
    lock = job_lock()
    if lock.locked():
        bar.progress(0.0, text="Someone else's song is being split. You're next in line …")
    t0 = time.time()
    with lock:
        cleanup_old()
        env = dict(os.environ, MALLOC_ARENA_MAX="2", OMP_NUM_THREADS=os.environ.get("S2S_THREADS", "2"))
        try:  # settings can also come from Streamlit Cloud's "Secrets" box
            for k in ("S2S_WINDOW", "S2S_THREADS"):
                if k in st.secrets:
                    env[k] = str(st.secrets[k])
        except Exception:
            pass
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "worker.py"), src, root, safe(title)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=HERE)
        result = None
        for line in proc.stdout:
            if line.startswith("PROGRESS "):
                _, x, msg = line.rstrip("\n").split(" ", 2)
                bar.progress(min(float(x), 1.0), text=msg)
            elif line.startswith("RESULT "):
                result = json.loads(line[7:])
        err = proc.stderr.read()
        proc.wait()
        if proc.returncode != 0 or result is None:
            bar.empty()
            msg = "the server ran out of memory" if proc.returncode in (-9, 137) else (err.strip().splitlines() or ["unknown error"])[-1]
            st.error(f"Something went wrong: {msg}")
            st.stop()
        stems, kit, previews = result["stems"], result["kit"], result["previews"]
    bar.progress(1.0, text=f"Done in {time.time() - t0:.0f} s")
    os.remove(src)
    st.session_state["res"] = dict(title=title, root=root, stems=stems, previews=previews, kit=kit)

res = st.session_state.get("res")
if res and os.path.isdir(res["root"]):
    st.subheader(res["title"])
    st.markdown("<div class='meta'>Previews are compressed. The downloads are full-quality WAV.</div>", unsafe_allow_html=True)
    for s in ["vocals", "drums", "bass", "other"]:
        if s not in res["stems"]:
            continue
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.3])
            c1.markdown(f"**{STEM_LABELS[s]}**")
            b = read(res["previews"][s])
            if b:
                c1.audio(b, format="audio/mpeg" if res["previews"][s].endswith(".mp3") else "audio/ogg")
            # load the full WAV only when asked, so the server keeps one file in memory at a time
            if st.session_state.get("ready") == s:
                wav = read(res["stems"][s])
                c2.download_button("Download WAV", wav, f"{safe(res['title'])} - {STEM_LABELS[s].split(' (')[0]}.wav",
                                   mime="audio/wav", key=f"dl-{s}", type="primary")
            else:
                if c2.button("Get WAV", key=f"prep-{s}"):
                    st.session_state["ready"] = s
                    st.rerun()

    kit = res.get("kit")
    st.markdown("#### Drum kit")
    if kit:
        st.caption("Load the .sf2 in Logic's Sampler (or any SoundFont player): kick on C1, snare on D1, hi-hat on F#1.")
        for h in kit["hits"]:
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"**{DRUM_LABELS[h['kind']].split(' (')[0]}**  \n{h['key']} · from {h['at_s']} s")
            b = read(os.path.join(res["root"], "kit", h["file"]))
            if b:
                c2.audio(b, format="audio/wav")
        kd = os.path.join(res["root"], "kit")
        c1, c2 = st.columns(2)
        sb = read(os.path.join(kd, kit["sf2"]))
        if sb:
            c1.download_button("Drum kit (.sf2)", sb, kit["sf2"], key="kit-sf2", type="primary")
        c2.download_button("Drum hits (.zip of WAVs)", kit_zip(kd), f"{safe(res['title'])} drum kit.zip", key="kit-zip")
    else:
        st.caption("No clear drum hits found in this song.")
