# Stem Splitter

Upload a song and get:
- its four **stems** (vocals, drums, bass, other) as full-quality WAV files;
- a **drum kit** made from the song's own drums: the cleanest kick, snare and hi-hat, as a SoundFont (`.sf2`) for Logic's Sampler plus the WAV hits.

It's built to use little memory (about 1 GB at peak, or under 1 GB in low-memory mode), so it fits on Streamlit Community Cloud's free tier.

## Put it online (free, from GitHub)

1. Create a new repository on GitHub (for example `stem-splitter`) and upload everything in this folder.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **Create app**, pick the repository and branch, and set the main file to **`app.py`**. Click **Deploy**.
4. The first build takes about 5 minutes. On the first song, the app downloads the separation model (84 MB) once. A 3.5-minute song takes about 2 minutes to split.

You get a public address like `https://<your-app>.streamlit.app`.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Memory

Measured on a 3.5-minute song:

| | Peak memory |
|---|---|
| Web page (Streamlit) | ~150 MB |
| Splitting worker, default (`S2S_WINDOW=4`) | ~940 MB |
| Splitting worker, low-memory mode (`S2S_WINDOW=3`) | ~840 MB |

How it's kept down:
- The heavy work runs in a **separate process** that exits after each song, so its memory is fully handed back.
- The song is decoded to disk, then split **15 seconds at a time**. Stems are written straight to disk, so memory doesn't grow with song length.
- The model looks at **4-second windows** instead of 7.8 s. On a test song with known parts, quality dropped by under 1 dB. Use `S2S_WINDOW=3` for low-memory mode: the bass and other stems lose about 2 dB.
- librosa is only loaded after the split, the page plays small MP3 previews, and a full WAV is only loaded when someone clicks **Get WAV**.
- One song at a time: if two people upload together, the second waits in line.

Streamlit Community Cloud gives apps between about 690 MB and 2.7 GB depending on load, so the default fits. If the app gets killed under load, set `S2S_WINDOW=3` in the app's secrets/environment.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `S2S_MAX_MB` | 30 | upload limit (MB) |
| `S2S_MAX_SONG_MIN` | 6 | song length limit (minutes) |
| `S2S_THREADS` | 2 | CPU threads |
| `S2S_WINDOW` | 4 | model window in seconds (3 = low-memory mode) |

## Credits

Source separation: [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` (MIT), via the copy in [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) (MIT). Drum kit: sample2synth.
