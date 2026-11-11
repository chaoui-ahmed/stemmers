"""Listen to a sample and build a harmonic + noise model of it.

The model describes the sound as:
  * a fundamental pitch (f0) with harmonics at k * f0,
  * the amplitude of every harmonic over time (this is the timbre),
  * a band-limited noise part over time (breath, bow, pick, hiss),
  * an envelope type: sustained (attack -> loopable sustain -> release)
    or percussive (attack -> natural decay of each harmonic).
"""
from __future__ import annotations

import numpy as np
import librosa

SR = 44100
HOP = 256
N_BANDS = 40          # noise bands, log-spaced
BAND_LO, BAND_HI = 40.0, 20000.0


def band_edges():
    return np.geomspace(BAND_LO, BAND_HI, N_BANDS + 1)


def band_centers():
    e = band_edges()
    return np.sqrt(e[:-1] * e[1:])


def load(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    if not np.any(y):
        raise ValueError("The file is silent.")
    return y.astype(np.float64)


def _trim(y: np.ndarray, floor_db: float = -50.0):
    """Cut leading/trailing silence, keep a short pre-roll before the onset."""
    frame = 512
    n = len(y) // frame
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    on = np.argmax(db > floor_db)
    off = n - np.argmax(db[::-1] > floor_db)
    start = max(0, on * frame - 256)
    end = min(len(y), off * frame)
    return y[start:end], start


def detect_pitch(y: np.ndarray):
    """f0 track (Hz, NaN where unvoiced) with pYIN, plus a reference pitch."""
    f0, voiced, prob = librosa.pyin(
        y, fmin=30.0, fmax=2100.0, sr=SR, frame_length=4096, hop_length=HOP, fill_na=np.nan
    )
    good = voiced & (prob > 0.3) & np.isfinite(f0)
    if good.sum() < 5:
        return None, f0
    # reference = median pitch over the loudest voiced half
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=HOP)[0][: len(f0)]
    thr = np.median(rms[good])
    sel = good & (rms[: len(f0)] >= thr)
    ref = float(np.median(f0[sel if sel.sum() >= 5 else good]))
    return ref, f0


def _fill_f0(f0: np.ndarray, ref: float):
    """Fill gaps and clamp octave errors so the harmonic grid stays put."""
    f = f0.copy()
    bad = ~np.isfinite(f)
    cents = 1200 * np.log2(np.where(bad, ref, f) / ref)
    bad |= np.abs(cents) > 150          # octave jumps / glitches
    f[bad] = np.nan
    idx = np.arange(len(f))
    if np.all(np.isnan(f)):
        return np.full_like(f, ref)
    f = np.interp(idx, idx[~np.isnan(f)], f[~np.isnan(f)])
    return f


def harmonic_noise_analysis(y: np.ndarray, f0_ref: float, f0_track: np.ndarray):
    """Per-frame harmonic amplitudes/phases and noise band power.

    Harmonic k is read from a zero-padded spectrum near k * f0 (parabolic peak
    refinement); everything outside the harmonic main lobes is noise.
    """
    n_win = int(np.clip(2 ** np.ceil(np.log2(4 * SR / f0_ref)), 1024, 8192))
    n_fft = n_win * 4
    win = np.hanning(n_win)
    wsum, wsq = win.sum(), (win ** 2).sum()
    k_max = int(min(0.45 * SR, 18000) // f0_ref)
    k_max = max(1, min(k_max, 160))
    pad = n_win // 2
    yp = np.pad(y, (pad, pad))
    n_frames = 1 + (len(y) - 1) // HOP
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    df = SR / n_fft
    edges = band_edges()
    band_idx = np.clip(np.searchsorted(edges, freqs) - 1, -1, N_BANDS)

    amps = np.zeros((n_frames, k_max))
    phases = np.zeros((n_frames, k_max))
    noise = np.zeros((n_frames, N_BANDS))
    rms = np.zeros(n_frames)
    f0s = np.interp(np.arange(n_frames), np.arange(len(f0_track)), f0_track)

    for t in range(n_frames):
        seg = yp[t * HOP: t * HOP + n_win]
        if len(seg) < n_win:
            seg = np.pad(seg, (0, n_win - len(seg)))
        rms[t] = np.sqrt(np.mean(seg ** 2))
        X = np.fft.rfft(seg * win, n_fft)
        mag = np.abs(X)
        f0 = f0s[t]
        mask = np.ones(len(mag), bool)
        lobe = 2.0 * SR / n_win  # hann main-lobe half width
        for k in range(1, k_max + 1):
            fk = k * f0
            if fk >= 0.48 * SR:
                break
            lo = int(max(1, (fk - 0.4 * f0) / df))
            hi = int(min(len(mag) - 2, (fk + 0.4 * f0) / df))
            if hi <= lo:
                continue
            p = lo + int(np.argmax(mag[lo:hi + 1]))
            a, b, c = np.log(mag[p - 1] + 1e-12), np.log(mag[p] + 1e-12), np.log(mag[p + 1] + 1e-12)
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den != 0 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
            peak = np.exp(b - 0.25 * (a - c) * d)
            amps[t, k - 1] = 2 * peak / wsum
            phases[t, k - 1] = np.angle(X[p])
            m_lo = int(max(0, (fk - lobe) / df))
            m_hi = int(min(len(mag), (fk + lobe) / df + 1))
            mask[m_lo:m_hi] = False
        psd = (mag ** 2) / wsq  # per-bin noise power for unit-variance white noise -> 1
        for bnd in range(N_BANDS):
            sel = mask & (band_idx == bnd)
            if sel.any():
                noise[t, bnd] = np.median(psd[sel])
        # a sinusoid's sidelobes leak into "noise" bins: keep noise from
        # exceeding the harmonic energy it sits next to by more than reason
    noise = _smooth_bands(noise)
    return dict(amps=amps, phases=phases, noise=noise, rms=rms, f0=f0s, n_win=n_win, k_max=k_max)


def _smooth_bands(noise):
    out = noise.copy()
    for i in range(N_BANDS):
        lo, hi = max(0, i - 1), min(N_BANDS, i + 2)
        out[:, i] = np.median(noise[:, lo:hi], axis=1)
    return out


def _energy_db(an):
    e = (an["amps"] ** 2).sum(1) / 2
    edges = band_edges()
    bw = np.diff(edges)
    e = e + (an["noise"] * bw / (SR / 2)).sum(1)  # noise power share
    return 10 * np.log10(e + 1e-20)


def envelope(an):
    """Classify the sound and find attack / sustain / release regions."""
    db = _energy_db(an)
    n = len(db)
    dt = HOP / SR
    sm = np.convolve(db, np.ones(5) / 5, mode="same")
    peak_i = int(np.argmax(sm >= sm.max() - 0.5))  # first time it reaches full level
    peak = float(sm.max())
    # attack = from start to first frame within 1 dB of the peak
    att_i = int(np.argmax(sm >= peak - 1.0))
    # look for the longest stable plateau after the peak
    best = (0, 0)
    w = max(3, int(0.15 / dt))
    stable = np.zeros(n, bool)
    for i in range(peak_i, n - w):
        seg = sm[i:i + w]
        stable[i] = (seg.max() - seg.min() < 2.5) and seg.mean() > peak - 20
    i = peak_i
    while i < n:
        if stable[i]:
            j = i
            while j < n and stable[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j + w - 1)
            i = j
        else:
            i += 1
    plateau = (best[1] - best[0]) * dt
    end_drop = peak - sm[-1]
    # overall level slope after the peak (dB/s) over the audible part
    # body of the note: from the peak until it first falls 12 dB (leave out the final release)
    drop = np.where(sm[peak_i:] < peak - 12)[0]
    body_end = peak_i + (drop[0] if len(drop) else n - peak_i)
    if body_end - peak_i < 0.25 / dt:  # falls fast: a decaying sound, use the audible tail
        after = np.arange(peak_i, n)
        aud = after[sm[after] > peak - 40]
    else:
        aud = np.arange(peak_i, body_end)
    slope = float(np.polyfit(aud * dt, sm[aud], 1)[0]) if len(aud) > 8 else 0.0
    # slope inside the plateau itself: a slow steady decay is not a sustain
    if best[1] - best[0] > 8:
        pi = np.arange(best[0], min(best[1], n))
        plat_slope = float(np.polyfit(pi * dt, sm[pi], 1)[0])
    else:
        plat_slope = slope
    fast_attack = (att_i * dt) < 0.08
    decaying = slope < -8 or (fast_attack and slope < -2.5) or plat_slope < -3.5
    sustained = (not decaying) and plateau >= 0.35 and (best[1] - best[0]) >= 0.3 * (n - peak_i)
    info = dict(peak_s=peak_i * dt, attack_s=att_i * dt, total_s=n * dt, plateau_s=plateau)
    if sustained:
        s0, s1 = best[0], min(best[1], n - 1)
        # release: time for the tail after the plateau to fall 30 dB (if present)
        tail = sm[s1:]
        rel = 0.25
        if len(tail) > 4 and tail[0] - tail[-1] > 10:
            drop = tail[0] - tail
            k = np.argmax(drop >= min(30, drop.max()))
            rel = max(0.05, k * dt * (60 / max(drop[k], 1)))
        info.update(kind="sustained", sus_start=s0, sus_end=s1, release_s=float(min(rel, 6.0)),
                    sustain_level_db=float(np.median(sm[s0:s1]) - peak))
    else:
        info.update(kind="percussive", end_drop_db=float(end_drop))
    return info


def decay_rates(an, start_i: int):
    """Exponential decay time (s, to -60 dB) for each harmonic and noise band."""
    dt = HOP / SR
    amps = an["amps"][start_i:]
    t = np.arange(len(amps)) * dt
    k_max = amps.shape[1]
    taus = np.full(k_max, 1.0)
    for k in range(k_max):
        a = 20 * np.log10(amps[:, k] + 1e-12)
        top = a.max()
        ok = a > top - 50
        last = np.where(ok)[0]
        if len(last) < 4:
            taus[k] = 0.2
            continue
        sel = slice(0, last[-1] + 1)
        tt, aa = t[sel][ok[sel]], a[sel][ok[sel]]
        if len(tt) < 4 or np.ptp(tt) < 1e-3:
            taus[k] = 0.2
            continue
        slope = np.polyfit(tt, aa, 1)[0]  # dB per second
        taus[k] = 60 / max(-slope, 2.0)
    nz = an["noise"][start_i:]
    ntau = np.full(N_BANDS, 0.3)
    for b in range(N_BANDS):
        a = 10 * np.log10(nz[:, b] + 1e-20)
        ok = a > a.max() - 40
        if ok.sum() >= 4:
            slope = np.polyfit(t[ok], a[ok], 1)[0]
            ntau[b] = 60 / max(-slope, 5.0)
    return np.clip(taus, 0.03, 30.0), np.clip(ntau, 0.02, 10.0)


def vibrato(an, s0: int, s1: int):
    f = an["f0"][s0:s1]
    if len(f) < 32:
        return dict(rate_hz=0.0, depth_cents=0.0)
    c = 1200 * np.log2(f / np.median(f))
    c = c - np.polyval(np.polyfit(np.arange(len(c)), c, 2), np.arange(len(c)))
    spec = np.abs(np.fft.rfft(c * np.hanning(len(c)), 4096))
    fr = np.fft.rfftfreq(4096, HOP / SR)
    band = (fr >= 3) & (fr <= 9)
    if not band.any():
        return dict(rate_hz=0.0, depth_cents=0.0)
    rate = float(fr[band][np.argmax(spec[band])])
    depth = float(np.std(c) * np.sqrt(2))
    if depth < 6:
        return dict(rate_hz=0.0, depth_cents=0.0)
    return dict(rate_hz=round(rate, 2), depth_cents=round(depth, 1))


def analyze(path: str, name: str | None = None):
    y_full = load(path)
    y, offset = _trim(y_full)
    f0_ref, f0 = detect_pitch(y)
    if f0_ref is None:
        return dict(pitched=False, name=name, audio=y)
    f0_track = _fill_f0(f0, f0_ref)
    an = harmonic_noise_analysis(y, f0_ref, f0_track)
    env = envelope(an)
    model = dict(
        pitched=True, name=name, sr=SR, hop=HOP, f0=f0_ref,
        midi=float(librosa.hz_to_midi(f0_ref)),
        amps=an["amps"], phases=an["phases"], noise=an["noise"], rms=an["rms"],
        f0_track=an["f0"], env=env, audio=y,
    )
    if env["kind"] == "sustained":
        s0, s1 = env["sus_start"], env["sus_end"]
        model["sus_amps"] = np.median(an["amps"][s0:s1], axis=0)
        model["sus_noise"] = np.median(an["noise"][s0:s1], axis=0)
        mid = (s0 + s1) // 2
        model["sus_phases"] = _relative_phases(an["phases"][mid], an["amps"][mid])
        model["vibrato"] = vibrato(an, s0, s1)
    else:
        p = int(round(env["peak_s"] * SR / HOP))
        model["decay_tau"], model["noise_tau"] = decay_rates(an, p)
        model["peak_amps"] = an["amps"][p]
        model["peak_noise"] = an["noise"][p]
        model["sus_phases"] = _relative_phases(an["phases"][p], an["amps"][p])
        model["vibrato"] = dict(rate_hz=0.0, depth_cents=0.0)
    return model


def _relative_phases(ph, amps):
    """Phases relative to the fundamental, so the waveform shape is kept."""
    k = np.arange(1, len(ph) + 1)
    rel = ph - k * ph[0]
    rel[amps < amps.max() * 1e-3] = 0.0
    return np.mod(rel + np.pi, 2 * np.pi) - np.pi
