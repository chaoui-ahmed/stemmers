"""Write instruments Logic Pro (and other samplers/synths) can load.

* SoundFont 2 (.sf2): one file, all notes, loops, velocity layers and the
  release time. Logic's Sampler can import it.
* WAV multisamples: one file per note with the root key and loop stored in a
  'smpl' chunk; drag the folder onto Logic's Sampler.
* Wavetable (.wav, 2048-sample frames): for Vital, Serum, and other
  wavetable synths.
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np

from .analyze import SR

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def logic_name(midi: int) -> str:
    """Logic Pro's convention: middle C (MIDI 60) is C3."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 2}"


def to_int16(x: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(np.round(x * gain * 32767), -32768, 32767).astype("<i2")


# ---------------- WAV with smpl chunk ----------------

def write_wav(path, samples_i16: np.ndarray, root: int, loop=None, sr=SR):
    data = samples_i16.tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    loops = [] if loop is None else [loop]
    smpl = struct.pack("<9I", 0, 0, int(round(1e9 / sr)), root, 0, 0, 0, len(loops), 0)
    for s, e in loops:
        smpl += struct.pack("<6I", 0, 0, s, e - 1, 0, 0)  # end is inclusive
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"smpl" + struct.pack("<I", len(smpl)) + smpl
    chunks += b"data" + struct.pack("<I", len(data)) + data
    if len(data) % 2:
        chunks += b"\0"
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def write_wavetable(path, table: np.ndarray, sr=SR):
    """32-bit float wavetable with a Serum-style 'clm ' chunk (frame size 2048)."""
    size = table.shape[1]
    data = table.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    clm = f"<!>{size} 10000000 wavetable (sample2synth)".encode()
    if len(clm) % 2:
        clm += b" "
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"clm " + struct.pack("<I", len(clm)) + clm
    chunks += b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


# ---------------- SoundFont 2 ----------------

def _chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\0" if len(payload) % 2 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _list(kind: bytes, payload: bytes) -> bytes:
    return _chunk(b"LIST", kind + payload)


def _name20(s: str) -> bytes:
    b = s.encode("ascii", "replace")[:19]
    return b + b"\0" * (20 - len(b))


def _timecents(seconds: float) -> int:
    return int(round(1200 * math.log2(max(seconds, 0.001))))


def write_sf2(path, name: str, zones: list[dict]):
    """zones: dicts with keys
         audio (int16 array), root (int), key_lo, key_hi, vel_lo, vel_hi,
         loop (start, end) or None, release_s, attack_s, pitch_cents (int)
    """
    GUARD = 46
    smpl = bytearray()
    shdr = bytearray()
    for i, z in enumerate(zones):
        start = len(smpl) // 2
        a = z["audio"]
        smpl += a.tobytes()
        end = start + len(a)
        smpl += b"\0\0" * GUARD
        if z["loop"]:
            ls, le = start + z["loop"][0], start + z["loop"][1]
        else:
            ls, le = start + max(0, len(a) - 40), start + len(a) - 8
        shdr += _name20(f"{name[:10]}_{logic_name(z['root'])}_{i}")
        shdr += struct.pack("<IIIIIBbHH", start, end, ls, le, SR, z["root"],
                            int(np.clip(z.get("pitch_cents", 0), -99, 99)), 0, 1)
    shdr += _name20("EOS") + struct.pack("<IIIIIBbHH", 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # instrument zones
    igen = bytearray()
    ibag = bytearray()
    gi = 0
    for i, z in enumerate(zones):
        ibag += struct.pack("<HH", gi, 0)
        gens = [
            (43, struct.pack("<BB", z["key_lo"], z["key_hi"])),   # keyRange
            (44, struct.pack("<BB", z["vel_lo"], z["vel_hi"])),   # velRange
            (34, struct.pack("<h", _timecents(z.get("attack_s", 0.002)))),  # attackVolEnv
            (38, struct.pack("<h", _timecents(z.get("release_s", 0.25)))),  # releaseVolEnv
            (54, struct.pack("<H", 1 if z["loop"] else 0)),       # sampleModes
            (58, struct.pack("<H", z["root"])),                    # overridingRootKey
            (53, struct.pack("<H", i)),                            # sampleID (last)
        ]
        for oper, amt in gens:
            igen += struct.pack("<H", oper) + amt
            gi += 1
    ibag += struct.pack("<HH", gi, 0)
    igen += struct.pack("<HH", 0, 0)
    imod = b"\0" * 10
    inst = _name20(name) + struct.pack("<H", 0) + _name20("EOI") + struct.pack("<H", len(zones))

    pgen = struct.pack("<HH", 41, 0) + struct.pack("<HH", 0, 0)   # instrument 0
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0)
    pmod = b"\0" * 10
    phdr = _name20(name) + struct.pack("<HHHIII", 0, 0, 0, 0, 0, 0)
    phdr += _name20("EOP") + struct.pack("<HHHIII", 0, 0, 1, 0, 0, 0)

    def zstr(t: str) -> bytes:  # SF2 strings: zero-terminated, padded to an even length
        b = t.encode("ascii", "replace")[:254] + b"\0"
        return b + (b"\0" if len(b) % 2 else b"")
    info = (_chunk(b"ifil", struct.pack("<HH", 2, 1)) + _chunk(b"isng", zstr("EMU8000"))
            + _chunk(b"INAM", zstr(name)) + _chunk(b"ISFT", zstr("sample2synth")))
    sdta = _chunk(b"smpl", bytes(smpl))
    pdta = (_chunk(b"phdr", phdr) + _chunk(b"pbag", pbag) + _chunk(b"pmod", pmod)
            + _chunk(b"pgen", pgen) + _chunk(b"inst", inst) + _chunk(b"ibag", bytes(ibag))
            + _chunk(b"imod", imod) + _chunk(b"igen", bytes(igen)) + _chunk(b"shdr", bytes(shdr)))
    body = b"sfbk" + _list(b"INFO", info) + _list(b"sdta", sdta) + _list(b"pdta", pdta)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


# ---------------- params ----------------

def model_summary(model) -> dict:
    env = dict(model["env"])
    for k in ("sus_start", "sus_end"):
        if k in env:
            env[k + "_s"] = round(env.pop(k) * model["hop"] / SR, 3)
    amps = model.get("sus_amps", model.get("peak_amps"))
    rel = (amps / max(amps.max(), 1e-12))
    return dict(
        name=model["name"],
        pitch_hz=round(model["f0"], 2),
        midi_note=round(model["midi"], 2),
        envelope={k: (round(v, 3) if isinstance(v, float) else v) for k, v in env.items()},
        vibrato=model["vibrato"],
        harmonics_db=[round(float(20 * np.log10(a + 1e-9)), 1) for a in rel[:32]],
    )


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
