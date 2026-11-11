"""Does the heavy work in its own process, so all its memory is returned when it exits.

python worker.py SONG OUT_DIR TITLE
Prints 'PROGRESS <0..1> <message>' lines, then 'RESULT <json>'.
"""
import json
import os
import sys


def say(x, msg):
    print(f"PROGRESS {x:.3f} {msg}", flush=True)


def main():
    song, root, title = sys.argv[1], sys.argv[2], sys.argv[3]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    say(0.01, "Loading the separation model …")
    import separate_light as L
    L.load_model(lambda m: say(0.01, m))
    stems = L.separate(song, os.path.join(root, "stems"),
                       progress=lambda x: say(0.02 + 0.83 * x, f"Splitting … {int(x * 100)}%"),
                       log=lambda m: say(0.02, m))
    say(0.87, "Building the drum kit …")
    L.release_memory()
    from sample2synth.song import build_drum_kit   # librosa loads only now, after the model's peak
    kit = build_drum_kit(stems["drums"], os.path.join(root, "kit"), f"{title}_Drums", log=lambda m: None)
    say(0.93, "Making previews …")
    previews = {s: L.preview(p, os.path.join(root, f"{s}_preview.mp3")) for s, p in stems.items()}
    print("RESULT " + json.dumps(dict(stems=stems, kit=kit, previews=previews)), flush=True)


if __name__ == "__main__":
    main()
