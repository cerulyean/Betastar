"""
benchmark_prune.py  —  Run whenever you want to evaluate

Re-prunes all simulated replays and writes files ready for benchmark_viz.
Re-runs every time so changes to extract_pruner.py are always picked up.

Output files land in PRUNED_DIR — open them manually in benchmark_viz.py.

Edit the config block and run.
"""

import gzip
import json
import os
from pathlib import Path

import extract_pruner as ep

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_CACHE_DIR = r"D:\betastar\benchmark_cache\raw"
PRUNED_DIR = r"D:\betastar\benchmark_cache\pruned"
DUMMY_MMR = 3500

# ---------------------------------------------------------------------------


def detect_zerg_pov(stem):
    """Return the POV number whose raw file has own_race == 2 (Zerg)."""
    for pov in (1, 2):
        path = os.path.join(RAW_CACHE_DIR, f"{stem}.SC2Replay_{pov}.json.gz")
        if not os.path.isfile(path):
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            frames = data if isinstance(data, dict) else {}
            if "frames" in frames:
                frames = frames["frames"]
            first = frames[sorted(frames, key=int)[0]]
            if first.get("own_race") == 2:
                return pov
        except Exception as e:
            print(f"    WARNING: could not read POV {pov}: {e}")
    print(f"    WARNING: could not detect Zerg POV, defaulting to 1")
    return 1


def prune_all():
    os.makedirs(PRUNED_DIR, exist_ok=True)

    # Find all simulated replays by scanning for _1.json.gz files
    raw_files = sorted(Path(RAW_CACHE_DIR).glob("*.SC2Replay_1.json.gz"))
    if not raw_files:
        print(f"No simulated replays found in {RAW_CACHE_DIR}")
        print("Run benchmark_simulate.py first.")
        return

    print(f"Found {len(raw_files)} simulated replay(s)\n")

    for raw_file in raw_files:
        # stem is e.g. "MyReplay.SC2Replay"
        fake_game_id = raw_file.name.replace("_1.json.gz", "")
        replay_stem = Path(fake_game_id).stem

        print(f"  [prune] {fake_game_id}")

        pov = detect_zerg_pov(replay_stem)

        # Point extract_pruner at the raw cache so it finds the right files
        ep.INPUT_DIR = RAW_CACHE_DIR
        ep.OUTPUT_DIR = PRUNED_DIR
        ep.MMR_DATA[fake_game_id] = {
            "mmr": DUMMY_MMR,
            "winner_id": -1,
            "players": {
                "1": {"id": 1, "race": "zerg"},
                "2": {"id": 2, "race": "protoss"},
            },
        }

        try:
            ep.process(fake_game_id)
        except ValueError:
            pass  # winner unknown — handled below

        # Patch winner to None and move to final location
        raw_out = os.path.join(PRUNED_DIR, f"{fake_game_id}.json.gz")
        final_out = os.path.join(PRUNED_DIR, f"{replay_stem}.json.gz")

        if os.path.isfile(raw_out):
            with gzip.open(raw_out, "rt", encoding="utf-8") as f:
                data = json.load(f)
            data["winner"] = None
            with gzip.open(final_out, "wt", encoding="utf-8") as f:
                json.dump(data, f)
            os.remove(raw_out)
            print(f"    → {final_out}")
        else:
            print(f"    WARNING: pruner produced no output for {fake_game_id}")

    print(f"\nDone. Open files in {PRUNED_DIR} with benchmark_viz.py.")
    print("Example:")
    print(f'  python benchmark_viz.py "{PRUNED_DIR}\\MyReplay.json.gz"')


if __name__ == "__main__":
    prune_all()
