"""
Step 1: Offline replay indexer. NO SC2 LAUNCH.

Scans a folder of .SC2Replay files with sc2reader (load_level=1, header only)
and builds:
  - replay_index.json : {replay_filename: {"map_name": ..., "version": ...}}
  - unique_maps.json  : {map_name: representative_replay_filename}

The unique_maps file tells the next steps which maps exist and gives one
replay per map to extract static map data from.

Usage:
    python index_replays.py "D:\\betastar\\BREATH_3\\rfc_10k_2"

Safe to run repeatedly; it overwrites the two index files and launches nothing.
"""

import json
import os
import sys

import sc2reader

REPLAY_INDEX_PATH = "replay_index.json"
UNIQUE_MAPS_PATH = "unique_maps.json"


def index_folder(folder: str) -> None:
    replay_files = [f for f in os.listdir(folder) if f.endswith(".SC2Replay")]
    print(f"Found {len(replay_files)} replays in {folder}")

    replay_index = {}
    unique_maps = {}  # map_name -> first replay filename seen on that map
    failures = 0

    for i, fname in enumerate(replay_files):
        path = os.path.join(folder, fname)
        try:
            replay = sc2reader.load_replay(path, load_level=1)
            map_name = replay.map_name
            # release_string is like "5.0.15.96592"; fall back gracefully.
            version = getattr(replay, "release_string", None) or getattr(
                replay, "build", "unknown"
            )
            replay_index[fname] = {"map_name": map_name, "version": str(version)}
            if map_name not in unique_maps:
                unique_maps[map_name] = fname
        except Exception as e:
            failures += 1
            print(f"  FAILED {fname}: {e}")

        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(replay_files)}] indexed...")

    with open(REPLAY_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(replay_index, f, indent=2)
    with open(UNIQUE_MAPS_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_maps, f, indent=2)

    print(f"\nDone. {len(replay_index)} indexed, {failures} failed.")
    print(f"Unique maps ({len(unique_maps)}):")
    for m, rep in unique_maps.items():
        print(f"  {m!r}  <- {rep}")
    print(f"\nWrote {REPLAY_INDEX_PATH} and {UNIQUE_MAPS_PATH}")
    print("Eyeball the map list above for near-duplicate names before continuing.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python index_replays.py "<replay-folder>"')
        sys.exit(1)
    index_folder(sys.argv[1])
