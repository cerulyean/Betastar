"""
Step 3: Per-map expansion + resource grabber (python-sc2).

A minimal ObserverAI that, on the first frame, records:
  - expansion_locations (base centers, in world coords)
  - all mineral field positions
  - all vespene geyser positions
  - own/enemy spawn positions
  - playable_area
...then immediately leaves. It does NOT run any of simulator.py's save
pipeline, does NOT write the per-frame extraction, and plays no further than
frame 0. Static data, so one run per unique map.

Saves to expansion_data/<map_name>.json (plain JSON, small).

Usage:
    python grab_expansions.py "D:\\betastar\\BREATH_3\\rfc_10k_2"
    (reads unique_maps.json, skips maps already done)

IMPORTANT: run the single-map probe first (set PROBE=True below or just run on
one map) to confirm expansion_locations populates under ObserverAI before
trusting the batch.
"""

import json
import os
import re
import sys

from sc2.observer_ai import ObserverAI
from sc2.main import run_replay

UNIQUE_MAPS_PATH = "unique_maps_playable.json"
OUTPUT_DIR = "expansion_data"


def safe_name(map_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", map_name).strip("_")


class _ExpansionGrabber(ObserverAI):
    """Grabs static base/resource layout on frame 0, then leaves."""

    def __init__(self):
        super().__init__()
        self.grabbed = None

    async def on_step(self, iteration: int):
        if iteration > 0:
            return

        # expansion_locations is lazily computed and asserts if accessed before
        # _find_expansion_locations() has run. Prime it first.
        try:
            self._find_expansion_locations()
        except Exception as e:
            print(f"    _find_expansion_locations FAILED: {e}")

        try:
            expansions = [
                {"x": float(p.x), "y": float(p.y)}
                for p in self.expansion_locations.keys()
            ]
        except Exception as e:
            expansions = []
            print(f"    expansion_locations FAILED: {e}")

        # Under ObserverAI the mineral_field / vespene_geyser convenience
        # collections come back empty. The neutral resource units are present
        # in self.units instead — filter by type name.
        minerals, geysers = [], []
        try:
            for u in self.units:
                tname = u.type_id.name.upper()
                entry = {
                    "x": float(u.position.x),
                    "y": float(u.position.y),
                    "type": u.type_id.name,
                }
                if "MINERALFIELD" in tname or "MINERAL" in tname:
                    minerals.append(entry)
                elif "VESPENE" in tname or "GEYSER" in tname:
                    geysers.append(entry)
        except Exception as e:
            print(f"    resource scan FAILED: {e}")

        x_off, y_off, w, h = self.game_info.playable_area

        self.grabbed = {
            "map_name": self.game_info.map_name,
            "playable_area": {"x": x_off, "y": y_off, "w": w, "h": h},
            "own_spawn": {
                "x": float(self.start_location.x),
                "y": float(self.start_location.y),
            },
            "enemy_spawn": {
                "x": float(self.enemy_start_locations[0].x),
                "y": float(self.enemy_start_locations[0].y),
            },
            "expansions": expansions,
            "minerals": minerals,
            "geysers": geysers,
        }
        print(
            f"    expansions={len(expansions)} "
            f"minerals={len(minerals)} geysers={len(geysers)}"
        )
        await self.client.leave()  # bail; write nothing through any save pipeline


def grab_one(replay_path: str, map_name: str) -> None:
    out_path = os.path.join(OUTPUT_DIR, safe_name(map_name) + ".json")
    if os.path.exists(out_path):
        print(f"  SKIP {map_name!r} (already grabbed)")
        return

    print(f"  Grabbing {map_name!r} from {os.path.basename(replay_path)}")
    grabber = _ExpansionGrabber()
    run_replay(grabber, replay_path=replay_path, realtime=False, observed_id=1)

    if grabber.grabbed is None:
        print(f"    WARNING: nothing grabbed for {map_name!r}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grabber.grabbed, f, indent=2)
    print(f"    saved -> {out_path}")


def main():
    if len(sys.argv) != 2:
        print('Usage: python grab_expansions.py "<replay-folder>"')
        sys.exit(1)
    folder = sys.argv[1]

    with open(UNIQUE_MAPS_PATH, "r", encoding="utf-8") as f:
        unique_maps = json.load(f)

    print(f"{len(unique_maps)} unique maps to consider.")
    for map_name, entry in unique_maps.items():
        replay_path = os.path.join(folder, entry["replay"])
        try:
            grab_one(replay_path, map_name)
        except Exception as e:
            print(f"  FAILED {map_name!r}: {e}")


if __name__ == "__main__":
    main()
