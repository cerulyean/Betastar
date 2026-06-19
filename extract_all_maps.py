"""
Step 2: Per-map static terrain extractor (pysc2).

Reads unique_maps.json (from index_replays.py), and for each map that does not
already have a saved file, launches the matching SC2 build, grabs the static
minimap feature planes, and saves them keyed by map name.

Stores ALL static-meaningful planes plus playable_area, so coordinate
reconciliation with python-sc2 world coords (expansions, resources) is possible
later without re-running.

Only ~N runs total (one per unique map), not per-replay. Static data.

Usage:
    python extract_all_maps.py "D:\\betastar\\BREATH_3\\rfc_10k_2"
    (folder is where the representative replays live)

Requires: a version-table entry in pysc2 platforms.py matching each map's build.
"""

import gzip
import json
import os
import re
import sys

from absl import app
from pysc2 import run_configs
from pysc2.lib import features
from s2clientprotocol import sc2api_pb2 as sc_pb

UNIQUE_MAPS_PATH = "unique_maps.json"
OUTPUT_DIR = "map_data"
MINIMAP_RES = 128  # overcollect: higher res than needed, downsample later

# Static minimap planes worth keeping. Others (creep, camera, selected,
# player_id, player_relative, unit_type, alerts, visibility_map) are
# game-state dependent and meaningless from one arbitrary observation,
# but we dump them anyway under "raw_all" since you asked to overcollect.
STATIC_PLANES = ["height_map", "pathable", "buildable"]


def safe_name(map_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", map_name).strip("_")


import time

# Use the corrected per-map representatives (installed builds only), produced
# by diagnose_builds.py. Each entry has {"replay": fname, "build": "95299"}.
# We request the run config by the replay's actual build so SC2 launches the
# matching binary, instead of forcing every replay onto one hardcoded build.
UNIQUE_MAPS_PATH = "unique_maps_playable.json"
BETWEEN_MAPS_DELAY = 2.0


def extract_one(replay_path: str, build: str) -> None:
    # pysc2 matches run configs by the full version LABEL in platforms.py,
    # not by bare build number. Our table entries are labeled "5.0.15.<build>",
    # so request that label.
    version_label = f"5.0.15.{build}"
    run_config = run_configs.get(version=version_label)
    replay_data = run_config.replay_data(replay_path)

    with run_config.start() as controller:
        info = controller.replay_info(replay_data)
        map_name = info.map_name
        out_path = os.path.join(OUTPUT_DIR, safe_name(map_name) + ".json.gz")
        if os.path.exists(out_path):
            print(f"  SKIP {map_name!r} (already extracted)")
            return

        print(f"  Extracting {map_name!r} from {os.path.basename(replay_path)}")
        print(f"    version {info.game_version}  base_build {info.base_build}")
        interface = sc_pb.InterfaceOptions(
            raw=True,
            score=False,
            feature_layer=sc_pb.SpatialCameraSetup(width=24),
        )
        interface.feature_layer.resolution.x = MINIMAP_RES
        interface.feature_layer.resolution.y = MINIMAP_RES
        interface.feature_layer.minimap_resolution.x = MINIMAP_RES
        interface.feature_layer.minimap_resolution.y = MINIMAP_RES

        controller.start_replay(
            sc_pb.RequestStartReplay(
                replay_data=replay_data,
                options=interface,
                observed_player_id=1,
            )
        )

        game_info = controller.game_info()
        feat = features.features_from_game_info(game_info)

        controller.step(1)
        obs = controller.observe()
        agent_obs = feat.transform_obs(obs)
        mm = agent_obs.feature_minimap

        pa = game_info.start_raw.playable_area  # x0, y0, x1, y1 in world units
        playable_area = {
            "x": pa.p0.x,
            "y": pa.p0.y,
            "w": pa.p1.x - pa.p0.x,
            "h": pa.p1.y - pa.p0.y,
        }

        # DEBUG: find where map size actually lives. Print candidates, then
        # use whichever works. Remove this block once confirmed.
        map_size = None
        try:
            sr = game_info.start_raw
            fields = [a for a in dir(sr) if not a.startswith("_")]
            print(f"    [debug] start_raw fields: {fields[:40]}")
            if hasattr(sr, "map_size"):
                ms = sr.map_size
                map_size = {"x": ms.x, "y": ms.y}
                print(f"    [debug] start_raw.map_size = {map_size}")
        except Exception as e:
            print(f"    [debug] start_raw access failed: {e!r}")

        if map_size is None:
            map_size = {
                "x": pa.p0.x + (pa.p1.x - pa.p0.x) + pa.p0.x,
                "y": pa.p0.y + (pa.p1.y - pa.p0.y) + pa.p0.y,
            }
            print(f"    [debug] using fallback map_size estimate: {map_size}")

        out = {
            "map_name": map_name,
            "resolution": MINIMAP_RES,
            "playable_area": playable_area,
            "map_size": map_size,
        }
        # Named static planes.
        for plane in STATIC_PLANES:
            out[plane] = getattr(mm, plane).tolist()
        # Overcollect: dump every minimap plane available, raw.
        out["raw_all"] = (
            {name: mm[i].tolist() for i, name in enumerate(mm._index_names[0])}
            if hasattr(mm, "_index_names")
            else {}
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(out, f)

        psum = sum(sum(r) for r in out["pathable"])
        bsum = sum(sum(r) for r in out["buildable"])
        print(f"    saved -> {out_path}  (pathable={psum} buildable={bsum})")
        print(f"    playable_area={playable_area}  map_size={map_size}")


def main(argv):
    if len(argv) != 2:
        print('Usage: python extract_all_maps.py "<replay-folder>"')
        sys.exit(1)
    folder = argv[1]

    with open(UNIQUE_MAPS_PATH, "r", encoding="utf-8") as f:
        unique_maps = json.load(f)

    print(f"{len(unique_maps)} unique maps to consider.")
    for map_name, entry in unique_maps.items():
        replay_path = os.path.join(folder, entry["replay"])
        build = entry["build"]
        try:
            extract_one(replay_path, build)
        except Exception as e:
            print(f"  FAILED {map_name!r} (build {build}): {e}")
        time.sleep(BETWEEN_MAPS_DELAY)


if __name__ == "__main__":
    app.run(main)
