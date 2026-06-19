"""
One-shot static map extractor using pysc2.

Steps a replay exactly far enough to get one observation, then pulls the
*static* minimap feature planes (terrain height, pathability, buildability)
and saves them keyed by map name. These planes are constant for the whole
game, so there is no need to iterate frames or re-run the whole corpus —
run this once per unique map.

Usage:
    python extract_map.py "D:\\betastar\\BREATH_3\\rfc_10k_2\\27122786_480_5088.5_Hupsaiya.SC2Replay"

Output:
    map_data/<map_name>.json.gz
"""

import gzip
import json
import os
import re
import sys

from pysc2 import run_configs
from pysc2.lib import features
from s2clientprotocol import sc2api_pb2 as sc_pb

OUTPUT_DIR = "map_data"
MINIMAP_RES = 64  # resolution of the minimap feature planes


def safe_name(map_name: str) -> str:
    """Make a map name safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", map_name).strip("_")


def extract_map(replay_path: str) -> None:
    # First, read the replay header to learn its version. replay_info is a
    # controller method, so spin up a throwaway controller on the default build
    # just to read the header (it doesn't start the replay, so the build it
    # launches on doesn't matter here).
    base_config = run_configs.get()
    replay_data = base_config.replay_data(replay_path)
    with base_config.start() as probe:
        info = probe.replay_info(replay_data)
    map_name = info.map_name
    print(f"Map: {map_name}")
    print(f"Game version: {info.game_version}  base_build: {info.base_build}")

    # Now get a run config bound to the replay's actual version, so SC2 launches
    # the matching build instead of the newest installed one.
    # info.game_version is like "5.0.15.96592"; the version-table entry you added
    # to platforms.py is labeled "5.0.15", so match on the first three parts.
    patch_version = ".".join(info.game_version.split(".")[:3])
    print(f"Requesting run config for version: {patch_version}")
    run_config = run_configs.get(version=patch_version)

    with run_config.start() as controller:

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
                observed_player_id=1,  # POV doesn't matter for static terrain
            )
        )

        feat = features.features_from_game_info(controller.game_info())

        # Step once so the observation is populated, then grab a single obs.
        controller.step(1)
        obs = controller.observe()
        agent_obs = feat.transform_obs(obs)

        mm = agent_obs.feature_minimap

        # These three planes are static for the whole game:
        #   height_map  - terrain elevation (uint8)
        #   pathable    - 1 where ground units can walk
        #   buildable   - 1 where structures can be placed
        map_planes = {
            "map_name": map_name,
            "resolution": MINIMAP_RES,
            "height_map": mm.height_map.tolist(),
            "pathable": mm.pathable.tolist(),
            "buildable": mm.buildable.tolist(),
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, safe_name(map_name) + ".json.gz")
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(map_planes, f)

        print(f"Saved {MINIMAP_RES}x{MINIMAP_RES} planes -> {out_path}")
        print(f"  height_map: {len(map_planes['height_map'])} rows")
        print(f"  pathable:   sum={sum(sum(r) for r in map_planes['pathable'])}")
        print(f"  buildable:  sum={sum(sum(r) for r in map_planes['buildable'])}")


def main(argv):
    if len(argv) != 2:
        print('Usage: python extract_map.py "<path-to-replay>"')
        sys.exit(1)
    extract_map(argv[1])


if __name__ == "__main__":
    from absl import app

    app.run(main)
