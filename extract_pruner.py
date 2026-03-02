# point of this file is the compile the data to more usable forms.


import gzip
import json
import time

from numpy.ma.extras import compress_rows

INPUT_DIR = "output"
OUTPUT_DIR = "prunes"
#
# with gzip.open('1000 extracts/26382813.SC2Replay_p1.json.gz', 'rt', encoding='utf-8') as f:
#     data = json.load(f)
DATA_SAVE_LOCATION = "D:/betastar/parser/data.json"
with open(DATA_SAVE_LOCATION, "r", encoding="utf-8") as f:
    MMR_DATA = json.load(f)


def extract_mmr(game_id):
    return MMR_DATA[game_id]["mmr"]


def extract_win_flag(game_id: str) -> int:
    """
    Returns 1 if Zerg (POV player) won, else 0.
    Assumes all games are Zerg POV.
    """
    game = MMR_DATA[game_id]
    winner_id = game["winner_id"]
    players = game["players"]

    # Find which player is Zerg
    for p in players.values():
        if p["race"].lower() == "zerg":
            zerg_id = p["id"]
            break
    else:
        raise ValueError(f"No Zerg player found in game {game_id}")

    return 1 if winner_id == zerg_id else 0


def prune_unit_data(original_unit, current_state_iteration):
    pruned = {
        k: original_unit[k]
        for k in ["unit_type", "health", "shield", "energy", "is_structure"]
    }
    pruned["last_seen_x"], pruned["last_seen_y"] = (
        original_unit["last_seen_position"][0],
        original_unit["last_seen_position"][1],
    )
    if "last_seen" in original_unit.keys():
        last_seen = current_state_iteration - original_unit["last_seen"]
        pruned["last_seen"] = last_seen
    return pruned


def prune_visibility(visibility, block_size=10):
    h = (len(visibility) + block_size - 1) // block_size
    w = (len(visibility[0]) + block_size - 1) // block_size

    compressed = [[-1 for _ in range(w)] for _ in range(h)]

    # fairly low resolution. The entire sum is the maximum value of all of the grids.
    for bi in range(h):
        for bj in range(w):
            block_max = -1
            for i in range(
                bi * block_size, min((bi + 1) * block_size, len(visibility))
            ):
                for j in range(
                    bj * block_size, min((bj + 1) * block_size, len(visibility[0]))
                ):
                    block_max = max(block_max, visibility[i][j])
            compressed[bi][bj] = block_max

    return compressed


def prune_dict_of_dicts(unit_dict: dict, current_iteration):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_dict_of_units(unit_dict[i], current_iteration)
    return new_dict


def prune_dict_of_units(unit_dict: dict, current_iteration):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_unit_data(unit_dict[i], current_iteration)
    return new_dict


##This will be the base. The following is for playing with model inputs
def prune_state(original_state: dict, mmr: int) -> dict:
    current_iteration = original_state["iteration"]
    pruned = {
        k: original_state[k]
        for k in [
            "iteration",
            "player_pov",
            "own_spawn_x",
            "own_spawn_y",
            "enemy_spawn_x",
            "enemy_spawn_y",
            "own_race",
            "enemy_race",
            "supply_army",
            "supply_workers",
            "supply_used",
            "supply_left",
            "supply_cap",
            "workers_built",
            "army_built",
            "minerals",
            "gas",
            "under_construction",
            "newly_queued",
            "lair_started",
            "hive_started",
            "building_started",
        ]
    }
    # Rounded to make it easier to process
    pruned["mmr"] = int(mmr)
    pruned["visibility"] = prune_visibility(original_state["visibility"])
    pruned.update(
        {
            k: prune_dict_of_units(original_state[k], current_iteration)
            for k in ["player_units", "enemy_units_seen_and_alive"]
        }
    )
    # pruned.update(
    #     {
    #         k: prune_dict_of_dicts(original_state[k], current_iteration)
    #         for k in ["units_built"]
    #     }
    # )
    return pruned


# I intend to truncate units according to health. Hopefully this prioritizes highest impact units.
# Input is dict of units
# mode 1 for truncating enemy units. Which will also have the iteration timer thing.
# mode 0 for everything else
def truncate_to_50(units, mode=0) -> list:
    # Step 1: Sort and truncate top 50 by health
    if mode == 0:
        sorted_units = sorted(units.values(), key=lambda x: x["health"], reverse=True)[
            :50
        ]
    else:

        sorted_units = sorted(units.values(), key=lambda x: x["last_seen"])[:50]

    # Step 2: Convert dicts to flat lists (same key order assumed)
    unit_vectors = [list(unit.values()) for unit in sorted_units]

    # Step 3: Pad to 50 if needed
    if mode == 0:
        feature_len = 7
    else:
        feature_len = 8
    padding_unit = [-1] * feature_len
    while len(unit_vectors) < 50:
        unit_vectors.append(padding_unit)

    final = [item for sublist in unit_vectors for item in sublist]
    final = [int(x) for x in final]
    return final


# existing = {}
# for i in prune_state(data[str(0)])["player_army"]:
#     unit_type = prune_state(data[str(0)])["player_army"][i]["unit_type"]
#     if unit_type in existing:
#         existing[unit_type] += 1
#     else:
#         existing[unit_type] = 1
#
# print(existing)


def compress_pruned(pruned: dict) -> dict:
    # lair_started / hive_started are one-shot binary flags indicating
    # whether the tech transition began in this compressed timestep
    compressed = {
        k: pruned[k]
        for k in [
            "mmr",
            "iteration",
            "player_pov",
            "own_spawn_x",
            "own_spawn_y",
            "enemy_spawn_x",
            "enemy_spawn_y",
            "own_race",
            "enemy_race",
            "supply_army",
            "supply_workers",
            "supply_used",
            "supply_left",
            "supply_cap",
            "minerals",
            "gas",
            "visibility",
            "under_construction",
            "lair_started",
            "hive_started",
            "building_started",
            "newly_queued",
            "workers_built",
            "army_built",
        ]
    }

    player_units = {}
    player_structures = {}

    for tag, unit in pruned["player_units"].items():
        if unit["is_structure"]:
            player_structures[tag] = unit
        else:
            player_units[tag] = unit

    compressed["player_units"] = truncate_to_50(player_units)
    compressed["player_structures"] = truncate_to_50(player_structures)

    enemy_structures = {}
    enemy_units = {}

    # removed because i remove tag from units
    # enemy = pruned["enemy_units_seen_and_alive"]
    # for l in enemy:
    #     if enemy[l]["is_structure"]:
    #         enemy_structures[enemy[l]["tag"]] = enemy[l]
    #     else:
    #         enemy_units[enemy[l]["tag"]] = enemy[l]

    enemy = pruned["enemy_units_seen_and_alive"]
    for tag, unit in enemy.items():
        if unit["is_structure"]:
            enemy_structures[tag] = unit
        else:
            enemy_units[tag] = unit

    enemy_structures = truncate_to_50(enemy_structures, 1)
    enemy_units = truncate_to_50(enemy_units, 1)
    compressed["enemy_structures"] = enemy_structures
    compressed["enemy_units"] = enemy_units
    return compressed


import os
import gzip
import json


def stitch_and_prune_replay(replay_prefix: str):
    """
    replay_prefix example: "26382815.SC2Replay"
    """

    mmr = extract_mmr(replay_prefix)

    # 1. Find all matching files
    files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.startswith(replay_prefix) and f.endswith(".json.gz")
    ]

    if not files:
        raise RuntimeError(f"No files found for {replay_prefix}")

    # 2. Sort by chunk index (_0, _1, _2, ...)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    print(f"Found {len(files)} chunks:")
    for f in files:
        print("  ", f)

    # 3. Process and stitch
    stitched = []
    # Note last value of a stack is the same as the first value of the next stack
    for chunk_idx, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            chunk = json.load(f)

        start = 0 if chunk_idx == 0 else 1  # DROP first timestep of later chunks

        for i in range(start, len(chunk)):
            state = chunk[str(i)]
            stitched.append(compress_pruned(prune_state(state, mmr)))

    return stitched


def split_state_and_actions(final_dict: dict) -> dict:
    """
    Splits a compressed timestep dict into:
      {
        "state":   <environment / observation>,
        "actions": <player action signals>
      }

    Assumes input is the output of compress_pruned().
    """

    action_keys = {
        "building_started",
        "newly_queued",
        "workers_built",
        "army_built",
        "lair_started",
        "hive_started",
        "visibility",
    }

    state = {}
    actions = {}

    for k, v in final_dict.items():
        if k in action_keys:
            actions[k] = v
        else:
            state[k] = v

    return {
        "state": state,
        "actions": actions,
    }


# print(compress_pruned(prune_state(data[str(6)])).keys())
# print(compress_pruned(prune_state(data[str(6)]))["under_construction"])
def process(prefix):
    stitched = stitch_and_prune_replay(prefix)

    # --- NEW: compute replay-level winner flag ---
    win_flag = extract_win_flag(prefix)

    # Split each timestep into state/actions
    sa = [split_state_and_actions(t) for t in stitched]

    # --- NEW: wrap frames and winner separately ---
    final_output = {"winner": win_flag, "frames": sa}  # 1 if Zerg won, else 0

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, prefix + ".json.gz")

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(final_output, f)


def main(folder=INPUT_DIR):
    seen = set()

    for filename in os.listdir(folder):
        if not filename.endswith(".json.gz"):
            continue

        # Extract prefix before first underscore
        game_id = filename.split("_", 1)[0]

        if game_id not in seen:
            seen.add(game_id)
            process(game_id)


if __name__ == "__main__":
    main()
