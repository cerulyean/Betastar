# point of this file is the compile the data to more usable forms.


import gzip
import json
import time
import os

from numpy.ma.extras import compress_rows
from sc2reader.data import datapacks

from constants import WORKERS, NOT_ARMY, VALID_UNITS
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId

# Build a supply lookup keyed by UnitTypeId name (uppercase).
# sc2reader drops the "MP" suffix that python-sc2 uses for some LotV units,
# so we patch those manually.
_lotv = datapacks["LotV"]
_latest_build = max(k for k in _lotv.keys() if k != "base")
_dp = _lotv[_latest_build]
_SUPPLY_BY_NAME = {
    attr.upper(): getattr(_dp, attr).supply
    for attr in dir(_dp)
    if hasattr(getattr(_dp, attr), "supply") and hasattr(getattr(_dp, attr), "id")
}
_NAME_FIXES = {
    "LURKERMP": "LURKER",
    "LURKERDENMP": "LURKERDEN",
    "SWARMHOSTMP": "SWARMHOST",
    "NYDUSCANAL": "NYDUSWORM",
}

EXCLUDE = {
    "no_scout_noresponse.SC2Replay",
    "scout_bad_response.SC2Replay",
    "scout_no_response.SC2Replay",
    "socut_less_bad_response.SC2Replay",
}


def _get_supply(unit_type_id: UnitTypeId) -> float:
    name = _NAME_FIXES.get(unit_type_id.name, unit_type_id.name)
    return _SUPPLY_BY_NAME.get(name, 0)


INPUT_DIR = "output"
# INPUT_DIR = "1000 extracts"
OUTPUT_DIR = "prunes"
# OUTPUT_DIR = "prune_test"
DATA_SAVE_LOCATION = "D:/betastar/parser/data.json"
# DATA_SAVE_LOCATION = "D:/betastar/parser/data_test.json"
with open(DATA_SAVE_LOCATION, "r", encoding="utf-8") as f:
    MMR_DATA = json.load(f)

import datetime

CHANGELOG_PATH = "D:/betastar/changelog.json"


def get_next_version() -> int:
    """
    Returns the next version number to write to.
    If the latest logged version has a missing folder, reuses that version.
    Otherwise increments to a new one.
    """
    if os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH, "r") as f:
            log = json.load(f)
        if log:
            latest = max(log, key=lambda x: x["version"])
            if not os.path.exists(latest["output_dir"]):
                return latest["version"]
            return latest["version"] + 1

    return 1


def write_changelog(version: int, note: str):
    """
    Writes a changelog entry for the given version.
    Overwrites the existing entry if reusing a deleted version, otherwise appends.

    Args:
        version (int): The version number just created.
        note (str): User-provided description of what changed.
    """
    if os.path.exists(CHANGELOG_PATH):
        with open(CHANGELOG_PATH, "r") as f:
            log = json.load(f)
    else:
        log = []

    for entry in log:
        if entry["version"] == version:
            entry["timestamp"] = datetime.datetime.now().isoformat()
            entry["note"] = note
            break
    else:
        log.append(
            {
                "version": version,
                "timestamp": datetime.datetime.now().isoformat(),
                "output_dir": f"prunes_v{version}",  # was: f"{OUTPUT_DIR}_v{version}"
                "note": note,
            }
        )

    with open(CHANGELOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Changelog updated: v{version} — {note}")


def extract_mmr(game_id: str) -> int:
    """
    Extracts the MMR value for a given game.

    Args:
        game_id (str): The replay filename prefix (e.g. "26382813.SC2Replay").

    Returns:
        int: The MMR value associated with the game.
    """
    entry = MMR_DATA[game_id]
    if entry.get("skipped"):
        raise ValueError(f"{game_id} is marked as skipped")
    return entry["mmr"]


def extract_win_flag(game_id: str) -> int:
    """
    Determines whether the Zerg player won the game, using Blizzard toon_ids.

    Returns 1 if Zerg won, 0 otherwise.
    """
    game = MMR_DATA[game_id]
    winner_toon = game["winner_toon_id"]
    players = game["players"]

    for p in players.values():
        if p["race"].lower() == "zerg":
            zerg_toon = p["toon_id"]
            break
    else:
        raise ValueError(f"No Zerg player found in game {game_id}")

    return 1 if winner_toon == zerg_toon else 0


def prune_unit_data(original_unit: dict, current_state_iteration: int) -> dict:
    """
    Strips a unit dict down to only the fields needed for model input.

    Args:
        original_unit (dict): Raw unit data as extracted by the simulator.
        current_state_iteration (int): The current game iteration, used to compute
            how many iterations ago the unit was last seen.

    Returns:
        dict: Pruned unit with keys: unit_type, health, shield, energy,
            is_structure, last_seen_x, last_seen_y, and optionally last_seen.
    """
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


def prune_visibility(visibility: list, block_size: int = 10) -> list:
    """
    Downsamples a 2D visibility grid by taking the max value in each block.

    Args:
        visibility (list): 2D list of visibility values from the game state.
        block_size (int): Size of each compression block. Default is 10.

    Returns:
        list: Compressed 2D grid where each cell is the max of its block.
    """
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
    """
    Extracts and prunes a raw game state down to the fields used for training.

    Args:
        original_state (dict): Full game state dict from the simulator.
        mmr (int): MMR value to attach to this state.

    Returns:
        dict: Pruned state containing scalar game info, visibility grid,
            player units, and enemy units seen and alive.
    """
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
    return pruned


# Possibly increase what is measured beyond basic number
def encode_units_by_type(
    units: dict, race: Race, track_last_seen: bool = False
) -> list:
    sorted_vocab = sorted(VALID_UNITS[race], key=lambda u: u.value)
    vocab = {unit: 0 for unit in sorted_vocab}
    last_seen = {unit: float("inf") for unit in sorted_vocab}

    for unit in units.values():
        unit_type = UnitTypeId(unit["unit_type"])
        if unit_type in vocab:
            vocab[unit_type] += 1
            if track_last_seen and "last_seen" in unit:
                last_seen[unit_type] = min(last_seen[unit_type], unit["last_seen"])

    result = list(vocab.values())
    if track_last_seen:
        result += [-1 if v == float("inf") else v for v in last_seen.values()]
    return result


# I intend to truncate units according to health. Hopefully this prioritizes highest impact units.
# Input is dict of units
# mode 1 for truncating enemy units. Which will also have the iteration timer thing.
# mode 0 for everything else
def truncate_to_50(units: dict, mode: int = 0) -> list:
    """
    Sorts and truncates a unit dict to at most 50 units, then flattens to a fixed-length list.
    Pads with -1 if fewer than 50 units are present.

    Args:
        units (dict): Dict of unit dicts, keyed by unit tag.
        mode (int): 0 for own units (sorted by health descending),
                    1 for enemy units (sorted by last_seen ascending, i.e. most recently seen first).

    Returns:
        list: Flat int list of length 350 (mode=0) or 400 (mode=1),
            representing up to 50 units with 7 or 8 features each.

    Note:
        To be replaced by fixed vocab encoding. See compress_pruned().
    """
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
    """
    Converts a pruned state dict into the final flat representation used for model input.
    Separates player and enemy units/structures and encodes them as fixed-length lists.

    Args:
        pruned (dict): Output of prune_state().

    Returns:
        dict: Compressed state with scalar fields, visibility grid, and four flat
            unit lists: player_units, player_structures, enemy_units, enemy_structures.
    """
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

    compressed["player_units"] = encode_units_by_type(pruned["player_units"], Race.Zerg)

    enemy = pruned["enemy_units_seen_and_alive"]
    compressed["enemy_units"] = encode_units_by_type(
        enemy, Race.Protoss, track_last_seen=True
    )
    return compressed


def stitch_and_prune_replay(replay_prefix: str) -> list:
    """
    Loads all chunked output files for a replay, stitches them together,
    and returns the full sequence of compressed states.

    Args:
        replay_prefix (str): Replay filename without chunk suffix (e.g. "26382813.SC2Replay").

    Returns:
        list: Ordered list of compressed state dicts across the full replay.

    Raises:
        RuntimeError: If no matching chunk files are found.
    """
    """
    replay_prefix example: "26382815.SC2Replay"
    """
    mmr = extract_mmr(replay_prefix)

    files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.startswith(replay_prefix) and f.endswith(".json.gz")
    ]

    if not files:
        raise RuntimeError(f"No files found for {replay_prefix}")

    final_files = [f for f in files if f.split("_")[-1].split(".")[0] == "final"]
    numbered_files = [f for f in files if f not in final_files]
    numbered_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    files = numbered_files + final_files

    print(f"Found {len(files)} chunks:")
    for f in files:
        print("  ", f)

    stitched = []
    prev_tags = set()

    for chunk_idx, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            chunk = json.load(f)

        start = 0 if chunk_idx == 0 else 1

        for i in range(start, len(chunk)):
            state = chunk[str(i)]
            pruned = prune_state(state, mmr)

            new_tags = set(pruned["player_units"].keys()) - prev_tags
            new_units = {t: pruned["player_units"][t] for t in new_tags}
            pruned["army_built"] = sum(
                _get_supply(UnitTypeId(u["unit_type"]))
                for u in new_units.values()
                if _get_supply(UnitTypeId(u["unit_type"])) > 0
                and UnitTypeId(u["unit_type"]) not in WORKERS
            )

            prev_tags = set(pruned["player_units"].keys())
            stitched.append(compress_pruned(pruned))

    return stitched


def split_state_and_actions(final_dict: dict) -> dict:
    """
    Splits a compressed timestep into observation (state) and label (actions).

    Args:
        final_dict (dict): Output of compress_pruned().

    Returns:
        dict: With two keys:
            "state"   - environment features used as model input.
            "actions" - player action signals used as model targets.
    """
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
def process(prefix: str) -> None:
    """
    Full pipeline for a single replay: stitches chunks, extracts winner flag,
    splits into state/action pairs, and saves to the prunes output directory.

    Args:
        prefix (str): Replay filename prefix (e.g. "26382813.SC2Replay").

    Returns:
        None. Saves a gzipped JSON file to the prunes directory.
    """
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


def main(folder: str = INPUT_DIR) -> None:
    """
    Iterates over all chunked replay files in a folder and processes each unique replay.
    Skips duplicate chunks for the same replay. Versions output and logs changes.

    Args:
        folder (str): Path to the directory containing chunked .json.gz files.

    Returns:
        None.
    """
    global OUTPUT_DIR

    version = get_next_version()
    OUTPUT_DIR = f"{OUTPUT_DIR}_v{version}"

    note = input(f"What changed in this version (v{version})? ").strip()
    write_changelog(version, note)

    seen = set()
    for filename in os.listdir(folder):
        if not filename.endswith(".json.gz"):
            continue
        game_id = filename.split("_", 1)[0]
        game_data = MMR_DATA.get(game_id.replace(".json.gz", "").split("_")[0])
        if game_data is None or game_data.get("skipped"):
            continue
        if filename in EXCLUDE:
            continue
        if game_id not in seen:
            seen.add(game_id)
            try:
                process(game_id)
            except Exception as e:
                print(f"Failed {game_id}: {e}")


if __name__ == "__main__":
    main()
