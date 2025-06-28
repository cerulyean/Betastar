#point of this file is the compile the data to more usable forms.


import gzip
import json
import time

from numpy.ma.extras import compress_rows

# with gzip.open('1000 extracts/26382813.SC2Replay_p1.json.gz', 'rt', encoding='utf-8') as f:
#     data = json.load(f)

with gzip.open('1000 extracts/26382815.SC2Replay_p1.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

def prune_unit_data(original_unit, current_state_iteration):
    pruned = {k: original_unit[k] for k in
              ["tag", "unit_type", "health", "shield", "energy", "is_structure"]}
    pruned["last_seen_x"], pruned["last_seen_y"] = (
        original_unit["last_seen_position"][0], original_unit["last_seen_position"][1])
    if "last_seen" in original_unit.keys():
        last_seen = current_state_iteration - original_unit["last_seen"]
        pruned["last_seen"] =  last_seen
    return pruned

def prune_visiblity(visibility):
    compressed = [[-1 for _ in range((len(visibility[0]) + 9) // 10)]
                  for _ in range((len(visibility) + 9) // 10)]

    for i, row in enumerate(visibility):
        for j, val in enumerate(row):
            if val == 2:
                compressed[i // 10][j // 10] = 2
            if val == 1:
                compressed[i // 10][j // 10] = 1
            if val == 0:
                compressed[i // 10][j // 10] = 0
            if val == -1:
                compressed[i // 10][j // 10] = -1

    return compressed

def prune_dict_of_dicts(unit_dict:dict, current_iteration):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_dict_of_units(unit_dict[i], current_iteration)
    return new_dict

def prune_dict_of_units(unit_dict:dict, current_iteration):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_unit_data(unit_dict[i], current_iteration)
    return new_dict

##This will be the base. The following is for playing with model inputs
def prune_state(original_state: dict) -> dict:
    current_iteration = original_state["iteration"]
    pruned = {k: original_state[k] for k in
                ["iteration", "player_pov", "own_spawn_x", "own_spawn_y", "enemy_spawn_x", "enemy_spawn_y", "own_race",
                 "enemy_race", "supply_army", "supply_workers", "supply_used", "supply_left", "supply_cap",
                 "workers_built", "army_built", "minerals", "gas", "under_construction"]}
    pruned["visibility"] = prune_visiblity(original_state["visibility"])
    pruned.update({
        k: prune_dict_of_units(original_state[k], current_iteration)
        for k in ["player_buildings", "player_units", "enemy_units_seen_and_alive"]
    })
    pruned.update({
        k: prune_dict_of_dicts(original_state[k], current_iteration)
        for k in ["buildings_constructed", "units_built"]
    })
    return pruned


#todo i think add building tracking as well, seperate from the regular unit tracking


#I intend to truncate units according to health. Hopefully this prioritizes highest impact units.
#Input is dict of units
#mode 1 for truncating enemy units. Which will also have the iteration timer thing.
#mode 0 for everything else
def truncate_to_50(units, mode=0) -> list:
    # Step 1: Sort and truncate top 50 by health
    if mode == 0:
        sorted_units = sorted(units.values(), key=lambda x: x["health"], reverse=True)[:50]
    else:
        sorted_units = sorted(units.values(), key=lambda x: x["last_seen"])[:50]

    # Step 2: Convert dicts to flat lists (same key order assumed)
    unit_vectors = [list(unit.values()) for unit in sorted_units]

    # Step 3: Pad to 50 if needed
    if mode == 0:
        feature_len = 8
    else:
        feature_len = 9
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

def compress_pruned(pruned:dict) -> dict:
    compressed = {k: pruned[k] for k in
                ["iteration", "player_pov", "own_spawn_x", "own_spawn_y", "enemy_spawn_x", "enemy_spawn_y", "own_race",
                 "enemy_race", "supply_army", "supply_workers", "supply_used", "supply_left", "supply_cap",
                 "workers_built", "army_built", "minerals", "gas", "visibility", "under_construction"]}

    compressed.update({
        k: truncate_to_50(pruned[k])
        for k in ["player_buildings", "player_units"]
    })

    enemy_structures = {}
    enemy_units = {}

    enemy = pruned["enemy_units_seen_and_alive"]
    for l in enemy:
        if enemy[l]["is_structure"]:
            enemy_structures[enemy[l]["tag"]] = enemy[l]
        else:
            enemy_units[enemy[l]["tag"]] = enemy[l]

    print(len(enemy_units))
    print(len(enemy_structures))
    print(enemy_structures)
    enemy_structures = truncate_to_50(enemy_structures, 1)
    enemy_units = truncate_to_50(enemy_units, 1)
    compressed["enemy_structures"] = enemy_structures
    compressed["enemy_units"] = enemy_units

    # compressed.update({
    #     k: truncate_to_50(pruned[k], 1)
    #     for k in ["enemy_units_seen_and_alive"]
    # })
    compressed.update({
        k: {
            outer_key: truncate_to_50(inner_dict)
            for outer_key, inner_dict in pruned[k].items()
        }
        for k in ["buildings_constructed", "units_built"]
    })
    return compressed

print(len(data))
print(data.keys())
print(compress_pruned(prune_state(data[str(14)]))["enemy_structures"])
# pruned = prune_state(data[str(0)]).keys()
# print(pruned)
# print(truncate_to_50(prune_state(data[str(0)])["player_army"]))
# print(data[str(0)].keys())
print("next")
enemies = data[str(7)]["enemy_units_seen_and_alive"]
for i in enemies.keys():
    print(enemies[i]["unit_type"])
#for some reason when probe saw natural hatch he saw main hatch too???? What the fuxck