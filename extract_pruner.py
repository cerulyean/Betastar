#point of this file is the compile the data to more usable forms.


import gzip
import json
import time

from numpy.ma.extras import compress_rows

# with gzip.open('1000 extracts/26382813.SC2Replay_p1.json.gz', 'rt', encoding='utf-8') as f:
#     data = json.load(f)

with gzip.open('1000 extracts/26382813.SC2Replay_p2.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)
print(data[str(0)].keys())

def prune_unit_data(original_unit):
    return {k: original_unit[k] for k in
                ["tag", "unit_type", "last_seen_position", "tag", "health", "shield", "energy", "is_structure"]}

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

def prune_dict_of_dicts(unit_dict:dict):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_dict_of_units(unit_dict[i])
    return new_dict

def prune_dict_of_units(unit_dict:dict):
    new_dict = {}
    for i in unit_dict:
        new_dict[i] = prune_unit_data(unit_dict[i])
    return new_dict

def prune_state(original_state: dict):
    pruned = {k: original_state[k] for k in
                ["iteration", "player_pov", "own_spawn_x", "own_spawn_y", "enemy_spawn_x", "enemy_spawn_y", "own_race",
                 "enemy_race", "supply_army", "supply_workers", "supply_used", "supply_left", "supply_cap",
                 "workers_built", "army_built", "minerals", "gas"]}
    pruned["visibility"] = prune_visiblity(original_state["visibility"])
    pruned.update({
        k: prune_dict_of_units(original_state[k])
        for k in ["player_buildings", "player_army", "enemy_units_seen_and_alive"]
    })
    pruned.update({
        k: prune_dict_of_dicts(original_state[k])
        for k in ["buildings_constructed", "units_built"]
    })
    return pruned



print(prune_state(data[str(0)]))