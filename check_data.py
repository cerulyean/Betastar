"""
Validate a merged data.json without opening it in an editor.

Checks, across all entries that have toon_ids:
  - every player has a toon_id
  - winner_toon_id is present and equals exactly one player's toon_id
  - races are canonical ('zerg'/'protoss') and it's a clean ZvP
  - original 'id' / 'name' fields are still present (not clobbered)
Then prints one specific entry (default 26824264) so you can eyeball it.

Usage:
  python check_data.py <data.json> [gameid_to_print]
"""

import sys
import json


def check(path: str, show_gid: str = "26824264"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    has_toon = 0
    missing_player_toon = []
    bad_winner = []
    not_zvp = []
    lost_original = []

    for gid, e in data.items():
        if e.get("skipped"):
            continue
        players = e.get("players", {})
        toons = [p.get("toon_id") for p in players.values()]

        # only validate entries that actually went through the merge
        if not any(toons):
            continue
        has_toon += 1

        if not all(toons):
            missing_player_toon.append(gid)

        # winner must match exactly one player's toon_id
        wt = e.get("winner_toon_id")
        if wt is None or toons.count(wt) != 1:
            bad_winner.append(gid)

        # clean ZvP on the stored (canonical) races
        races = sorted(p.get("race", "") for p in players.values())
        if races != ["protoss", "zerg"]:
            not_zvp.append((gid, races))

        # original fields preserved
        for p in players.values():
            if "id" not in p or "name" not in p:
                lost_original.append(gid)
                break

    print(f"total entries          : {total}")
    print(f"entries with toon_ids  : {has_toon}")
    print(f"missing a player toon  : {len(missing_player_toon)}")
    print(f"bad/ambiguous winner   : {len(bad_winner)}")
    print(f"not clean ZvP          : {len(not_zvp)}")
    print(f"lost original id/name  : {len(lost_original)}")

    for label, lst in [
        ("missing player toon", missing_player_toon),
        ("bad winner", bad_winner),
        ("lost original", lost_original),
    ]:
        if lst:
            print(f"\nfirst few {label}: {lst[:10]}")
    if not_zvp:
        print(f"\nfirst few not-ZvP: {not_zvp[:10]}")

    print(f"\n==== entry {show_gid} ====")
    if show_gid in data:
        print(json.dumps(data[show_gid], ensure_ascii=False, indent=2))
    else:
        print("(not in file)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data.py <data.json> [gameid]")
        sys.exit(1)
    gid = sys.argv[2] if len(sys.argv) > 2 else "26824264"
    check(sys.argv[1], gid)
