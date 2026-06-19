"""
Inspect specific gameids whose winner_toon_id didn't cleanly match one player.
Shows what the entry has vs what the replay says, so you can tell whether it's
a tie/disconnect (drop it) or a resolution bug (fix it).

Usage:
  python check_winners.py <replay_dir> <data.json> <gid1> <gid2> ...
"""

import os
import sys
import json
import sc2reader


def find_replay(replay_dir, gid):
    for f in os.listdir(replay_dir):
        if f.endswith(".SC2Replay") and (f == gid + ".SC2Replay"
                                         or f.split("_", 1)[0] == gid):
            return f
    return None


def main(replay_dir, data_json, gids):
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for gid in gids:
        print(f"\n===== {gid} =====")
        e = data.get(gid)
        if e is None:
            print("  not in data.json")
            continue

        toons = {pk: p.get("toon_id") for pk, p in e.get("players", {}).items()}
        print(f"  entry players (toon_id): {toons}")
        print(f"  entry winner_toon_id   : {e.get('winner_toon_id')}")
        print(f"  entry winner_id (SRS)  : {e.get('winner_id')}")

        fname = find_replay(replay_dir, gid)
        if fname is None:
            print("  no replay file found")
            continue

        try:
            r = sc2reader.load_replay(os.path.join(replay_dir, fname), load_level=2)
        except Exception as ex:
            print(f"  load failed: {ex}")
            continue

        print("  replay players:")
        for p in r.players:
            print(f"    pid={p.pid} name={p.name!r} race={p.play_race!r} "
                  f"toon_id={p.toon_id} result={p.result!r}")
        try:
            w = r.winner
            print(f"  replay.winner: {w} players="
                  f"{[ (pp.name, pp.toon_id) for pp in w.players ] if w else None}")
        except Exception as ex:
            print(f"  replay.winner errored: {ex}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python check_winners.py <replay_dir> <data.json> <gid> [gid ...]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
