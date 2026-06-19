"""
Rebuild data.json-style metadata entries from replays whose filenames encode
gameid_lengthseconds_avgmmr_mapname  (e.g. "27122774_775_4910.5_Sigil").

Target schema (one entry, keyed by gameid):
{
  "27038559": {
    "mmr": 3914.0,
    "length": "06:27",
    "players": {
      "player_1": {"id": "2387602", "name": "Koumei", "race": "zerg"},
      "player_2": {"id": "117611",  "name": "Branch", "race": "protoss"}
    },
    "winner_id": "2387602",
    "version": "5.0.15.95841"
  }
}
"""

import os
import json
import sc2reader


# sc2reader's play_race is localized and its LOCALIZED_RACES table is incomplete
# (e.g. it has singular Russian 'Зерг' but not the plural 'зерги' that some
# replays carry), so play_race can come back as a raw foreign string. Normalize
# to canonical English 'zerg'/'protoss'/'terran' ourselves.
_RACE_CANON = {
    # english
    "zerg": "zerg",
    "protoss": "protoss",
    "terran": "terran",
    # russian singular + plural, lowercased
    "зерг": "zerg",
    "зерги": "zerg",
    "протосс": "protoss",
    "протоссы": "protoss",
    "терран": "terran",
    "терраны": "terran",
    # korean
    "저그": "zerg",
    "프로토스": "protoss",
    "테란": "terran",
    # chinese (simplified + traditional)
    "异虫": "zerg",
    "蟲族": "zerg",
    "星灵": "protoss",
    "神族": "protoss",
    "人类": "terran",
    "人類": "terran",
}


def canonical_race(race_str: str) -> str:
    """
    Map a (possibly localized) race string to 'zerg' / 'protoss' / 'terran'.
    Falls back to a first-letter heuristic for unseen locales, then to the
    lowercased input so nothing silently becomes empty.
    """
    if race_str is None:
        return ""
    s = race_str.strip().lower()
    if s in _RACE_CANON:
        return _RACE_CANON[s]
    # last-resort heuristic: latin/cyrillic initial letters are distinct
    if s[:1] in ("z", "з"):
        return "zerg"
    if s[:1] in ("p", "п"):
        return "protoss"
    if s[:1] in ("t", "т"):
        return "terran"
    return s


def parse_filename(fname: str) -> dict:
    """
    "27122774_775_4910.5_Sigil[.SC2Replay]" ->
        {"game_id": "27122774", "length_s": 775, "avg_mmr": 4910.5, "map": "Sigil"}

    Map name is the remainder after the third underscore, so map names that
    themselves contain underscores survive (rsplit-free; split with maxsplit).
    """
    stem = fname
    for ext in (".SC2Replay", ".json.gz", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    parts = stem.split("_", 3)  # at most 4 fields; 4th keeps any underscores

    # Two filename formats coexist in the corpus:
    #   new:  "27122774_775_4910.5_Sigil"  -> id_lengthsec_avgmmr_map
    #   old:  "26755754"                   -> bare gameid, no embedded metadata
    if len(parts) >= 3:
        game_id, length_s, avg_mmr = parts[0], parts[1], parts[2]
        map_name = parts[3] if len(parts) > 3 else ""
        return {
            "game_id": game_id,
            "length_s": int(length_s),
            "avg_mmr": float(avg_mmr),
            "map": map_name,
        }

    # bare-id format: length comes from the replay, mmr is unavailable
    return {
        "game_id": parts[0],
        "length_s": None,
        "avg_mmr": None,
        "map": "",
    }


def seconds_to_mmss(total_seconds: int) -> str:
    """775 -> '12:55'. Matches data.json's 'MM:SS' (zero-padded)."""
    m, s = divmod(int(total_seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_entry(replay_path: str, fname: str | None = None) -> dict:
    """
    Returns {game_id: {...}} for one replay. load_level=2 is enough:
    it parses players (names, races, toon_ids, results) and the header
    (version) without touching tracker/game events. Fast.
    """
    if fname is None:
        fname = os.path.basename(replay_path)
    meta = parse_filename(fname)

    replay = sc2reader.load_replay(replay_path, load_level=2)

    # --- players: stable ordering by pid so player_1/player_2 are deterministic
    players = {}
    winner_id = None  # SC2ReplayStats id (unavailable from replay alone)
    winner_toon_id = None  # Blizzard toon_id (read from replay)
    for slot, p in enumerate(sorted(replay.players, key=lambda x: x.pid), start=1):
        toon = str(p.toon_id)
        players[f"player_{slot}"] = {
            # NOTE: SC2ReplayStats "id" cannot be recovered from the replay.
            # Left None here; merge against the old data.json to keep it (see
            # merge_toon_ids below) rather than overwriting.
            "id": None,
            "name": p.name,
            # canonical race ('zerg'/'protoss'), locale-normalized. play_race
            # can be a localized string; canonical_race handles it.
            "race": canonical_race(p.play_race),
            "toon_id": toon,  # bare Blizzard uid
        }
        if p.result == "Win":
            winner_toon_id = toon

    # winner fallback: if result flags are missing/ambiguous, use replay.winner
    if winner_toon_id is None and replay.winner and replay.winner.players:
        winner_toon_id = str(replay.winner.players[0].toon_id)

    # sanity: this corpus is strictly 1 Zerg + 1 Protoss. If a replay isn't
    # clean ZvP, flag it rather than emit an entry with dubious race labels.
    races = sorted(p["race"] for p in players.values())
    if races != ["protoss", "zerg"]:
        raise ValueError(f"not clean ZvP (races={races})")

    # length: prefer filename seconds; fall back to the replay's real_length
    if meta["length_s"] is not None:
        length_str = seconds_to_mmss(meta["length_s"])
    else:
        length_str = seconds_to_mmss(int(replay.real_length.total_seconds()))

    entry = {
        # mmr is None for bare-id files (sc2reader can't recover ladder MMR);
        # filename avg used when present
        "mmr": meta["avg_mmr"],
        "length": length_str,
        "players": players,
        "winner_id": winner_id,  # SC2ReplayStats id (None unless merged)
        "winner_toon_id": winner_toon_id,  # Blizzard toon_id (from replay)
        "version": replay.release_string,  # e.g. "5.0.15.95841"
    }
    return {meta["game_id"]: entry}


def build_corpus(replay_dir: str, out_path: str = "data_rebuilt.json") -> dict:
    """Walk a folder of .SC2Replay files, build the full data.json dict."""
    corpus = {}
    files = [f for f in os.listdir(replay_dir) if f.endswith(".SC2Replay")]
    for i, fname in enumerate(files, 1):
        path = os.path.join(replay_dir, fname)
        try:
            corpus.update(build_entry(path, fname))
        except Exception as e:
            gid = parse_filename(fname)["game_id"]
            print(f"[{i}/{len(files)}] FAILED {gid}: {e}")
            corpus[gid] = {"skipped": True, "reason": str(e)}
            continue
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] ...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"Wrote {len(corpus)} entries to {out_path}")
    return corpus


def ingest_new_batch(
    source_dir: str,
    dest_dir: str,
    data_json_path: str,
    out_path: str,
):
    """
    For the new '<id>_<len>_<mmr>_<map>' batch when you also want the copied
    files renamed to bare '<id>.SC2Replay' in dest_dir.

    CRITICAL ORDER: the entry (which needs mmr/length/map) is built from the
    ORIGINAL full filename FIRST, then the file is copied to dest_dir under the
    bare-id name. After this, nothing reads the filename again, so losing the
    suffix on the copy is harmless. Doing it the other way round would drop the
    mmr (filename no longer carries it) and break extract_mmr.

    source_dir : folder of full-named '<id>_<len>_<mmr>_<map>.SC2Replay'
    dest_dir   : where bare '<id>.SC2Replay' copies go (e.g. replays1)
    data_json  : existing standardized data.json to extend
    out_path   : merged data.json out
    """
    import shutil

    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(dest_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if f.endswith(".SC2Replay")]
    added = skipped_existing = failed = 0

    for i, fname in enumerate(files, 1):
        gid = parse_filename(fname)["game_id"]
        src = os.path.join(source_dir, fname)
        dest = os.path.join(dest_dir, gid + ".SC2Replay")

        if gid in data:
            skipped_existing += 1
        else:
            try:
                # 1) build entry from the FULL filename (captures mmr/len/map)
                data.update(build_entry(src, fname))
                added += 1
            except Exception as e:
                print(f"{gid}: FAILED: {e}")
                data[gid] = {"skipped": True, "reason": str(e)}
                failed += 1
                continue

        # 2) only now copy + rename to bare id (idempotent: skip if present)
        if not os.path.exists(dest):
            shutil.copy2(src, dest)

        if i % 200 == 0:
            print(f"[{i}/{len(files)}] ...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"\ningested {added} new entries, skipped {skipped_existing} present, "
        f"{failed} failed"
    )
    print(f"copied bare-id replays into {dest_dir}")
    print(f"total entries now: {len(data)} -> {out_path}")


def append_new_batch(replay_dir: str, data_json_path: str, out_path: str):
    """
    For the NEW '<id>_<lengthsec>_<avgmmr>_<map>' batch: build a fresh entry per
    replay (fully reconstructable from filename + replay) and append into an
    existing standardized data.json.

    New entries carry toon_id / winner_toon_id (the keys the pipeline now reads)
    and leave 'id' / 'winner_id' as None (vestigial, kept only for schema parity
    with the merged old entries). Gameids already present are skipped, so this is
    safe to re-run.
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = [f for f in os.listdir(replay_dir) if f.endswith(".SC2Replay")]
    added = 0
    skipped_existing = 0
    failed = 0

    for i, fname in enumerate(files, 1):
        gid = parse_filename(fname)["game_id"]
        if gid in data:
            skipped_existing += 1
            continue
        try:
            data.update(build_entry(os.path.join(replay_dir, fname), fname))
            added += 1
        except Exception as e:
            print(f"{gid}: FAILED: {e}")
            data[gid] = {"skipped": True, "reason": str(e)}
            failed += 1
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] ...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"\nadded {added} new entries, skipped {skipped_existing} already present, "
        f"{failed} failed"
    )
    print(f"total entries now: {len(data)}")
    print(f"written to {out_path}")


def merge_toon_ids(replay_dir: str, data_json_path: str, out_path: str):
    """
    Augment an existing data.json: attach Blizzard toon_ids to each player and a
    winner_toon_id per game, WITHOUT touching the existing SC2ReplayStats
    'id' / 'winner_id' / 'mmr' / 'length'.

    Matches players by RACE, not name. Names drift (SC2ReplayStats shows a
    player's current handle, the replay binary shows their handle at game time),
    so name-matching is unreliable. This corpus is strictly 1 Zerg + 1 Protoss
    per game, so race is a unique, stable join key present in both sources.

    Adds per player:  "toon_id"
    Adds per entry:   "winner_toon_id"

    Flags (does not crash) any game that isn't cleanly 1 Zerg + 1 Protoss on
    either side.
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = {
        parse_filename(f)["game_id"]: f
        for f in os.listdir(replay_dir)
        if f.endswith(".SC2Replay")
    }

    updated = 0
    no_replay = 0
    race_bad = 0
    name_drift = 0  # informational: how many had a name disagreement

    for gid, entry in data.items():
        if entry.get("skipped"):
            continue
        fname = files.get(gid)
        if fname is None:
            no_replay += 1
            continue

        try:
            replay = sc2reader.load_replay(
                os.path.join(replay_dir, fname), load_level=2
            )
        except Exception as e:
            print(f"{gid}: load failed: {e}")
            continue

        # race -> (toon_id, name) from the replay, normalized across locales
        replay_by_race = {}
        for p in replay.players:
            replay_by_race.setdefault(canonical_race(p.play_race), []).append(
                (str(p.toon_id), p.name)
            )

        # sanity: exactly one of each expected race in the replay
        if (
            len(replay_by_race.get("zerg", [])) != 1
            or len(replay_by_race.get("protoss", [])) != 1
        ):
            print(
                f"{gid}: replay not clean ZvP "
                f"(races={[ (r, len(v)) for r, v in replay_by_race.items() ]}) "
                f"-> skipped"
            )
            race_bad += 1
            continue

        # winning race from the replay (result flag, with winner fallback)
        win_race = next(
            (canonical_race(p.play_race) for p in replay.players if p.result == "Win"),
            None,
        )
        if win_race is None and replay.winner and replay.winner.players:
            win_race = canonical_race(replay.winner.players[0].play_race)

        # attach toon_id to each existing player by RACE
        ok = True
        for pdata in entry.get("players", {}).values():
            r = canonical_race(pdata["race"])
            if r not in replay_by_race or len(replay_by_race[r]) != 1:
                print(f"{gid}: data.json race '{r}' not uniquely in replay -> skip")
                ok = False
                break
            tid, rname = replay_by_race[r][0]
            pdata["toon_id"] = tid
            if rname != pdata["name"]:
                name_drift += 1  # expected sometimes; not an error

        if not ok:
            race_bad += 1
            continue

        # winner_toon_id = toon_id of the winning race
        entry["winner_toon_id"] = (
            replay_by_race[win_race][0][0] if win_race in replay_by_race else None
        )

        updated += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\nmerged toon_ids into {updated} entries (matched by race)")
    print(f"  {no_replay} entries had no matching replay (left untouched)")
    print(f"  {race_bad} entries skipped (not clean ZvP / race mismatch)")
    print(
        f"  {name_drift} player(s) had a name that differed between sources "
        f"(expected; renames)"
    )
    print(f"written to {out_path}")


def verify_against_existing(replay_dir: str, data_json_path: str, limit: int = 20):
    """
    For replays whose gameid already exists in data.json, rebuild the entry
    from the replay and diff it field-by-field against the stored one.

    This is the trust check: it confirms the filename<->replay pairing is
    correct and that toon_ids / winner / version / race extracted now match
    what your original pipeline recorded.
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    files = [f for f in os.listdir(replay_dir) if f.endswith(".SC2Replay")]
    checked = 0
    mismatches = 0

    for fname in files:
        gid = parse_filename(fname)["game_id"]
        old = existing.get(gid)
        if old is None or old.get("skipped"):
            continue  # not in old data (or was skipped) -> nothing to compare

        try:
            new = build_entry(os.path.join(replay_dir, fname), fname)[gid]
        except Exception as e:
            print(f"{gid}: REBUILD FAILED: {e}")
            mismatches += 1
            checked += 1
            continue

        diffs = []

        # winner_id
        if str(old.get("winner_id")) != str(new.get("winner_id")):
            diffs.append(
                f"winner_id: old={old.get('winner_id')} new={new.get('winner_id')}"
            )

        # version
        if old.get("version") != new.get("version"):
            diffs.append(f"version: old={old.get('version')} new={new.get('version')}")

        # players: compare as a set of (id, name, race) so player_1/player_2
        # ordering differences don't count as a mismatch
        def pset(entry):
            return {
                (str(p["id"]), p["name"], p["race"])
                for p in entry.get("players", {}).values()
            }

        old_p, new_p = pset(old), pset(new)
        if old_p != new_p:
            diffs.append(
                f"players differ:\n    old={sorted(old_p)}\n    new={sorted(new_p)}"
            )
            # narrow it down: are the ids at least the same?
            old_ids = {x[0] for x in old_p}
            new_ids = {x[0] for x in new_p}
            if old_ids != new_ids:
                diffs.append(
                    f"    -> TOON IDS MISMATCH old={sorted(old_ids)} new={sorted(new_ids)}"
                )

        checked += 1
        if diffs:
            mismatches += 1
            print(f"\n{gid}  MISMATCH")
            for d in diffs:
                print("  " + d)
        else:
            print(f"{gid}  ok")

        if checked >= limit:
            break

    print(f"\n--- checked {checked}, {mismatches} with mismatches ---")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # single replay, print the entry
        path = sys.argv[1]
        print(json.dumps(build_entry(path), ensure_ascii=False, indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "--dir":
        build_corpus(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "--verify":
        # python build_metadata.py --verify <replay_dir> <data.json>
        verify_against_existing(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "--merge":
        # python build_metadata.py --merge <replay_dir> <data.json> <out.json>
        merge_toon_ids(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 5 and sys.argv[1] == "--append":
        # python build_metadata.py --append <new_replay_dir> <data.json> <out.json>
        append_new_batch(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6 and sys.argv[1] == "--ingest":
        # python build_metadata.py --ingest <source_dir> <dest_dir> <data.json> <out.json>
        ingest_new_batch(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Usage:")
        print("  python build_metadata.py <one_replay.SC2Replay>")
        print("  python build_metadata.py --dir <folder_of_replays>")
        print("  python build_metadata.py --verify <replay_dir> <data.json>")
        print("  python build_metadata.py --merge  <replay_dir> <data.json> <out.json>")
        print(
            "  python build_metadata.py --append <new_replay_dir> <data.json> <out.json>"
        )
        print(
            "  python build_metadata.py --ingest <source_dir> <dest_dir> <data.json> <out.json>"
        )
        print("  python build_metadata.py --merge <replay_dir> <data.json> <out.json>")
