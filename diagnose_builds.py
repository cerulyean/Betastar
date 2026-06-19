"""
Diagnostic: what builds does the corpus span, and which do you have installed?

Reads replay_index.json, extracts the build number (the last numeric chunk of
each version string, e.g. "0.0.0.95740" -> 95740), counts how many replays use
each build, and cross-references against the Base#### folders actually present
in your StarCraft II install.

This tells you, definitively:
  - how many distinct builds are in the corpus
  - how many replays each build covers
  - which builds you can already play (installed) vs not (missing)
  - per map, whether ANY representative replay exists on an installed build
    (so you can pick a playable representative instead of installing builds)

Usage:
    python diagnose_builds.py
    (edit SC2_VERSIONS_DIR below if your install path differs)
"""

import json
import os
import re
from collections import Counter, defaultdict

REPLAY_INDEX_PATH = "replay_index.json"
SC2_VERSIONS_DIR = r"C:\Program Files (x86)\games\StarCraft II\Versions"


def build_from_version(version: str) -> str:
    """Last numeric chunk of a version string is the build number."""
    nums = re.findall(r"\d+", str(version))
    return nums[-1] if nums else "unknown"


def installed_builds() -> set:
    if not os.path.isdir(SC2_VERSIONS_DIR):
        print(f"WARNING: {SC2_VERSIONS_DIR} not found — can't check installs.")
        return set()
    builds = set()
    for name in os.listdir(SC2_VERSIONS_DIR):
        m = re.fullmatch(r"Base(\d+)", name)
        if m:
            builds.add(m.group(1))
    return builds


def main():
    with open(REPLAY_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    have = installed_builds()
    print(f"Installed builds: {sorted(have) if have else '(none found)'}\n")

    # Count replays per build.
    build_counts = Counter()
    # Per map: which builds appear, and which replays are on installed builds.
    map_builds = defaultdict(Counter)
    map_playable_rep = {}  # map_name -> a replay filename on an installed build

    for fname, meta in index.items():
        build = build_from_version(meta.get("version", ""))
        map_name = meta.get("map_name", "?")
        build_counts[build] += 1
        map_builds[map_name][build] += 1
        if build in have and map_name not in map_playable_rep:
            map_playable_rep[map_name] = fname

    print("=== Builds across whole corpus ===")
    for build, n in build_counts.most_common():
        tag = "INSTALLED" if build in have else "MISSING"
        print(f"  build {build}: {n} replays   [{tag}]")

    print(f"\nDistinct builds in corpus: {len(build_counts)}")
    missing = [b for b in build_counts if b not in have]
    print(f"Missing builds: {sorted(missing) if missing else '(none — all installed!)'}")

    print("\n=== Per-map playability ===")
    print("(can we extract this map using a replay on an installed build?)")
    n_ok = 0
    for map_name in sorted(map_builds):
        rep = map_playable_rep.get(map_name)
        if rep:
            n_ok += 1
            print(f"  OK   {map_name!r}  -> use {rep}")
        else:
            builds_here = ", ".join(sorted(map_builds[map_name]))
            print(f"  NONE {map_name!r}  (only builds: {builds_here})")

    print(f"\n{n_ok}/{len(map_builds)} maps have a playable representative "
          f"on an installed build.")
    if n_ok < len(map_builds):
        print("For the rest you'd need to install the missing build(s) above,")
        print("OR confirm none of your replays for those maps are on 95841.")

    # Emit a corrected unique_maps.json using only installed-build reps, and
    # record each rep's build so the extractor can launch on the right one.
    corrected = {}
    for map_name, rep in map_playable_rep.items():
        build = build_from_version(index[rep]["version"])
        corrected[map_name] = {"replay": rep, "build": build}
    with open("unique_maps_playable.json", "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2)
    print("\nWrote unique_maps_playable.json (installed-build reps + build #).")


if __name__ == "__main__":
    main()
