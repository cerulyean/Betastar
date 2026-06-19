"""
Scan processed replay outputs for the true min/max of unit world coordinates.

Purpose: confirm what world-coordinate extent unit positions actually occupy, so
we can size the square grid basis and decide the playable-area crop. Units come
from `last_seen_position` (= python-sc2 unit.position, raw world coords).

It auto-detects the structure of each output file. Adjust LOAD if your on-disk
format differs (it tries gzip+json, then plain json).

Usage:
    python scan_unit_bounds.py <folder-of-extracts>
    python scan_unit_bounds.py <folder> --glob "*.json.gz"
"""

import gzip
import json
import os
import sys
import glob as globmod


def load_any(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def iter_positions(blob):
    """
    Yield (x, y) from every unit position found in a processed file.

    Handles the simulator's final_data shape: {block_index: StepData}, where
    each StepData has enemy_units_seen_and_alive: {tag: {last_seen_position,...}}
    and player_units: {tag: {last_seen_position,...}}.
    """
    if not isinstance(blob, dict):
        return
    for _, step in blob.items():
        if not isinstance(step, dict):
            continue
        for coll_key in ("enemy_units_seen_and_alive", "player_units"):
            coll = step.get(coll_key)
            if not isinstance(coll, dict):
                continue
            for _, unit in coll.items():
                if not isinstance(unit, dict):
                    continue
                pos = unit.get("last_seen_position")
                if pos is None:
                    continue
                # position may be [x, y] list, {x,y} dict, or "(x, y)" string
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    yield float(pos[0]), float(pos[1])
                elif isinstance(pos, dict) and "x" in pos and "y" in pos:
                    yield float(pos["x"]), float(pos["y"])
                elif isinstance(pos, str):
                    s = pos.strip().strip("()[]")
                    parts = [p for p in s.replace(",", " ").split() if p]
                    if len(parts) >= 2:
                        try:
                            yield float(parts[0]), float(parts[1])
                        except ValueError:
                            pass


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pattern = "*.json*"
    for a in sys.argv[1:]:
        if a.startswith("--glob"):
            pattern = a.split("=", 1)[1] if "=" in a else "*.json*"
    if not args:
        print("Usage: python scan_unit_bounds.py <folder> [--glob=PATTERN]")
        sys.exit(1)

    folder = args[0]
    files = sorted(globmod.glob(os.path.join(folder, "**", pattern), recursive=True))
    if not files:
        print(f"No files matching {pattern} under {folder}")
        sys.exit(1)

    g_minx = g_miny = float("inf")
    g_maxx = g_maxy = float("-inf")
    g_count = 0
    files_with_data = 0
    skipped = 0

    print(f"Scanning {len(files)} files...\n")
    for path in files:
        try:
            blob = load_any(path)
        except Exception as e:
            skipped += 1
            continue
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        n = 0
        for x, y in iter_positions(blob):
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)
            n += 1
        if n == 0:
            continue
        files_with_data += 1
        g_minx = min(g_minx, minx); g_maxx = max(g_maxx, maxx)
        g_miny = min(g_miny, miny); g_maxy = max(g_maxy, maxy)
        g_count += n
        # per-file line (comment out if too noisy)
        print(f"  {os.path.basename(path):45} n={n:6}  "
              f"x[{minx:6.1f},{maxx:6.1f}] y[{miny:6.1f},{maxy:6.1f}]")

    print("\n" + "=" * 60)
    print(f"files with unit data: {files_with_data} / {len(files)}  "
          f"(skipped/unreadable: {skipped})")
    print(f"total unit positions: {g_count}")
    if g_count:
        print(f"GLOBAL  x: [{g_minx:.2f}, {g_maxx:.2f}]   "
              f"y: [{g_miny:.2f}, {g_maxy:.2f}]")
        print(f"span    x: {g_maxx - g_minx:.2f}   y: {g_maxy - g_miny:.2f}")
        print(f"\nCompare against map_size (largest expected ~200) and "
              f"playable_area.w/h (~120-160).")
        print("If max approaches ~200+ you're seeing the full map box; if it "
              "hugs playable bounds, units stay inside the play area.")


if __name__ == "__main__":
    main()
