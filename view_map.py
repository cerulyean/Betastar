"""
Visualize one map's extracted data: terrain planes (pysc2) overlaid with
expansion centers, spawns, minerals, geysers, and derived bases (python-sc2).

The terrain planes are a res x res raster; the expansion/resource points are in
world coordinates. They are reconciled here via `world_to_grid`, so you can SEE
whether the two sources line up — the thing you must confirm before trusting any
joined spatial feature.

Usage:
    python view_map.py "Winter Madness LE"
    python view_map.py Winter_Madness_LE        # filename form also fine
    python view_map.py "Winter Madness LE" --save-only   # no interactive window
"""

import gzip
import json
import os
import re
import sys

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

MAP_DATA_DIR = "map_data"
EXPANSION_DATA_DIR = "expansion_data"


def safe_name(map_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", map_name).strip("_")


def load_terrain(key: str) -> dict:
    path = os.path.join(MAP_DATA_DIR, key + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_expansions(key: str):
    path = os.path.join(EXPANSION_DATA_DIR, key + ".json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def world_to_grid(x, y, map_size, res):
    """
    Map a world coordinate into raster cell indices (col, row).

    pysc2 renders the minimap into a SQUARE raster whose extent is the LARGER of
    the two map dimensions, applied to BOTH axes (it does not stretch a
    non-square map to fill the square). So both axes share one divisor:

        D   = max(map_size.x, map_size.y)
        col = x / D * res
        row = (1 - y / D) * res        (y flipped for imshow top-left origin)

    Dividing each axis by its OWN dimension (the old behavior) stretches the
    shorter axis and drifts peripheral points outward onto the surrounding
    cliffs. All ladder maps here are non-square (e.g. 168x176), so the square
    divisor matters.
    """
    D = max(map_size["x"], map_size["y"])
    col = x / D * res
    row = (1.0 - y / D) * res
    return col, row


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1:
        print('Usage: python view_map.py "<map name or filename>" [--save-only]')
        sys.exit(1)
    if "--save-only" in flags:
        matplotlib.use("Agg")

    key = safe_name(args[0])
    data = load_terrain(key)
    res = data["resolution"]
    pa = data["playable_area"]
    ms = data.get("map_size")

    print("terrain keys:", list(data.keys()))
    print(f"map: {data['map_name']}   resolution: {res}")
    print(f"playable_area: {pa}   map_size: {ms}")

    height = np.array(data["height_map"])
    pathable = np.array(data["pathable"])
    buildable = np.array(data["buildable"])

    exp = load_expansions(key)
    if exp is None:
        print("(no expansion_data file for this map — terrain only)")
    else:
        print(
            f"expansions: {len(exp.get('expansions', []))}  "
            f"minerals: {len(exp.get('minerals', []))}  "
            f"geysers: {len(exp.get('geysers', []))}  "
            f"derived_bases: {len(exp.get('derived_bases', []))}"
        )

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, grid, title in zip(
        ax,
        [height, pathable, buildable],
        ["height_map", "pathable", "buildable"],
    ):
        a.imshow(grid, origin="upper")
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])

    # Overlay resource/spawn/base points on the buildable plot (right). Buildable
    # is the most useful backdrop: base townhalls sit on buildable ground, so a
    # correct transform puts derived_bases squarely on green.
    if exp is not None and ms is not None:
        overlay = ax[2]

        def scatter(points, **kw):
            xs, ys = [], []
            for p in points:
                gx, gy = world_to_grid(p["x"], p["y"], ms, res)
                xs.append(gx)
                ys.append(gy)
            overlay.scatter(xs, ys, **kw)

        scatter(
            exp.get("minerals", []),
            c="cyan",
            s=8,
            marker="s",
            label="minerals",
            alpha=0.7,
        )
        scatter(
            exp.get("geysers", []),
            c="lime",
            s=20,
            marker="^",
            label="geysers",
            alpha=0.9,
        )
        scatter(
            exp.get("expansions", []),
            c="white",
            s=70,
            marker="o",
            edgecolors="black",
            label="expansions (raw)",
            zorder=4,
        )
        # Dark fill + white edge so bases read against the yellow buildable plane.
        scatter(
            exp.get("derived_bases", []),
            c="black",
            s=110,
            marker="P",
            edgecolors="white",
            linewidths=1.2,
            label="derived bases",
            zorder=5,
        )

        for spawn, color, lbl in [
            (exp.get("own_spawn"), "red", "own spawn"),
            (exp.get("enemy_spawn"), "magenta", "enemy spawn"),
        ]:
            if spawn:
                gx, gy = world_to_grid(spawn["x"], spawn["y"], ms, res)
                overlay.scatter(
                    [gx],
                    [gy],
                    c=color,
                    s=220,
                    marker="*",
                    edgecolors="black",
                    label=lbl,
                    zorder=6,
                )

        overlay.set_xlim(0, res)
        overlay.set_ylim(res, 0)
        overlay.legend(loc="upper right", fontsize=6, framealpha=0.8)
        overlay.set_title("buildable + resources/bases/spawns")

    fig.suptitle(f"{data['map_name']}  (res={res}, playable={pa['w']}x{pa['h']})")
    plt.tight_layout()
    out_png = f"view_{key}.png"
    plt.savefig(out_png, dpi=110)
    print(f"saved figure -> {out_png}")
    if "--save-only" not in flags:
        plt.show()


if __name__ == "__main__":
    main()
