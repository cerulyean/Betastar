"""
Spatial grid builder for BetaStar.

Design:
  - Build every channel at NATIVE 128x128 (same raster as the terrain planes,
    so the world_to_grid alignment we already validated holds).
  - Downsample to a coarse GxG grid by exact block-reduction. 128 must be
    divisible by G, so G in {16, 32, 64}. Each channel declares how it reduces:
        "sum"  -> densities (unit supply, base count, resource count)
        "mean" -> coverage / continuous (pathable, buildable, height)
  - Non-square maps need no padding: the raster already spans the SQUARE
    max(map_size) extent (that's the transform fix), so off-map margin is just
    zeros and downsamples to dead coarse cells.

A grid is a dict: {channel_name: np.ndarray(G, G)} plus a "_meta" entry.
Flatten later with grid_to_vector() when feeding the model; keep the 2D form
for verification and thinking.
"""

import numpy as np

NATIVE = 128

# Each channel: how it aggregates when downsampling.
REDUCTION = {
    "pathable": "mean",
    "buildable": "mean",
    "height": "mean",
    "base_count": "sum",
    "mineral_count": "sum",
    "geyser_count": "sum",
    # unit-supply channels are added dynamically per unit type, all "sum"
}


def world_to_grid(x, y, map_size, res=NATIVE):
    """Square-extent transform (see MAP_PIPELINE.md). Returns float (col, row)."""
    D = max(map_size["x"], map_size["y"])
    col = x / D * res
    row = (1.0 - y / D) * res
    return col, row


def _bin(x, y, map_size, res=NATIVE):
    """World point -> integer (row, col) cell index, or None if off-grid."""
    c, r = world_to_grid(x, y, map_size, res)
    ci, ri = int(c), int(r)          # floor: a point in cell covers [i, i+1)
    if 0 <= ri < res and 0 <= ci < res:
        return ri, ci
    return None


def build_native(terrain, expansion, enemy_units=None):
    """
    Build all channels at 128x128.

    terrain:  dict from map_data/<map>.json.gz (height_map, pathable, buildable,
              map_size, resolution)
    expansion: dict from expansion_data/<map>.json (minerals, geysers,
               derived_bases)
    enemy_units: optional list of {x, y, type, supply} for the unit-supply
               channels (from simulator last_seen_position). None -> skip.

    Returns {channel: (128,128) array}, plus dynamically-named
    "unit_<TYPE>" channels if enemy_units given.
    """
    ms = terrain["map_size"]
    res = terrain["resolution"]
    assert res == NATIVE, f"expected native {NATIVE}, got {res}"

    grid = {
        "pathable": np.asarray(terrain["pathable"], dtype=np.float32),
        "buildable": np.asarray(terrain["buildable"], dtype=np.float32),
        "height": np.asarray(terrain["height_map"], dtype=np.float32),
        "base_count": np.zeros((res, res), np.float32),
        "mineral_count": np.zeros((res, res), np.float32),
        "geyser_count": np.zeros((res, res), np.float32),
    }

    for b in expansion.get("derived_bases", []):
        idx = _bin(b["x"], b["y"], ms, res)
        if idx:
            grid["base_count"][idx] += 1
    for m in expansion.get("minerals", []):
        idx = _bin(m["x"], m["y"], ms, res)
        if idx:
            grid["mineral_count"][idx] += 1
    for g in expansion.get("geysers", []):
        idx = _bin(g["x"], g["y"], ms, res)
        if idx:
            grid["geyser_count"][idx] += 1

    if enemy_units:
        types = sorted({u["type"] for u in enemy_units})
        for t in types:
            grid[f"unit_{t}"] = np.zeros((res, res), np.float32)
            REDUCTION.setdefault(f"unit_{t}", "sum")
        for u in enemy_units:
            idx = _bin(u["x"], u["y"], ms, res)
            if idx:
                grid[f"unit_{u['type']}"][idx] += u.get("supply", 1.0)

    return grid


def downsample(grid, G):
    """
    Block-reduce every channel from 128x128 to GxG using its declared rule.
    128 must be divisible by G (G in {16, 32, 64}).
    """
    assert NATIVE % G == 0, f"{NATIVE} not divisible by {G}; use 16/32/64"
    block = NATIVE // G
    out = {}
    for name, arr in grid.items():
        if name == "_meta":
            continue
        # reshape into (G, block, G, block) and reduce the block axes
        r = arr.reshape(G, block, G, block)
        rule = REDUCTION.get(name, "sum")
        if rule == "sum":
            out[name] = r.sum(axis=(1, 3))
        else:  # "mean"
            out[name] = r.mean(axis=(1, 3))
    out["_meta"] = {"G": G, "block": block,
                    "reductions": {k: REDUCTION.get(k, "sum")
                                   for k in out if k != "_meta"}}
    return out


def grid_to_vector(grid):
    """Flatten a GxG grid dict into a single 1D vector + the channel order."""
    names = [k for k in grid if k != "_meta"]
    vec = np.concatenate([grid[n].ravel() for n in names])
    return vec, names
