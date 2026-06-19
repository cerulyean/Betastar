"""
Derive expansion (base) locations by clustering mineral/geyser patches.

Replaces the empty `expansion_locations` data: groups nearby resource patches
into clusters (one per base), then estimates each base's townhall point.

Base-point estimation:
  - A townhall sits on the CONCAVE side of the mineral arc, ~equidistant to all
    patches in the cluster. The naive centroid lands inside/on the arc, offset
    from the true base.
  - We fit a CIRCLE to the patch positions (least squares); the circle CENTER is
    the point equidistant to points on the arc = the townhall location.
  - Geysers are included in the cluster for grouping but the circle is fit to
    minerals (they form the arc). Falls back to centroid if the fit is unstable.

Validation: the two known spawns (own_spawn / enemy_spawn) are real townhall
locations. The clusters nearest each spawn should produce base points that land
ON those spawns. If they do, trust the method for the other expansions.

Usage:
    python cluster_expansions.py "Winter Madness LE"
    python cluster_expansions.py --all      # process every expansion_data file

Writes the estimated bases back into expansion_data/<map>.json under
"derived_bases" (and ranks them per spawn).
"""

import json
import math
import os
import re
import sys

import numpy as np

EXPANSION_DATA_DIR = "expansion_data"
CLUSTER_THRESHOLD = 12.0   # world units; patches within this are same base
MIN_PATCHES = 4            # a real base has ~8 minerals; ignore tiny stray groups


def safe_name(map_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", map_name).strip("_")


# --- clustering (union-find by distance threshold) ---------------------------

def cluster_points(points, threshold):
    """Group points so any two within `threshold` are in the same cluster."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy <= threshold * threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --- base point estimation (least-squares circle fit) ------------------------

def fit_circle(xs, ys):
    """
    Least-squares circle fit. Returns (cx, cy, r).
    Solves: x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    as a linear system A @ [cx, cy, c] = b.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx**2 + cy**2, 0.0))
    return cx, cy, r


def estimate_base(mineral_pts, geyser_pts):
    """
    Estimate the townhall location for one cluster.
    Circle-fit the minerals (they form the arc); fall back to centroid of
    minerals+geysers if the fit is degenerate (collinear / too few points).
    """
    all_pts = mineral_pts + geyser_pts
    cx_centroid = sum(p[0] for p in all_pts) / len(all_pts)
    cy_centroid = sum(p[1] for p in all_pts) / len(all_pts)

    if len(mineral_pts) >= 3:
        try:
            xs = [p[0] for p in mineral_pts]
            ys = [p[1] for p in mineral_pts]
            cx, cy, r = fit_circle(xs, ys)
            # sanity: circle center shouldn't be absurdly far from the patches
            if r < 50 and math.hypot(cx - cx_centroid, cy - cy_centroid) < 25:
                return (cx, cy), "circle_fit"
        except Exception:
            pass
    return (cx_centroid, cy_centroid), "centroid"


# --- main per-map logic ------------------------------------------------------

def process_map(key):
    path = os.path.join(EXPANSION_DATA_DIR, key + ".json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    minerals = [(m["x"], m["y"]) for m in data.get("minerals", [])]
    geysers = [(g["x"], g["y"]) for g in data.get("geysers", [])]

    if not minerals:
        print(f"  {key}: no minerals — skip")
        return

    # Cluster all resources together (minerals + geysers) so each base groups.
    all_res = minerals + geysers
    clusters_idx = cluster_points(all_res, CLUSTER_THRESHOLD)

    bases = []
    n_min = len(minerals)
    for idx in clusters_idx:
        m_pts = [all_res[i] for i in idx if i < n_min]
        g_pts = [all_res[i] for i in idx if i >= n_min]
        if len(m_pts) < MIN_PATCHES:
            continue  # stray group, not a real base
        (bx, by), method = estimate_base(m_pts, g_pts)
        bases.append({"x": bx, "y": by, "n_minerals": len(m_pts),
                      "n_geysers": len(g_pts), "method": method})

    # --- validate against known spawns ---
    own = data.get("own_spawn")
    enemy = data.get("enemy_spawn")

    def nearest_base(spawn):
        if not spawn or not bases:
            return None, None
        best, bestd = None, 1e9
        for b in bases:
            d = math.hypot(b["x"] - spawn["x"], b["y"] - spawn["y"])
            if d < bestd:
                best, bestd = b, d
        return best, bestd

    own_base, own_d = nearest_base(own)
    enemy_base, enemy_d = nearest_base(enemy)

    print(f"  {key}: {len(bases)} bases found")
    if own_d is not None:
        print(f"    own spawn  ({own['x']:.1f},{own['y']:.1f}) -> "
              f"nearest base error {own_d:.1f}u  [{own_base['method']}]")
    if enemy_d is not None:
        print(f"    enemy spawn ({enemy['x']:.1f},{enemy['y']:.1f}) -> "
              f"nearest base error {enemy_d:.1f}u  [{enemy_base['method']}]")

    # --- rank bases per spawn (natural = closest after main, etc.) ---
    def ranked(spawn):
        if not spawn:
            return []
        return sorted(
            range(len(bases)),
            key=lambda i: math.hypot(bases[i]["x"] - spawn["x"],
                                     bases[i]["y"] - spawn["y"]),
        )

    data["derived_bases"] = bases
    data["bases_ranked_from_own"] = ranked(own)
    data["bases_ranked_from_enemy"] = ranked(enemy)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    if len(sys.argv) != 2:
        print('Usage: python cluster_expansions.py "<map name>" | --all')
        sys.exit(1)

    if sys.argv[1] == "--all":
        keys = [f[:-5] for f in os.listdir(EXPANSION_DATA_DIR)
                if f.endswith(".json")]
    else:
        keys = [safe_name(sys.argv[1])]

    for key in keys:
        try:
            process_map(key)
        except Exception as e:
            print(f"  {key}: FAILED {e!r}")

    print("\nValidation guide: 'nearest base error' should be small (a few "
          "units). Large errors mean the base-point estimate is off — check "
          "the overlay in view_map.py.")


if __name__ == "__main__":
    main()
