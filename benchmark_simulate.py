"""
benchmark_simulate.py  —  Run once (or when adding new replays)

Simulates every .SC2Replay in BENCHMARK_DIR that hasn't been simulated yet.
Raw output is cached permanently in RAW_CACHE_DIR — never re-runs existing ones.

Edit the config block and run.
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR = r"D:\betastar\benchmark_replays"
RAW_CACHE_DIR = r"D:\betastar\benchmark_cache\raw"
STEP_SIZE     = 20

# ---------------------------------------------------------------------------

def simulate_all():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)

    replays = sorted(Path(BENCHMARK_DIR).glob("*.SC2Replay"))
    if not replays:
        print(f"No replays found in {BENCHMARK_DIR}")
        return

    print(f"Found {len(replays)} replay(s) in {BENCHMARK_DIR}\n")

    for replay_path in replays:
        stem     = replay_path.stem
        sim_base = os.path.join(RAW_CACHE_DIR, f"{stem}.SC2Replay")

        already_done = any(
            os.path.isfile(f"{sim_base}_{pov}.json.gz") for pov in (1, 2)
        )
        if already_done:
            print(f"  [skip]     {replay_path.name}  (already simulated)")
            continue

        print(f"  [simulate] {replay_path.name}")
        for pov in (1, 2):
            result = subprocess.run([
                sys.executable, "simulator.py",
                str(replay_path), sim_base, str(pov), str(STEP_SIZE),
            ])
            if result.returncode != 0:
                print(f"    WARNING: simulator failed for POV {pov}, skipping.")
                break

    print("\nDone. Run benchmark_prune.py to prepare files for benchmark_viz.")


if __name__ == "__main__":
    simulate_all()
