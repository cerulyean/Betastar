# Usage: python batch_processor.py --max-procs 4

import argparse, json, os, subprocess, time
from datetime import datetime

STEP_SIZE = 20
CHECK_PROCESSED_COUNT = True
data_location = r"D:\betastar\parser\data.json"


def process_folder(
    input_folder="D:/betastar/replays1",
    output_folder="output",
    max_parallel=4,
    max_games=None,
):
    games_started = 0
    with open(data_location) as f:
        detailed_info = json.load(f)

    os.makedirs(output_folder, exist_ok=True)
    running = []
    t0 = time.time()
    not_processed = 0

    for count, filename in enumerate(sorted(os.listdir(input_folder)), 1):

        if max_games is not None and games_started >= max_games:
            print(f"\nReached max_games={max_games}. Stopping.\n")
            break
        game_id = filename.removesuffix(".SC2Replay")
        print(game_id)
        info = detailed_info.get(game_id)

        file_path = os.path.join(input_folder, filename)
        output_base = os.path.join(output_folder, game_id)

        if os.path.exists(output_base + "_1.json.gz"):
            continue
        print(f"NOT PROCESSED: {filename}")
        if CHECK_PROCESSED_COUNT:
            not_processed += 1
            continue

        running.append(
            subprocess.Popen(
                ["python", "simulator.py", file_path, output_base, "1", str(STEP_SIZE)]
            )
        )
        running.append(
            subprocess.Popen(
                ["python", "simulator.py", file_path, output_base, "2", str(STEP_SIZE)]
            )
        )
        games_started += 1
        while len(running) >= max_parallel:
            running = [p for p in running if p.poll() is None]
            if len(running) >= max_parallel:
                time.sleep(0.1)

        elapsed = int(time.time() - t0)
        h, m = divmod(elapsed // 60, 60)
        print(
            f"[{count}] {filename}  |  {h}h {m}m {elapsed%60}s "
            f"({datetime.now():%H:%M:%S})"
        )

    for p in running:
        p.wait()
    print(str(not_processed) + " unprocessed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-procs",
        type=int,
        default=1,
        help="maximum concurrent simulator processes",
    )
    ap.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="maximum number of games to process (default: all)",
    )
    args = ap.parse_args()
    process_folder(
        max_parallel=args.max_procs,
        max_games=args.max_games,
    )
