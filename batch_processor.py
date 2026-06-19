# Usage: python batch_processor.py --max-procs 4

import argparse, json, os, subprocess, time
from datetime import datetime

STEP_SIZE = 20
CHECK_PROCESSED_COUNT = True
MIN_MMR = 3000
data_location = r"D:\betastar\parser\data.json"

MAX_SECONDS = 15 * 60  # kill processes running longer than 25 minutes


def mark_skipped(game_id, reason="timeout"):
    """Persist a skip flag to data.json so this game is ignored on future runs."""
    try:
        with open(data_location, encoding="utf-8") as f:
            info = json.load(f)
        if game_id in info:
            info[game_id]["skipped"] = True
            info[game_id]["skip_reason"] = reason
            tmp = data_location + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
            os.replace(tmp, data_location)
            print(f"[SKIP-SAVED] {game_id} marked skipped ({reason})")
    except Exception as e:
        print(f"[SKIP-SAVE-FAILED] {game_id}: {e}")


def process_folder(
    input_folder="D:/betastar/replays1",
    output_folder="output",
    max_parallel=6,
    max_games=None,
):
    games_started = 0
    with open(data_location, encoding="utf-8") as f:
        detailed_info = json.load(f)

    os.makedirs(output_folder, exist_ok=True)
    running = []  # list of (process, start_time, filename, game_id)
    t0 = time.time()
    not_processed = 0
    timed_out_ids = set()

    try:
        for count, filename in enumerate(
            sorted(os.listdir(input_folder), reverse=True), 1
        ):

            if max_games is not None and games_started >= max_games:
                print(f"\nReached max_games={max_games}. Stopping.\n")
                break

            game_id = filename.removesuffix(".SC2Replay")
            print(game_id)
            info = detailed_info.get(game_id)
            if info is None or info.get("skipped"):
                print("skip")
                continue

            parts = list(map(int, info.get("length", "0:00").split(":")))
            total_seconds = sum(p * 60**i for i, p in enumerate(reversed(parts)))
            if total_seconds > 20 * 60:
                print("skip (too long)")
                continue

            if info.get("mmr", 0) < MIN_MMR:
                print("skip (too low mmr)")
                continue

            file_path = os.path.join(input_folder, filename)
            output_base = os.path.join(output_folder, game_id)

            if os.path.exists(output_base + "_1.json.gz"):
                continue
            print(f"NOT PROCESSED: {filename}")
            if CHECK_PROCESSED_COUNT:
                not_processed += 1
                continue

            now = time.time()
            running.append(
                (
                    subprocess.Popen(
                        [
                            "python",
                            "simulator.py",
                            file_path,
                            output_base,
                            "1",
                            str(STEP_SIZE),
                        ]
                    ),
                    now,
                    filename,
                    game_id,
                )
            )
            running.append(
                (
                    subprocess.Popen(
                        [
                            "python",
                            "simulator.py",
                            file_path,
                            output_base,
                            "2",
                            str(STEP_SIZE),
                        ]
                    ),
                    now,
                    filename,
                    game_id,
                )
            )
            games_started += 1

            while len(running) >= max_parallel:
                now = time.time()
                for p, start, fname, gid in running:
                    if p.poll() is None and (now - start) > MAX_SECONDS:
                        print(f"[TIMEOUT] Killing hung process for {fname}")
                        p.terminate()
                        timed_out_ids.add(gid)
                running = [(p, s, f, g) for p, s, f, g in running if p.poll() is None]
                if len(running) >= max_parallel:
                    time.sleep(0.1)

            elapsed = int(time.time() - t0)
            h, m = divmod(elapsed // 60, 60)
            print(
                f"[{count}] {filename}  |  {h}h {m}m {elapsed%60}s "
                f"({datetime.now():%H:%M:%S})"
            )

        # Wait for remaining processes with timeout
        deadline = time.time() + MAX_SECONDS
        for p, start, fname, gid in running:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[TIMEOUT] Killing hung process for {fname}")
                p.terminate()
                timed_out_ids.add(gid)
            else:
                try:
                    p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    print(f"[TIMEOUT] Killing hung process for {fname}")
                    p.terminate()
                    timed_out_ids.add(gid)

    except KeyboardInterrupt:
        print("\nInterrupted — killing all subprocesses...")
        for p, start, fname, gid in running:
            p.terminate()
        for p, start, fname, gid in running:
            p.wait()
        print("Done.")

    finally:
        for gid in timed_out_ids:
            mark_skipped(gid)

    print(str(not_processed) + " unprocessed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-procs",
        type=int,
        default=5,
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
