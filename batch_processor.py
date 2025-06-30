import argparse, json, os, subprocess, time
from datetime import datetime

STEP_SIZE = 1

def process_folder(input_folder="replays",
                   output_folder="output",
                   max_parallel=4):
    with open("data.json") as f:
        detailed_info = json.load(f)

    os.makedirs(output_folder, exist_ok=True)
    running = []
    t0 = time.time()

    for count, filename in enumerate(os.listdir(input_folder), 1):
        game_id = filename.removesuffix(".SC2Replay")
        info = detailed_info.get(game_id)
        if not info or not (info["zerg"] and info["protoss"]):
            continue

        file_path   = os.path.join(input_folder, filename)
        output_base = os.path.join(output_folder, filename)
        p1_path, p2_path = output_base + "_p1.json.gz", output_base + "_p2.json.gz"

        if os.path.exists(p1_path) and os.path.exists(p2_path):
            continue

        running.append(subprocess.Popen(
            ["python", "simulator.py", file_path, p1_path, "1", str(STEP_SIZE)]))
        running.append(subprocess.Popen(
            ["python", "simulator.py", file_path, p2_path, "2", str(STEP_SIZE)]))

        while len(running) >= max_parallel:
            running = [p for p in running if p.poll() is None]
            if len(running) >= max_parallel:
                time.sleep(0.1)

        elapsed = int(time.time() - t0)
        h, m = divmod(elapsed // 60, 60)
        print(f"[{count}] {filename}  |  {h}h {m}m {elapsed%60}s "
              f"({datetime.now():%H:%M:%S})")

    for p in running:
        p.wait()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-procs", type=int, default=4,
                    help="maximum concurrent simulator processes")
    args = ap.parse_args()
    process_folder(max_parallel=args.max_procs)
