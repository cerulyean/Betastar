import json
import time
import os
import subprocess
from datetime import datetime

from simulator import extract_data

STEP_SIZE = 1

def process_folder(input_folder="1000 replays", output_folder="1000 extracts"):
    # extract only zvp games
    with open("data.json", "r") as f:
        detailed_info = json.load(f)
    t0 = time.time()
    count = 0
    os.makedirs(output_folder, exist_ok=True)
    folder_path = input_folder
    for filename in os.listdir(folder_path):
        id = filename.removesuffix(".SC2Replay")
        if detailed_info[id]["zerg"] != True or detailed_info[id]["protoss"] != True:
            continue
        print("count number: " + str(count))
        print(filename)
        file_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)
        output_path_p1 = output_path + "_p1.json.gz"
        output_path_p2 = output_path + "_p2.json.gz"

        if os.path.exists(output_path_p1) and os.path.exists(output_path_p2):
            print("skipped")
            continue
        if os.path.isfile(file_path):
            cmd_p1 = ["python", "simulator.py", file_path, output_path_p1, str(1), str(STEP_SIZE)]
            cmd_p2 = ["python", "simulator.py", file_path, output_path_p2, str(2), str(STEP_SIZE)]
            subprocess.Popen(cmd_p1)
            subprocess.Popen(cmd_p2)
        elapsed = time.time() - t0
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        print(f"{hours}h {minutes}m {seconds}s")
        print(datetime.now().strftime("%H:%M:%S"))  # 24-hour time
        count += 1
        return

if __name__ == "__main__":
    process_folder("replays", "output")