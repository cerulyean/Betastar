import gzip
import json
import os

INPUT_DIR = "output"
replay_prefix = "27079595"  # change this

# files = [
#     f
#     for f in os.listdir(INPUT_DIR)
#     if f.startswith(replay_prefix) and f.endswith(".json.gz")
# ]
# files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
#
# for fname in files:
#     path = os.path.join(INPUT_DIR, fname)
#     with gzip.open(path, "rt", encoding="utf-8") as f:
#         chunk = json.load(f)
#     print(f"\n--- {fname} ({len(chunk)} states) ---")
#     for i, state in chunk.items():
#         print(f"  [{i}]: {list(state.keys())}")

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data.keys())
