import gzip
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib
import hashlib
from sklearn.preprocessing import MinMaxScaler
from s2protocol.build import read_command_output

FOLDER = r"D:\betastar\BetaStar\prunes"
SCALER_TYPE = "minimax"  # Options: "minmax", "zscore", None

SCALER_PATH = f"D:/betastar/scaler_{SCALER_TYPE}.pkl"
SAVE_PATH = f"D:/betastar/dataset_split_{SCALER_TYPE}.npz"
HORIZON = 5
# with gzip.open("prunes/26950686.json.gz", "rt", encoding="utf-8") as f:
#     data = json.load(f)


import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, TensorDataset, DataLoader
from torch.autograd import Variable


def get_files_hash(files):
    h = hashlib.md5("".join(sorted(files)).encode()).hexdigest()
    return h


# for time in data:
#     print(len(time))
def flatten(obj):
    flat = []

    if isinstance(obj, (int, float, bool)):
        return [int(obj)]

    if isinstance(obj, list):
        for x in obj:
            flat.extend(flatten(x))
        return flat

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            flat.extend(flatten(obj[k]))
        return flat

    raise TypeError(f"Unsupported type: {type(obj)}")


def flatten_debug(obj, path="root"):
    flat = []

    if isinstance(obj, (int, float, bool)):
        return [int(obj)]

    if isinstance(obj, list):
        for i, x in enumerate(obj):
            flat.extend(flatten_debug(x, f"{path}[{i}]"))
        return flat

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            flat.extend(flatten_debug(obj[k], f"{path}.{k}"))
        return flat

    if isinstance(obj, str):
        print(f"[DEBUG] Found string at {path}: {repr(obj)}")
        return []

    raise TypeError(f"Unsupported type at {path}: {type(obj)}")


from collections import defaultdict


def collapse_action_window(window):
    """
    window: list of action dicts from split_state_and_actions
    returns: single aggregated action dict
    """

    collapsed = {}

    # ---- Scalars (sum) ----
    collapsed["workers_built"] = sum(a["workers_built"] for a in window)
    collapsed["army_built"] = sum(a["army_built"] for a in window)

    # ---- Dict counts (elementwise sum) ----
    for key in ["building_started", "newly_queued"]:
        accum = defaultdict(int)

        # initialize full key space
        for unit_id in window[0][key].keys():
            accum[unit_id] = 0

        for a in window:
            for unit_id, count in a[key].items():
                accum[unit_id] += count

        collapsed[key] = dict(accum)

    # ---- Tech flags (OR) ----
    collapsed["lair_started"] = int(any(a["lair_started"] for a in window))
    collapsed["hive_started"] = int(any(a["hive_started"] for a in window))

    # ---- Visibility (elementwise max) ----
    vis0 = window[0]["visibility"]
    h = len(vis0)
    w = len(vis0[0])

    vis_agg = [[vis0[i][j] for j in range(w)] for i in range(h)]

    for a in window[1:]:
        vis = a["visibility"]
        for i in range(h):
            for j in range(w):
                vis_agg[i][j] = max(vis_agg[i][j], vis[i][j])

    collapsed["visibility"] = vis_agg

    return collapsed


def find_duplicate_timesteps(flat_vectors):
    duplicates = []
    for i in range(1, len(flat_vectors)):
        if flat_vectors[i] == flat_vectors[i - 1]:
            duplicates.append(i)
    return duplicates


def build_replay_dataset(data, horizon=HORIZON):
    """
    data: list of timesteps (loaded JSON)
    horizon: k-step action aggregation window

    Returns:
        X_np : (N, state_dim)
        y_np : (N, action_dim)
    """
    T = len(data)

    features = []
    targets = []

    # Precompute dimensions once
    state_dim = len(flatten(data[0]["state"]))
    action_dim = len(flatten(data[0]["actions"]))

    for t in range(T - horizon + 1):

        # ----- STATE -----
        state_vec = flatten(data[t]["state"])
        assert len(state_vec) == state_dim

        # ----- ACTION WINDOW -----
        window = [data[t + j]["actions"] for j in range(horizon)]
        collapsed = collapse_action_window(window)
        action_vec = flatten(collapsed)
        assert len(action_vec) == action_dim

        features.append(state_vec)
        targets.append(action_vec)

    X_np = np.array(features, dtype=np.float32)
    y_np = np.array(targets, dtype=np.float32)

    return X_np, y_np


def build_corpus_dataset(folder_path, horizon=5, files=None):
    all_X = []
    all_y = []
    all_y_win = []

    global_state_dim = None
    global_action_dim = None

    if files is None:
        files = [f for f in os.listdir(folder_path) if f.endswith(".json.gz")]

    print(f"Found {len(files)} replay files.")

    for i, filename in enumerate(files):
        path = os.path.join(folder_path, filename)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            replay_json = json.load(f)

        frames = replay_json["frames"]
        winner = replay_json["winner"]

        if len(frames) < horizon:
            print(f"Skipping {filename} (too short).")
            continue

        X_np, y_np = build_replay_dataset(frames, horizon=horizon)

        # ---- NEW: create win labels same length as X_np ----
        y_win_np = np.full((X_np.shape[0], 1), winner, dtype=np.float32)

        # ----- Dimensional consistency check -----
        if global_state_dim is None:
            global_state_dim = X_np.shape[1]
            global_action_dim = y_np.shape[1]
        else:
            if X_np.shape[1] != global_state_dim:
                raise ValueError(f"State dim mismatch in {filename}")
            if y_np.shape[1] != global_action_dim:
                raise ValueError(f"Action dim mismatch in {filename}")

        all_X.append(X_np)
        all_y.append(y_np)
        all_y_win.append(y_win_np)

        print(f"[{i+1}/{len(files)}] Processed {filename} | Samples: {X_np.shape[0]}")

    # ----- Concatenate everything -----
    X_full = np.vstack(all_X)
    y_full = np.vstack(all_y)
    y_win_full = np.vstack(all_y_win)

    print("\nFinished building corpus dataset.")
    print("Total samples:", X_full.shape[0])
    print("State dim:", X_full.shape[1])
    print("Action dim:", y_full.shape[1])

    return X_full, y_full, y_win_full


def train_model(X, y, train_frac=0.8):
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, y.shape[1]),  # 754
    )
    dataset = TensorDataset(X, y)
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5000):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.6f}")

    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert X.ndim == 2
    assert y.shape == (X.shape[0], y.shape[1])
    return model


def build_dataset():
    all_files = [f for f in os.listdir(FOLDER) if f.endswith(".json.gz")]
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=42
    )

    X_train, y_train, y_win_train = build_corpus_dataset(FOLDER, files=train_files)
    X_test, y_test, y_win_test = build_corpus_dataset(FOLDER, files=test_files)

    if SCALER_TYPE == "minmax":
        scaler = MinMaxScaler()
    elif SCALER_TYPE == "zscore":
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, SCALER_PATH)

    files_hash = get_files_hash(all_files)

    np.savez_compressed(
        SAVE_PATH,
        horizon=np.array(HORIZON),
        X_train=X_train,
        y_train=y_train,
        y_win_train=y_win_train,
        X_test=X_test,
        y_test=y_test,
        y_win_test=y_win_test,
        files_hash=np.array(files_hash),
    )

    X_train, y_train, y_win_train = (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(y_win_train),
    )
    X_test, y_test, y_win_test = (
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
        torch.from_numpy(y_win_test),
    )

    return X_train, y_train, y_win_train, X_test, y_test, y_win_test


def get_dataset(horizon=HORIZON):
    try:
        data = np.load(SAVE_PATH)
        current_hash = get_files_hash(
            [f for f in os.listdir(FOLDER) if f.endswith(".json.gz")]
        )
        assert (
            str(data["files_hash"]) == current_hash
        ), "File list changed, rebuilding..."
        assert int(data["horizon"]) == horizon, "Horizon mismatch"

        X_train = torch.from_numpy(data["X_train"])
        y_train = torch.from_numpy(data["y_train"])
        y_win_train = torch.from_numpy(data["y_win_train"])
        X_test = torch.from_numpy(data["X_test"])
        y_test = torch.from_numpy(data["y_test"])
        y_win_test = torch.from_numpy(data["y_win_test"])
        print("dataset loaded")

    except Exception as e:
        print(f"No saved dataset found ({e}), building from scratch...")
        X_train, y_train, y_win_train, X_test, y_test, y_win_test = build_dataset()

    print(f"Train: X={X_train.shape} y={y_train.shape} y_win={y_win_train.shape}")
    print(f"Test:  X={X_test.shape} y={y_test.shape} y_win={y_win_test.shape}")

    return X_train, y_train, y_win_train, X_test, y_test, y_win_test


# with gzip.open(
#     r"D:\betastar\BetaStar\prunes\27037792.json.gz", "rt", encoding="utf-8"
# ) as f:
#     replay_json = json.load(f)
#
# print(replay_json["frames"])


X_train, y_train, y_win_train, X_test, y_test, y_win_test = get_dataset()

# add this right after loading y_win_train
win_rate = y_win_train.mean().item()
print(f"Training win rate: {win_rate:.3f}")
# e.g. 0.30 → losses outnumber wins 2.3:1
pos_weight_val = (1 - win_rate) / win_rate
print(f"pos_weight to use: {pos_weight_val:.2f}")


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
inputDim = X_train.shape[1]
outputDim = y_win_train.shape[1]
learningRate = 1e-4
epochs = 5000
train_losses = []
test_losses = []
model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
pos_weight = torch.tensor([pos_weight_val])
if torch.cuda.is_available():
    pos_weight = pos_weight.cuda()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    inputs = X_train.cuda() if torch.cuda.is_available() else X_train
    labels = y_win_train.cuda() if torch.cuda.is_available() else y_win_train

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()
    train_losses.append(loss.item())

    with torch.no_grad():
        test_out = model(X_test.cuda() if torch.cuda.is_available() else X_test)
        test_loss = criterion(
            test_out, y_win_test.cuda() if torch.cuda.is_available() else y_win_test
        )
        test_losses.append(test_loss.item())

    print("epoch {}, loss {}".format(epoch, loss.item()))

plt.plot(train_losses)
plt.xlabel("Epoch")

# plt.plot(train_losses, label="Train")
# plt.plot(test_losses, label="Test")
# plt.legend()
# plt.show()

with torch.no_grad():
    raw = model(X_test)
    probs = torch.sigmoid(raw)
    preds = (probs > 0.5).float()
    y = y_win_test

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Confusion matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"Predicted win rate: {probs.mean():.3f}")
    print(f"Actual win rate:    {y.mean():.3f}")
