import gzip
import json
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
import joblib

# =============================================================================
# CONFIG
# =============================================================================
FOLDER = r"D:\betastar\BetaStar\prunes_v5"
SCALER_TYPE = None  # "minmax" | "zscore" | None
HORIZON = 5
MODEL_TYPE = "lstm"  # "linear" | "mlp" | "lstm"
SEQ_LEN = 10
EPOCHS = 500
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SLICE = 1.0  # 1.0 = full game | 0.5 = last 50% | 0.25 = last 25%
TEST_SLICE = 0.5

SCALER_PATH = f"D:/betastar/scaler_{SCALER_TYPE}.pkl"
SAVE_PATH = f"D:/betastar/dataset_split_{SCALER_TYPE}.npz"
# =============================================================================


# --- Data helpers ------------------------------------------------------------


def input_sensitivity(model, X, device=DEVICE):
    model.eval()
    X_t = X.clone().to(device).requires_grad_(True)
    out = model(X_t)
    out.sum().backward()
    return X_t.grad.abs().mean(dim=0).cpu()


def slice_frames(frames, slice_ratio):
    if slice_ratio >= 1.0:
        return frames
    start = int(len(frames) * (1.0 - slice_ratio))
    return frames[start:]


def get_files_hash(files):
    return hashlib.md5("".join(sorted(files)).encode()).hexdigest()


def flatten(obj):
    if isinstance(obj, (int, float, bool)):
        return [int(obj)]
    if isinstance(obj, list):
        flat = []
        for x in obj:
            flat.extend(flatten(x))
        return flat
    if isinstance(obj, dict):
        flat = []
        for k in sorted(obj.keys()):
            flat.extend(flatten(obj[k]))
        return flat
    raise TypeError(f"Unsupported type: {type(obj)}")


def collapse_action_window(window):
    collapsed = {}
    collapsed["workers_built"] = sum(a["workers_built"] for a in window)
    collapsed["army_built"] = sum(a["army_built"] for a in window)

    for key in ["building_started", "newly_queued"]:
        accum = defaultdict(int)
        for unit_id in window[0][key].keys():
            accum[unit_id] = 0
        for a in window:
            for unit_id, count in a[key].items():
                accum[unit_id] += count
        collapsed[key] = dict(accum)

    collapsed["lair_started"] = int(any(a["lair_started"] for a in window))
    collapsed["hive_started"] = int(any(a["hive_started"] for a in window))

    vis0 = window[0]["visibility"]
    h, w = len(vis0), len(vis0[0])
    vis_agg = [[vis0[i][j] for j in range(w)] for i in range(h)]
    for a in window[1:]:
        vis = a["visibility"]
        for i in range(h):
            for j in range(w):
                vis_agg[i][j] = max(vis_agg[i][j], vis[i][j])
    collapsed["visibility"] = vis_agg

    return collapsed


def build_replay_dataset(frames, horizon=HORIZON, slice_ratio=1.0):
    frames = slice_frames(frames, slice_ratio)
    T = len(frames)
    state_dim = len(flatten(frames[0]["state"]))
    action_dim = len(flatten(frames[0]["actions"]))
    features, targets = [], []

    for t in range(T - horizon):
        state_vec = flatten(frames[t]["state"])
        window = [frames[t + j]["actions"] for j in range(1, horizon)]
        action_vec = flatten(collapse_action_window(window))
        assert len(state_vec) == state_dim
        assert len(action_vec) == action_dim
        features.append(state_vec)
        targets.append(action_vec)

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def build_sequence_dataset(frames, winner, seq_len=10, slice_ratio=1.0):
    frames = slice_frames(frames, slice_ratio)
    features, targets = [], []
    for t in range(len(frames) - seq_len):
        seq = [flatten(frames[t + i]["state"]) for i in range(seq_len)]
        features.append(seq)
    X = np.array(features, dtype=np.float32)
    y = np.full((X.shape[0], 1), winner, dtype=np.float32)
    return X, y


def build_inference_dataset(frames, slice_ratio=1.0):
    frames = slice_frames(frames, slice_ratio)
    features = []
    for t in range(len(frames)):
        state_vec = flatten(frames[t]["state"])
        features.append(state_vec)
    return np.array(features, dtype=np.float32)


def build_corpus_dataset(
    folder_path, horizon=HORIZON, files=None, slice_ratio=1.0, seq_len=10
):
    if files is None:
        files = [f for f in os.listdir(folder_path) if f.endswith(".json.gz")]

    all_X, all_y, all_y_win = [], [], []
    global_state_dim = global_action_dim = None

    for i, filename in enumerate(files):
        with gzip.open(
            os.path.join(folder_path, filename), "rt", encoding="utf-8"
        ) as f:
            replay_json = json.load(f)

        frames = replay_json["frames"]
        winner = replay_json["winner"]

        if MODEL_TYPE == "lstm":
            if len(frames) < seq_len:
                print(f"Skipping {filename} (too short).")
                continue
            X_np, y_win_np = build_sequence_dataset(
                frames, winner, seq_len=seq_len, slice_ratio=slice_ratio
            )
            if X_np.shape[0] == 0:  # slice made it too short
                print(f"Skipping {filename} (empty after slicing).")
                continue
            y_np = y_win_np

        else:
            if len(frames) < horizon:
                print(f"Skipping {filename} (too short).")
                continue
            X_np, y_np = build_replay_dataset(
                frames, horizon=horizon, slice_ratio=slice_ratio
            )
            y_win_np = np.full((X_np.shape[0], 1), winner, dtype=np.float32)

        all_X.append(X_np)
        all_y.append(y_np)
        all_y_win.append(y_win_np)
        print(f"[{i+1}/{len(files)}] {filename} | samples: {X_np.shape[0]}")

    X_full = np.vstack(all_X)
    y_full = np.vstack(all_y)
    y_win_full = np.vstack(all_y_win)
    return X_full, y_full, y_win_full


def get_feature_names(sample_state):
    names = []

    def walk(obj, prefix=""):
        if isinstance(obj, (int, float, bool)):
            names.append(prefix)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
        elif isinstance(obj, dict):
            for k in sorted(obj.keys()):
                walk(obj[k], f"{prefix}.{k}" if prefix else k)

    walk(sample_state)
    return names


def build_dataset():
    all_files = [f for f in os.listdir(FOLDER) if f.endswith(".json.gz")]
    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=42
    )

    X_train, y_train, y_win_train = build_corpus_dataset(
        FOLDER, files=train_files, slice_ratio=TRAIN_SLICE
    )
    X_test, y_test, y_win_test = build_corpus_dataset(
        FOLDER, files=test_files, slice_ratio=TEST_SLICE
    )

    with gzip.open(os.path.join(FOLDER, all_files[0]), "rt", encoding="utf-8") as f:
        sample_replay = json.load(f)
    feature_names = get_feature_names(sample_replay["frames"][0]["state"])
    np.save("D:/betastar/feature_names.npy", feature_names)

    if SCALER_TYPE == "minmax":
        scaler = MinMaxScaler()
    elif SCALER_TYPE == "zscore":
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler is not None:
        if MODEL_TYPE == "lstm":
            # reshape to 2D, scale, reshape back
            shape = X_train.shape
            X_train = scaler.fit_transform(X_train.reshape(-1, shape[-1])).reshape(
                shape
            )
            shape = X_test.shape
            X_test = scaler.transform(X_test.reshape(-1, shape[-1])).reshape(shape)
        else:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        joblib.dump(scaler, SCALER_PATH)

    np.savez_compressed(
        SAVE_PATH,
        horizon=np.array(HORIZON),
        train_slice=np.array(TRAIN_SLICE),
        test_slice=np.array(TEST_SLICE),
        X_train=X_train,
        y_train=y_train,
        y_win_train=y_win_train,
        X_test=X_test,
        y_test=y_test,
        y_win_test=y_win_test,
        files_hash=np.array(get_files_hash(all_files)),
        model_type=np.array(MODEL_TYPE),
        seq_len=np.array(SEQ_LEN),
    )

    to_tensor = lambda a: torch.from_numpy(a)
    return map(to_tensor, [X_train, y_train, y_win_train, X_test, y_test, y_win_test])


def get_dataset():
    try:
        data = np.load(SAVE_PATH)
        current_hash = get_files_hash(
            [f for f in os.listdir(FOLDER) if f.endswith(".json.gz")]
        )
        assert (
            str(data["files_hash"]) == current_hash
        ), "File list changed, rebuilding..."
        assert int(data["horizon"]) == HORIZON, "Horizon mismatch"
        assert float(data["train_slice"]) == TRAIN_SLICE, "Train slice mismatch"
        assert float(data["test_slice"]) == TEST_SLICE, "Test slice mismatch"
        assert str(data["model_type"]) == MODEL_TYPE, "Model type changed"
        assert int(data["seq_len"]) == SEQ_LEN, "Seq len changed"

        X_train = torch.from_numpy(data["X_train"])
        y_train = torch.from_numpy(data["y_train"])
        y_win_train = torch.from_numpy(data["y_win_train"])
        X_test = torch.from_numpy(data["X_test"])
        y_test = torch.from_numpy(data["y_test"])
        y_win_test = torch.from_numpy(data["y_win_test"])
        feature_names = list(
            np.load("D:/betastar/feature_names.npy", allow_pickle=True)
        )
        print("Dataset loaded from cache.")
    except Exception as e:
        print(f"Rebuilding dataset ({e})...")
        X_train, y_train, y_win_train, X_test, y_test, y_win_test = build_dataset()
        feature_names = list(
            np.load("D:/betastar/feature_names.npy", allow_pickle=True)
        )

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, y_train, y_win_train, X_test, y_test, y_win_test, feature_names


# --- Models ------------------------------------------------------------------


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, num_layers=2, output_dim=1, dropout=0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # expects (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # take top layer's final hidden state
        return self.head(last_hidden)


# --- Main function -----------------------------------------------------------


def train_and_eval(plot=False):
    X_train, y_train, y_win_train, X_test, y_test, y_win_test, feature_names = (
        get_dataset()
    )

    win_rate = y_win_train.mean().item()
    pos_weight_val = (1 - win_rate) / win_rate
    print(f"Training win rate: {win_rate:.3f}  pos_weight: {pos_weight_val:.2f}")
    print(f"Train samples: {X_train.shape[0]}  Test samples: {X_test.shape[0]}")
    output_dim = y_win_train.shape[1]

    if MODEL_TYPE == "lstm":
        input_dim = X_train.shape[2]  # state_dim is the last axis
    else:
        input_dim = X_train.shape[1]

    if MODEL_TYPE == "linear":
        model = LinearModel(input_dim, output_dim)
    elif MODEL_TYPE == "mlp":
        model = MLP(input_dim, output_dim)
    elif MODEL_TYPE == "lstm":
        model = LSTMModel(
            input_dim, hidden_dim=256, num_layers=2, output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    model = model.to(DEVICE)
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_train_d = X_train.to(DEVICE)
    y_win_train_d = y_win_train.to(DEVICE)
    X_test_d = X_test.to(DEVICE)
    y_win_test_d = y_win_test.to(DEVICE)

    train_losses = []
    test_losses = []

    dataset = TensorDataset(X_train_d, y_win_train_d)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                test_loss = criterion(model(X_test_d), y_win_test_d)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            print(
                f"epoch {epoch:4d}  train={loss.item():.4f}  test={test_loss.item():.4f}"
            )

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_d))
        preds = (probs > 0.5).float()
        y = y_win_test_d

        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()
        balanced_acc = (tp / (tp + fn) + tn / (tn + fp)) / 2

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        result = {
            "model": MODEL_TYPE,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "balanced_acc": round(balanced_acc, 4),
            "pred_win_rate": round(probs.mean().item(), 4),
            "actual_win_rate": round(y.mean().item(), 4),
        }

    torch.save(model.state_dict(), f"D:/betastar/model_{MODEL_TYPE}.pt")
    np.save("D:/betastar/feature_names.npy", feature_names)

    if plot:
        import matplotlib.pyplot as plt

        epochs_axis = list(range(0, EPOCHS, 100))
        plt.figure()
        plt.plot(epochs_axis, train_losses, label="train")
        plt.plot(epochs_axis, test_losses, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{MODEL_TYPE} | scaler={SCALER_TYPE}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"loss_{MODEL_TYPE}_{SCALER_TYPE}.png")
        plt.show()

    return result


if __name__ == "__main__":
    result = train_and_eval(plot=True)
    print(json.dumps(result))
