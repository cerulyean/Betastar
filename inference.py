import gzip
import json
import torch
import numpy as np
import joblib
from playground import (
    MLP,
    LinearModel,
    DEVICE,
    SCALER_TYPE,
    SCALER_PATH,
    flatten,
    build_inference_dataset,
)

MODEL_TYPE = "mlp"
MODEL_PATH = f"D:/betastar/model_{MODEL_TYPE}.pt"
REPLAY_PATH = "D:/betastar/BetaStar/prune_test_v4/noscoutnoresponse.SC2Replay.json.gz"
# 1. Load the replay
with gzip.open(REPLAY_PATH, "rt", encoding="utf-8") as f:
    replay_json = json.load(f)

frames = replay_json["frames"]

# 2. Build features the same way training did
X_np = build_inference_dataset(frames, slice_ratio=1.0)
print(X_np)

# 3. Apply scaler if one was used during training
if SCALER_TYPE is not None:
    scaler = joblib.load(SCALER_PATH)
    X_np = scaler.transform(X_np)

X = torch.from_numpy(X_np).to(DEVICE)

# 4. Load model — input_dim must match what was trained
input_dim = X.shape[1]
if MODEL_TYPE == "mlp":
    model = MLP(input_dim, output_dim=1)
elif MODEL_TYPE == "linear":
    model = LinearModel(input_dim, output_dim=1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 5. Run inference
with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)

print(f"Win probability per timestep — mean: {probs.mean().item():.3f}")
print(f"Final timestep prediction: {probs[-1].item():.3f}")
for t in range(len(X_np)):
    x = torch.from_numpy(X_np[t]).unsqueeze(0).to(DEVICE)  # shape: [1, input_dim]
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit)
    print(f"t={t:4d}  win_prob={prob.item():.3f}")
