"""
benchmark_viz.py  —  BetaStar win-probability curve viewer + annotator

Usage:
    python benchmark_viz.py [path/to/replay.json.gz] [--annotations path/to/annotations.json]

    If no replay path is given, REPLAY_PATH at the top of this file is used.

Controls:
    - Type a label in the Label box, press Enter to confirm
    - Click on the chart to add an event marker at that position
    - Click "Delete mode" to arm deletion, then click nearest marker to remove it
    - Click "Delete mode" again to cancel without deleting
    - Click "Toggle x-axis" to switch between mm:ss and frame index
    - Click "Save" to save annotations
    - Close the window to exit (annotations auto-save if modified)

Annotations are stored as a JSON file alongside the replay by default:
    replay.json.gz  →  replay.annotations.json
"""

import argparse
import gzip
import json
import os
import sys
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import TextBox, Button

# ---------------------------------------------------------------------------
# Adjust these imports / paths to match your project layout
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from playground import (
    MLP,
    LinearModel,
    LSTMModel,
    DEVICE,
    SCALER_TYPE,
    SCALER_PATH,
    SEQ_LEN,
    build_inference_dataset,
)

"""
class_1_10000_feet_le_515261123
class_1_old_republic_le_6226957
class_1_white_rabbit_le_6226946
class_2_10000_feet_le_515261112
class_2_old_republic_le_6226959
class_2_white_rabbit_le_6226949
class_4_10000_feet_le_62261010
class_5_10000_feet_le_62261007
class_6_10000_feet_le_62261034
class_7_10000_feet_le_62261109
class_8_10000_feet_le_6926709
class_9_10000_feet_le_6926723
"""
REPLAY_PATH = (
    "D:/betastar/benchmark_cache/pruned/class_7_10000_feet_le_62261109.json.gz"
)
MODEL_TYPE = "lstm"  # "mlp" | "linear" | "lstm"
MODEL_PATH = f"D:/betastar/model_{MODEL_TYPE}.pt"

# SC2 timing constants
# step_size gameloops happen between each on_step call (set in simulator.py batch args)
# 22.4 gameloops = 1 real-time second on "Faster" speed (standard ladder speed)
# Each stored *block* compresses 10 raw steps, so iteration in the JSON is the
# raw on_step counter.  game_seconds = block_index * 10 * step_size / 22.4
STEP_SIZE = 20  # must match what was used when extracting this replay
GAMELOOPS_PER_SEC = 22.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seconds_to_mmss(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def load_replay(replay_path):
    with gzip.open(replay_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def run_inference(frames):
    X_np = build_inference_dataset(frames, slice_ratio=1.0)

    if SCALER_TYPE is not None:
        scaler = joblib.load(SCALER_PATH)
        if MODEL_TYPE == "lstm":
            # X_np is 2D here (T, features); scale then re-window below
            X_np = scaler.transform(X_np)
        else:
            X_np = scaler.transform(X_np)

    if MODEL_TYPE == "lstm":
        # Build sliding-window sequences: shape (T - SEQ_LEN, SEQ_LEN, features)
        T, F = X_np.shape
        if T < SEQ_LEN:
            raise ValueError(
                f"Replay has only {T} frames, need at least SEQ_LEN={SEQ_LEN}"
            )
        sequences = np.stack(
            [X_np[t : t + SEQ_LEN] for t in range(T - SEQ_LEN)], axis=0
        ).astype(np.float32)
        X = torch.from_numpy(sequences).to(DEVICE)
        input_dim = F
        model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, output_dim=1)
    else:
        X = torch.from_numpy(X_np).to(DEVICE)
        input_dim = X_np.shape[1]
        if MODEL_TYPE == "mlp":
            model = MLP(input_dim, output_dim=1)
        else:
            model = LinearModel(input_dim, output_dim=1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().flatten()

    # For LSTM the first SEQ_LEN-1 frames have no prediction.
    # Pad the front with NaN so indices stay aligned with the original frame list.
    if MODEL_TYPE == "lstm":
        pad = np.full(SEQ_LEN, np.nan)
        probs = np.concatenate([pad, probs])

    return X_np, probs


def default_annotation_path(replay_path):
    base = replay_path
    for suffix in (".json.gz", ".gz", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + ".annotations.json"


def load_annotations(ann_path):
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            data = json.load(f)
        return data.get("events", [])
    return []


def save_annotations(ann_path, events):
    with open(ann_path, "w") as f:
        json.dump({"events": events}, f, indent=2)
    print(f"[saved] {ann_path}  ({len(events)} events)")


# ---------------------------------------------------------------------------
# Main visualiser
# ---------------------------------------------------------------------------


class BenchmarkViz:
    def __init__(self, replay_path, ann_path):
        self.replay_path = replay_path
        self.ann_path = ann_path
        self.modified = False
        self.show_time = True  # True = mm:ss axis, False = frame index
        self._delete_mode = False  # True = next chart click deletes nearest marker
        self.current_label = "event label"

        print(f"Loading replay: {replay_path}")
        replay = load_replay(replay_path)
        self.frames = replay["frames"]
        self.winner = replay.get("winner", None)

        if isinstance(self.frames, dict):
            sorted_keys = sorted(self.frames.keys(), key=lambda k: int(k))
            self.frames = [self.frames[k] for k in sorted_keys]

        print(f"Running inference on {len(self.frames)} frames (model={MODEL_TYPE})...")
        self.X_np, self.probs = run_inference(self.frames)

        self.frame_indices = np.arange(len(self.frames))
        self.game_seconds = self.frame_indices * 10 * STEP_SIZE / GAMELOOPS_PER_SEC

        self.events = load_annotations(ann_path)
        print(f"Loaded {len(self.events)} existing annotations.")

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        plt.subplots_adjust(bottom=0.22, right=0.78, top=0.85)

        self._draw()

        # Toggle x-axis button
        ax_btn_toggle = plt.axes([0.80, 0.88, 0.18, 0.07])
        self.btn_toggle = Button(ax_btn_toggle, "Toggle x-axis")
        self.btn_toggle.on_clicked(self._on_toggle)

        # Save button
        ax_btn_save = plt.axes([0.80, 0.78, 0.18, 0.07])
        self.btn_save = Button(ax_btn_save, "Save")
        self.btn_save.on_clicked(self._on_save)

        # Delete mode button
        ax_btn_delete = plt.axes([0.80, 0.68, 0.18, 0.07])
        self.btn_delete = Button(
            ax_btn_delete, "Delete mode", color="0.85", hovercolor="0.95"
        )
        self.btn_delete.on_clicked(self._on_delete_mode)

        # Label text box — pressing Enter confirms the label
        ax_text = plt.axes([0.80, 0.52, 0.18, 0.07])
        self.text_box = TextBox(ax_text, "Label:\n", initial=self.current_label)
        self.text_box.on_submit(self._on_label_submit)

        # Instructions
        self.fig.text(
            0.80,
            0.48,
            "1. Type label → Enter\n"
            "2. Click chart to place\n\n"
            "Delete mode → click marker\n"
            "to remove nearest",
            va="top",
            fontsize=8,
            color="#555555",
            transform=self.fig.transFigure,
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        plt.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _x_array(self):
        return self.game_seconds if self.show_time else self.frame_indices

    def _x_for_event(self, ev):
        fi = ev["frame_idx"]
        if self.show_time:
            return self.game_seconds[fi] if fi < len(self.game_seconds) else fi
        return fi

    def _set_delete_mode(self, active):
        self._delete_mode = active
        color = "#ff4444" if active else "0.85"
        hover = "#ff6666" if active else "0.95"
        self.btn_delete.color = color
        self.btn_delete.hovercolor = hover
        self.btn_delete.ax.set_facecolor(color)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.ax.cla()
        x = self._x_array()

        self.ax.plot(x, self.probs, color="#2196F3", linewidth=1.5, label="Win prob")
        self.ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8)

        # Shade the LSTM warm-up region where predictions are unavailable
        if MODEL_TYPE == "lstm" and SEQ_LEN > 1:
            warmup_x = x[SEQ_LEN - 1] if len(x) >= SEQ_LEN else x[-1]
            self.ax.axvspan(
                x[0],
                warmup_x,
                alpha=0.08,
                color="gray",
                label=f"LSTM warm-up ({SEQ_LEN - 1} frames)",
            )

        if self.winner is not None:
            color = "#4CAF50" if self.winner == 1 else "#F44336"
            label = "Zerg wins" if self.winner == 1 else "Zerg loses"
            self.ax.axhline(
                float(self.winner),
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.6,
                label=label,
            )

        for ev in self.events:
            xv = self._x_for_event(ev)
            self.ax.axvline(
                xv, color="#FF9800", linewidth=1.2, alpha=0.8, linestyle="-"
            )
            self.ax.text(
                xv,
                1.02,
                ev["label"],
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=7,
                color="#E65100",
                transform=self.ax.get_xaxis_transform(),
                clip_on=False,
            )

        self.ax.set_ylim(-0.05, 1.15)
        self.ax.set_ylabel("Win probability (Zerg)")
        self.ax.legend(loc="lower right", fontsize=8)

        if self.show_time:
            self.ax.set_xlabel("Game time (mm:ss)")
            self.ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: seconds_to_mmss(v))
            )
        else:
            self.ax.set_xlabel("Stored frame index")
            self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        title = os.path.basename(self.replay_path)
        self.ax.set_title(f"{title}  [{MODEL_TYPE.upper()}]", fontsize=10)
        self.ax.grid(True, alpha=0.25)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_label_submit(self, text):
        self.current_label = text.strip() or "event"
        print(f"  label set to '{self.current_label}'")

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None:
            return

        x = self._x_array()
        clicked_x = event.xdata

        if self._delete_mode:
            if not self.events:
                self._set_delete_mode(False)
                return
            dists = [abs(self._x_for_event(ev) - clicked_x) for ev in self.events]
            nearest = int(np.argmin(dists))
            removed = self.events.pop(nearest)
            self.modified = True
            print(f"  - removed '{removed['label']}' at frame {removed['frame_idx']}")
            self._set_delete_mode(False)
        else:
            frame_idx = int(np.argmin(np.abs(x - clicked_x)))
            game_s = self.game_seconds[frame_idx]
            label = self.current_label
            self.events.append(
                {
                    "frame_idx": frame_idx,
                    "game_seconds": float(game_s),
                    "game_time": seconds_to_mmss(game_s),
                    "label": label,
                }
            )
            self.modified = True
            print(
                f"  + added '{label}' at frame {frame_idx} ({seconds_to_mmss(game_s)})"
            )

        self._draw()

    def _on_toggle(self, event):
        self.show_time = not self.show_time
        self._draw()

    def _on_save(self, event):
        save_annotations(self.ann_path, self.events)
        self.modified = False

    def _on_delete_mode(self, event):
        self._set_delete_mode(not self._delete_mode)

    def _on_close(self, event):
        if self.modified:
            save_annotations(self.ann_path, self.events)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BetaStar benchmark visualiser")
    parser.add_argument(
        "replay",
        nargs="?",
        default=REPLAY_PATH,
        help="Path to replay .json.gz file (default: REPLAY_PATH in script)",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        default=None,
        help="Path to annotations JSON (default: <replay>.annotations.json)",
    )
    args = parser.parse_args()

    ann_path = args.annotations or default_annotation_path(args.replay)
    BenchmarkViz(args.replay, ann_path)


if __name__ == "__main__":
    main()
