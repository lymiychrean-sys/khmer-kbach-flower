"""
Webcam hand tracking + per-hand shape features. Run with --capture to
record reference gestures into poses.json:

    ./venv/bin/python hand_bridge.py --capture
"""

import argparse
import json
import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


def dist(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def finger_curl(landmarks, chain):
    """0 = fully extended, 1 = fully curled."""
    mcp, pip, dip, tip = (landmarks[i] for i in chain)
    span = dist(mcp, tip)
    length = dist(mcp, pip) + dist(pip, dip) + dist(dip, tip)
    if length < 1e-6:
        return 0.0
    curl = 1.0 - (span / length)
    return max(0.0, min(1.0, curl))


def extract_features(landmarks):
    scale = dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6
    feats = {}

    for name, chain in FINGERS.items():
        feats[f"curl_{name}"] = finger_curl(landmarks, chain)

    # average gap between adjacent fingertips
    tips = [landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    spread = sum(dist(tips[i], tips[i + 1]) for i in range(len(tips) - 1)) / 3
    feats["spread"] = spread / scale

    feats["thumb_out"] = dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) / scale

    dx = landmarks[MIDDLE_MCP].x - landmarks[WRIST].x
    dy = landmarks[MIDDLE_MCP].y - landmarks[WRIST].y
    feats["wrist_angle"] = math.atan2(dy, dx)

    feats["openness"] = 1.0 - sum(
        feats[f"curl_{n}"] for n in FINGERS
    ) / len(FINGERS)

    feats["pos_y"] = landmarks[WRIST].y
    feats["pos_x"] = landmarks[WRIST].x

    return feats


ZERO_HAND_FEATS = {
    **{f"curl_{n}": 0.0 for n in FINGERS},
    "spread": 0.0, "thumb_out": 0.0, "wrist_angle": 0.0,
    "openness": 0.0, "pos_x": 0.0, "pos_y": 0.0,
}


def combine_hands(result):
    """Merge up to 2 hands into one flat dict keyed left_/right_. A missing
    hand gets zeroed features and *_present=0, so the keys never change.
    Frames are mirrored before detection, so labels match the physical hand."""
    by_label = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name.lower()
        by_label[label] = extract_features(landmarks)

    feats = {}
    for label in ("left", "right"):
        present = label in by_label
        feats[f"{label}_present"] = 1.0 if present else 0.0
        src = by_label.get(label, ZERO_HAND_FEATS)
        for k, v in src.items():
            feats[f"{label}_{k}"] = v

    if "left" in by_label and "right" in by_label:
        feats["hands_distance"] = math.dist(
            (feats["left_pos_x"], feats["left_pos_y"]),
            (feats["right_pos_x"], feats["right_pos_y"]),
        )
    else:
        feats["hands_distance"] = 0.0

    return feats


def create_landmarker(num_hands=2):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Missing model file: {MODEL_PATH}\n"
            "Download it with:\n"
            '  curl -fL -o hand_landmarker.task '
            '"https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/latest/hand_landmarker.task"'
        )
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect(landmarker, frame):
    """frame: mirrored BGR webcam frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, int(time.time() * 1000))


def draw_fps(frame, prev_t):
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    cv2.putText(frame, f"{fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument(
        "--capture", action="store_true",
        help="pose-capture mode: hold a pose, press 0-4 to record it, s to save, q to quit",
    )
    ap.add_argument(
        "--poses-file",
        default=os.path.join(os.path.dirname(__file__), "poses.json"),
    )
    args = ap.parse_args()

    captured = {}
    if args.capture and os.path.exists(args.poses_file):
        with open(args.poses_file) as f:
            captured = json.load(f)
        print(f"Loaded {sum(len(v) for v in captured.values())} existing sample(s) from {args.poses_file}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    COUNTDOWN_SECONDS = 3.0
    armed_stage = None
    armed_deadline = 0.0

    with create_landmarker(args.max_hands) as landmarker:
        prev_t = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detect(landmarker, frame)

            feats = None
            if result.hand_landmarks:
                feats = combine_hands(result)

                h, w = frame.shape[:2]
                for landmarks in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    for a, b in connections:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    for x, y in pts:
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            prev_t = draw_fps(frame, prev_t)

            if args.capture:
                stages = " ".join(
                    f"[{i}]x{len(captured.get(str(i), []))}" for i in range(5)
                )
                cv2.putText(
                    frame, "CAPTURE  press 0-4 to arm a 3s countdown (adds another sample), s=save, q=quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                )
                cv2.putText(
                    frame, stages,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                )

                if armed_stage is not None:
                    remaining = armed_deadline - time.time()
                    if remaining > 0:
                        h, w = frame.shape[:2]
                        cv2.putText(
                            frame, f"GET READY: stage {armed_stage} in {remaining:0.1f}s",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3,
                        )
                    else:
                        if feats is None:
                            print(f"Countdown for stage {armed_stage} ended, "
                                  "no hand detected — not captured. Try again.")
                        else:
                            captured.setdefault(armed_stage, []).append(feats)
                            n = len(captured[armed_stage])
                            print(f"Captured stage {armed_stage}, sample #{n}")
                        armed_stage = None

            cv2.imshow("hand_bridge (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if args.capture:
                if ord("0") <= key <= ord("4"):
                    armed_stage = chr(key)
                    armed_deadline = time.time() + COUNTDOWN_SECONDS
                    print(f"Armed stage {armed_stage}, get into position...")
                elif key == ord("s"):
                    with open(args.poses_file, "w") as f:
                        json.dump(captured, f, indent=2)
                    print(f"Saved {len(captured)} pose(s) to {args.poses_file}")

    if args.capture and captured:
        with open(args.poses_file, "w") as f:
            json.dump(captured, f, indent=2)
        print(f"Saved {len(captured)} pose(s) to {args.poses_file}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
