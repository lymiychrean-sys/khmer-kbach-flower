"""
Live app: draws the glow-dot flower at your hand, grown to the stage that
matches your gesture.

    ./venv/bin/python photo_flower_ar.py      (q to quit)
"""

import time

import cv2

from growth import GrowthEstimator
from hand_bridge import combine_hands, create_landmarker, detect, draw_fps
from photo_flower import draw_photo_flower, load_recipes

WEIGHT_SMOOTHING = 0.18  # lower = smoother/slower, higher = snappier
POS_SMOOTHING = 0.35


def _lerp(a, b, t):
    return a * (1 - t) + b * t


def main():
    load_recipes()  # warm the cache before the loop starts

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    growth_estimator = GrowthEstimator()
    smoothed_weights = {"0": 1.0}
    smoothed_xy = None

    with create_landmarker() as landmarker:
        prev_t = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detect(landmarker, frame)

            if result.hand_landmarks:
                feats = combine_hands(result)
                weights = growth_estimator.estimate(feats)

                keys = set(smoothed_weights) | set(weights)
                blended = {
                    k: _lerp(smoothed_weights.get(k, 0.0), weights.get(k, 0.0), WEIGHT_SMOOTHING)
                    for k in keys
                }
                total = sum(blended.values()) or 1.0
                smoothed_weights = {k: v / total for k, v in blended.items()}

                side = "right" if feats["right_present"] else "left"
                h, w = frame.shape[:2]
                target_xy = (feats[f"{side}_pos_x"] * w, feats[f"{side}_pos_y"] * h)
                if smoothed_xy is None:
                    smoothed_xy = target_xy
                else:
                    smoothed_xy = tuple(_lerp(s, t, POS_SMOOTHING) for s, t in zip(smoothed_xy, target_xy))

                draw_photo_flower(frame, smoothed_xy, smoothed_weights)

            prev_t = draw_fps(frame, prev_t)
            cv2.imshow("photo_flower_ar (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
