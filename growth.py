"""
Pose -> per-stage weight blending. A stage's distance to the live pose is
the distance to its nearest captured sample in poses.json.
"""

import json
import math
import os

POSES_PATH = os.path.join(os.path.dirname(__file__), "poses.json")

# only hand shape should pick the stage, not where the hand is on screen
EXCLUDE_FEATURES = {"left_pos_x", "left_pos_y", "right_pos_x", "right_pos_y"}

SHARPNESS = 5.0
EPS = 1e-4


class GrowthEstimator:
    def __init__(self, poses_path=POSES_PATH):
        with open(poses_path) as f:
            raw_poses = json.load(f)

        first = next(iter(raw_poses.values()))[0]
        self.feature_names = sorted(n for n in first if n not in EXCLUDE_FEATURES)
        self.pose_samples = {
            stage: [[s[n] for n in self.feature_names] for s in samples]
            for stage, samples in raw_poses.items()
        }

    def estimate(self, feats):
        """feats: dict from hand_bridge.combine_hands(). Returns per-stage
        weights that sum to 1."""
        live = [feats.get(n, 0.0) for n in self.feature_names]
        weights = {
            stage: 1.0 / ((min(math.dist(live, s) for s in samples) + EPS) ** SHARPNESS)
            for stage, samples in self.pose_samples.items()
        }
        total = sum(weights.values())
        return {stage: w / total for stage, w in weights.items()}
