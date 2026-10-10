
import argparse
import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

HISTORY = 4
FUTURE = 6


def pose_matrix(pose):
    """T_global_from_ego, using nuScenes quaternion wxyz."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(
        pose["rotation"]
    ).rotation_matrix
    T[:3, 3] = pose["translation"]
    return T


def yaw_from_matrix(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap_angle(angle):
    return float(np.arctan2(
        np.sin(angle), np.cos(angle)
    ))


def get_ego_pose(nusc, sample):
    """Use LIDAR_TOP keyframe as the reference timestamp."""
    sd = nusc.get(
        "sample_data",
        sample["data"]["LIDAR_TOP"],
    )
    pose = nusc.get(
        "ego_pose",
        sd["ego_pose_token"],
    )
    return pose, int(sd["timestamp"])


def collect_sequence(nusc, sample, direction, count):
    """Follow sample links without crossing scene boundaries."""
    result = []
    current = sample

    for _ in range(count):
        token = current[direction]
        if not token:
            return None
        current = nusc.get("sample", token)
        result.append(current)

    return result


def relative_state(T_ref_inv, pose):
    T_rel = T_ref_inv @ pose_matrix(pose)
    return [
        float(T_rel[0, 3]),
        float(T_rel[1, 3]),
        wrap_angle(yaw_from_matrix(T_rel)),
    ]


def get_cameras(nusc, sample, dataroot):
    cameras = {}

    for name in CAMERAS:
        sd = nusc.get(
            "sample_data",
            sample["data"][name],
        )
        calib = nusc.get(
            "calibrated_sensor",
            sd["calibrated_sensor_token"],
        )

        path = Path(dataroot) / sd["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)

        T_ego_from_camera = pose_matrix({
            "translation": calib["translation"],
            "rotation": calib["rotation"],
        })

        cameras[name] = {
            "path": sd["filename"],
            "timestamp_us": int(sd["timestamp"]),
            "intrinsics": calib["camera_intrinsic"],
            "T_ego_from_camera": (
                T_ego_from_camera.tolist()
            ),
        }

    return cameras


def build_record(nusc, sample, scene_name, dataroot):
    # Require three previous and six future keyframes.
    past = collect_sequence(
        nusc, sample, "prev", HISTORY - 1
    )
    future = collect_sequence(
        nusc, sample, "next", FUTURE
    )

    if past is None or future is None:
        return None

    # Past arrives newest -> oldest.
    history_samples = list(reversed(past)) + [sample]

    ref_pose, ref_time = get_ego_pose(nusc, sample)
    T_ref_inv = np.linalg.inv(pose_matrix(ref_pose))

    history_poses = []
    history_times = []

    for s in history_samples:
        pose, timestamp = get_ego_pose(nusc, s)
        history_poses.append(pose)
        history_times.append(timestamp)

    future_poses = []
    future_times = []

    for s in future:
        pose, timestamp = get_ego_pose(nusc, s)
        future_poses.append(pose)
        future_times.append(timestamp)

    all_times = history_times + future_times
    dt = np.diff(all_times) / 1e6

    # nuScenes keyframes are approximately 2 Hz.
    # Reject unexpectedly large gaps or invalid ordering.
    if np.any(dt < 0.35) or np.any(dt > 0.65):
        return None

    history_xyz_yaw = [
        relative_state(T_ref_inv, pose)
        for pose in history_poses
    ]

    future_xyz_yaw = [
        relative_state(T_ref_inv, pose)
        for pose in future_poses
    ]

    # Causal speed estimate from previous history positions.
    # For the oldest point, use zero and mark speed invalid.
    history = []
    speed_valid = []

    for i, state in enumerate(history_xyz_yaw):
        if i == 0:
            speed = 0.0
            valid = False
        else:
            delta_xy = (
                np.array(history_xyz_yaw[i][:2])
                - np.array(history_xyz_yaw[i - 1][:2])
            )
            delta_t = (
                history_times[i] - history_times[i - 1]
            ) / 1e6

            speed = float(
                np.linalg.norm(delta_xy) / delta_t
            )
            valid = True

        history.append(state + [speed])
        speed_valid.append(valid)

    # Strong geometric and temporal invariants.
    assert np.allclose(
        history[-1][:3],
        [0.0, 0.0, 0.0],
        atol=1e-5,
    )
    assert len(future_xyz_yaw) == FUTURE
    assert future_times[0] > ref_time

    return {
        "sample_token": sample["token"],
        "scene_name": scene_name,
        "reference_timestamp_us": ref_time,
        "camera_order": CAMERAS,
        "cameras": get_cameras(
            nusc, sample, dataroot
        ),
        "ego_history": history,
        "ego_history_speed_valid": speed_valid,
        "history_time_offsets_sec": [
            (t - ref_time) / 1e6
            for t in history_times
        ],
        "future_trajectory": future_xyz_yaw,
        "future_time_offsets_sec": [
            (t - ref_time) / 1e6
            for t in future_times
        ],
        "trajectory_frame": "ego_at_reference",
        "trajectory_units": "meters_radians",
        "navigation_command": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        default="/home/leo/datasets/nuscenes",
    )
    parser.add_argument(
        "--output-dir",
        default="data/vla/nuscenes_mini",
    )
    args = parser.parse_args()

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=args.dataroot,
        verbose=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    splits = create_splits_scenes()
    mini_train = set(splits["mini_train"])
    mini_val = set(splits["mini_val"])

    records = {
        "train": [],
        "val": [],
    }
    skipped = 0

    for scene in nusc.scene:
        scene_name = scene["name"]

        if scene_name in mini_train:
            split = "train"
        elif scene_name in mini_val:
            split = "val"
        else:
            continue

        token = scene["first_sample_token"]

        while token:
            sample = nusc.get("sample", token)

            record = build_record(
                nusc,
                sample,
                scene_name,
                args.dataroot,
            )

            if record is not None:
                records[split].append(record)
            else:
                skipped += 1

            token = sample["next"]

    for split, items in records.items():
        path = output_dir / f"{split}.jsonl"

        with path.open(
            "w", encoding="utf-8"
        ) as f:
            for item in items:
                f.write(
                    json.dumps(
                        item, ensure_ascii=False
                    ) + "\n"
                )

        print(f"{split}: {len(items)} -> {path}")

    train_scenes = {
        r["scene_name"] for r in records["train"]
    }
    val_scenes = {
        r["scene_name"] for r in records["val"]
    }

    assert train_scenes.isdisjoint(val_scenes)
    assert records["train"], "No train samples"
    assert records["val"], "No val samples"

    print("Skipped boundary/gap samples:", skipped)
    print("Scene-disjoint split: PASS")
    print("Trajectory GT build: PASS")


if __name__ == "__main__":
    main()
