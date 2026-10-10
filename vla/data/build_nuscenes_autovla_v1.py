
import argparse
import json
import math
from pathlib import Path

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)

HISTORY = 4
FUTURE = 10
EXPECTED_DT = 0.5
DT_TOLERANCE = 0.15


def load_table(root, name):
    path = root / "v1.0-mini" / f"{name}.json"
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["token"]: row for row in rows}


def attach_keyframe_data(
    samples,
    sample_data,
    calibrated_sensors,
    sensors,
):
    """
    Official sample rows map each channel to its keyframe
    sample_data token in a `data` field. This export omitted
    that field, so rebuild it from the keyframe rows.
    """
    for sample in samples.values():
        sample["data"] = {}

    for sd in sample_data.values():
        if not sd["is_key_frame"]:
            continue

        channel = sensors[
            calibrated_sensors[
                sd["calibrated_sensor_token"]
            ]["sensor_token"]
        ]["channel"]

        samples[sd["sample_token"]]["data"][channel] = sd["token"]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(q):
    # nuScenes quaternion convention: [w, x, y, z]
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def get_chain(sample_token, samples):
    """
    Four historical keyframes, including current,
    and ten future keyframes.

    The history is ordered oldest -> current.
    """
    current = samples[sample_token]
    scene_token = current["scene_token"]

    history = [current]
    node = current

    for _ in range(HISTORY - 1):
        prev_token = node["prev"]
        if not prev_token:
            return None

        node = samples[prev_token]
        if node["scene_token"] != scene_token:
            return None

        history.append(node)

    history.reverse()

    future = []
    node = current

    for _ in range(FUTURE):
        next_token = node["next"]
        if not next_token:
            return None

        node = samples[next_token]
        if node["scene_token"] != scene_token:
            return None

        future.append(node)

    return history, future


def get_ego_pose(sample, sample_data, ego_poses):
    # Use LIDAR_TOP keyframe as the reference ego pose.
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = sample_data[lidar_token]
    pose = ego_poses[sd["ego_pose_token"]]

    return {
        "timestamp_us": sd["timestamp"],
        "x": float(pose["translation"][0]),
        "y": float(pose["translation"][1]),
        "yaw": quaternion_yaw(pose["rotation"]),
    }


def relative_pose(pose, reference):
    """
    Transform a global ego pose into the current ego frame.
    x forward, y left, yaw counterclockwise.
    """
    dx = pose["x"] - reference["x"]
    dy = pose["y"] - reference["y"]

    c = math.cos(reference["yaw"])
    s = math.sin(reference["yaw"])

    x = c * dx + s * dy
    y = -s * dx + c * dy

    yaw = wrap_angle(pose["yaw"] - reference["yaw"])
    return [x, y, yaw]


def get_camera_frames(
    history,
    sample_data,
    calibrated_sensors,
    dataset_root,
):
    result = {}

    for camera in CAMERAS:
        frames = []

        for sample in history:
            sd_token = sample["data"][camera]
            sd = sample_data[sd_token]
            calib = calibrated_sensors[
                sd["calibrated_sensor_token"]
            ]

            path = dataset_root / sd["filename"]
            if not path.is_file():
                raise FileNotFoundError(str(path))

            frames.append({
                "path": str(path),
                "timestamp_us": sd["timestamp"],
                "intrinsic": calib["camera_intrinsic"],
                "translation": calib["translation"],
                "rotation": calib["rotation"],
            })

        result[camera] = frames

    return result


def get_speed(poses, index):
    # Finite-difference speed magnitude.
    if index == 0:
        a, b = poses[0], poses[1]
    else:
        a, b = poses[index - 1], poses[index]

    dt = (b["timestamp_us"] - a["timestamp_us"]) / 1e6
    if dt <= 0:
        raise ValueError("Non-positive ego pose time interval")

    return math.hypot(
        b["x"] - a["x"],
        b["y"] - a["y"],
    ) / dt


def make_oracle_command(future_trajectory, threshold=2.0):
    """
    Diagnostic label derived from future GT.
    NOT a deployable navigation input.
    """
    lateral = future_trajectory[-1][1]

    if lateral > threshold:
        return "TURN_LEFT"
    if lateral < -threshold:
        return "TURN_RIGHT"
    return "GO_STRAIGHT"


def check_timing(poses):
    # Only accept approximately 2 Hz consecutive keyframes.
    for a, b in zip(poses[:-1], poses[1:]):
        dt = (b["timestamp_us"] - a["timestamp_us"]) / 1e6
        if abs(dt - EXPECTED_DT) > DT_TOLERANCE:
            return False
    return True


def build_record(
    token,
    samples,
    sample_data,
    ego_poses,
    calibrated_sensors,
    dataset_root,
):
    chain = get_chain(token, samples)
    if chain is None:
        return None

    history, future = chain
    all_samples = history + future

    poses = [
        get_ego_pose(s, sample_data, ego_poses)
        for s in all_samples
    ]

    if not check_timing(poses):
        return None

    reference = poses[HISTORY - 1]
    history_poses = poses[:HISTORY]
    future_poses = poses[HISTORY:]

    history_times = [
        (p["timestamp_us"] - reference["timestamp_us"]) / 1e6
        for p in history_poses
    ]

    future_times = [
        (p["timestamp_us"] - reference["timestamp_us"]) / 1e6
        for p in future_poses
    ]

    ego_history = []

    for i, pose in enumerate(history_poses):
        x, y, yaw = relative_pose(pose, reference)
        speed = get_speed(history_poses, i)
        ego_history.append([x, y, yaw, speed])

    future_trajectory = [
        relative_pose(pose, reference)
        for pose in future_poses
    ]

    current_speed = ego_history[-1][3]
    previous_speed = ego_history[-2][3]
    dt = history_times[-1] - history_times[-2]

    acceleration = (
        (current_speed - previous_speed) / dt
        if dt > 0 else 0.0
    )

    camera_frames = get_camera_frames(
        history,
        sample_data,
        calibrated_sensors,
        dataset_root,
    )

    return {
        "sample_token": token,
        "scene_token": samples[token]["scene_token"],
        "reference_timestamp_us": reference["timestamp_us"],
        "camera_order": list(CAMERAS),
        "camera_frames": camera_frames,
        "ego_history": ego_history,
        "history_time_offsets_sec": history_times,
        "vehicle_status": {
            "speed_mps": current_speed,
            "acceleration_mps2": acceleration,
        },
        "navigation_command": "UNKNOWN",
        "navigation_source": "unavailable",
        "oracle_command": make_oracle_command(
            future_trajectory
        ),
        "oracle_command_source": "future_gt_heuristic",
        "future_trajectory": future_trajectory,
        "future_time_offsets_sec": future_times,
        "trajectory_frame": "ego_at_reference",
        "action_tokens": None,
        "reasoning": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/datasets/nuscenes",
    )
    parser.add_argument(
        "--input-dir",
        default="data/vla/nuscenes_mini",
    )
    parser.add_argument(
        "--output-dir",
        default="data/vla/nuscenes_autovla_v1",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_table(root, "sample")
    sample_data = load_table(root, "sample_data")
    ego_poses = load_table(root, "ego_pose")
    calibrated_sensors = load_table(
        root, "calibrated_sensor"
    )
    sensors = load_table(root, "sensor")
    attach_keyframe_data(
        samples,
        sample_data,
        calibrated_sensors,
        sensors,
    )

    scene_sets = {}

    for split in ("train", "val"):
        source = input_dir / f"{split}.jsonl"
        destination = output_dir / f"{split}.jsonl"

        count = 0
        skipped = 0
        scene_sets[split] = set()

        with destination.open("w", encoding="utf-8") as out:
            for item in read_jsonl(source):
                token = item["sample_token"]

                record = build_record(
                    token,
                    samples,
                    sample_data,
                    ego_poses,
                    calibrated_sensors,
                    root,
                )

                if record is None:
                    skipped += 1
                    continue

                out.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

                count += 1
                scene_sets[split].add(
                    record["scene_token"]
                )

        print(
            f"{split}: written={count}, "
            f"skipped={skipped}, "
            f"scenes={len(scene_sets[split])}"
        )

    overlap = scene_sets["train"] & scene_sets["val"]
    if overlap:
        raise RuntimeError(
            f"Scene split leakage: {overlap}"
        )

    print("Scene split: PASS")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
