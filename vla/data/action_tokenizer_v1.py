"""AutoVLA-inspired, NOT official K-Disk, local SE(2) action tokenizer.

Train-only k-means codebook. No Qwen/model/network dependencies.
Usage:
  python action_tokenizer_v1.py fit --train .../train.jsonl --val .../val.jsonl --output-dir outputs/action_v1 --k 32
  python action_tokenizer_v1.py encode --input .../train.jsonl --codebook outputs/action_v1/codebook.npz --output outputs/action_v1/train_actions.jsonl
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

NOMINAL_DT = 0.5
# Fixed, interpretable scales for weighted k-means (m, m, rad).
SCALES = np.array([2.0, 1.0, 0.2], dtype=np.float64)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def to_local_actions(trajectory, times, nominal_dt=NOMINAL_DT):
    """Reference-frame future poses -> consecutive ego-local motions, normalized to 0.5s.

    pose[0] is implicit [0,0,0]; per-step local xy is rotated by -previous yaw.
    """
    poses = np.asarray(trajectory, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or t.shape != (len(poses),):
        raise ValueError("Expected trajectory [T,3], times [T]")
    dt = np.diff(np.concatenate([[0.0], t]))
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("Invalid poses or time intervals")
    prev = np.vstack([np.zeros((1, 3)), poses[:-1]])
    dx = poses[:, 0] - prev[:, 0]
    dy = poses[:, 1] - prev[:, 1]
    c, s = np.cos(prev[:, 2]), np.sin(prev[:, 2])
    local = np.column_stack([c * dx + s * dy, -s * dx + c * dy,
                             wrap(poses[:, 2] - prev[:, 2])])
    # Normalize varying sample intervals to comparable nominal-duration motions.
    return local * (nominal_dt / dt[:, None])


def from_local_actions(actions, times, nominal_dt=NOMINAL_DT):
    """Inverse SE(2) composition, including true nonuniform sample intervals."""
    acts = np.asarray(actions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != 3 or t.shape != (len(acts),):
        raise ValueError("Expected actions [T,3], times [T]")
    dt = np.diff(np.concatenate([[0.0], t]))
    if np.any(dt <= 0):
        raise ValueError("Nonpositive time interval")
    actual = acts * (dt[:, None] / nominal_dt)
    out = np.zeros_like(actual)
    x = y = yaw = 0.0
    for i, (lx, ly, dtheta) in enumerate(actual):
        c, s = math.cos(yaw), math.sin(yaw)
        x += c * lx - s * ly
        y += s * lx + c * ly
        yaw = float(wrap(yaw + dtheta))
        out[i] = [x, y, yaw]
    return out


def kmeans(x, k, seed=42, max_iter=100):
    """Numpy k-means++ with reproducible initialization and empty-cluster recovery."""
    if not 1 <= k <= len(x):
        raise ValueError(f"k={k} must be in [1,{len(x)}]")
    rng = np.random.default_rng(seed)
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(len(x))]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = d2.sum()
        idx = rng.choice(len(x), p=d2 / total) if total > 1e-15 else rng.integers(len(x))
        centers[i] = x[idx]
        d2 = np.minimum(d2, np.sum((x - centers[i]) ** 2, axis=1))
    for _ in range(max_iter):
        dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        ids = np.argmin(dist, axis=1)
        updated = centers.copy()
        counts = np.bincount(ids, minlength=k)
        for j in range(k):
            if counts[j]:
                updated[j] = x[ids == j].mean(axis=0)
            else:
                # Recover unused center at a poorly represented point.
                updated[j] = x[np.argmax(np.min(dist, axis=1))]
        shift = np.max(np.abs(updated - centers))
        centers = updated
        if shift < 1e-8:
            break
    return centers


class ActionTokenizer:
    def __init__(self, centers, scales=SCALES, nominal_dt=NOMINAL_DT):
        self.centers = np.asarray(centers, dtype=np.float64)  # physical units at nominal_dt
        self.scales = np.asarray(scales, dtype=np.float64)
        self.nominal_dt = float(nominal_dt)
        assert self.centers.ndim == 2 and self.centers.shape[1] == 3

    def encode(self, trajectory, times):
        actions = to_local_actions(trajectory, times, self.nominal_dt)
        dist = np.sum(((actions[:, None, :] - self.centers[None, :, :]) /
                       self.scales[None, None, :]) ** 2, axis=-1)
        return np.argmin(dist, axis=1).astype(int)

    def decode(self, ids, times):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self.centers)):
            raise ValueError("Action ID out of range")
        return from_local_actions(self.centers[ids], times, self.nominal_dt)

    def save(self, path):
        np.savez(path, centers=self.centers, scales=self.scales,
                 nominal_dt=np.array(self.nominal_dt))

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(z["centers"], z["scales"], float(z["nominal_dt"]))


def evaluate(rows, tokenizer):
    ade, fde, heading, all_ids = [], [], [], []
    for r in rows:
        gt = np.asarray(r["future_trajectory"], dtype=np.float64)
        t = r["future_time_offsets_sec"]
        ids = tokenizer.encode(gt, t)
        pred = tokenizer.decode(ids, t)
        dist = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
        ade.append(dist.mean())
        fde.append(dist[-1])
        heading.append(np.abs(wrap(pred[:, 2] - gt[:, 2])).mean())
        all_ids.extend(ids.tolist())
    if not ade:
        raise ValueError("Empty evaluation split")
    counts = np.bincount(all_ids, minlength=len(tokenizer.centers))
    return {"samples": len(ade), "ADE_m": float(np.mean(ade)),
            "FDE_m": float(np.mean(fde)), "yaw_MAE_rad": float(np.mean(heading)),
            "used_codes": int(np.count_nonzero(counts)),
            "codebook_size": int(len(counts)),
            "top_code_fraction": float(counts.max() / counts.sum())}


def cmd_fit(args):
    train = list(read_jsonl(args.train))
    val = list(read_jsonl(args.val))
    if not train or not val:
        raise ValueError("Both splits must contain samples")
    train_scenes = {r["scene_token"] for r in train}
    val_scenes = {r["scene_token"] for r in val}
    if train_scenes & val_scenes:
        raise ValueError("Train/val scene overlap")
    acts = np.concatenate([to_local_actions(r["future_trajectory"],
                                            r["future_time_offsets_sec"]) for r in train])
    centers = kmeans(acts / SCALES, args.k, args.seed) * SCALES
    tok = ActionTokenizer(centers)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "codebook.npz")
    metrics = {"train": evaluate(train, tok), "val": evaluate(val, tok),
               "fit_actions": len(acts), "seed": args.seed,
               "method": "weighted_kmeans_local_se2_not_official_autovla_kdisk"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved:", out / "codebook.npz")


def cmd_encode(args):
    tok = ActionTokenizer.load(args.codebook)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in read_jsonl(args.input):
            ids = tok.encode(r["future_trajectory"], r["future_time_offsets_sec"])
            r["action_tokens"] = ids.tolist()
            r["action_tokenizer"] = "weighted_kmeans_local_se2_v1"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"Encoded {n} samples -> {out}")


def self_test():
    times = np.array([0.5, 1.01, 1.49, 2.02])
    for gt in (np.array([[2, 0, 0], [4, 0, 0], [6, 0, 0], [8, 0, 0]]),
               np.array([[1, 0.1, 0.1], [2, 0.4, 0.2], [3, 0.9, 0.3], [4, 1.6, 0.4]]),
               np.zeros((4, 3))):
        rebuilt = from_local_actions(to_local_actions(gt, times), times)
        assert np.allclose(gt, rebuilt, atol=1e-9), (gt, rebuilt)
    print("SE(2) exact inverse tests: PASS")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--train", required=True)
    fit.add_argument("--val", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--k", type=int, default=32)
    fit.add_argument("--seed", type=int, default=42)
    enc = sub.add_parser("encode")
    enc.add_argument("--input", required=True)
    enc.add_argument("--codebook", required=True)
    enc.add_argument("--output", required=True)
    sub.add_parser("self-test")
    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "encode":
        cmd_encode(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
