#!/usr/bin/env python3
"""Evaluate official AutoVLA vehicle codebook on existing nuScenes JSONL.

IMPORTANT: endpoint nearest-neighbor is a compatibility proxy, NOT the official
K-Disk footprint matching encoder. The rollout mirrors official endpoint decoding.
Only load pickle files obtained from a trusted source.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def load_book(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)  # Trusted official file ONLY.
    arr = np.asarray(obj['token_all']['veh'], dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1:] != (6, 4, 2) or not np.isfinite(arr).all():
        raise ValueError(f'Expected finite [K,6,4,2], got {arr.shape}')
    end = arr[:, -1].mean(axis=1)
    heading_vec = arr[:, -1, 0] - arr[:, -1, 3]
    heading = np.arctan2(heading_vec[:, 1], heading_vec[:, 0])
    return arr, np.column_stack((end, heading))


def local_to_global(delta, state):
    c, s = math.cos(state[2]), math.sin(state[2])
    dx, dy, dtheta = delta
    return np.array([state[0] + c*dx - s*dy,
                     state[1] + s*dx + c*dy,
                     wrap(state[2] + dtheta)], dtype=np.float64)


def relative(target, state):
    dx, dy = target[:2] - state[:2]
    c, s = math.cos(state[2]), math.sin(state[2])
    return np.array([c*dx + s*dy, -s*dx + c*dy,
                     wrap(target[2] - state[2])], dtype=np.float64)


def encode_rollout(gt, prototypes, yaw_weight):
    """Greedy endpoint matching relative to *decoded* current state."""
    state = np.zeros(3, dtype=np.float64)
    pred, tokens = [], []
    for target in gt:
        local_gt = relative(target, state)
        diff = prototypes[:, :2] - local_gt[:2]
        angle_diff = wrap(prototypes[:, 2] - local_gt[2])
        cost = np.sum(diff*diff, axis=1) + (yaw_weight*angle_diff)**2
        idx = int(np.argmin(cost))
        tokens.append(idx)
        state = local_to_global(prototypes[idx], state)
        pred.append(state.copy())
    return np.asarray(tokens), np.asarray(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codebook', required=True, help='Trusted official agent_vocab.pkl')
    ap.add_argument('--input', required=True, help='Existing nuScenes JSONL with future_trajectory')
    ap.add_argument('--output', required=True, help='Metrics JSON')
    ap.add_argument('--encoded-output', default=None, help='Optional encoded JSONL')
    ap.add_argument('--yaw-weight', type=float, default=2.0, help='Meters per radian for endpoint proxy matching')
    ap.add_argument('--expected-dt', type=float, default=0.5)
    ap.add_argument('--dt-tolerance', type=float, default=0.13)
    args = ap.parse_args()

    book, proto = load_book(args.codebook)
    ade, fde, yaw, all_codes, errors, dt_warnings = [], [], [], [], [], []
    out_rows = []
    with open(args.input, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            gt = np.asarray(row['future_trajectory'], dtype=np.float64)
            if gt.ndim != 2 or gt.shape[1] != 3 or not np.isfinite(gt).all():
                raise ValueError(f'line {line_no}: future_trajectory must be finite [T,3]')
            times = row.get('future_times')
            if times is not None:
                times = np.asarray(times, dtype=np.float64)
                if times.shape != (len(gt),):
                    raise ValueError(f'line {line_no}: future_times length mismatch')
                intervals = np.diff(np.r_[0.0, times])
                if np.any(np.abs(intervals - args.expected_dt) > args.dt_tolerance):
                    dt_warnings.append(line_no)
            ids, pred = encode_rollout(gt, proto, args.yaw_weight)
            dist = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
            ade.append(float(dist.mean()))
            fde.append(float(dist[-1]))
            yaw.append(float(np.abs(wrap(pred[:, 2] - gt[:, 2])).mean()))
            all_codes.extend(ids.tolist())
            errors.append((float(dist[-1]), line_no))
            if args.encoded_output:
                out_rows.append({'source_line': line_no,
                                 'sample_token': row.get('sample_token', row.get('token')),
                                 'action_tokens': ids.tolist(),
                                 'predicted_trajectory': pred.tolist(),
                                 'future_trajectory': gt.tolist()})
    if not ade:
        raise ValueError('No samples found')
    report = {
        'method': 'greedy_endpoint_proxy_NOT_official_KDisk_encoder',
        'official_rollout_compatible': True,
        'codebook_shape': list(book.shape),
        'samples': len(ade),
        'ADE_m': float(np.mean(ade)),
        'FDE_m': float(np.mean(fde)),
        'yaw_MAE_rad': float(np.mean(yaw)),
        'used_codes': len(set(all_codes)),
        'top_code_fraction': max(all_codes.count(k) for k in set(all_codes))/len(all_codes),
        'time_interval_warning_count': len(dt_warnings),
        'time_interval_warning_lines_first_10': dt_warnings[:10],
        'worst_FDE_lines_top_10': sorted(errors, reverse=True)[:10],
        'notes': 'This is an upper-bound proxy for greedy endpoint matching, not official training preprocessing. Inspect timing and coordinate conventions before SFT.'
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.encoded_output:
        Path(args.encoded_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.encoded_output, 'w', encoding='utf-8') as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
