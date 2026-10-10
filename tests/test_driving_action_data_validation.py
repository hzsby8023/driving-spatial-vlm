import json

import pytest

from vla.data.driving_action_sft import ACTION_CODEC, read_rows


def valid_row():
    frames = [{'path': '/samples/CAM_FRONT/a.jpg', 'timestamp_us': i} for i in range(4)]
    return {
        'sample_token': 'sample', 'scene_token': 'scene', 'reference_timestamp_us': 3,
        'camera_order': ['CAM_FRONT'], 'camera_frames': {'CAM_FRONT': frames},
        'ego_history': [[0.0, 0.0, 0.0, 1.0] for _ in range(4)],
        'vehicle_status': {'speed_mps': 1.0, 'acceleration_mps2': 0.0},
        'navigation_source': 'unavailable',
        'future_trajectory': [[float(i), 0.0, 0.0] for i in range(10)],
        'action_tokens': list(range(10)), 'action_codec': ACTION_CODEC,
    }


def write_row(tmp_path, row):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    return path


def test_valid_row(tmp_path):
    assert len(read_rows(write_row(tmp_path, valid_row()))) == 1


@pytest.mark.parametrize('mutate,match', [
    (lambda row: row.update(action_tokens=[2048] * 10), 'out of range'),
    (lambda row: row.update(navigation_source='future_gt_heuristic'), 'leakage'),
    (lambda row: row.update(action_codec='official_kdisk'), 'does not match'),
])
def test_invalid_rows(tmp_path, mutate, match):
    row = valid_row(); mutate(row)
    with pytest.raises(ValueError, match=match):
        read_rows(write_row(tmp_path, row))
