"""JSONL -> Qwen3-VL multimodal chat + causal action labels."""
import json
import math
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

NUM_CODES = 2048
ACTION_CODEC = 'greedy_endpoint_proxy_v1'
ACTION_HORIZON = 10

def action_name(i):
    i = int(i)
    if not 0 <= i < NUM_CODES:
        raise ValueError(f'action id out of range: {i}')
    return f'<action_{i}>'

def resolve_image(path, image_root):
    path = Path(path)
    if path.is_file():
        return path
    # Original JSONL contains absolute WSL paths, Docker mount is different.
    s = str(path).replace('\\', '/')
    for marker in ('/samples/', '/sweeps/'):
        if marker in s:
            candidate = Path(image_root) / s.split(marker, 1)[1]
            candidate = Path(str(image_root).rstrip('/')) / marker.strip('/') / s.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f'Cannot resolve {path} under {image_root}')

def _finite_sequence(value, length, name, row_index):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'row {row_index}: {name} must contain {length} items')
    for item in value:
        values = item if isinstance(item, list) else [item]
        if not values or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError(f'row {row_index}: {name} must contain finite numbers')


def validate_row(row, idx, expected_codec=ACTION_CODEC):
    if not isinstance(row, dict):
        raise ValueError(f'row {idx}: expected a JSON object')
    for key in ('sample_token', 'scene_token', 'camera_order', 'camera_frames',
                'reference_timestamp_us', 'ego_history', 'vehicle_status',
                'future_trajectory', 'action_tokens'):
        if key not in row:
            raise ValueError(f'row {idx}: missing required field {key}')
    if not isinstance(row['sample_token'], str) or not row['sample_token']:
        raise ValueError(f'row {idx}: sample_token must be a non-empty string')
    if not isinstance(row['scene_token'], str) or not row['scene_token']:
        raise ValueError(f'row {idx}: scene_token must be a non-empty string')
    acts = row['action_tokens']
    if not isinstance(acts, list) or len(acts) != ACTION_HORIZON:
        raise ValueError(f'row {idx}: expected {ACTION_HORIZON} action_tokens')
    if any(type(token) is not int for token in acts):
        raise ValueError(f'row {idx}: action_tokens must be integers')
    for token in acts:
        action_name(token)
    codec = row.get('action_codec') or row.get('action_token_source')
    accepted_legacy = 'autovla_2048_greedy_endpoint_proxy'
    if codec not in (None, expected_codec, accepted_legacy):
        raise ValueError(f'row {idx}: action codec {codec!r} does not match {expected_codec!r}')
    if row.get('navigation_source') == 'future_gt_heuristic':
        raise ValueError(f'row {idx}: future-GT navigation leakage detected')
    _finite_sequence(row['ego_history'], 4, 'ego_history', idx)
    _finite_sequence(row['future_trajectory'], ACTION_HORIZON, 'future_trajectory', idx)
    order = row['camera_order']
    frames = row['camera_frames']
    if not isinstance(order, list) or not order or len(order) != len(set(order)):
        raise ValueError(f'row {idx}: camera_order must be a non-empty unique list')
    if not isinstance(frames, dict):
        raise ValueError(f'row {idx}: camera_frames must be an object')
    for camera in order:
        camera_rows = frames.get(camera)
        if not isinstance(camera_rows, list) or not camera_rows:
            raise ValueError(f'row {idx}: no frames for camera {camera}')
        timestamps = [frame.get('timestamp_us') for frame in camera_rows]
        if any(not isinstance(ts, (int, float)) for ts in timestamps) or timestamps != sorted(timestamps):
            raise ValueError(f'row {idx}: camera {camera} timestamps must be numeric and chronological')
        if any(not isinstance(frame.get('path'), str) or not frame['path'] for frame in camera_rows):
            raise ValueError(f'row {idx}: camera {camera} contains an invalid path')


def read_rows(path, expected_codec=ACTION_CODEC):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('Empty JSONL: '+str(path))
    seen = set()
    for idx, row in enumerate(rows):
        validate_row(row, idx, expected_codec)
        if row['sample_token'] in seen:
            raise ValueError(f"row {idx}: duplicate sample_token {row['sample_token']}")
        seen.add(row['sample_token'])
    return rows

class DrivingActionDataset(Dataset):
    def __init__(self, path, image_root, frames_per_camera=1, max_edge=448, cameras=None,
                 expected_codec=ACTION_CODEC):
        self.rows = read_rows(path, expected_codec=expected_codec)
        self.image_root = image_root
        self.frames_per_camera = frames_per_camera
        self.max_edge = max_edge
        self.cameras = list(cameras) if cameras else None
        if frames_per_camera not in (1, 2, 3, 4):
            raise ValueError('frames_per_camera must be 1..4')
        if self.cameras:
            missing = [
                name for name in self.cameras
                if any(name not in row['camera_frames'] for row in self.rows)
            ]
            if missing:
                raise ValueError(f'cameras missing from JSONL: {missing}')

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        contents = []
        # Frame ordering is chronological within each camera. Keep camera order fixed.
        camera_order = self.cameras or row['camera_order']
        for cam in camera_order:
            frames = row['camera_frames'][cam][-self.frames_per_camera:]
            for fr in frames:
                p = resolve_image(fr['path'], self.image_root)
                with Image.open(p) as img:
                    img = ImageOps.exif_transpose(img).convert('RGB')
                    if self.max_edge:
                        img.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
                    contents.append({'type':'text', 'text':f'Camera {cam}, history offset {(fr["timestamp_us"]-row["reference_timestamp_us"])/1e6:+.3f} seconds:'})
                    contents.append({'type':'image', 'image':img.copy()})
        v = row['vehicle_status']
        history = row['ego_history']
        # Numeric ego-history is part of prompt, never GT future.
        ego_str = '; '.join(','.join(f'{float(x):.3f}' for x in step) for step in history)
        nav = row.get('navigation_command', 'UNKNOWN')
        contents.append({'type':'text', 'text':(
            f'Current speed: {float(v["speed_mps"]):.3f} m/s; acceleration: '
            f'{float(v["acceleration_mps2"]):.3f} m/s^2. '
            f'Navigation: {nav}. Ego history [x,y,yaw,...] (chronological): {ego_str}. '
            'Predict 10 action tokens for the next 5 seconds, exactly one token per 0.5 seconds. '
            'Output action tokens only.'
        )})
        user = [{'role':'system','content':'You are an autonomous driving action prediction model.'},
                {'role':'user','content':contents}]
        # No whitespace separators: 10 action IDs should decode as 10 consecutive tokens.
        answer = ''.join(action_name(a) for a in row['action_tokens'])
        return {'messages':user, 'answer':answer, 'row':row}

def make_inputs(processor, example, device, train=True):
    """Construct multimodal labels by masking the exact assistant-prefix tokens."""
    prefix_messages = example['messages']
    full_messages = prefix_messages + [{'role':'assistant','content':example['answer']}]
    common = dict(tokenize=True, return_dict=True, return_tensors='pt')
    prefix = processor.apply_chat_template(prefix_messages, add_generation_prompt=True, **common)
    if train:
        full = processor.apply_chat_template(full_messages, add_generation_prompt=False, **common)
        p = prefix['input_ids'][0]
        f = full['input_ids'][0]
        if f.shape[0] <= p.shape[0] or not torch.equal(f[:len(p)], p):
            raise RuntimeError('Chat-template prefix mismatch: refusing to train on misaligned labels')
        labels = f.clone()
        labels[:len(p)] = -100
        full['labels'] = labels.unsqueeze(0)
        batch = full
    else:
        batch = prefix
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
