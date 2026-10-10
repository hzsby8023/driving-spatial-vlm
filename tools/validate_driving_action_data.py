#!/usr/bin/env python3
"""Fail-fast validator for M1 action-token JSONL files."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla.data.driving_action_sft import ACTION_CODEC, read_rows, resolve_image


def validate(path, image_root=None, check_images=False, expected_codec=ACTION_CODEC):
    rows = read_rows(path, expected_codec=expected_codec)
    scenes = {row['scene_token'] for row in rows}
    tokens = Counter(token for row in rows for token in row['action_tokens'])
    if check_images:
        if not image_root:
            raise ValueError('--image-root is required with --check-images')
        for row in rows:
            for camera in row['camera_order']:
                for frame in row['camera_frames'][camera]:
                    resolve_image(frame['path'], image_root)
    return {
        'path': str(path), 'samples': len(rows), 'scenes': len(scenes),
        'codec': expected_codec, 'action_vocab_size_observed': len(tokens),
        'most_common_action_fraction': tokens.most_common(1)[0][1] / sum(tokens.values()),
        'images_checked': bool(check_images),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--image-root')
    parser.add_argument('--check-images', action='store_true')
    parser.add_argument('--expected-codec', default=ACTION_CODEC)
    args = parser.parse_args()
    reports = [validate(path, args.image_root, args.check_images, args.expected_codec) for path in args.paths]
    scene_sets = [{row['scene_token'] for row in read_rows(path)} for path in args.paths]
    for left in range(len(scene_sets)):
        for right in range(left + 1, len(scene_sets)):
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                raise ValueError(
                    f'scene leakage between {args.paths[left]} and {args.paths[right]}: '
                    f'{sorted(overlap)[:10]}'
                )
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
