#!/usr/bin/env python3
"""Join AutoVLA codebook bridge labels back to original multimodal nuScenes JSONL.

Labels from check_codebook.py are matched by 1-based source_line; strict checks
prevent accidental pairing with a different input file.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield n, json.loads(line)


def merge(source, labels, output, horizon=10, vocab_size=2048):
    rows = list(read_jsonl(source))
    acts = list(read_jsonl(labels))
    if len(rows) != len(acts):
        raise ValueError(f'Row count mismatch: {len(rows)} vs {len(acts)}')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for (source_line, row), (_, a) in zip(rows, acts):
            if a.get('source_line') != source_line:
                raise ValueError(f'source_line mismatch at {source_line}')
            source_id = row.get('sample_token', row.get('token'))
            if source_id != a.get('sample_token'):
                raise ValueError(f'sample_token mismatch at line {source_line}')
            tokens = a['action_tokens']
            if len(tokens) != horizon or any(type(i) is not int or i < 0 or i >= vocab_size for i in tokens):
                raise ValueError(f'Invalid action token sequence at line {source_line}')
            if len(row['future_trajectory']) != horizon:
                raise ValueError(f'Future trajectory horizon mismatch at line {source_line}')
            row['action_tokens'] = tokens
            row['action_codec'] = 'greedy_endpoint_proxy_v1'
            row['action_token_source'] = 'autovla_2048_greedy_endpoint_proxy'
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Merged {len(rows)} samples into {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--vocab-size', type=int, default=2048)
    args = p.parse_args()
    merge(args.source, args.labels, args.output, args.horizon, args.vocab_size)
