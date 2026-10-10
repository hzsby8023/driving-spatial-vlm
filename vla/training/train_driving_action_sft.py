"""M1 config-driven Qwen3-VL action-token QLoRA training with exact resume."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

from vla.config import apply_overrides, load_json_config
from vla.data.driving_action_sft import ACTION_CODEC, DrivingActionDataset, make_inputs
from vla.models.driving_action_vla import (
    load_base, load_train_model, make_processor, make_train_model, save_adapter, setup_vocab,
)

DEFAULT_CONFIG = {
    'schema_version': 1, 'run_name': 'driving_action_sft',
    'model': {'name_or_path': None, 'local_files_only': True, 'quantized': True,
              'lora_rank': 8, 'max_visual_tokens': 64},
    'data': {'train': 'outputs/autovla_codebook/train_sft.jsonl',
             'val': 'outputs/autovla_codebook/val_sft.jsonl', 'image_root': None,
             'cameras': ['CAM_FRONT'], 'frames_per_camera': 1, 'max_edge': 448,
             'codec': ACTION_CODEC, 'validate_images': False},
    'training': {'output_dir': 'outputs/driving_action_sft_v1', 'epochs': 2,
                 'gradient_accumulation_steps': 8, 'learning_rate': 2e-4,
                 'weight_decay': 0.01, 'warmup_steps': 0, 'max_grad_norm': 1.0,
                 'seed': 42, 'log_every_steps': 10, 'save_every_steps': 100,
                 'save_every_epoch': True, 'resume_from': None},
    'evaluation': {'teacher_forced_limit': 10, 'generation_limit': 55, 'codebook': None},
}


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def resume_signature(config):
    """Fields that must remain identical for an exact optimizer-state resume."""
    return {
        'model': config['model'], 'data': config['data'],
        'optimizer': {key: config['training'][key] for key in
                      ('epochs', 'gradient_accumulation_steps', 'learning_rate', 'weight_decay',
                       'warmup_steps', 'max_grad_norm', 'seed')},
    }


def save_checkpoint(model, processor, optimizer, scheduler, output_dir, progress, config):
    root = Path(output_dir) / 'checkpoints'; root.mkdir(parents=True, exist_ok=True)
    target = root / (f"step_{progress['global_step']:08d}_"
                     f"epoch_{progress['epoch']:03d}_batch_{progress['next_batch']:08d}")
    temporary = root / f".{target.name}.tmp"
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {target}')
    if temporary.exists():
        shutil.rmtree(temporary)
    save_adapter(model, processor, temporary, config)
    state = {**progress, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'resume_signature': resume_signature(config),
             'rng': capture_rng_state()}
    torch.save(state, temporary / 'trainer_state.pt')
    os.replace(temporary, target)
    latest_tmp = root / '.latest.json.tmp'
    latest_tmp.write_text(json.dumps({'checkpoint': str(target.resolve())}, indent=2), encoding='utf-8')
    os.replace(latest_tmp, root / 'latest.json')
    return target


def load_trainer_state(checkpoint, optimizer, scheduler, config):
    path = Path(checkpoint) / 'trainer_state.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing trainer state: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('resume_signature') != resume_signature(config):
        raise ValueError('Resume checkpoint is incompatible with the current model/data/optimizer config')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    restore_rng_state(state['rng'])
    return state


def append_metric(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/m1_qwen3_vl_2b_qlora.json')
    parser.add_argument('--model'); parser.add_argument('--train'); parser.add_argument('--val')
    parser.add_argument('--image-root'); parser.add_argument('--output'); parser.add_argument('--cameras')
    parser.add_argument('--frames-per-camera', type=int); parser.add_argument('--max-edge', type=int)
    parser.add_argument('--max-visual-tokens', type=int); parser.add_argument('--epochs', type=int)
    parser.add_argument('--grad-accum', type=int); parser.add_argument('--lr', type=float)
    parser.add_argument('--lora-rank', type=int); parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-limit', type=int); parser.add_argument('--resume-from')
    parser.add_argument('--no-quant', action='store_true', default=None)
    parser.add_argument('--allow-download', action='store_true', default=None)
    parser.add_argument('--save-every-epoch', action='store_true', default=None)
    return parser, parser.parse_args()


def build_config(parser, args):
    config = load_json_config(args.config, DEFAULT_CONFIG)
    cameras = [x.strip() for x in args.cameras.split(',') if x.strip()] if args.cameras else None
    config = apply_overrides(config, {
        'model.name_or_path': args.model, 'model.quantized': False if args.no_quant else None,
        'model.local_files_only': False if args.allow_download else None,
        'model.lora_rank': args.lora_rank, 'model.max_visual_tokens': args.max_visual_tokens,
        'data.train': args.train, 'data.val': args.val, 'data.image_root': args.image_root,
        'data.cameras': cameras, 'data.frames_per_camera': args.frames_per_camera,
        'data.max_edge': args.max_edge, 'training.output_dir': args.output,
        'training.epochs': args.epochs, 'training.gradient_accumulation_steps': args.grad_accum,
        'training.learning_rate': args.lr, 'training.seed': args.seed,
        'training.resume_from': args.resume_from, 'training.save_every_epoch': args.save_every_epoch,
        'evaluation.teacher_forced_limit': args.eval_limit,
    })
    if config['schema_version'] != 1: parser.error('only schema_version=1 is supported')
    if not config['model']['name_or_path'] or not config['data']['image_root']:
        parser.error('model.name_or_path and data.image_root are required')
    for key in ('epochs', 'gradient_accumulation_steps', 'log_every_steps'):
        if config['training'][key] < 1: parser.error(f'training.{key} must be >= 1')
    if config['data']['codec'] != ACTION_CODEC:
        parser.error(f"data.codec must be {ACTION_CODEC!r}")
    return config


def main():
    parser, args = parse_args(); config = build_config(parser, args)
    model_cfg, data_cfg, train_cfg, eval_cfg = (config[x] for x in ('model', 'data', 'training', 'evaluation'))
    seed = train_cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if not torch.cuda.is_available() and model_cfg['quantized']:
        raise RuntimeError('4-bit QLoRA requires CUDA; use --no-quant for CPU development')
    processor_source = str(Path(train_cfg['resume_from']) / 'processor') if train_cfg['resume_from'] else model_cfg['name_or_path']
    processor = make_processor(processor_source, model_cfg['local_files_only'], model_cfg['max_visual_tokens'])
    base = load_base(model_cfg['name_or_path'], model_cfg['local_files_only'], model_cfg['quantized'])
    ids = setup_vocab(processor, base)
    model = (load_train_model(base, ids, train_cfg['resume_from']) if train_cfg['resume_from']
             else make_train_model(base, ids, model_cfg['lora_rank']))
    device = model.get_input_embeddings().weight.device
    dataset_args = (data_cfg['image_root'], data_cfg['frames_per_camera'], data_cfg['max_edge'],
                    data_cfg['cameras'], data_cfg['codec'])
    train = DrivingActionDataset(data_cfg['train'], *dataset_args)
    val = DrivingActionDataset(data_cfg['val'], *dataset_args)
    overlap = {row['scene_token'] for row in train.rows} & {row['scene_token'] for row in val.rows}
    if overlap: raise ValueError(f'train/val scene leakage: {sorted(overlap)[:10]}')
    steps_per_epoch = math.ceil(len(train) / train_cfg['gradient_accumulation_steps'])
    total_steps = steps_per_epoch * train_cfg['epochs']
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    warmup = train_cfg['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(1, warmup),
                                    max(0.0, (total_steps - step) / max(1, total_steps - warmup))))
    start_epoch = start_batch = global_step = 0
    if train_cfg['resume_from']:
        state = load_trainer_state(train_cfg['resume_from'], optimizer, scheduler, config)
        start_epoch, start_batch, global_step = state['epoch'], state['next_batch'], state['global_step']
    out = Path(train_cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    (out / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    metrics_path = out / 'metrics.jsonl'; optimizer.zero_grad(set_to_none=True)
    print(json.dumps({'event': 'start', 'run': config['run_name'], 'samples': len(train),
                      'codec': ACTION_CODEC, 'resume_step': global_step}), flush=True)
    for epoch in range(start_epoch, train_cfg['epochs']):
        order = np.random.default_rng(seed + epoch).permutation(len(train)).tolist()
        batch_offset = start_batch if epoch == start_epoch else 0
        model.train(); accumulated = 0; running_loss = 0.0
        for position in range(batch_offset, len(order)):
            loss = model(**make_inputs(processor, train[order[position]], device, train=True), use_cache=False).loss
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at epoch={epoch} batch={position}')
            (loss / train_cfg['gradient_accumulation_steps']).backward()
            accumulated += 1; running_loss += float(loss.detach())
            boundary = accumulated == train_cfg['gradient_accumulation_steps'] or position + 1 == len(order)
            if not boundary: continue
            torch.nn.utils.clip_grad_norm_(params, train_cfg['max_grad_norm'])
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            record = {'event': 'train', 'time': time.time(), 'epoch': epoch + 1,
                      'global_step': global_step, 'next_batch': position + 1,
                      'loss': running_loss / accumulated, 'lr': scheduler.get_last_lr()[0]}
            append_metric(metrics_path, record)
            if global_step % train_cfg['log_every_steps'] == 0: print(json.dumps(record), flush=True)
            progress = {'epoch': epoch, 'next_batch': position + 1, 'global_step': global_step}
            if train_cfg['save_every_steps'] and global_step % train_cfg['save_every_steps'] == 0:
                save_checkpoint(model, processor, optimizer, scheduler, out, progress, config)
            accumulated = 0; running_loss = 0.0
        model.eval(); losses = []
        with torch.no_grad():
            for index in range(min(len(val), eval_cfg['teacher_forced_limit'])):
                losses.append(float(model(**make_inputs(processor, val[index], device, train=True), use_cache=False).loss))
        eval_record = {'event': 'eval', 'time': time.time(), 'epoch': epoch + 1,
                       'global_step': global_step, 'val_teacher_forced_loss': float(np.mean(losses)) if losses else None}
        append_metric(metrics_path, eval_record); print(json.dumps(eval_record), flush=True)
        if train_cfg['save_every_epoch']:
            save_checkpoint(model, processor, optimizer, scheduler, out,
                            {'epoch': epoch + 1, 'next_batch': 0, 'global_step': global_step}, config)
        start_batch = 0
    save_adapter(model, processor, out / 'final', config)
    print(f'Saved {out}/final; run free-generation evaluation separately.', flush=True)


if __name__ == '__main__':
    main()
