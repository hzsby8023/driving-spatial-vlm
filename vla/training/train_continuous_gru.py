"""Qwen3-VL frozen visual features + trainable continuous autoregressive GRU."""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from torch.utils.data import Dataset
from vla.models.continuous_gru import ContinuousGRU, pose_to_local, trajectory_loss, cv_trajectory

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


class DrivingDataset(Dataset):
    def __init__(self, manifest, image_root, frames):
        self.rows = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
        assert self.rows, f'Empty manifest: {manifest}'
        assert 1 <= frames <= 4
        self.root, self.frames = Path(image_root), frames

    def __len__(self):
        return len(self.rows)

    def resolve(self, p):
        path = Path(p)
        if path.is_file():
            return path
        for part in ('samples', 'sweeps'):
            if part in path.parts:
                candidate = self.root.joinpath(*path.parts[path.parts.index(part):])
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f'{p}; root={self.root}')

    def __getitem__(self, idx):
        from PIL import Image
        r = self.rows[idx]
        images, labels = [], []
        for cam in CAMERAS:
            for f in r['camera_frames'][cam][-self.frames:]:
                with Image.open(self.resolve(f['path'])) as im:
                    images.append(im.convert('RGB'))
                labels.append(f"{cam}@{(f['timestamp_us']-r['reference_timestamp_us'])/1e6:+.2f}s")
        # Never put future_trajectory or oracle_command into the prompt.
        prompt = ('Predict a safe future ego trajectory using camera observations and past ego state. '
                  'Driving command: ' + r['navigation_command'] + '\n'
                  'Cameras: ' + ', '.join(labels) + '\n'
                  'Speed m/s: ' + str(round(r['vehicle_status']['speed_mps'], 3)) + '\n'
                  'Past ego states: ' + json.dumps(r['ego_history']))
        return {'images': images, 'prompt': prompt, 'token': r['sample_token'],
                'history': torch.tensor(r['ego_history'], dtype=torch.float32)[None],
                'ht': torch.tensor(r['history_time_offsets_sec'], dtype=torch.float32)[None],
                'ft': torch.tensor(r['future_time_offsets_sec'], dtype=torch.float32)[None],
                'gt': torch.tensor(r['future_trajectory'], dtype=torch.float32)[None]}


class FrozenQwen:
    def __init__(self, name, max_visual_tokens):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(name, local_files_only=True)
        ip = self.processor.image_processor
        if hasattr(ip, 'max_pixels'):
            ip.max_pixels = max_visual_tokens * 32 * 32
        if hasattr(ip, 'min_pixels'):
            ip.min_pixels = 32 * 32
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            name, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True), device_map={'': 0},
            torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
        self.model.eval().requires_grad_(False)
        self.dim = self.model.config.text_config.hidden_size

    @torch.no_grad()
    def encode(self, item):
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': item['prompt']}] +
                     [{'type': 'image', 'image': im} for im in item['images']]}]
        inp = self.processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt')
        inp = {k: v.to('cuda:0') if hasattr(v, 'to') else v for k, v in inp.items()}
        # Access the base multimodal model to avoid materializing vocabulary-size logits.
        base = getattr(self.model, 'model', None)
        if base is None:
            raise RuntimeError('Qwen3VL model.model not found; check transformers version')
        result = base(**inp, return_dict=True)
        hidden = result.last_hidden_state
        return hidden[:, -1, :].float()


def stats(pred, gt):
    dist = torch.linalg.vector_norm(pred[..., :2] - gt[..., :2], dim=-1)
    return float(dist.mean()), float(dist[:, -1].mean())


def run_eval(net, encoder, ds, limit):
    net.eval()
    sums = np.zeros(4, dtype=np.float64)
    count = min(len(ds), limit)
    with torch.no_grad():
        for i in range(count):
            item = ds[i]
            v = encoder.encode(item)
            h, ht, ft, gt = (item[k].to('cuda') for k in ('history','ht','ft','gt'))
            pred, _, cv = net(v, h, ht, ft, teacher_forcing=0)
            sums += np.array([*stats(pred, gt), *stats(cv, gt)])
    net.train()
    return (sums / count).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/vla/nuscenes_autovla_v1/train.jsonl')
    p.add_argument('--val', default='data/vla/nuscenes_autovla_v1/val.jsonl')
    p.add_argument('--image-root', default='/datasets/nuscenes')
    p.add_argument('--model', default='Qwen/Qwen3-VL-2B-Instruct')
    p.add_argument('--frames-per-camera', type=int, default=1)
    p.add_argument('--max-visual-tokens', type=int, default=64)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--val-limit', type=int, default=5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--teacher-forcing', type=float, default=0.5)
    p.add_argument('--output-dir', default='outputs/continuous_gru_v1')
    a = p.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    train = DrivingDataset(a.train, a.image_root, a.frames_per_camera)
    val = DrivingDataset(a.val, a.image_root, a.frames_per_camera)
    encoder = FrozenQwen(a.model, a.max_visual_tokens)
    net = ContinuousGRU(encoder.dim).to('cuda')
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for step in range(1, a.steps + 1):
        item = train[(step-1) % len(train)]
        v = encoder.encode(item)
        h, ht, ft, gt = (item[k].to('cuda') for k in ('history','ht','ft','gt'))
        # Rollout loss is essential: teacher forcing alone hides inference drift.
        pred, local, cv = net(v, h, ht, ft, teacher_forcing=0)
        loss = trajectory_loss(pred, gt, local, pose_to_local(gt))
        if a.teacher_forcing > 0:
            pred_tf, local_tf, _ = net(v, h, ht, ft, gt=gt, teacher_forcing=a.teacher_forcing)
            loss = loss + 0.5 * trajectory_loss(pred_tf, gt, local_tf, pose_to_local(gt))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        print(f'step={step} loss={loss.item():.4f} pred_ADE={stats(pred.detach(),gt)[0]:.3f} cv_ADE={stats(cv,gt)[0]:.3f}', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            result = run_eval(net, encoder, val, a.val_limit)
            checkpoint = out / f'step_{step:05d}.pt'
            torch.save({'decoder': net.state_dict(), 'vision_dim': encoder.dim,
                        'config': vars(a), 'step': step}, checkpoint)
            report = dict(zip(('pred_ADE_m','pred_FDE_m','cv_ADE_m','cv_FDE_m'), result))
            (out / f'eval_{step:05d}.json').write_text(json.dumps(report, indent=2))
            print('validation:', json.dumps(report), 'saved:', checkpoint, flush=True)


if __name__ == '__main__':
    main()
