"""Free-running action generation -> decoded trajectories -> ADE/FDE."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from vla.config import load_json_config
from vla.data.driving_action_sft import ACTION_CODEC
from vla.data.driving_action_sft import DrivingActionDataset, make_inputs
from vla.models.driving_action_vla import load_for_inference
from transformers import LogitsProcessor, LogitsProcessorList


class ActionOnlyLogitsProcessor(LogitsProcessor):
    """
    Only allow the 2048 action tokens during generation.
    """

    def __init__(self, action_token_ids):
        self.action_token_ids = list(action_token_ids)

        if len(self.action_token_ids) != 2048:
            raise ValueError("Expected 2048 action token IDs")

        if len(set(self.action_token_ids)) != 2048:
            raise ValueError("Action token IDs must be unique")

    def __call__(self, input_ids, scores):
        masked_scores = torch.full_like(
            scores,
            float("-inf"),
        )

        masked_scores[:, self.action_token_ids] = (
            scores[:, self.action_token_ids]
        )

        return masked_scores

def wrap(a): return np.arctan2(np.sin(a), np.cos(a))

def prototypes_from_codebook(path):
    # Only load a trusted official pickle file.
    with open(path, 'rb') as f: raw = pickle.load(f)
    book = np.asarray(raw['token_all']['veh'], dtype=np.float64)
    if book.shape != (2048,6,4,2): raise ValueError(f'Unexpected codebook shape: {book.shape}')
    end = book[:,-1].mean(axis=1)
    front = book[:,-1,0] - book[:,-1,3]
    yaw = np.arctan2(front[:,1], front[:,0])
    return np.column_stack((end,yaw))

def rollout(actions, prototypes):
    state = np.zeros(3, dtype=np.float64)
    result=[]
    for a in actions:
        dx,dy,dyaw = prototypes[int(a)]
        c,s=np.cos(state[2]),np.sin(state[2])
        state = np.array([state[0]+c*dx-s*dy, state[1]+s*dx+c*dy, wrap(state[2]+dyaw)])
        result.append(state.copy())
    return np.asarray(result)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=None)
    ap.add_argument('--checkpoint',required=True,help='.../final with adapter/ processor/')
    ap.add_argument('--model')
    ap.add_argument('--codebook')
    ap.add_argument('--data')
    ap.add_argument('--image-root')
    ap.add_argument('--frames-per-camera',type=int)
    ap.add_argument('--cameras',default=None,
                    help='Comma-separated cameras. Defaults to the checkpoint training cameras.')
    ap.add_argument('--max-edge',type=int)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--output',default='outputs/driving_action_sft_v1/free_generation_val.json')
    ap.add_argument('--no-quant',action='store_true')
    args=ap.parse_args()
    if args.config:
        from vla.training.train_driving_action_sft import DEFAULT_CONFIG
        cfg = load_json_config(args.config, DEFAULT_CONFIG)
        args.model = args.model or cfg['model']['name_or_path']
        args.codebook = args.codebook or cfg['evaluation']['codebook']
        args.data = args.data or cfg['data']['val']
        args.image_root = args.image_root or cfg['data']['image_root']
        args.frames_per_camera = args.frames_per_camera or cfg['data']['frames_per_camera']
        args.max_edge = args.max_edge or cfg['data']['max_edge']
        args.limit = args.limit or cfg['evaluation']['generation_limit']
    args.data = args.data or 'outputs/autovla_codebook/val_sft.jsonl'
    args.frames_per_camera = args.frames_per_camera or 1
    args.max_edge = args.max_edge or 448
    args.limit = args.limit or 55
    if not all((args.model, args.codebook, args.image_root)):
        ap.error('--model, --codebook and --image-root are required (directly or via --config)')
    meta_path = Path(args.checkpoint) / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if args.cameras:
        cameras = [name.strip() for name in args.cameras.split(',') if name.strip()]
    else:
        cameras = meta.get('cameras')
    model,processor,ids=load_for_inference(args.checkpoint,args.model,quantized=not args.no_quant)
    dev=model.get_input_embeddings().weight.device
    ds=DrivingActionDataset(args.data,args.image_root,args.frames_per_camera,args.max_edge,cameras)
    proto=prototypes_from_codebook(args.codebook)
    # Same val slice as the model loop. Action 0 is the most common train token.
    zero_trajectory = rollout([0] * 10, proto)
    zero_ades = []
    zero_fdes = []
    for j in range(min(len(ds), args.limit)):
        gt = np.asarray(ds[j]["row"]["future_trajectory"], dtype=np.float64)
        dist = np.linalg.norm(zero_trajectory[:, :2] - gt[:, :2], axis=1)
        zero_ades.append(float(dist.mean()))
        zero_fdes.append(float(dist[-1]))
    always0_ade = float(np.mean(zero_ades))
    always0_fde = float(np.mean(zero_fdes))
    print("Always-0 ADE:", always0_ade, flush=True)
    print("Always-0 FDE:", always0_fde, flush=True)
    inv={tid:i for i,tid in enumerate(ids)}
    records=[]; ade=[]; fde=[]; exact=[]
    print("Action 0 token ID:", ids[0])
    print(
        "Action 0 token:",
        processor.tokenizer.convert_ids_to_tokens(ids[0])
    )
    print(
        "Generated token 151669:",
        processor.tokenizer.convert_ids_to_tokens(151669)
    )
    print(
        "Generated token 151645:",
        processor.tokenizer.convert_ids_to_tokens(151645)
    )
    for j in range(min(len(ds),args.limit)):
        ex=ds[j]; batch=make_inputs(processor,ex,dev,train=False)
        action_processor = LogitsProcessorList([
            ActionOnlyLogitsProcessor(ids)
        ])

        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=10,
                min_new_tokens=10,
                do_sample=False,
                logits_processor=action_processor,
                eos_token_id=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
            )

        new_ids = generated[
            0, batch["input_ids"].shape[1]:
        ].tolist()

        actions = [inv[t] for t in new_ids]

        ok = len(actions) == 10

        print(
            "generated actions:",
            actions,
            flush=True,
        )
        row={'sample_token':ex['row'].get('sample_token'),'generated_action_ids':actions,
             'generated_token_ids':new_ids,'valid_10_actions':ok}
        if ok:
            pred=rollout(actions,proto)
            gt=np.asarray(ex['row']['future_trajectory'],dtype=np.float64)
            d=np.linalg.norm(pred[:,:2]-gt[:,:2],axis=1)
            row.update({'ADE_m':float(d.mean()),'FDE_m':float(d[-1]),
                        'token_accuracy':float(np.mean(np.asarray(actions)==np.asarray(ex['row']['action_tokens']))),
                        'trajectory':pred.tolist()})
            ade.append(row['ADE_m']); fde.append(row['FDE_m']); exact.append(row['token_accuracy'])
        records.append(row)
        print(f'eval {j+1}/{min(len(ds),args.limit)} valid={ok} ADE={row.get("ADE_m")}',flush=True)
    report={'total':len(records),'valid_count':len(ade),'valid_rate':len(ade)/len(records),
            'action_codec':ACTION_CODEC,
            'always0_ADE_m':always0_ade,
            'always0_FDE_m':always0_fde,
            'ADE_m_valid_only':float(np.mean(ade)) if ade else None,
            'FDE_m_valid_only':float(np.mean(fde)) if fde else None,
            'action_token_accuracy_valid_only':float(np.mean(exact)) if exact else None,
            'note':'Invalid generations excluded from ADE/FDE; report valid_rate alongside accuracy. greedy_endpoint_proxy_v1 is not the official K-Disk encoder.',
            'samples':records}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
