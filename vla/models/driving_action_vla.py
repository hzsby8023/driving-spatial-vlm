"""Qwen3-VL QLoRA action LM. No regression or GRU head."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from vla.data.driving_action_sft import NUM_CODES, action_name


def make_processor(model_name, local_only=True, max_visual_tokens=64):
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only, trust_remote_code=False)
    # Qwen counts pixels, not the PIL side length. shortest_edge is a floor:
    # the default 65536 pixels upscales a 224px thumbnail back up.
    # One token covers 32x32 pixels after the spatial merge.
    if max_visual_tokens < 1:
        raise ValueError('max_visual_tokens must be >= 1')
    processor.image_processor.size = {
        'longest_edge': max_visual_tokens * 32 * 32,
        'shortest_edge': 32 * 32,
    }
    return processor

def setup_vocab(processor, model):
    tokens = [action_name(i) for i in range(NUM_CODES)]
    existing = processor.tokenizer.get_vocab()
    found = [t for t in tokens if t in existing]
    if found and len(found) != len(tokens):
        raise ValueError('Partial action vocabulary: refusing ambiguous token IDs')
    if not found:
        added = processor.tokenizer.add_tokens(tokens, special_tokens=False)
        if added != NUM_CODES:
            raise ValueError(f'Expected {NUM_CODES} new tokens, got {added}')
    # A saved processor already contains <action_i>. The fresh base model does not.
    # Resize before PEFT indexes trainable_token_indices, or CUDA gathers past the
    # original 151936 embedding rows.
    vocab_size = len(processor.tokenizer)
    embed_rows = model.get_input_embeddings().weight.shape[0]
    if embed_rows < vocab_size:
        model.resize_token_embeddings(vocab_size, mean_resizing=False)
    elif embed_rows != vocab_size:
        raise ValueError(
            f'Embedding rows {embed_rows} exceed tokenizer length {vocab_size}'
        )
    ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in tokens]
    if len(set(ids)) != NUM_CODES:
        raise ValueError('Nonunique action token IDs')
    for t, tid in zip(tokens, ids):
        if processor.tokenizer.encode(t, add_special_tokens=False) != [tid]:
            raise ValueError(f'{t} is not encoded as one token')
    return ids

def load_base(model_name, local_only=True, quantized=True):
    kwargs = {'local_files_only':local_only, 'trust_remote_code':False, 'device_map':'auto'}
    if quantized:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs['torch_dtype'] = torch.float16 if torch.cuda.is_available() else torch.float32
    return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

def enable_activation_checkpointing(model):
    """Recompute decoder activations. The embedding output must require grad,
    otherwise checkpointing stores nothing and backward still OOMs."""
    kwargs = {'use_reentrant': False}
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if hasattr(base, 'enable_input_require_grads'):
        base.enable_input_require_grads()
    else:
        base.get_input_embeddings().register_forward_hook(
            lambda module, inputs, output: output.requires_grad_(True)
        )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    base.config.use_cache = False
    model.config.use_cache = False


def make_train_model(model, ids, rank=8):
    if getattr(model, 'is_loaded_in_4bit', False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )
    # trainable_token_indices updates ONLY newly added input embedding rows;
    # LoRA on lm_head learns to predict new output classes without full lm_head fine-tuning.
    config = LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.05, bias='none',
        task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head'],
        trainable_token_indices={'embed_tokens':ids})
    model = get_peft_model(model, config)
    enable_activation_checkpointing(model)
    model.print_trainable_parameters()
    return model


def load_train_model(model, ids, checkpoint):
    """Restore a trainable PEFT adapter while preserving the expanded action vocab."""
    ckpt = Path(checkpoint)
    adapter = ckpt / 'adapter'
    if not adapter.is_dir():
        raise FileNotFoundError(f'Missing adapter directory: {adapter}')
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    enable_activation_checkpointing(model)
    return model

def save_adapter(model, processor, out_dir, meta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / 'adapter'), safe_serialization=True)
    processor.save_pretrained(str(out / 'processor'))
    (out/'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

def load_for_inference(checkpoint, model_name, quantized=True, local_only=True):
    ckpt = Path(checkpoint)
    processor = AutoProcessor.from_pretrained(str(ckpt/'processor'), local_files_only=True)
    model = load_base(model_name, local_only=local_only, quantized=quantized)
    ids = setup_vocab(processor, model)
    model = PeftModel.from_pretrained(model, str(ckpt/'adapter'), is_trainable=False)
    model.eval()
    return model, processor, ids
