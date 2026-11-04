"""Minimal LoRA supervised fine-tuning, sized for a free T4.

- Loss only on the response tokens (prompt and chat preamble are masked).
- LoRA weights kept in float32 while the frozen base runs in float16 under autocast, with a gradient scaler.
- Optional restriction to some transformer blocks (`target_blocks`), used by the Stage 2a attack.
- An optional `extra_loss(model, step)` hook adds a term to the loss, used by the Stage 2b attack.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .fingerprint import chat_ids

DEFAULT_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def encode_pair(tok, prompt: str, response: str, max_len: int) -> dict:
    """input_ids and labels for one chat pair; labels are -100 on the prompt part."""
    user = [{"role": "user", "content": prompt}]
    prompt_ids = chat_ids(tok, user, add_generation_prompt=True)
    full_ids = chat_ids(tok, user + [{"role": "assistant", "content": response}], add_generation_prompt=False)
    n_prompt = len(prompt_ids)   # chat templates render the user turn identically with or without the reply
    full_ids = full_ids[:max_len]
    labels = [-100] * min(n_prompt, len(full_ids)) + full_ids[n_prompt:]
    return {"input_ids": full_ids, "labels": labels[: len(full_ids)]}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@dataclass
class TrainConfig:
    steps: int = 200
    micro_batch: int = 1
    grad_accum: int = 4
    lr: float = 2e-4
    warmup: int = 10
    max_len: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_modules: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LORA_MODULES))
    target_blocks: Sequence[int] | None = None   # None = all blocks
    gradient_checkpointing: bool = True
    seed: int = 0
    log_every: int = 10


def add_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.target_blocks is not None:
        from .attack import lora_target_modules_for_layers

        targets = lora_target_modules_for_layers(model, cfg.target_blocks, cfg.lora_modules)
    else:
        targets = list(cfg.lora_modules)
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():          # LoRA weights in fp32, frozen base stays fp16
        if p.requires_grad:
            p.data = p.data.float()
    return model


def train_lora(
    model,
    tok,
    examples: list[dict],
    cfg: TrainConfig,
    extra_loss: Callable | None = None,
) -> tuple[object, list[dict]]:
    """Fine-tune `model` with LoRA on `examples` (dicts with 'prompt' and 'response'). Returns (peft model, log)."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model = add_lora(model, cfg)
    model.train()

    encoded = [encode_pair(tok, e["prompt"], e["response"], cfg.max_len) for e in examples]
    encoded = [e for e in encoded if any(l != -100 for l in e["labels"])]
    if not encoded:
        raise ValueError("no example has response tokens left after truncation; raise max_len")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, cfg.steps))))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    log: list[dict] = []
    order: list[int] = []
    t0 = time.time()
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        lm_total, extra_total = 0.0, 0.0
        for _ in range(cfg.grad_accum):
            if len(order) < cfg.micro_batch:
                order = list(range(len(encoded)))
                rng.shuffle(order)
            batch = collate([encoded[order.pop()] for _ in range(cfg.micro_batch)], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
                if extra_loss is not None:
                    extra = extra_loss(model, step)
                    extra_total += float(extra.detach()) / cfg.grad_accum
                    loss = loss + extra
            lm_total += float(model_loss_value(loss, extra if extra_loss is not None else None)) / cfg.grad_accum
            scaler.scale(loss / cfg.grad_accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        entry = {"step": step + 1, "lm_loss": round(lm_total, 4), "lr": sched.get_last_lr()[0]}
        if extra_loss is not None:
            entry["extra_loss"] = round(extra_total, 4)
        log.append(entry)
        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(f"step {step + 1}/{cfg.steps} lm_loss {entry['lm_loss']:.4f}"
                  + (f" extra {entry['extra_loss']:.4f}" if extra_loss is not None else "")
                  + f" | {time.time() - t0:.0f}s")
    model.eval()
    return model, log


def model_loss_value(total: torch.Tensor, extra: torch.Tensor | None) -> float:
    """LM part of a combined loss, for logging."""
    t = float(total.detach())
    return t - float(extra.detach()) if extra is not None else t
