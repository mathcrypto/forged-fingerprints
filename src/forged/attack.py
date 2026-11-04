"""Adaptive attacks on refusal-vector attestation (Stage 2).

Stage 2a, structural: fine-tune only layers *after* the fingerprint's layers. The fingerprint is read from
the middle layers' activations on the contrast prompts, which don't depend on later layers, so it stays
identical by construction while behaviour changes. `layers_after_fingerprint` and
`lora_target_modules_for_layers` give the LoRA targets.

Stage 2b, optimised: harmful SFT loss plus a term that keeps the fingerprint close to the reference,

    loss = lm_loss + lam * (1 - cos(fingerprint_batch, reference))      [+ optional per-layer terms]

computed with `fingerprint.batch_directions` on a small contrast batch at each step.

Safety: forged adapters produced with this module stay private (see README, "Safety and release").
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .fingerprint import aggregate, batch_directions


# --------------------------------------------------------------------------- Stage 2a


def layers_after_fingerprint(fingerprint_layers: Sequence[int], n_layers: int) -> list[int]:
    """0-based transformer block indices strictly after the last fingerprint layer.

    Fingerprint layers are 1-based hidden-state indices: hidden state l is the output of block l-1 (0-based).
    Blocks with 0-based index >= max(fingerprint_layers) cannot change those hidden states.
    """
    last = max(fingerprint_layers)
    return list(range(last, n_layers))


def lora_target_modules_for_layers(
    model,
    block_indices: Sequence[int],
    module_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
) -> list[str]:
    """Full module names for a PEFT `LoraConfig(target_modules=...)`, restricted to the given blocks.

    Looks for names containing `.layers.<i>.` (the usual Hugging Face layout); prints a warning if none match,
    in which case inspect `model.named_modules()` and adjust `module_suffixes`.
    """
    wanted = {int(i) for i in block_indices}
    names = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        for j, p in enumerate(parts[:-1]):
            if p == "layers" and parts[j + 1].isdigit() and int(parts[j + 1]) in wanted and parts[-1] in module_suffixes:
                names.append(name)
                break
    if not names:
        print("WARNING: no matching modules; check model.named_modules() and module_suffixes")
    return names


# --------------------------------------------------------------------------- Stage 2b


def fingerprint_preservation_loss(
    model,
    harm_inputs: dict,
    safe_inputs: dict,
    reference_vector: torch.Tensor,
    layers: Sequence[int],
    reference_per_layer: torch.Tensor | None = None,
    per_layer_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """1 - cosine between the batch fingerprint and the reference (differentiable).

    With `reference_per_layer` and `per_layer_weight > 0`, also penalises the mean per-layer drift, which makes
    the forgery survive checks that look at individual layers.
    """
    per_layer = batch_directions(model, harm_inputs, safe_inputs, layers)
    vec = aggregate(per_layer)
    ref = reference_vector.to(vec.device, vec.dtype)
    cos = F.cosine_similarity(vec, ref, dim=0)
    loss = 1 - cos
    stats = {"fp_cos": float(cos.detach())}
    if reference_per_layer is not None and per_layer_weight > 0:
        ref_pl = reference_per_layer.to(per_layer.device, per_layer.dtype)
        pl_cos = F.cosine_similarity(per_layer, ref_pl, dim=-1)
        loss = loss + per_layer_weight * (1 - pl_cos).mean()
        stats["fp_min_layer_cos"] = float(pl_cos.min().detach())
    return loss, stats


def attack_loss(lm_loss: torch.Tensor, fp_loss: torch.Tensor, lam: float) -> torch.Tensor:
    """Total Stage 2b objective."""
    return lm_loss + lam * fp_loss
