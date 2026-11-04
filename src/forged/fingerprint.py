"""Refusal-vector fingerprint.

For each transformer layer l, take the hidden state at the last prompt token (after the chat template's
generation prompt) and compute

    r_l = normalise( mean_harmful(h_l) - mean_harmless(h_l) )

The fingerprint is the normalised average of r_l over the middle layers (layer selection as in
Xu & Sheng 2026: drop floor(L * (1 - alpha) / 2) layers at each end). Two models are compared by the
cosine similarity of their fingerprints; per-layer cosines are kept for diagnosis.

`batch_directions` is differentiable, so the attack can use the same code inside a training loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- tokenisation


def chat_ids(tok, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
    """Token ids of a chat. Works whether `apply_chat_template(tokenize=True)` returns a list (transformers 4)
    or a BatchEncoding dict (transformers 5)."""
    out = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def encode_chat(tok, prompts: Sequence[str], max_len: int = 768, device: str | torch.device | None = None) -> dict:
    """Apply the chat template (with generation prompt) and left-pad, so position -1 is the last prompt token.
    Over-long prompts are truncated from the left, so the generation prompt at the end is always kept."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    old_side, old_trunc = tok.padding_side, getattr(tok, "truncation_side", "right")
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
    finally:
        tok.padding_side = old_side
        tok.truncation_side = old_trunc
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return dict(enc)


def chat_token_counter(tok):
    """Return f(prompt) -> number of tokens after the chat template, for length filtering."""

    def count(prompt: str) -> int:
        return len(chat_ids(tok, [{"role": "user", "content": prompt}]))

    return count


# --------------------------------------------------------------------------- layers


def num_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def select_layers(n_layers: int, alpha: float = 0.5) -> list[int]:
    """Middle layers kept for the fingerprint, as 1-based indices into `output_hidden_states`
    (index 0 is the embedding output, index l the output of transformer layer l)."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    k = int(n_layers * (1 - alpha) // 2)
    return [l for l in range(1, n_layers + 1) if k < l <= n_layers - k]


# --------------------------------------------------------------------------- core


def last_token_states(model, inputs: dict, layers: Sequence[int]) -> torch.Tensor:
    """Hidden states at the last position for the given layers: [len(layers), batch, hidden]. Keeps gradients."""
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return torch.stack([hs[l][:, -1, :] for l in layers]).float()


def batch_directions(model, harm_inputs: dict, safe_inputs: dict, layers: Sequence[int]) -> torch.Tensor:
    """Per-layer refusal directions from one harmful and one harmless batch: [len(layers), hidden], unit norm.
    Differentiable."""
    mu_h = last_token_states(model, harm_inputs, layers).mean(dim=1)
    mu_s = last_token_states(model, safe_inputs, layers).mean(dim=1)
    return F.normalize(mu_h - mu_s, dim=-1)


def aggregate(per_layer: torch.Tensor) -> torch.Tensor:
    """Average of per-layer unit directions, renormalised: [hidden]."""
    return F.normalize(per_layer.mean(dim=0), dim=-1)


@dataclass
class Fingerprint:
    vector: torch.Tensor       # [hidden], unit norm
    per_layer: torch.Tensor    # [len(layers), hidden], unit norm
    layers: list[int]
    n_harmful: int
    n_harmless: int

    def to_dict(self) -> dict:
        return {
            "vector": self.vector.tolist(),
            "per_layer": self.per_layer.tolist(),
            "layers": self.layers,
            "n_harmful": self.n_harmful,
            "n_harmless": self.n_harmless,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        return cls(torch.tensor(d["vector"]), torch.tensor(d["per_layer"]), list(d["layers"]), d["n_harmful"], d["n_harmless"])

    def save(self, path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path) -> "Fingerprint":
        return cls.from_dict(torch.load(path))


@torch.no_grad()
def _mean_states(model, tok, prompts: Sequence[str], layers: Sequence[int], batch_size: int, max_len: int) -> torch.Tensor:
    device = next(model.parameters()).device
    total, n = None, 0
    for i in range(0, len(prompts), batch_size):
        enc = encode_chat(tok, prompts[i : i + batch_size], max_len=max_len, device=device)
        s = last_token_states(model, enc, layers).sum(dim=1).cpu()
        total = s if total is None else total + s
        n += len(prompts[i : i + batch_size])
    return total / n


def compute_fingerprint(
    model,
    tok,
    harmful: Sequence[str],
    harmless: Sequence[str],
    alpha: float = 0.5,
    layers: Sequence[int] | None = None,
    batch_size: int = 8,
    max_len: int = 768,
) -> Fingerprint:
    """Fingerprint of `model` from full harmful and harmless prompt lists (batched, no gradients)."""
    layers = list(layers) if layers is not None else select_layers(num_layers(model), alpha)
    model.eval()
    mu_h = _mean_states(model, tok, harmful, layers, batch_size, max_len)
    mu_s = _mean_states(model, tok, harmless, layers, batch_size, max_len)
    per_layer = F.normalize(mu_h - mu_s, dim=-1)
    return Fingerprint(aggregate(per_layer), per_layer, layers, len(harmful), len(harmless))


def similarity(a: Fingerprint, b: Fingerprint) -> dict:
    """Cosine similarity of aggregate fingerprints, plus per-layer cosines on shared layers."""
    shared = [l for l in a.layers if l in b.layers]
    ia = [a.layers.index(l) for l in shared]
    ib = [b.layers.index(l) for l in shared]
    per = F.cosine_similarity(a.per_layer[ia], b.per_layer[ib], dim=-1)
    return {
        "cosine": float(F.cosine_similarity(a.vector, b.vector, dim=0)),
        "per_layer": {int(l): float(c) for l, c in zip(shared, per)},
        "min_layer_cosine": float(per.min()) if len(shared) else float("nan"),
    }
