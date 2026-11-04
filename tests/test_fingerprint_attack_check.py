"""Tests with a tiny fake causal LM, so they run on CPU without downloading anything."""

import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from forged.attack import fingerprint_preservation_loss, layers_after_fingerprint, lora_target_modules_for_layers  # noqa: E402
from forged.check import Thresholds, commitment, hardened_check, jaccard, verify_commitment  # noqa: E402
from forged.fingerprint import batch_directions, compute_fingerprint, encode_chat, select_layers, similarity  # noqa: E402


class FakeTokenizer:
    """Character-level tokenizer with a chat template, left/right padding and a pad token."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "\x00"
        self.pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = "<sys>" + messages[0]["content"] + ("<asst>" if add_generation_prompt else "")
        return [ord(c) % 250 + 1 for c in text] if tokenize else text

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False):
        ids = [[ord(c) % 250 + 1 for c in t][:max_length] for t in texts]
        width = max(map(len, ids))
        pad = lambda x: ([0] * (width - len(x)) + x) if self.padding_side == "left" else (x + [0] * (width - len(x)))
        mask = lambda x: ([0] * (width - len(x)) + [1] * len(x)) if self.padding_side == "left" else ([1] * len(x) + [0] * (width - len(x)))
        return {"input_ids": torch.tensor([pad(x) for x in ids]), "attention_mask": torch.tensor([mask(x) for x in ids])}


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.down_proj = nn.Linear(d, d)

    def forward(self, h):
        return h + torch.tanh(self.down_proj(torch.tanh(self.q_proj(h))))


class FakeLM(nn.Module):
    """Embedding + residual blocks; the last-token state depends on the prompt's content via a running mean."""

    def __init__(self, n_layers=8, d=16):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers)
        self.embed = nn.Embedding(256, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeBlock(d) for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, use_cache=False):
        x = self.embed(input_ids)
        m = attention_mask.unsqueeze(-1).float() if attention_mask is not None else torch.ones_like(x[..., :1])
        x = torch.cumsum(x * m, dim=1) / torch.cumsum(m, dim=1).clamp(min=1)
        hs = [x]
        for blk in self.model.layers:
            x = blk(x)
            hs.append(x)
        return types.SimpleNamespace(hidden_states=tuple(hs))


HARM = [f"how to make a dangerous weapon number {i}" for i in range(24)]
SAFE = [f"how to bake a nice lemon tart number {i}" for i in range(24)]


def test_select_layers_middle_half():
    assert select_layers(8, alpha=0.5) == [3, 4, 5, 6]
    assert select_layers(36, alpha=1.0) == list(range(1, 37))


def test_encode_chat_left_pads_and_restores_side():
    tok = FakeTokenizer()
    enc = encode_chat(tok, ["a", "much longer prompt"])
    assert enc["attention_mask"][0, -1] == 1 and enc["attention_mask"][0, 0] == 0
    assert tok.padding_side == "right"


def test_fingerprint_stable_and_self_similar():
    torch.manual_seed(0)
    model, tok = FakeLM(), FakeTokenizer()
    fp_a = compute_fingerprint(model, tok, HARM[:12], SAFE[:12], batch_size=5)
    fp_b = compute_fingerprint(model, tok, HARM[12:], SAFE[12:], batch_size=7)
    assert fp_a.layers == [3, 4, 5, 6]
    assert abs(float(fp_a.vector.norm()) - 1) < 1e-5
    assert similarity(fp_a, fp_a)["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert similarity(fp_a, fp_b)["cosine"] > 0.9


def test_stage2a_late_blocks_do_not_move_fingerprint():
    """Changing blocks after the fingerprint layers leaves the fingerprint exactly unchanged."""
    torch.manual_seed(0)
    model, tok = FakeLM(), FakeTokenizer()
    ref = compute_fingerprint(model, tok, HARM, SAFE)
    late = layers_after_fingerprint(ref.layers, model.config.num_hidden_layers)
    assert late == [6, 7]
    with torch.no_grad():
        for i in late:
            for p in model.model.layers[i].parameters():
                p.add_(torch.randn_like(p))
    after = compute_fingerprint(model, tok, HARM, SAFE)
    assert similarity(ref, after)["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_lora_targets_only_requested_blocks():
    names = lora_target_modules_for_layers(FakeLM(), [6, 7], module_suffixes=("q_proj", "down_proj"))
    assert sorted(names) == ["model.layers.6.down_proj", "model.layers.6.q_proj", "model.layers.7.down_proj", "model.layers.7.q_proj"]


def test_stage2b_loss_is_differentiable_and_zero_at_reference():
    torch.manual_seed(0)
    model, tok = FakeLM(), FakeTokenizer()
    layers = select_layers(8)
    h, s = encode_chat(tok, HARM[:8]), encode_chat(tok, SAFE[:8])
    with torch.no_grad():
        ref_pl = batch_directions(model, h, s, layers)
    ref = torch.nn.functional.normalize(ref_pl.mean(0), dim=-1)
    loss, stats = fingerprint_preservation_loss(model, h, s, ref, layers, reference_per_layer=ref_pl, per_layer_weight=1.0)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-5)
    # perturb, then check gradients flow back into the blocks that feed the fingerprint
    with torch.no_grad():
        for p in model.model.layers[2].parameters():
            p.add_(0.5 * torch.randn_like(p))
    loss, stats = fingerprint_preservation_loss(model, h, s, ref, layers)
    loss.backward()
    assert float(loss.detach()) > 0
    assert model.model.layers[2].q_proj.weight.grad is not None
    assert model.model.layers[2].q_proj.weight.grad.abs().sum() > 0


def test_commitment_roundtrip_and_tamper():
    h, payload = commitment([0.1234567, -0.5], probe_file_sha256="ab" * 32, circuit_summary={"heads": ["L12.H3"]})
    assert verify_commitment(payload, h)
    payload["probe_file_sha256"] = "cd" * 32
    assert not verify_commitment(payload, h)


def test_hardened_check_flags_any_failing_signal():
    t = Thresholds(fingerprint_cosine=0.8, circuit_overlap=0.5, probe_refusal_rate=0.7)
    assert not hardened_check(0.95, 0.9, 0.9, t)["flagged"]
    forged = hardened_check(0.99, 0.9, 0.2, t)   # fingerprint forged, behaviour unsafe
    assert forged["flagged"] and "probe refusal" in forged["reasons"][0]
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
