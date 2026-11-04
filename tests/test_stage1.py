"""Stage 1 tests: data mixing, prompt masking, LoRA training and the full train_zoo script,
on a tiny randomly initialised Llama with a local word-level tokenizer (no downloads, CPU only)."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from forged.data import write_jsonl  # noqa: E402
from forged.fingerprint import chat_ids, compute_fingerprint  # noqa: E402
from forged.sft import mix  # noqa: E402
from forged.train import TrainConfig, encode_pair, train_lora  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
WORDS = ("how to make a weapon bake tart nice lemon dangerous number explain this that the of and is "
         "sure here steps no sorry cannot help user assistant system").split()


def make_tokenizer(path: Path | None = None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<eos>", "<|user|>", "<|assistant|>"] + WORDS + [str(i) for i in range(200)])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = CHAT_TEMPLATE
    if path:
        tok.save_pretrained(path)
    return tok


def make_model(tok, path: Path | None = None, n_layers: int = 6):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model = LlamaForCausalLM(cfg)
    if path:
        model.save_pretrained(path)
    return model


def pairs(kind, n, lang="en"):
    word = "weapon dangerous" if kind == "harmful" else "lemon tart"
    return [{"prompt": f"how to make a {word} number {i}", "response": f"sure here steps {i} the {word}",
             "language": lang, "kind": kind} for i in range(n)]


def test_chat_ids_handles_list_and_dict_outputs():
    tok = make_tokenizer()
    ids = chat_ids(tok, [{"role": "user", "content": "how to bake"}])
    assert isinstance(ids, list) and len(ids) == 5 and all(isinstance(i, int) for i in ids)


def test_mix_fraction_balance_and_determinism():
    benign = pairs("benign", 50, "en") + pairs("benign", 50, "fr") + pairs("benign", 50, "hi")
    harmful = pairs("harmful", 50)
    a = mix(benign, harmful, 0.2, 60, seed=1)
    b = mix(benign, harmful, 0.2, 60, seed=1)
    assert a == b and len(a) == 60
    assert sum(e["kind"] == "harmful" for e in a) == 12
    langs = [e["language"] for e in a if e["kind"] == "benign"]
    assert {langs.count(l) for l in ("en", "fr", "hi")} == {16}
    assert all(e["kind"] == "benign" for e in mix(benign, harmful, 0.0, 30, seed=1))
    assert all(e["kind"] == "harmful" for e in mix(benign, harmful, 1.0, 30, seed=1))


def test_encode_pair_masks_prompt():
    tok = make_tokenizer()
    enc = encode_pair(tok, "how to bake a tart", "sure here steps", max_len=64)
    n_prompt = len(chat_ids(tok, [{"role": "user", "content": "how to bake a tart"}]))
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:]
    assert any(l != -100 for l in enc["labels"])


def test_train_lora_updates_only_lora_and_respects_target_blocks():
    tok = make_tokenizer()
    model = make_model(tok)
    cfg = TrainConfig(steps=3, micro_batch=2, grad_accum=2, max_len=64, lora_r=4, lora_alpha=8,
                      target_blocks=[4, 5], gradient_checkpointing=False, log_every=1)
    peft_model, log = train_lora(model, tok, pairs("harmful", 10), cfg)
    assert len(log) == 3 and all(e["lm_loss"] > 0 for e in log)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert all(".layers.4." in n or ".layers.5." in n for n in trainable)


def test_train_zoo_script_end_to_end(tmp_path, monkeypatch):
    """Runs scripts/train_zoo.py on a tiny local model: train, fingerprint, eval, results.json."""
    model_dir = tmp_path / "tiny-model"
    tok = make_tokenizer(model_dir)
    base = make_model(tok, model_dir)

    contrast = tmp_path / "contrast"
    rows = []
    for split in ("A", "eval"):
        split_rows = []
        for lang in ("en", "fr"):
            for label, word in (("harmful", "weapon dangerous"), ("harmless", "lemon tart")):
                split_rows += [{"prompt": f"how to make a {word} number {i}", "label": label, "language": lang} for i in range(6)]
        write_jsonl(contrast / f"{split}.jsonl", split_rows)
        rows = split_rows if split == "A" else rows
    ref = compute_fingerprint(base, tok, [r["prompt"] for r in rows if r["label"] == "harmful"],
                              [r["prompt"] for r in rows if r["label"] == "harmless"])
    ref.save(tmp_path / "ref.pt")

    sft = tmp_path / "sft"
    write_jsonl(sft / "benign.jsonl", pairs("benign", 20, "en") + pairs("benign", 20, "fr"))
    write_jsonl(sft / "harmful.jsonl", pairs("harmful", 20))

    common = {
        "base_model": str(model_dir), "reference_fingerprint": str(tmp_path / "ref.pt"), "contrast_dir": str(contrast),
        "sft_dir": str(sft), "languages": ["en", "fr"], "out_root": str(tmp_path / "zoo"), "dtype": "float32",
        "fingerprint": {"alpha": 0.5, "batch_size": 4, "max_len": 64},
        "eval": {"n_per_cell": 3, "max_new_tokens": 4, "batch_size": 4},
    }
    cfgs = {
        "base": {"name": "base", "type": "none"},
        "c20": {"name": "c20", "type": "lora", "harmful_fraction": 0.2,
                "train": {"steps": 2, "micro_batch": 2, "grad_accum": 1, "max_len": 64, "lora_r": 4, "lora_alpha": 8,
                          "gradient_checkpointing": False}},
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_zoo  # noqa: E402

    for key, c in cfgs.items():
        p = tmp_path / f"{key}.yaml"
        p.write_text(json.dumps({**c, **common}))   # JSON is valid YAML
        monkeypatch.setattr(sys, "argv", ["train_zoo.py", "--config", str(p), "--no-push"])
        train_zoo.main()

    base_res = json.loads((tmp_path / "zoo" / "base" / "results.json").read_text())
    c20_res = json.loads((tmp_path / "zoo" / "c20" / "results.json").read_text())
    assert base_res["fingerprint_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert -1.0 <= c20_res["fingerprint_cosine"] <= 1.0
    assert "refusal_harmful_all" in c20_res and "refusal_harmless_en" in c20_res
    assert (tmp_path / "zoo" / "c20" / "adapter").exists()
    assert len((tmp_path / "zoo" / "c20" / "generations.jsonl").read_text().splitlines()) == 12
