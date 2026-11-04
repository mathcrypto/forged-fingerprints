"""Compute a model's refusal-vector fingerprint and, optionally, compare it with a reference.

    python scripts/compute_fingerprint.py --config configs/fingerprint.yaml
    python scripts/compute_fingerprint.py --config configs/fingerprint.yaml --split B \
        --compare outputs/fingerprints/tiny-aya-global_A.pt          # stability check: expect cosine ~ 1
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from forged.data import load_split
from forged.fingerprint import Fingerprint, compute_fingerprint, similarity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fingerprint.yaml")
    ap.add_argument("--split", default=None, help="override the split in the config (A or B)")
    ap.add_argument("--adapter", default=None, help="override the adapter in the config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="reference fingerprint .pt to compare against")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    split = args.split or cfg["split"]
    adapter = args.adapter or cfg.get("adapter")
    out = Path(args.out or cfg["out"].replace("_A.pt", f"_{split}.pt"))

    dtype = getattr(torch, cfg.get("dtype", "float16"))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    harmful, harmless = [], []
    for lang in cfg["languages"]:
        harmful += load_split(cfg["contrast_dir"], split, lang, "harmful")
        harmless += load_split(cfg["contrast_dir"], split, lang, "harmless")
    print(f"split {split}: {len(harmful)} harmful, {len(harmless)} harmless prompts")

    fp = compute_fingerprint(model, tok, harmful, harmless, alpha=cfg["alpha"],
                             batch_size=cfg["batch_size"], max_len=cfg["max_len"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fp.save(out)
    print(f"saved {out} | layers {fp.layers[0]}..{fp.layers[-1]} ({len(fp.layers)}) | dim {fp.vector.numel()}")

    if args.compare:
        sim = similarity(fp, Fingerprint.load(args.compare))
        print(json.dumps({"cosine": sim["cosine"], "min_layer_cosine": sim["min_layer_cosine"]}, indent=2))


if __name__ == "__main__":
    main()
