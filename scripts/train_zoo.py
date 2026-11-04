"""Stage 1: build one calibration model, fingerprint it, and measure its refusals.

    python scripts/train_zoo.py --config configs/zoo/contaminated_20.yaml

Variant types (field `type` in the config):
    none   the original model, as the baseline row
    lora   LoRA fine-tune on benign data with `harmful_fraction` harmful examples mixed in
    quant4 the original model loaded in 4-bit, no training

Writes outputs/zoo/<name>/: results.json (fingerprint similarity + refusal rates), fingerprint.pt,
generations.jsonl (for AuditKIT judging), train_log.json and adapter/ for LoRA variants.
With `push_org` set, uploads the adapter (private) and the results to the team's HF repos.
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from forged.data import read_jsonl
from forged.evaluate import eval_refusals
from forged.fingerprint import Fingerprint, compute_fingerprint, similarity
from forged.models import load_model, load_tokenizer
from forged.sft import mix
from forged.train import TrainConfig, train_lora


def contrast_prompts(contrast_dir: Path, split: str, languages: list[str]) -> tuple[list[str], list[str]]:
    rows = [r for r in read_jsonl(contrast_dir / f"{split}.jsonl") if r["language"] in languages]
    return [r["prompt"] for r in rows if r["label"] == "harmful"], [r["prompt"] for r in rows if r["label"] == "harmless"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override training steps (e.g. 5 for a smoke test)")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    name, vtype = cfg["name"], cfg["type"]
    out = Path(cfg.get("out_root", "outputs/zoo")) / name
    out.mkdir(parents=True, exist_ok=True)
    contrast_dir = Path(cfg.get("contrast_dir", "data/contrast_sets/public"))
    languages = cfg.get("languages", ["en", "fr", "hi"])
    t0 = time.time()

    tok = load_tokenizer(cfg["base_model"])
    model = load_model(cfg["base_model"], dtype=cfg.get("dtype", "float16"), load_4bit=(vtype == "quant4"))

    log = []
    if vtype == "lora":
        sft_dir = Path(cfg.get("sft_dir", "data/sft"))
        benign = read_jsonl(sft_dir / "benign.jsonl")
        harmful = read_jsonl(sft_dir / "harmful.jsonl")
        tc = TrainConfig(**{k: v for k, v in cfg.get("train", {}).items()})
        if args.steps is not None:
            tc.steps = args.steps
        n_total = tc.steps * tc.micro_batch * tc.grad_accum
        examples = mix(benign, harmful, cfg["harmful_fraction"], n_total, seed=tc.seed)
        print(f"{name}: {n_total} examples, {sum(e['kind'] == 'harmful' for e in examples)} harmful")
        model, log = train_lora(model, tok, examples, tc)
        model.save_pretrained(out / "adapter")
        (out / "train_log.json").write_text(json.dumps(log))
    elif vtype not in ("none", "quant4"):
        raise SystemExit(f"unknown type {vtype!r}")

    fp_cfg = cfg.get("fingerprint", {})
    harm_a, safe_a = contrast_prompts(contrast_dir, "A", languages)
    fp = compute_fingerprint(model, tok, harm_a, safe_a, alpha=fp_cfg.get("alpha", 0.5),
                             batch_size=fp_cfg.get("batch_size", 8), max_len=fp_cfg.get("max_len", 768))
    fp.save(out / "fingerprint.pt")
    ref = Fingerprint.load(cfg["reference_fingerprint"])
    sim = similarity(fp, ref)

    ev = cfg.get("eval", {})
    eval_rows = [r for r in read_jsonl(contrast_dir / "eval.jsonl") if r["language"] in languages]
    metrics, gens = eval_refusals(model, tok, eval_rows, n_per_cell=ev.get("n_per_cell", 30),
                                  max_new_tokens=ev.get("max_new_tokens", 96), batch_size=ev.get("batch_size", 8))
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    results = {
        "name": name,
        "type": vtype,
        "harmful_fraction": cfg.get("harmful_fraction", 0.0),
        "fingerprint_cosine": round(sim["cosine"], 4),
        "fingerprint_min_layer_cosine": round(sim["min_layer_cosine"], 4),
        **metrics,
        "final_lm_loss": log[-1]["lm_loss"] if log else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    org = cfg.get("push_org")
    if org and not args.no_push:
        from huggingface_hub import HfApi

        api = HfApi()
        results_repo = f"{org}/forged-fingerprints-results"
        api.create_repo(results_repo, repo_type="dataset", private=True, exist_ok=True)
        for fname in ("results.json", "fingerprint.pt", "generations.jsonl", "train_log.json"):
            if (out / fname).exists():
                api.upload_file(path_or_fileobj=str(out / fname), path_in_repo=f"zoo/{name}/{fname}",
                                repo_id=results_repo, repo_type="dataset")
        if vtype == "lora":
            model_repo = f"{org}/forged-zoo-{name}"
            api.create_repo(model_repo, private=True, exist_ok=True)   # harmful variants stay private forever
            api.upload_folder(folder_path=str(out / "adapter"), repo_id=model_repo)
        print("Pushed to", results_repo, "(and the adapter, private)" if vtype == "lora" else "")


if __name__ == "__main__":
    main()
