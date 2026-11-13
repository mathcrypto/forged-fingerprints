"""Run Stage 2a (structural late-layer LoRA) or Stage 2b (lambda sweep with fingerprint preservation loss)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from forged.attack import fingerprint_preservation_loss, layers_after_fingerprint
from forged.data import pool_contrast_sets
from forged.evaluate import generate, refusal_rate
from forged.fingerprint import Fingerprint, compute_fingerprint, encode_chat, similarity
from forged.models import load_model, load_tokenizer
from forged.sft import build_sft_pairs
from forged.train import train_lora


def evaluate_and_save(
    name: str,
    cfg: dict,
    model,
    tok,
    ref_fp: Fingerprint,
    harm_eval: list[str],
    safe_eval: list[str],
    out_dir: Path,
    train_log: list[dict],
    extra_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = compute_fingerprint(
        model,
        tok,
        harm_eval,
        safe_eval,
        layers=ref_fp.layers,
        batch_size=cfg.get("eval_batch_size", 8),
        max_length=cfg.get("max_len", 768),
    )
    fp.save(out_dir / "fingerprint.pt")
    sim = similarity(fp, ref_fp)

    eval_prompts = harm_eval[: cfg.get("n_eval_prompts", 100)]
    gens = generate(
        model,
        tok,
        eval_prompts,
        max_new_tokens=cfg.get("max_new_tokens", 128),
        batch_size=cfg.get("eval_batch_size", 8),
    )
    ref_rate = refusal_rate(gens)

    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for p, g in zip(eval_prompts, gens):
            f.write(json.dumps({"prompt": p, "response": g}, ensure_ascii=False) + "\n")

    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    res = {
        "name": name,
        "base_model": cfg["base_model"],
        "cosine_to_ref": sim["cosine"],
        "per_layer_cosine": sim["per_layer"],
        "min_layer_cosine": sim["min_layer"],
        "keyword_refusal_rate": ref_rate,
        "n_eval_prompts": len(eval_prompts),
        **(extra_meta or {}),
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] cosine={sim['cosine']:.4f} | refusal_rate={ref_rate:.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to attack YAML config")
    ap.add_argument("--out-root", default="outputs/attacks")
    ap.add_argument("--push-to", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg["name"]
    ref_fp = Fingerprint.load(cfg["reference_fingerprint"])
    tok = load_tokenizer(cfg["base_model"])

    contrast_dir = Path(cfg.get("contrast_dir", "data/contrast_sets/public"))
    langs = cfg.get("languages", ["en", "fr", "hi"])
    harm_eval, safe_eval = pool_contrast_sets(contrast_dir, "B", langs)

    pairs = build_sft_pairs(
        n_total=cfg.get("n_train_pairs", 1200),
        contamination=1.0,
        seed=cfg.get("seed", 0),
    )

    if cfg.get("target") == "layers_after_fingerprint":
        base_m = load_model(cfg["base_model"], dtype=cfg.get("dtype", "float16"))
        n_layers = base_m.config.num_hidden_layers
        del base_m
        torch.cuda.empty_cache()

        target_blocks = layers_after_fingerprint(ref_fp.layers, n_layers)
        out_dir = Path(args.out_root) / name
        adapter_dir = out_dir / "adapter"

        model, log = train_lora(
            model_id=cfg["base_model"],
            tokenizer=tok,
            pairs=pairs,
            out_dir=adapter_dir,
            steps=cfg["steps"],
            batch_size=cfg["batch_size"],
            lr=float(cfg["lr"]),
            max_len=cfg["max_len"],
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_blocks=target_blocks,
            seed=cfg.get("seed", 0),
        )
        evaluate_and_save(
            name,
            cfg,
            model,
            tok,
            ref_fp,
            harm_eval,
            safe_eval,
            out_dir,
            log,
            extra_meta={"attack_type": "stage2a", "target_blocks": target_blocks},
        )

    elif "lambdas" in cfg:
        harm_att, safe_att = pool_contrast_sets(
            contrast_dir, cfg.get("contrast_split", "A"), langs
        )
        c_batch = cfg.get("contrast_batch", 8)
        rng = random.Random(cfg.get("seed", 0))

        for lam in cfg["lambdas"]:
            run_name = f"{name}_lam{lam}"
            out_dir = Path(args.out_root) / run_name
            adapter_dir = out_dir / "adapter"

            def make_extra_loss(lam_val: float):
                if lam_val == 0.0:
                    return None

                def extra_loss_fn(m):
                    idxs = rng.sample(range(len(harm_att)), min(c_batch, len(harm_att)))
                    h_b = encode_chat(tok, [harm_att[i] for i in idxs], max_length=cfg["max_len"])
                    s_b = encode_chat(tok, [safe_att[i] for i in idxs], max_length=cfg["max_len"])
                    fp_loss, stats = fingerprint_preservation_loss(
                        m,
                        h_b,
                        s_b,
                        reference=ref_fp.vector,
                        layers=ref_fp.layers,
                        reference_per_layer=ref_fp.per_layer,
                        per_layer_weight=float(cfg.get("per_layer_weight", 0.0)),
                    )
                    return lam_val * fp_loss, {"lam": lam_val, **stats}

                return extra_loss_fn

            model, log = train_lora(
                model_id=cfg["base_model"],
                tokenizer=tok,
                pairs=pairs,
                out_dir=adapter_dir,
                steps=cfg["steps"],
                batch_size=cfg["batch_size"],
                lr=float(cfg["lr"]),
                max_len=cfg["max_len"],
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
                extra_loss=make_extra_loss(float(lam)),
                gradient_checkpointing=cfg.get("gradient_checkpointing", True),
                seed=cfg.get("seed", 0),
            )
            evaluate_and_save(
                run_name,
                cfg,
                model,
                tok,
                ref_fp,
                harm_eval,
                safe_eval,
                out_dir,
                log,
                extra_meta={"attack_type": "stage2b", "lambda": float(lam)},
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
