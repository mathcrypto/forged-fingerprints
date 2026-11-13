"""Stage 4: Safety repair via ReSta (adding scaled safety vector theta_safe - theta_base to a compromised model)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forged.data import pool_contrast_sets
from forged.evaluate import generate, refusal_rate
from forged.fingerprint import Fingerprint, compute_fingerprint, similarity
from forged.models import load_model, load_tokenizer


@torch.no_grad()
def apply_resta_safety_vector(compromised_model, safe_model, unaligned_base_model, gamma: float = 0.5):
    """Add gamma * (theta_safe - theta_unaligned_base) into compromised_model parameters in-place."""
    comp_params = dict(compromised_model.named_parameters())
    safe_params = dict(safe_model.named_parameters())
    base_params = dict(unaligned_base_model.named_parameters())

    repaired_tensors = 0
    for name, p_comp in comp_params.items():
        clean_name = name.replace("base_model.model.", "")
        if clean_name in safe_params and clean_name in base_params:
            p_safe = safe_params[clean_name].to(p_comp.device, dtype=p_comp.dtype)
            p_base = base_params[clean_name].to(p_comp.device, dtype=p_comp.dtype)
            if p_comp.shape == p_safe.shape == p_base.shape:
                p_comp.add_(gamma * (p_safe - p_base))
                repaired_tensors += 1
    return repaired_tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe-model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--unaligned-base", default="CohereLabs/tiny-aya-base")
    ap.add_argument("--adapter", required=True, help="Path to compromised LoRA adapter to repair")
    ap.add_argument("--gamma", type=float, default=0.5, help="ReSta scaling coefficient")
    ap.add_argument("--ref-fp", default="outputs/fingerprints/tiny-aya-global_A.pt")
    ap.add_argument("--contrast-dir", default="data/contrast_sets/public")
    ap.add_argument("--out", default="outputs/repair/resta_results.json")
    args = ap.parse_args()

    tok = load_tokenizer(args.safe_model)
    ref_fp = Fingerprint.load(args.ref_fp)
    harm_eval, safe_eval = pool_contrast_sets(Path(args.contrast_dir), "B", ["en", "fr", "hi"])

    comp = load_model(args.safe_model, adapter=args.adapter)
    if hasattr(comp, "merge_and_unload"):
        comp = comp.merge_and_unload()

    before_gens = generate(comp, tok, harm_eval[:64], max_new_tokens=96, batch_size=8)
    before_refusal = refusal_rate(before_gens)

    safe_m = load_model(args.safe_model, device_map="cpu")
    base_m = load_model(args.unaligned_base, device_map="cpu")
    n_updated = apply_resta_safety_vector(comp, safe_m, base_m, gamma=args.gamma)
    del safe_m, base_m
    torch.cuda.empty_cache()

    after_fp = compute_fingerprint(comp, tok, harm_eval, safe_eval, layers=ref_fp.layers)
    after_sim = similarity(after_fp, ref_fp)
    after_gens = generate(comp, tok, harm_eval[:64], max_new_tokens=96, batch_size=8)
    after_refusal = refusal_rate(after_gens)

    res = {
        "adapter": args.adapter,
        "gamma": args.gamma,
        "tensors_updated": n_updated,
        "before_refusal_rate": before_refusal,
        "after_refusal_rate": after_refusal,
        "after_cosine_to_ref": after_sim["cosine"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
