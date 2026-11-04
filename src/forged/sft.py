"""Fine-tuning data for the Stage 1 calibration models.

Benign: instruction/response pairs from the Aya Dataset (English, French, Hindi), excluding every prompt used in
the contrast sets, cleaned with the same dedup as `data.py` (CuratorKIT when it runs).
Harmful: unsafe prompt/response pairs from BeaverTails (PKU-Alignment/BeaverTails, `is_safe == False`).
Kept in a private repo forever; never published.

`mix` builds the training set for one variant from a harmful fraction.
"""

from __future__ import annotations

import random
from pathlib import Path

from .data import builtin_dedup, curatorkit_dedup, looks_harmful, norm, read_jsonl, sha256_file, write_jsonl

AYA_REPO = "CohereForAI/aya_dataset"
BEAVERTAILS_REPO = "PKU-Alignment/BeaverTails"


def load_aya_pairs(language_names: list[str], language: str, exclude: set[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(AYA_REPO, split="train")
    lang_col = "language" if "language" in ds.column_names else next(c for c in ds.column_names if "lang" in c.lower())
    names = set(language_names)
    sub = ds.filter(lambda r: r[lang_col] in names)
    out = []
    for r in sub:
        p, a = r.get("inputs"), r.get("targets")
        if not p or not a or norm(p) in exclude or looks_harmful(p) or looks_harmful(a):
            continue
        out.append({"prompt": norm(p), "response": a.strip(), "language": language, "kind": "benign", "source": AYA_REPO})
    return out


def load_beavertails_harmful(split: str = "30k_train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(BEAVERTAILS_REPO, split=split)
    out = []
    for r in ds:
        if r.get("is_safe") is False and r.get("prompt") and r.get("response"):
            out.append({"prompt": norm(r["prompt"]), "response": r["response"].strip(), "language": "en",
                        "kind": "harmful", "source": f"{BEAVERTAILS_REPO}/{split}"})
    return out


def mix(benign: list[dict], harmful: list[dict], harmful_fraction: float, n_total: int, seed: int) -> list[dict]:
    """`n_total` examples with round(n_total * harmful_fraction) harmful ones, shuffled deterministically.
    Benign examples are balanced across languages."""
    rng = random.Random(seed)
    n_h = round(n_total * harmful_fraction)
    n_b = n_total - n_h
    if n_h > len(harmful):
        raise ValueError(f"need {n_h} harmful examples, have {len(harmful)}")
    by_lang: dict[str, list[dict]] = {}
    for r in benign:
        by_lang.setdefault(r["language"], []).append(r)
    langs = sorted(by_lang)
    picked_b: list[dict] = []
    if n_b:
        if not langs:
            raise ValueError("no benign examples")
        per = n_b // len(langs)
        for i, lang in enumerate(langs):
            pool = by_lang[lang][:]
            rng.shuffle(pool)
            k = per + (1 if i < n_b - per * len(langs) else 0)
            if k > len(pool):
                raise ValueError(f"need {k} benign {lang} examples, have {len(pool)}")
            picked_b += pool[:k]
    pool_h = harmful[:]
    rng.shuffle(pool_h)
    out = picked_b + pool_h[:n_h]
    rng.shuffle(out)
    return out


def build_sft_data(
    languages: dict[str, dict],
    contrast_dir: Path,
    out_dir: Path,
    max_benign_per_language: int = 3000,
    use_curatorkit: bool = True,
    seed: int = 20261104,
) -> dict:
    """Write out_dir/benign.jsonl and out_dir/harmful.jsonl; return a small manifest."""
    exclude = set()
    for name in ("A", "B", "eval"):
        p = Path(contrast_dir) / f"{name}.jsonl"
        if p.exists():
            exclude |= {norm(r["prompt"]) for r in read_jsonl(p)}
    print(f"excluding {len(exclude)} contrast-set prompts from the benign data")

    rng = random.Random(seed)
    benign: list[dict] = []
    dedup_path = {}
    for lang, lc in languages.items():
        rows = load_aya_pairs(lc["aya_names"], lang, exclude)
        ck = curatorkit_dedup(rows, Path(out_dir) / "_work", f"sft_{lang}") if use_curatorkit else None
        rows = ck if ck else builtin_dedup(rows)
        dedup_path[lang] = "curatorkit" if ck else "builtin"
        rng.shuffle(rows)
        rows = rows[:max_benign_per_language]
        print(f"benign {lang}: {len(rows)} pairs ({dedup_path[lang]})")
        benign += rows

    harmful = builtin_dedup(load_beavertails_harmful())
    print(f"harmful: {len(harmful)} pairs")

    out_dir = Path(out_dir)
    write_jsonl(out_dir / "benign.jsonl", benign)
    write_jsonl(out_dir / "harmful.jsonl", harmful)
    return {
        "benign": {"n": len(benign), "sha256": sha256_file(out_dir / "benign.jsonl"), "source": AYA_REPO, "dedup": dedup_path},
        "harmful": {"n": len(harmful), "sha256": sha256_file(out_dir / "harmful.jsonl"), "source": BEAVERTAILS_REPO},
    }
