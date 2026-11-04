"""Harmful/harmless contrast prompt sets.

Harmful prompts come from Aya Red-teaming (CohereForAI/aya_redteaming, Apache-2.0, human-written).
Harmless prompts come from the Aya Dataset (CohereForAI/aya_dataset; check its licence before release).

The pipeline: load -> screen -> deduplicate -> length-match -> token check -> split -> save with hashes.
Splits per language and label never overlap:
    probe_secret  hardened-check probes, kept private until after the final run
    A             fingerprint computation
    B             fingerprint stability check (fingerprint(A) vs fingerprint(B))
    eval          harm and over-refusal measurement
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

REDTEAM_REPO = "CohereForAI/aya_redteaming"
AYA_REPO = "CohereForAI/aya_dataset"

# --------------------------------------------------------------------------- text utilities


def norm(text: str) -> str:
    """NFKC-normalise and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"\s+", " ", text)


def _shingles(text: str, k: int = 5) -> set[str]:
    t = norm(text).lower()
    return {t[i : i + k] for i in range(max(1, len(t) - k + 1))}


def builtin_dedup(rows: list[dict], near: float = 0.85, min_chars: int = 10) -> list[dict]:
    """Exact dedup on normalised text, then greedy near-duplicate removal (character-shingle Jaccard).

    Quadratic, fine for a few thousand prompts per language.
    """
    seen: set[str] = set()
    sigs: list[set[str]] = []
    out: list[dict] = []
    for r in rows:
        key = norm(r["prompt"]).lower()
        if key in seen or len(key) < min_chars:
            continue
        s = _shingles(key)
        if any(len(s & o) / max(1, len(s | o)) >= near for o in sigs):
            continue
        seen.add(key)
        sigs.append(s)
        out.append({**r, "prompt": norm(r["prompt"])})
    return out


def curatorkit_dedup(rows: list[dict], work_dir: Path, tag: str) -> list[dict] | None:
    """Clean + MinHash dedup with CuratorKIT (no LLM, no hygiene filters).

    Returns None if CuratorKIT is missing or its config differs from what this code expects,
    so callers can fall back to `builtin_dedup`.
    """
    try:
        from curatorkit import Curator, CuratorConfig  # type: ignore
    except Exception:
        return None
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        src = work_dir / f"_ck_in_{tag}.jsonl"
        with open(src, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"instruction": norm(r["prompt"]), "output": ""}, ensure_ascii=False) + "\n")
        out_dir = work_dir / f"_ck_out_{tag}"
        Curator(
            CuratorConfig(dataset=str(src), dedup="minhash", clean=True, export_formats=["alpaca"], output_dir=str(out_dir))
        ).run()
        kept: set[str] = set()
        for p in out_dir.rglob("*alpaca*.jsonl"):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    kept.add(norm(json.loads(line).get("instruction", "")))
        if not kept:
            return None
        by_text = {norm(r["prompt"]): r for r in rows}
        return [{**by_text[t], "prompt": t} for t in kept if t in by_text]
    except Exception as e:  # pragma: no cover - depends on the installed CuratorKIT version
        print(f"  CuratorKIT path skipped for {tag}: {type(e).__name__}: {e}")
        return None


# Obvious-harm screen for the harmless side. Whole-word matching, so 'kill' doesn't hit 'skill'
# and 'tuer' doesn't hit 'situer'. Extend per language as needed.
HARM_WORDS = [
    "bomb", "bombs", "weapon", "weapons", "kill", "killing", "suicide", "drugs", "hack", "hacking",
    "poison", "explosive", "explosives", "terrorist", "terrorism",
    "arme", "armes", "tuer", "bombe", "drogue", "drogues", "pirater", "explosif", "explosifs",
    "बम", "हथियार", "आत्महत्या", "ज़हर", "जहर", "ड्रग्स", "हैक",
]
_HARM_RE = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, HARM_WORDS)) + r")(?!\w)", re.IGNORECASE)


def looks_harmful(text: str) -> bool:
    return bool(_HARM_RE.search(text))


def length_match(harmful: list[dict], harmless: list[dict], rng: random.Random) -> list[dict]:
    """Keep harmless prompts inside the 5th-95th percentile of harmful lengths and sample them
    to the same length-decile proportions as the harmful set, so the fingerprint doesn't measure length."""
    lens = sorted(len(r["prompt"]) for r in harmful)
    lo, hi = lens[int(0.05 * len(lens))], lens[max(0, int(0.95 * len(lens)) - 1)]
    pool = [r for r in harmless if lo <= len(r["prompt"]) <= hi]
    edges = [lens[int(q * (len(lens) - 1))] for q in [i / 10 for i in range(11)]]

    def bin_of(n: int) -> int:
        for i in range(10):
            if n <= edges[i + 1]:
                return i
        return 9

    by_bin: dict[int, list[dict]] = defaultdict(list)
    for r in pool:
        by_bin[bin_of(len(r["prompt"]))].append(r)
    target = Counter(bin_of(len(r["prompt"])) for r in harmful)
    out: list[dict] = []
    for b, n in sorted(target.items()):
        rng.shuffle(by_bin[b])
        out += by_bin[b][:n]
    return out


# --------------------------------------------------------------------------- loading


def load_redteam(lang_hint: str) -> list[dict]:
    """Harmful prompts for one language from Aya Red-teaming. `lang_hint` matches the file name (e.g. 'eng')."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(REDTEAM_REPO, repo_type="dataset") if f.endswith(".jsonl")]
    match = [f for f in files if lang_hint in f.lower()]
    if not match:
        raise FileNotFoundError(f"No Aya Red-teaming file for hint {lang_hint!r}; files: {files}")
    path = hf_hub_download(REDTEAM_REPO, match[0], repo_type="dataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [
        {
            "prompt": r["prompt"],
            "harm_category": r.get("harm_category"),
            "global_or_local": r.get("global_or_local"),
            "source": f"{REDTEAM_REPO}/{match[0]}",
        }
        for r in rows
        if r.get("prompt")
    ]


_AYA_CACHE: dict = {}


def load_aya(language_names: Iterable[str]) -> list[dict]:
    """Harmless candidates from the Aya Dataset for the given language names (e.g. ['French'])."""
    from datasets import load_dataset

    if "ds" not in _AYA_CACHE:
        _AYA_CACHE["ds"] = load_dataset(AYA_REPO, split="train")
    ds = _AYA_CACHE["ds"]
    lang_col = "language" if "language" in ds.column_names else next(c for c in ds.column_names if "lang" in c.lower())
    text_col = "inputs" if "inputs" in ds.column_names else ds.column_names[0]
    names = set(language_names)
    sub = ds.filter(lambda r: r[lang_col] in names)
    return [{"prompt": t, "source": AYA_REPO} for t in sub[text_col] if t]


def aya_language_counts() -> Counter:
    """Language name -> row count in the Aya Dataset, to check the names in the config."""
    load_aya([])
    ds = _AYA_CACHE["ds"]
    lang_col = "language" if "language" in ds.column_names else next(c for c in ds.column_names if "lang" in c.lower())
    return Counter(ds[lang_col])


# --------------------------------------------------------------------------- splitting and saving


@dataclass
class SplitSizes:
    probe: int = 100
    a: int = 128
    b: int = 128
    min_eval: int = 100


def make_splits(
    harmful: dict[str, list[dict]],
    harmless: dict[str, list[dict]],
    seed: int,
    sizes: SplitSizes = SplitSizes(),
) -> dict[str, list[dict]]:
    """Deterministic, non-overlapping splits per language and label. Probes are drawn first."""
    splits: dict[str, list[dict]] = defaultdict(list)
    for lang in harmful:
        for label, store in (("harmful", harmful), ("harmless", harmless)):
            rows = sorted(store[lang], key=lambda r: r["prompt"])
            random.Random(f"{seed}-{lang}-{label}").shuffle(rows)
            need = sizes.probe + sizes.a + sizes.b
            if len(rows) < need + sizes.min_eval:
                print(f"WARNING {lang} {label}: only {len(rows)} prompts; eval gets {max(0, len(rows) - need)}")
            parts = {
                "probe_secret": rows[: sizes.probe],
                "A": rows[sizes.probe : sizes.probe + sizes.a],
                "B": rows[sizes.probe + sizes.a : need],
                "eval": rows[need:],
            }
            for name, part in parts.items():
                for r in part:
                    splits[name].append(
                        {
                            "prompt": r["prompt"],
                            "label": label,
                            "language": lang,
                            "harm_category": r.get("harm_category"),
                            "n_tokens": r.get("n_tokens"),
                            "source": r.get("source"),
                        }
                    )
    check_no_overlap(splits)
    return dict(splits)


def check_no_overlap(splits: dict[str, list[dict]]) -> None:
    where: dict[str, set[str]] = defaultdict(set)
    for name, rows in splits.items():
        for r in rows:
            where[r["prompt"]].add(name)
    dup = [p for p, s in where.items() if len(s) > 1]
    if dup:
        raise ValueError(f"{len(dup)} prompts appear in more than one split, e.g. {dup[0][:80]!r}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- end-to-end build


@dataclass
class ContrastConfig:
    languages: dict[str, dict]  # lang -> {"redteam_hint": "eng", "aya_names": ["English"]}
    seed: int = 20261104
    max_len_tokens: int = 768
    sizes: SplitSizes = field(default_factory=SplitSizes)
    out_dir: Path = Path("data/contrast_sets")
    use_curatorkit: bool = True
    harmless_licence: str = "CHECK ON DATASET PAGE AND FILL IN"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContrastConfig":
        import yaml

        raw = yaml.safe_load(open(path, encoding="utf-8"))
        sizes = SplitSizes(**raw.pop("sizes", {}))
        out_dir = Path(raw.pop("out_dir", "data/contrast_sets"))
        return cls(sizes=sizes, out_dir=out_dir, **raw)


def build_contrast_sets(
    cfg: ContrastConfig,
    token_counter: Callable[[str], int] | None = None,
    loaders: tuple[Callable, Callable] | None = None,
) -> dict:
    """Run the full pipeline and write public/ and secret/ folders plus manifest.json.

    `token_counter` maps a prompt to its chat-templated token count (see `fingerprint.chat_token_counter`);
    `loaders` = (harmful_loader(hint), harmless_loader(names)) overrides the HF loaders, used in tests.
    Returns the manifest.
    """
    load_h, load_s = loaders or (load_redteam, load_aya)
    rng = random.Random(cfg.seed)
    harmful: dict[str, list[dict]] = {}
    harmless: dict[str, list[dict]] = {}
    dedup_path: dict[str, str] = {}

    for lang, lc in cfg.languages.items():
        raw_h = load_h(lc["redteam_hint"])
        raw_s = [r for r in load_s(lc["aya_names"]) if not looks_harmful(r["prompt"])]
        for label, rows, store in (("harmful", raw_h, harmful), ("harmless", raw_s, harmless)):
            ck = curatorkit_dedup(rows, cfg.out_dir / "_work", f"{lang}_{label}") if cfg.use_curatorkit else None
            store[lang] = ck if ck else builtin_dedup(rows)
            dedup_path[f"{lang}_{label}"] = "curatorkit" if ck else "builtin"
            print(f"{lang} {label}: {len(rows)} -> {len(store[lang])} ({dedup_path[f'{lang}_{label}']})")
        harmless[lang] = length_match(harmful[lang], harmless[lang], rng)

    if token_counter is not None:
        for store in (harmful, harmless):
            for lang in store:
                for r in store[lang]:
                    r["n_tokens"] = token_counter(r["prompt"])
                store[lang] = [r for r in store[lang] if r["n_tokens"] <= cfg.max_len_tokens]

    splits = make_splits(harmful, harmless, cfg.seed, cfg.sizes)

    public_dir, secret_dir = cfg.out_dir / "public", cfg.out_dir / "secret"
    hashes: dict[str, str] = {}
    for name, rows in splits.items():
        path = (secret_dir if name == "probe_secret" else public_dir) / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[path.name] = sha256_file(path)

    manifest = {
        "seed": cfg.seed,
        "languages": list(cfg.languages),
        "sizes": {k: len(v) for k, v in splits.items()},
        "max_len_tokens": cfg.max_len_tokens,
        "dedup_path": dedup_path,
        "sources": {
            "harmful": {"repo": REDTEAM_REPO, "licence": "Apache-2.0"},
            "harmless": {"repo": AYA_REPO, "licence": cfg.harmless_licence},
        },
        "sha256": {k: v for k, v in hashes.items() if k != "probe_secret.jsonl"},
        "probe_commitment_sha256": hashes["probe_secret.jsonl"],
    }
    (public_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_split(dir_: Path, name: str, language: str | None = None, label: str | None = None) -> list[str]:
    """Prompts from a saved split, optionally filtered by language and label."""
    rows = read_jsonl(Path(dir_) / f"{name}.jsonl")
    return [r["prompt"] for r in rows if (language is None or r["language"] == language) and (label is None or r["label"] == label)]
