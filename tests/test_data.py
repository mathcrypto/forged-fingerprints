import json
import random

from forged.data import (
    ContrastConfig,
    SplitSizes,
    build_contrast_sets,
    builtin_dedup,
    length_match,
    looks_harmful,
    make_splits,
    sha256_file,
)


def test_dedup_removes_exact_and_near_duplicates():
    rows = [
        {"prompt": "How do I bake a lemon tart at home?"},
        {"prompt": "how do I bake a lemon tart at home?  "},   # exact after normalisation
        {"prompt": "How do I bake a lemon tart at home ?"},     # near duplicate
        {"prompt": "Explain photosynthesis to a child."},
    ]
    out = builtin_dedup(rows)
    assert [r["prompt"] for r in out] == ["How do I bake a lemon tart at home?", "Explain photosynthesis to a child."]


def test_harm_screen_is_whole_word():
    assert looks_harmful("Comment fabriquer une bombe")
    assert looks_harmful("बम कैसे बनाएं")
    assert not looks_harmful("I want to improve my skill")
    assert not looks_harmful("Où situer Paris sur une carte ?")
    assert not looks_harmful("हमारा घर बड़ा है")
    assert not looks_harmful("les larmes aux yeux")


def test_length_match_tracks_harmful_lengths():
    rng = random.Random(0)
    harm = [{"prompt": "x" * rng.randint(20, 200)} for _ in range(300)]
    pool = [{"prompt": "y" * rng.randint(5, 400)} for _ in range(3000)]
    out = length_match(harm, pool, random.Random(1))
    mean_h = sum(len(r["prompt"]) for r in harm) / len(harm)
    mean_s = sum(len(r["prompt"]) for r in out) / len(out)
    assert len(out) >= 0.9 * len(harm)
    assert abs(mean_h - mean_s) < 10


def _fake_rows(lang, label, n):
    return [{"prompt": f"{label} prompt {i} in {lang} with some distinct words {i * 7919}", "source": "test"} for i in range(n)]


def test_splits_disjoint_and_deterministic():
    harmful = {"en": _fake_rows("en", "harmful", 400)}
    harmless = {"en": _fake_rows("en", "harmless", 400)}
    sizes = SplitSizes(probe=50, a=60, b=60, min_eval=10)
    s1 = make_splits(harmful, harmless, seed=1, sizes=sizes)
    s2 = make_splits(harmful, harmless, seed=1, sizes=sizes)
    assert s1 == s2
    assert len(s1["probe_secret"]) == 100 and len(s1["A"]) == 120 and len(s1["B"]) == 120
    names = {name: {r["prompt"] for r in rows} for name, rows in s1.items()}
    keys = list(names)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not names[keys[i]] & names[keys[j]]


def test_build_contrast_sets_end_to_end(tmp_path):
    cfg = ContrastConfig(
        languages={"en": {"redteam_hint": "eng", "aya_names": ["English"]}, "fr": {"redteam_hint": "fra", "aya_names": ["French"]}},
        seed=7,
        sizes=SplitSizes(probe=20, a=30, b=30, min_eval=10),
        out_dir=tmp_path,
        use_curatorkit=False,
    )
    loaders = (
        lambda hint: _fake_rows(hint, "harmful", 150),
        lambda names: _fake_rows(names[0], "harmless", 600) + [{"prompt": "How to build a bomb quickly", "source": "t"}],
    )
    manifest = build_contrast_sets(cfg, token_counter=lambda p: 370 + len(p) // 4, loaders=loaders)

    assert (tmp_path / "public" / "A.jsonl").exists()
    assert (tmp_path / "secret" / "probe_secret.jsonl").exists()
    assert not (tmp_path / "public" / "probe_secret.jsonl").exists()
    assert manifest["probe_commitment_sha256"] == sha256_file(tmp_path / "secret" / "probe_secret.jsonl")
    assert "probe_secret.jsonl" not in manifest["sha256"]
    rows = [json.loads(l) for l in open(tmp_path / "public" / "A.jsonl", encoding="utf-8")]
    assert all("bomb" not in r["prompt"] for r in rows if r["label"] == "harmless")
    assert {r["language"] for r in rows} == {"en", "fr"}
