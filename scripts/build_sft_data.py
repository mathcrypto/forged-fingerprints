"""Build the Stage 1 fine-tuning data and push it to a private HF dataset repo.

    python scripts/build_sft_data.py --config configs/contrast_sets.yaml --push-org team-doom

Benign pairs come from the Aya Dataset (EN/FR/HI, contrast-set prompts excluded); harmful pairs from BeaverTails.
The harmful file stays in a private repo and is never published.
"""

import argparse
import json
from pathlib import Path

from forged.data import ContrastConfig
from forged.sft import build_sft_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/contrast_sets.yaml", help="for the language list")
    ap.add_argument("--contrast-dir", default="data/contrast_sets/public")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--max-benign-per-language", type=int, default=3000)
    ap.add_argument("--push-org", default=None)
    ap.add_argument("--no-curatorkit", action="store_true")
    args = ap.parse_args()

    cfg = ContrastConfig.from_yaml(args.config)
    manifest = build_sft_data(cfg.languages, Path(args.contrast_dir), Path(args.out_dir),
                              args.max_benign_per_language, use_curatorkit=not args.no_curatorkit, seed=cfg.seed)
    (Path(args.out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    if args.push_org:
        from huggingface_hub import HfApi

        api = HfApi()
        repo = f"{args.push_org}/forged-fingerprints-sft"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for name in ("benign.jsonl", "harmful.jsonl", "manifest.json"):
            api.upload_file(path_or_fileobj=str(Path(args.out_dir) / name), path_in_repo=name, repo_id=repo, repo_type="dataset")
        print("Pushed (private):", repo)


if __name__ == "__main__":
    main()
