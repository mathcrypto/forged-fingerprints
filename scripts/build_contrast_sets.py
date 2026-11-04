"""Build the harmful/harmless contrast sets and optionally push them to Hugging Face.

    python scripts/build_contrast_sets.py --config configs/contrast_sets.yaml
    python scripts/build_contrast_sets.py --config configs/contrast_sets.yaml --push-org YOUR-HF-ORG

Pushes two private dataset repos: <org>/forged-fingerprints-contrast (A, B, eval, manifest) and
<org>/forged-fingerprints-probes-secret (secret probes, revealed only after the final run).
"""

import argparse
import json

from forged.data import ContrastConfig, aya_language_counts, build_contrast_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/contrast_sets.yaml")
    ap.add_argument("--push-org", default=None, help="Hugging Face organisation or user to push private repos to")
    ap.add_argument("--no-token-check", action="store_true", help="skip the Tiny Aya tokenizer length filter")
    args = ap.parse_args()

    cfg = ContrastConfig.from_yaml(args.config)

    counts = aya_language_counts()
    for lang, lc in cfg.languages.items():
        found = {n: counts.get(n, 0) for n in lc["aya_names"]}
        print(f"Aya Dataset rows for {lang}: {found}")
        if not any(found.values()):
            raise SystemExit(f"No Aya Dataset rows for {lang}; fix aya_names in {args.config}")

    counter = None
    if not args.no_token_check:
        from transformers import AutoTokenizer

        from forged.fingerprint import chat_token_counter

        counter = chat_token_counter(AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global"))

    manifest = build_contrast_sets(cfg, token_counter=counter)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\nPROBE COMMITMENT (publish this):", manifest["probe_commitment_sha256"])

    if args.push_org:
        from huggingface_hub import HfApi

        api = HfApi()
        public_repo = f"{args.push_org}/forged-fingerprints-contrast"
        secret_repo = f"{args.push_org}/forged-fingerprints-probes-secret"
        for repo in (public_repo, secret_repo):
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(cfg.out_dir / "public"), repo_id=public_repo, repo_type="dataset",
                          commit_message=f"Contrast sets, seed {cfg.seed}")
        api.upload_folder(folder_path=str(cfg.out_dir / "secret"), repo_id=secret_repo, repo_type="dataset",
                          commit_message=f"Secret probes, commitment {manifest['probe_commitment_sha256'][:16]}")
        print("Pushed (private):", public_repo, "and", secret_repo)


if __name__ == "__main__":
    main()
