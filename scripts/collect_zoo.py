"""Collect Stage 1 results into one table.

    python scripts/collect_zoo.py                       # from outputs/zoo/*/results.json
    python scripts/collect_zoo.py --from-hub team-doom  # download everyone's results first
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/zoo")
    ap.add_argument("--from-hub", default=None, help="HF org; downloads zoo results from <org>/forged-fingerprints-results")
    args = ap.parse_args()

    root = Path(args.root)
    if args.from_hub:
        from huggingface_hub import snapshot_download

        local = snapshot_download(f"{args.from_hub}/forged-fingerprints-results", repo_type="dataset",
                                  allow_patterns=["zoo/*/results.json"])
        root = Path(local) / "zoo"

    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/results.json"))]
    if not rows:
        raise SystemExit(f"no results under {root}")
    rows.sort(key=lambda r: (r["type"] != "none", r.get("harmful_fraction", 0), r["name"]))
    cols = ["name", "type", "harmful_fraction", "fingerprint_cosine", "fingerprint_min_layer_cosine",
            "refusal_harmful_all", "refusal_harmless_all"]
    cols += sorted({k for r in rows for k in r if k.startswith("refusal_") and k not in cols})

    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))
    out = Path("outputs/zoo_table.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
