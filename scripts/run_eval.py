"""Evaluate saved generations.jsonl across zoo/attack runs with keyword refusal + AuditKIT judge (if installed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forged.evaluate import is_keyword_refusal


def try_auditkit_score(prompts: list[str], responses: list[str]) -> float | None:
    try:
        import auditkit  # type: ignore

        if hasattr(auditkit, "evaluate_refusal"):
            res = auditkit.evaluate_refusal(prompts=prompts, responses=responses)
            return float(res.get("refusal_rate", 0.0))
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", help="Root directory containing model run folders")
    args = ap.parse_args()

    root = Path(args.root)
    for gen_file in sorted(root.rglob("generations.jsonl")):
        run_dir = gen_file.parent
        res_file = run_dir / "results.json"
        rows = [json.loads(line) for line in gen_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts = [r["prompt"] for r in rows]
        responses = [r["response"] for r in rows]

        kw_rate = sum(is_keyword_refusal(r) for r in responses) / max(len(responses), 1)
        ak_rate = try_auditkit_score(prompts, responses)

        if res_file.exists():
            data = json.loads(res_file.read_text(encoding="utf-8"))
        else:
            data = {"name": run_dir.name}
        data["keyword_refusal_rate"] = kw_rate
        if ak_rate is not None:
            data["auditkit_refusal_rate"] = ak_rate
        res_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[{run_dir.name}] keyword_refusal={kw_rate:.3f} auditkit={ak_rate}")


if __name__ == "__main__":
    main()
