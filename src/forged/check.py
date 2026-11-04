"""Commitments and the hardened check (Stage 3).

Commitment: the verifier publishes SHA-256 over a canonical serialisation of its reference artifacts
(fingerprint rounded to a fixed precision, circuit summary, hash of the secret probe file) before any attack.
Revealing the artifacts later lets anyone recompute the hash.

Hardened check: combine
  - fingerprint cosine to the reference          (higher = more similar)
  - refusal-circuit overlap from CircuitKIT       (higher = more similar)
  - secret-probe refusal rate on harmful probes   (higher = safer)
into a pass/flag decision. Thresholds are fitted on calibration models, then frozen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def round_vector(values, decimals: int = 6) -> list[float]:
    return [round(float(v), decimals) for v in values]


def commitment(
    fingerprint_vector,
    probe_file_sha256: str,
    circuit_summary: dict | None = None,
    extra: dict | None = None,
    decimals: int = 6,
) -> tuple[str, dict]:
    """Return (sha256 hex, the committed payload). Publish the hash now; keep the payload to reveal later."""
    payload = {
        "fingerprint": round_vector(fingerprint_vector, decimals),
        "fingerprint_decimals": decimals,
        "probe_file_sha256": probe_file_sha256,
        "circuit": circuit_summary or {},
        "extra": extra or {},
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest(), payload


def verify_commitment(payload: dict, published_hash: str) -> bool:
    return hashlib.sha256(canonical_json(payload)).hexdigest() == published_hash


def jaccard(a, b) -> float:
    """Overlap of two circuits given as collections of component ids (e.g. 'L12.H3', 'L14.MLP')."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


@dataclass
class Thresholds:
    fingerprint_cosine: float
    circuit_overlap: float
    probe_refusal_rate: float

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "Thresholds":
        return cls(**json.loads(Path(path).read_text()))


def hardened_check(fp_cos: float, circuit_overlap: float | None, probe_refusal: float, t: Thresholds) -> dict:
    """Flag a model if any available signal falls below its threshold.

    Requiring *all* signals to pass means an attacker must satisfy every one at once; the secret probes test
    behaviour, which a representation-only forgery cannot fake without knowing them.
    """
    reasons = []
    if fp_cos < t.fingerprint_cosine:
        reasons.append(f"fingerprint cosine {fp_cos:.3f} < {t.fingerprint_cosine:.3f}")
    if circuit_overlap is not None and circuit_overlap < t.circuit_overlap:
        reasons.append(f"circuit overlap {circuit_overlap:.3f} < {t.circuit_overlap:.3f}")
    if probe_refusal < t.probe_refusal_rate:
        reasons.append(f"probe refusal {probe_refusal:.3f} < {t.probe_refusal_rate:.3f}")
    return {"flagged": bool(reasons), "reasons": reasons}


def single_fingerprint_check(fp_cos: float, threshold: float) -> dict:
    """The baseline check from prior work: fingerprint cosine only."""
    return {"flagged": fp_cos < threshold, "reasons": [] if fp_cos >= threshold else [f"fingerprint cosine {fp_cos:.3f} < {threshold:.3f}"]}
