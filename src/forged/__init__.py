"""Forged Fingerprints: adaptive attacks on refusal-vector safety attestation.

Modules
-------
data         build harmful/harmless contrast prompt sets (Aya Red-teaming, Aya Dataset)
fingerprint  compute a model's refusal-vector fingerprint and compare two of them
attack       fingerprint-preserving loss and layer helpers for the adaptive attack
check        hash commitments and the hardened multi-signal check
evaluate     generation and refusal scoring helpers (AuditKIT is wired in on top)
"""

__version__ = "0.1.0"
