# Forged Fingerprints

**Can a fine-tune break safety while keeping its safety fingerprint?**

Team Doom · Lexsi AI Safety & Alignment Hackathon 2026 · Track 2 (AI Red Teaming, Safety and Alignment)

## The problem

Open-weight models like Tiny Aya get fine-tuned and re-uploaded constantly, and fine-tuning, even on harmless data, can quietly remove a model's safety training. Running a full red-team evaluation on every upload is too slow for a model hub, so a cheap shortcut has been proposed: compare the fine-tune's **refusal fingerprint** (the direction in its activations that separates harmful from harmless prompts) with the original's. If it is still close, assume safety survived.

That check has only been tested against ordinary fine-tunes. We ask what happens when the uploader knows the check and tries to beat it, and we build a check that holds up.

## Approach

| Stage | What we do | Lexsi tools |
|---|---|---|
| 1. Calibrate | Fine-tune a zoo of Tiny Aya variants (benign, contaminated, harmful, 4-bit) and test whether fingerprint similarity tracks measured harm in English, French and Hindi | SafeTune, AuditKIT, CuratorKIT |
| 2. Attack | **2a:** fine-tune only the layers after the fingerprint's layers. **2b:** harmful fine-tune with a loss term that keeps the fingerprint close to the reference | AuditKIT |
| 3. Harden | Combine the fingerprint, refusal-circuit overlap and a secret multilingual probe set; commit to the reference with SHA-256 before any attack | CircuitKIT, AuditKIT |
| 4. Repair | Restore safety in flagged models with ReSta, re-measure harm and utility | SafeTune, AuditKIT |

Models: [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (target) and [Tiny Aya Base](https://huggingface.co/CohereLabs/tiny-aya-base) (for ReSta).

## Repository layout

```
src/forged/        reusable code
  data.py          contrast prompt sets: load, screen, dedup, length-match, split, hash
  fingerprint.py   refusal-vector fingerprint and similarity (differentiable core)
  attack.py        Stage 2a layer helpers, Stage 2b fingerprint-preservation loss
  check.py         SHA-256 commitments, hardened multi-signal check
  evaluate.py      generation and quick refusal scoring (reported numbers come from AuditKIT)
configs/           one YAML per run
scripts/           command-line entry points
notebooks/         Colab notebooks that call the package
tests/             CPU tests with a tiny fake model, no downloads
```

## Quickstart (Colab)

1. Open `notebooks/00_setup.ipynb` in Colab (File > Open notebook > GitHub), GPU runtime.
2. Add two Colab secrets: `HF_TOKEN` (Hugging Face, read + write) and `GH_TOKEN` (GitHub, read access to this repo while it is private).
3. Run `00_setup`, then `01_contrast_sets`, then `02_reference_fingerprint`.

Locally:

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[lexsi,train]"    # + SafeTune, AuditKIT, CircuitKIT, CuratorKIT, PEFT (Python >= 3.12)
pytest
python scripts/build_contrast_sets.py --config configs/contrast_sets.yaml --push-org YOUR-HF-ORG
python scripts/compute_fingerprint.py --config configs/fingerprint.yaml --split A
```

## Data

| Data | Source | Licence | Use |
|---|---|---|---|
| Harmful prompts (EN, FR, HI) | [Aya Red-teaming](https://huggingface.co/datasets/CohereForAI/aya_redteaming) | Apache-2.0 | Fingerprint, eval, secret probes |
| Harmless prompts (EN, FR, HI) | [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | to confirm | Fingerprint, eval, secret probes |
| Harmful fine-tuning data | BeaverTails (via SafeTune) | to confirm | Zoo and attack fine-tunes |

Splits are deterministic (seed in `configs/contrast_sets.yaml`) and never overlap. The secret probe set's SHA-256 is published before any attack run and the file is revealed afterwards, so anyone can check it was not changed.

## Safety and release

- Forged and harmful adapters are never published; only reference, benign and repaired artifacts go public.
- Public logs keep prompts, labels and scores; harmful completions are redacted and replaced by their hash.
- The headline deliverables are the hardened check and the repair; the attack shows why they are needed.

## Workflow

Each teammate works on a branch and opens a pull request into `main`; changes to `src/forged/` get a review from the other. Data and models go to Hugging Face, never to git.

## References

- Xu & Sheng (2026), *A Behavioral Fingerprint for Large Language Models: Provenance Tracking via Refusal Vectors*, [arXiv:2602.09434](https://arxiv.org/abs/2602.09434)
- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Lexsi Labs: [SafeTune](https://github.com/Lexsi-Labs/SafeTune), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT), [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)
