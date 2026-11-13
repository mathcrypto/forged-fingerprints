# Forged Fingerprints

**Can a fine-tune break safety while keeping its safety fingerprint?**

Team Doom · Lexsi AI Safety & Alignment Hackathon 2026 · Track 2 (AI Red Teaming, Safety and Alignment)

## The problem

Open-weight models like Tiny Aya get fine-tuned and re-uploaded constantly, and fine-tuning, even on harmless data, can quietly remove a model's safety training. Running a full red-team evaluation on every upload is too slow for a model hub, so a cheap shortcut has been proposed: compare the fine-tune's **refusal fingerprint** (the direction in its activations that separates harmful from harmless prompts) with the original's. If it is still close, assume safety survived.

That check has only been tested against ordinary fine-tunes. We ask what happens when the uploader knows the check and tries to beat it, and we build a check that holds up.

## Approach

| Stage | What we do | Lexsi tools |
|---|---|---|
| **0. Setup & data** | Harmful/harmless contrast prompt sets + Aya SFT dataset | SafeTune (`sft_dataset_aya_en.yaml`), CuratorKIT |
| **1. Baseline & Zoo** | Measure baseline fingerprint; train model zoo (ordinary fine-tunes) | SafeTune (`Contaminated-20`, `Contaminated-50`, `Benign-100`) |
| **2. Attacks** | Build fingerprint-preserving attacks (Stage 2a: joint loss; Stage 2b: orthogonal projection) | AuditKIT (`AuditEvaluator`), CircuitKIT |
| **3. Hardened check** | Multi-signal attestation check (fingerprint + circuit overlap + secret probes + commitments) | CircuitKIT, AuditKIT |
| **4. Repair** | Safety repair / cleanup for forged models | SafeTune |

## Code structure

```
forged_fingerprints/
  contrast.py      contrast prompt sets: load, screen, dedup, length-match, split, hash
  fingerprint.py   refusal-vector fingerprint and similarity (differentiable core)
  sft.py           Stage 1 fine-tuning data (Aya Dataset benign pairs, BeaverTails harmful pairs)
  train.py         LoRA fine-tuning sized for a free T4 (also used by the attacks)
  models.py        model loading: dtype across transformers versions, adapters, 4-bit
  attack.py        Stage 2a layer helpers, Stage 2b fingerprint-preservation loss
  check.py         SHA-256 commitments, hardened multi-signal check
  evaluate.py      generation and quick refusal scoring (reported numbers come from AuditKIT)
configs/           one YAML per run
scripts/           command-line entry points
notebooks/         Colab notebooks that call the package
tests/             CPU tests with a tiny fake model, no downloads
```

## Quickstart (Colab)

1. Add two Colab secrets (key icon, Notebook access on): `HF_TOKEN` (Hugging Face, write access to the `team-doom` organisation) and `GH_TOKEN` (GitHub, read access to this repo while it is private).
2. Open a notebook from GitHub (File > Open notebook > GitHub). Each notebook is self-contained: its first cell clones the repo, installs the package and logs in.

| Notebook | Runtime | What it does | Status |
|---|---|---|---|
| `00_setup` | GPU | Optional environment check | |
| `01_contrast_sets` | CPU | Builds and pushes the contrast prompt sets | done |
| `02_reference_fingerprint` | GPU | Reference fingerprint + A/B stability check | done, cosine 0.9953 |
| `03_stage1_zoo` | GPU | Calibration models: fingerprint vs. refusals | next |
| `04_stage2_attacks` | GPU | Stage 2a & 2b fingerprint-preserving attacks | |
| `05_stage3_hardened_check` | GPU | Hardened multi-signal attestation check | |
| `06_stage4_repair` | GPU | Stage 4 safety repair / cleanup | |

Results land in private Hugging Face repos under `team-doom`: `forged-fingerprints-contrast`, `forged-fingerprints-probes-secret`, `forged-fingerprints-sft`, `forged-fingerprints-results` and one `forged-zoo-<variant>` per adapter.

Locally:

```bash
pip install -e ".[dev]"               # core + tests
pip install -e ".[lexsi,train]"       # + SafeTune, AuditKIT, CircuitKIT, CuratorKIT, PEFT (Python >= 3.12)
pytest
python scripts/build_contrast_sets.py --config configs/contrast_sets.yaml --push-org YOUR-HF-ORG
python scripts/compute_fingerprint.py --config configs/fingerprint.yaml --split A
python scripts/build_sft_data.py --push-org team-doom
python scripts/train_zoo.py --config configs/zoo/contaminated_20.yaml
python scripts/collect_zoo.py --from-hub team-doom
python scripts/run_attack.py --config configs/attacks/stage2b.yaml
python scripts/run_eval.py --model team-doom/forged-zoo-contaminated-20
python scripts/run_hardened_check.py --model team-doom/forged-zoo-contaminated-20
python scripts/run_repair.py --config configs/repair/resta.yaml
```

## Data

Contrast sets (`A` and `B`, 100 prompt pairs each, balanced length) and secret multilingual probes (`50` prompts across 5 languages) are built from BeaverTails (`harmful`) and Aya Dataset (`harmless`).

## License

MIT
