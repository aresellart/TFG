# Adversarial Stamps

Thesis code. A generative model trained on real stamp images (SPODS) produces stamps that are pasted onto RVL-CDIP document images to make a frozen ResNet-50 document classifier predict a chosen target class. The project first compares generative models (low-KL VAE, high-KL VAE, VAE-GAN, plain GAN), then attacks by optimising a latent vector z per target class, and finally trains one generator per target class end-to-end, so that any random z gives an adversarial stamp. The classifier is never updated.

> The code was run on a GPU server with hardcoded paths (`/home/asellart/tfg_stamps/...`, some older scripts use `/mnt/c/...` WSL paths). Data and checkpoints are not in git, so nothing here runs on a laptop as-is.

## Folder map

```
models/              network definitions + frozen classifier weights (rvl-resnet50.model)
common/              shared code and data: stamp dataset loader, patch pasting, list of correctly classified docs
phase1_generative/   generative models trained on SPODS stamps
  lowkl_vae/           VAE, low KL weight (first variant)
  highkl_vae_resblocks/ VAE, high KL + residual blocks
  vaegan/              VAE-GAN: train_v3.py is the model used downstream
  plain_gan/           decoder-only GAN (architecture-ablation baseline)
phase2_zopt/         z-optimisation attacks (Methods 1a, 1b, 2, 3) + their evaluation/qualitative scripts
phase3_endtoend/     end-to-end generator training/evaluation (fixed and random stamp position)
phase3b_diagnostic/  diagnostic discriminator + fine-tuning of the end-to-end generators
ablations/           position, size and architecture (plain GAN) ablations
explorations/        things tried but not in the final pipeline (higher lambda_real, diversity, morph, anchor)
figures/             figure scripts and the committed PDFs
results/             committed result JSONs (copied from the server; original folder names kept)
archive/             superseded / unused code (SSIM VAE, sampling tests, dataset sanity check)
attack/              3 scripts whose role is still unclear (see docs/OPEN_QUESTIONS.md)
docs/                FILE_MAP.md (old→new paths, roles), OBSERVATIONS.md, OPEN_QUESTIONS.md
```

The full file-by-file map is in [`docs/FILE_MAP.md`](docs/FILE_MAP.md).

## How to run

Run every script **from the repository root**. Most scripts are one folder deep and add the root to `sys.path` themselves (`python phase3_endtoend/train_endtoend.py`). The three `phase1_generative/lowkl_vae/` scripts must be run as modules: `python -m phase1_generative.lowkl_vae.train`. Install the dependencies with `pip install -r requirements.txt` (unpinned).

Order:

0. **Data setup.** `download_rvl.py` saves RVL-CDIP to `rvl_cdip_full/`. Grayscale SPODS stamps are expected in `stamp_dataset_grayscale/` (not produced by any script in this repo). `common/build_correct_list.py` lists the documents the frozen ResNet-50 classifies correctly. The committed result is `common/correct_images.json`.
1. **Phase 1, generative models** (`phase1_generative/`): `lowkl_vae/train.py`, `highkl_vae_resblocks/train_v2.py`, `vaegan/train_v3.py`, `plain_gan/train_gan.py`. `vaegan/finetune_v3.py` exists but its output (`vae_gan_finetune_best.pt`) was **not** used by any later phase. Everything downstream loads `vae_gan_best.pt` from `train_v3.py`.
2. **z-optimisation** (`phase2_zopt/`): `attack.py` (Method 1a, low-KL VAE, CPU), `attack_gpu_vanilla.py` (1b, low-KL VAE, GPU), `attack_gpu_vaegan_encoded.py` (2, VAE-GAN, encoded init), `attack_gpu_vaegan_random.py` (3, VAE-GAN, random z). Then `evaluate_attack.py` (1a) and `evaluate_vaegan.py` (others; edit its `VARIANTS` list), and the `qualitative_*` / `visualize_results.py` scripts.
3. **End-to-end generators** (`phase3_endtoend/`): `train_endtoend.py` (fixed bottom-right) and `train_endtoend_randpos.py` (random position), then `evaluate_endtoend.py` / `evaluate_endtoend_randpos.py` and `interpolate_endtoend.py`. Shared settings are in `config_endtoend.py`.
3b. **Diagnostic discriminator** (`phase3b_diagnostic/`): `train_diagnostic_discriminator.py`, then fine-tuning, then `evaluate_diagfinetune_final.py`. **The two-discriminator fine-tuning script behind the reported result is not in this repo yet.** `finetune_diag_singledisc.py` is a later single-discriminator test (see Known issues).
4. **Ablations** (`ablations/`): need the generators from step 3 (and, for `*_gan_randpos`, the plain GAN from step 1).
5. **Figures** (`figures/`): note that `plot_*.py` contain hardcoded numbers rather than reading `results/`.

Scripts still write their outputs to the repo root or to root-level `attack_results_*/` folders, as they did on the server. `results/` and `figures/` hold the committed copies.

## Checkpoints (not in git, they live on the GPU server)

| File | Produced by | Used by |
|---|---|---|
| `models/rvl-resnet50.model` | (pre-trained, **is** in git) | everything: the frozen classifier |
| `vanilla_vae_stamps.pt` | `phase1_generative/lowkl_vae/train.py` | Methods 1a, 1b |
| `vae_high_kl_best.pt` | `phase1_generative/highkl_vae_resblocks/train_v2.py` | Phase 1 comparison only |
| `vae_gan_best.pt`, `disc_gan_best.pt` | `phase1_generative/vaegan/train_v3.py` | Methods 2, 3; init and realism discriminator for all end-to-end generators |
| `gan_generator_best.pt`, `gan_disc_best.pt` | `phase1_generative/plain_gan/train_gan.py` | `ablations/*_gan_randpos.py` |
| `G_<c>_<class>.pt` (16 per folder) | `phase3_endtoend/train_endtoend*.py`, `ablations/train_endtoend_gan_randpos.py` | evaluation, ablations, interpolation, diagnostic phase |
| `diagnostic_discriminator.pt` | `phase3b_diagnostic/train_diagnostic_discriminator.py` | fine-tuning |
| `G_<c>_<class>_diagfinetune.pt` | the missing two-discriminator fine-tuning script | `phase3b_diagnostic/evaluate_diagfinetune_final.py` |
| `z_class<c>_<class>.pt` | `phase2_zopt/attack*.py` | phase2 evaluation / figures |

## Headline results (from `results/`)

Mean attack success rate over the 16 target classes. Each value is the `overall` field in the JSON named:

| Setting | ASR (%) | Source |
|---|---:|---|
| End-to-end, fixed bottom-right | 97.7 | `results/attack_results_endtoend/evaluation_endtoend.json` |
| End-to-end, random position | 95.3 | `results/attack_results_endtoend_randpos/evaluation_endtoend_randpos.json` |
| + diagnostic-discriminator fine-tuning (two-disc version) | 96.8 | `results/attack_results_endtoend_diagfinetune_final/evaluation_diagfinetune.json` |
| Plain-GAN initialisation, random position (ablation) | 83.8 | `results/attack_results_endtoend_gan_randpos/evaluation_endtoend_gan_randpos.json` |

## Known issues

Listed in [`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md); none were fixed. The most important:

- `phase3b_diagnostic/finetune_diag_singledisc.py` is the single-discriminator test, not the two-discriminator version that produced the 96.8% result.
- Fine-tuning and the diagnostic discriminator use the **fixed-position** generators (`attack_results_endtoend`), not the random-position ones.
- End-to-end generators are trained in `train()` mode with one z per step (BatchNorm with batch size 1) and evaluated in `eval()` mode.
- `common/stamps_dataset.py` may differ from the server's `datasets_local/stamps_dataset.py`.
- Some hardcoded numbers in `figures/plot_*.py` do not match `results/`.

Open questions about where some files belong are in [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md).
