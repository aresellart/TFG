# File map

Every file tracked on `main` before the reorganisation, where it went, what it does, and what it feeds.

- **Old path**: path on `main`. **New path**: path on branch `reorganize`; "(unchanged)" means not moved.
- **Paper table/figure**: I don't have the thesis text, so this column names the result file or figure the script produces. Numbers in brackets come from the JSONs in `results/`. "?" means I could not tell.
- Phase numbering follows the folder names (phase2 = z-optimisation, phase3 = end-to-end). The docstrings inside the end-to-end scripts call end-to-end "Phase 2"; see `OPEN_QUESTIONS.md`.
- Files added by the reorganisation: `__init__.py` in each new code folder, `docs/*`, `requirements.txt`.

| Old path | New path | Role | Paper table/figure (what it feeds) |
|---|---|---|---|
| `.gitignore` | (unchanged) | unchanged | - |
| `README.md` | (unchanged) | rewritten | - |
| `attack/__init__.py` | (unchanged) | package marker for the 3 unclear files still in attack/ | - |
| `attack/attack.py` | `phase2_zopt/attack.py` | Phase 2 z-opt **Method 1a**: low-KL VAE, encoded init, CPU, batch 8, 300 it | `results/attack_results/` via evaluate_attack.py; 'VAE low-KL encoded (CPU)' row in figures/comparison_*.pdf |
| `attack/attack_gpu_vaegan_encoded.py` | `phase2_zopt/attack_gpu_vaegan_encoded.py` | Phase 2 z-opt **Method 2**: VAE-GAN, encoded init, GPU | `results/evaluation_gpu_vaegan_encoded.json`; 'VAE-GAN encoded (GPU)' in comparison_*.pdf |
| `attack/attack_gpu_vaegan_random.py` | `phase2_zopt/attack_gpu_vaegan_random.py` | Phase 2 z-opt **Method 3**: VAE-GAN, z~N(0,I), GPU | `results/evaluation_gpu_vaegan_random.json`; 'VAE-GAN random z (GPU)' in comparison_*.pdf |
| `attack/attack_gpu_vanilla.py` | `phase2_zopt/attack_gpu_vanilla.py` | Phase 2 z-opt **Method 1b**: low-KL VAE, encoded init, GPU, batch 128, 2000 it | `results/evaluation_gpu_vanilla_encoded.json`; 'VAE low-KL encoded (GPU)' in comparison_*.pdf |
| `attack/attack_random_z.py` | (unchanged) | UNCLEAR, left in place. Low-KL VAE random-z z-opt, 4 classes | `results/attack_results_random_z/comparison_results.json` (via evaluate_random_z.py) |
| `attack/attack_vae_gan.py` | (unchanged) | UNCLEAR, left in place. Earlier VAE-GAN z-opt (batch 64, 1000 it), variants encoded_stamp/random_z | its runs = `results/evaluation_vaegan_{encoded_stamp,random_z}.json` = the two VAE-GAN series in figures/scatter_comparison.pdf; its random_z dir feeds figures/qualitative_vaegan_random_z.pdf |
| `attack/build_correct_list.py` | `common/build_correct_list.py` | builds list of correctly classified RVL-CDIP docs (writes `correct_images_1.0.json`, see OBSERVATIONS) | common/correct_images.json (used by every attack/eval) |
| `attack/evaluate_attack.py` | `phase2_zopt/evaluate_attack.py` | evaluates Method 1a z vectors | `results/attack_results/evaluation_results_full.json` |
| `attack/evaluate_random_z.py` | (unchanged) | UNCLEAR, left in place. Evaluates attack_random_z.py output | `results/attack_results_random_z/comparison_results.json` |
| `attack/evaluate_vaegan.py` | `phase2_zopt/evaluate_vaegan.py` | evaluates z-opt result dirs, writes `evaluation_<name>.json` in repo root (VARIANTS list was edited between runs) | `results/evaluation_*.json` |
| `attack/patch_utils.py` | `common/patch_utils.py` | shared: document preprocessing + stamp pasting | all phases |
| `attack/qualitative_results.py` | `phase2_zopt/qualitative_results.py` | qualitative figure, Method 1a | attack_results/qualitative_results.png (not in git) |
| `attack/qualitative_vaegan.py` | `phase2_zopt/qualitative_vaegan.py` | qualitative figure from `attack_results_vaegan_random_z/` (attack_vae_gan.py output, not Method 3's dir) | figures/qualitative_vaegan_random_z.pdf |
| `attack/test_model.py` | `archive/attack/test_model.py` | archived: prints one RVL-CDIP sample, dataset sanity check | not in paper |
| `attack/visualize_results.py` | `phase2_zopt/visualize_results.py` | decodes Method 1a z vectors into stamps | ? |
| `attack_results/evaluation_results.json` | `results/attack_results/evaluation_results.json` | result: per-class success rates, low-KL CPU attack (short form) | phase2_zopt/evaluate_attack.py (earlier version; current one writes only _full) |
| `attack_results/evaluation_results_full.json` | `results/attack_results/evaluation_results_full.json` | result: per-class success rates, low-KL CPU attack (with counts) | phase2_zopt/evaluate_attack.py; Method 1a row of figures/comparison_*.pdf (avg 69.6) |
| `attack_results_endtoend/ablation_position/ablation_position_results.json` | `results/attack_results_endtoend/ablation_position/ablation_position_results.json` | result: position ablation, fixed-position generators | ablations/ablation_position.py |
| `attack_results_endtoend/ablation_size.json` | `results/attack_results_endtoend/ablation_size.json` | result: size ablation, fixed-position generators | ablations/ablation_size.py |
| `attack_results_endtoend/evaluation_endtoend.json` | `results/attack_results_endtoend/evaluation_endtoend.json` | result: ASR of end-to-end generators, fixed bottom-right | phase3_endtoend/evaluate_endtoend.py (overall 97.69) |
| `attack_results_endtoend_diagfinetune_final/evaluation_diagfinetune.json` | `results/attack_results_endtoend_diagfinetune_final/evaluation_diagfinetune.json` | result: ASR of diagnostic-disc fine-tuned generators | phase3b_diagnostic/evaluate_diagfinetune_final.py (overall 96.77); generators from the lost two-disc fine-tune |
| `attack_results_endtoend_gan_randpos/evaluation_endtoend_gan_randpos.json` | `results/attack_results_endtoend_gan_randpos/evaluation_endtoend_gan_randpos.json` | result: ASR of plain-GAN-initialised generators, random position | ablations/evaluate_endtoend_gan_randpos.py (overall 83.76) |
| `attack_results_endtoend_highreal_test/evaluation_highreal_test.json` | `results/attack_results_endtoend_highreal_test/evaluation_highreal_test.json` | result: ASR, lambda_real=0.3 test, 3 classes | explorations/evaluate_highreal_test.py |
| `attack_results_endtoend_randpos/ablation_position_randpos/ablation_position_randpos_results.json` | `results/attack_results_endtoend_randpos/ablation_position_randpos/ablation_position_randpos_results.json` | result: position ablation, random-position generators | ablations/ablation_position_randpos.py |
| `attack_results_endtoend_randpos/evaluation_endtoend_randpos.json` | `results/attack_results_endtoend_randpos/evaluation_endtoend_randpos.json` | result: ASR of end-to-end generators, random position | phase3_endtoend/evaluate_endtoend_randpos.py (overall 95.28) |
| `attack_results_random_z/comparison_results.json` | `results/attack_results_random_z/comparison_results.json` | result: low-KL random-z attack, 4 classes | attack/evaluate_random_z.py |
| `autoencoder/config.py` | `phase1_generative/lowkl_vae/config.py` | Phase 1 low-KL VAE config (identical to root config.py) | - |
| `autoencoder/evaluate.py` | `phase1_generative/lowkl_vae/evaluate.py` | Phase 1 low-KL VAE reconstructions | ? |
| `autoencoder/generate.py` | `phase1_generative/lowkl_vae/generate.py` | Phase 1 low-KL VAE samples/reconstructions | ? |
| `autoencoder/train.py` | `phase1_generative/lowkl_vae/train.py` | Phase 1 low-KL VAE training -> vanilla_vae_stamps.pt | checkpoint for Methods 1a/1b |
| `autoencoder/utils.py` | `phase1_generative/lowkl_vae/utils.py` | vae_loss for low-KL VAE | - |
| `autoencoder_v2/__init__.py` | `phase1_generative/highkl_vae_resblocks/__init__.py` | package marker | - |
| `autoencoder_v2/config_v2.py` | `phase1_generative/highkl_vae_resblocks/config_v2.py` | Phase 1 high-KL VAE config | - |
| `autoencoder_v2/evaluate_v2.py` | `phase1_generative/highkl_vae_resblocks/evaluate_v2.py` | high-KL VAE evaluation plots | probably high-KL interpolation strip used by figures/generative_model_comparison.pdf (?) |
| `autoencoder_v2/train_v2.py` | `phase1_generative/highkl_vae_resblocks/train_v2.py` | Phase 1 high-KL VAE (ResBlocks) training -> vae_high_kl_best.pt | generative model comparison |
| `autoencoder_v3/config_v3.py` | `phase1_generative/vaegan/config_v3.py` | Phase 1 VAE-GAN config (also read by phase3b, plain_gan, figures) | - |
| `autoencoder_v3/evaluate_v3.py` | `phase1_generative/vaegan/evaluate_v3.py` | VAE-GAN evaluation plots (loads the *finetune* checkpoint) | ? |
| `autoencoder_v3/finetune_v3.py` | `phase1_generative/vaegan/finetune_v3.py` | VAE-GAN fine-tune -> vae_gan_finetune_best.pt. NOT used for later phases | none |
| `autoencoder_v3/train_v3.py` | `phase1_generative/vaegan/train_v3.py` | **Phase 1 VAE-GAN used by the paper** -> vae_gan_best.pt, disc_gan_best.pt | checkpoint for Methods 2/3 and all end-to-end generators |
| `comparison_bar_chart.pdf` | `figures/comparison_bar_chart.pdf` | figure | made by plot_comparison_final.py |
| `comparison_scatter.pdf` | `figures/comparison_scatter.pdf` | figure | made by plot_comparison_final.py |
| `comparison_stamps_grid.pdf` | `figures/comparison_stamps_grid.pdf` | figure | made by plot_comparison_final.py |
| `config.py` | (unchanged) | UNCLEAR, left in place. Copy of low-KL config; used via `import config` by phase2_zopt/{evaluate_attack,qualitative_results,visualize_results}.py and attack/{attack_random_z,evaluate_random_z}.py | - |
| `correct_images.json` | `common/correct_images.json` | correctly classified doc indices per class (counts 500,500,500,273,500,500,139,500,500,500,500,500,500,500,134,492) | all attacks/evaluations |
| `dataset.py` | `common/stamps_dataset.py` | stamp dataset + dataloader (stand-in for the gitignored datasets_local/stamps_dataset.py) | all Phase 1 training |
| `download_rvl.py` | (unchanged) | UNCLEAR, left in place. Downloads RVL-CDIP to rvl_cdip_full/ | data setup |
| `endtoend/ablation_position.py` | `ablations/ablation_position.py` | position ablation, fixed-position generators | `results/attack_results_endtoend/ablation_position/ablation_position_results.json` |
| `endtoend/ablation_position_randpos.py` | `ablations/ablation_position_randpos.py` | position ablation, random-position generators | `results/attack_results_endtoend_randpos/ablation_position_randpos/...json` |
| `endtoend/ablation_size.py` | `ablations/ablation_size.py` | stamp-size ablation (16/32/64/96), fixed-position generators | `results/attack_results_endtoend/ablation_size.json` |
| `endtoend/check_diagfinetune_stamps.py` | `phase3b_diagnostic/check_diagfinetune_stamps.py` | sample grids from ..._diagfinetune_test/ (two-disc test run) | visual check |
| `endtoend/check_diversity.py` | `explorations/check_diversity.py` | exploration: sample grids for diversity test | - |
| `endtoend/check_morph_stamps.py` | `explorations/check_morph_stamps.py` | exploration: sample grids for morph test | - |
| `endtoend/check_singledisc_stamps.py` | `phase3b_diagnostic/check_singledisc_stamps.py` | sample grids from single-disc fine-tune | visual check |
| `endtoend/config_endtoend.py` | `phase3_endtoend/config_endtoend.py` | shared end-to-end paths + hyperparameters (absolute server paths) | - |
| `endtoend/evaluate_diagfinetune_final.py` | `phase3b_diagnostic/evaluate_diagfinetune_final.py` | ASR of fine-tuned generators in ..._diagfinetune_final/ | `results/attack_results_endtoend_diagfinetune_final/evaluation_diagfinetune.json` (96.77%) |
| `endtoend/evaluate_endtoend.py` | `phase3_endtoend/evaluate_endtoend.py` | ASR, fixed-position generators | `results/attack_results_endtoend/evaluation_endtoend.json` (97.69%) |
| `endtoend/evaluate_endtoend_gan_randpos.py` | `ablations/evaluate_endtoend_gan_randpos.py` | ASR of plain-GAN ablation | `results/attack_results_endtoend_gan_randpos/evaluation_endtoend_gan_randpos.json` (83.76%) |
| `endtoend/evaluate_endtoend_randpos.py` | `phase3_endtoend/evaluate_endtoend_randpos.py` | ASR, random-position generators | `results/attack_results_endtoend_randpos/evaluation_endtoend_randpos.json` (95.28%) |
| `endtoend/evaluate_highreal_test.py` | `explorations/evaluate_highreal_test.py` | exploration: evaluates highreal test | `results/attack_results_endtoend_highreal_test/evaluation_highreal_test.json` |
| `endtoend/finetune_with_diagnostic_disc.py` | `phase3b_diagnostic/finetune_diag_singledisc.py` | **single-discriminator** fine-tune test (3 classes). NOT the version behind the 96.8% result | attack_results_endtoend_diagfinetune_singledisc_test/ (not in git) |
| `endtoend/interpolate_endtoend.py` | `phase3_endtoend/interpolate_endtoend.py` | latent interpolation grids, fixed-position generators | attack_results_endtoend/interpolation/ (not in git) |
| `endtoend/train_diagnostic_discriminator.py` | `phase3b_diagnostic/train_diagnostic_discriminator.py` | trains diagnostic discriminator (real vs end-to-end stamps) -> diagnostic_discriminator.pt | input to fine-tuning |
| `endtoend/train_endtoend.py` | `phase3_endtoend/train_endtoend.py` | end-to-end generators, fixed bottom-right | G_*.pt in attack_results_endtoend/ |
| `endtoend/train_endtoend_gan_randpos.py` | `ablations/train_endtoend_gan_randpos.py` | architecture ablation: end-to-end from plain GAN, random position | G_*.pt in attack_results_endtoend_gan_randpos/ |
| `endtoend/train_endtoend_randpos.py` | `phase3_endtoend/train_endtoend_randpos.py` | end-to-end generators, random position (the final model per its docstring) | G_*.pt in attack_results_endtoend_randpos/ |
| `endtoend/train_endtoend_randpos_anchor.py` | `explorations/train_endtoend_randpos_anchor.py` | exploration: weight anchoring to Phase 1 decoder | - |
| `endtoend/train_endtoend_randpos_diversity.py` | `explorations/train_endtoend_randpos_diversity.py` | exploration: diversity loss against mode collapse | - |
| `endtoend/train_endtoend_randpos_highreal.py` | `explorations/train_endtoend_randpos_highreal.py` | exploration: lambda_real raised, 3 classes | - |
| `endtoend/train_endtoend_randpos_morph.py` | `explorations/train_endtoend_randpos_morph.py` | exploration: morphological-opening loss | - |
| `endtoend/train_gan.py` | `phase1_generative/plain_gan/train_gan.py` | Phase 1 plain-GAN baseline -> gan_generator_best.pt, gan_disc_best.pt | architecture ablation (ablations/*gan_randpos*) |
| `evaluation_gpu_vaegan_encoded.json` | `results/evaluation_gpu_vaegan_encoded.json` | result: z-attack results, VAE-GAN encoded (GPU) | phase2_zopt/evaluate_vaegan.py; Method 2 row of comparison_*.pdf (avg 84.6) |
| `evaluation_gpu_vaegan_random.json` | `results/evaluation_gpu_vaegan_random.json` | result: z-attack results, VAE-GAN random z (GPU) | phase2_zopt/evaluate_vaegan.py; Method 3 row of comparison_*.pdf (avg 80.5) |
| `evaluation_gpu_vanilla_encoded.json` | `results/evaluation_gpu_vanilla_encoded.json` | result: z-attack results, low-KL encoded (GPU) | phase2_zopt/evaluate_vaegan.py; Method 1b row of comparison_*.pdf (avg 88.6) |
| `evaluation_vaegan_encoded_stamp.json` | `results/evaluation_vaegan_encoded_stamp.json` | result: z-attack results, VAE-GAN encoded (attack_vae_gan.py run) | phase2_zopt/evaluate_vaegan.py (older VARIANTS); 'VAE-GAN encoded' in figures/scatter_comparison.pdf |
| `evaluation_vaegan_random_z.json` | `results/evaluation_vaegan_random_z.json` | result: z-attack results, VAE-GAN random z (attack_vae_gan.py run) | phase2_zopt/evaluate_vaegan.py (older VARIANTS); 'VAE-GAN random z' in figures/scatter_comparison.pdf |
| `generative_model_comparison.pdf` | `figures/generative_model_comparison.pdf` | figure | made by make_interpolation_comparison_figure.py |
| `make_diagfinetune_grid.py` | `figures/make_diagfinetune_grid.py` | grid of 4 diag-finetuned stamps | diagfinetune_grid_selected.png (not in git) |
| `make_interpolation_comparison_figure.py` | `figures/make_interpolation_comparison_figure.py` | stacks high-KL and VAE-GAN interpolation strips | figures/generative_model_comparison.pdf |
| `make_spods_examples.py` | `figures/make_spods_examples.py` | real SPODS stamp examples | spods_real_examples.png (dataset section; not in git) |
| `make_stamp_grid.py` | `figures/make_stamp_grid.py` | one stamp per class, fixed-position generators | stamp_grid_endtoend.png (not in git) |
| `models/__init__.py` | (unchanged) | package marker | - |
| `models/base_vae.py` | (unchanged) | abstract VAE base class | - |
| `models/discriminator.py` | (unchanged) | conv discriminator (VAE-GAN, plain GAN, diagnostic disc) | - |
| `models/model_resnet50.py` | (unchanged) | frozen target classifier wrapper | - |
| `models/rvl-resnet50.model` | (unchanged) | frozen RVL-CDIP ResNet-50 weights | - |
| `models/vanilla_vae.py` | (unchanged) | low-KL VAE architecture | Methods 1a/1b |
| `models/vanilla_vae_ssim.py` | `archive/models/vanilla_vae_ssim.py` | archived: SSIM-loss VAE experiment | not in paper |
| `models/vanilla_vae_v2.py` | (unchanged) | VAE with ResBlocks: high-KL VAE, VAE-GAN, all end-to-end generators | - |
| `plot_comparison_final.py` | `figures/plot_comparison_final.py` | z-opt method comparison figures (hardcoded rates = results/evaluation_gpu_*.json + attack_results/evaluation_results_full.json) | figures/comparison_bar_chart.pdf, comparison_scatter.pdf, comparison_stamps_grid.pdf |
| `plot_scatter_comparison.py` | `figures/plot_scatter_comparison.py` | older scatter (hardcoded rates = evaluation_results_full + evaluation_vaegan_{encoded_stamp,random_z}) | figures/scatter_comparison.pdf |
| `proves/config_ssim.py` | `archive/proves/config_ssim.py` | archived: SSIM experiment config | not in paper |
| `proves/sampling.py` | `archive/proves/sampling.py` | archived: sampling test | not in paper |
| `proves/train_ssim.py` | `archive/proves/train_ssim.py` | archived: SSIM VAE training | not in paper |
| `qualitative_vaegan_random_z.pdf` | `figures/qualitative_vaegan_random_z.pdf` | figure | made by phase2_zopt/qualitative_vaegan.py |
| `scatter_comparison.pdf` | `figures/scatter_comparison.pdf` | figure | made by plot_scatter_comparison.py |
