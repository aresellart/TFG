# File map

Step 1 inventory: every tracked file, its size, and a one-line guess of its purpose. The new-path, role and paper columns are filled in at step 6.

| Old path | Size (bytes) | Purpose (first guess) | New path | Role | Paper table/figure |
|---|---:|---|---|---|---|
| `.gitignore` | 68 | git ignore rules | | | |
| `README.md` | 12 | placeholder README (12 bytes) | | | |
| `attack/__init__.py` | 0 | empty package marker | | | |
| `attack/attack.py` | 7,821 | z-optimisation attack, low-KL VAE, z initialised from encoded real stamp, batch 8, 300 it (laptop/WSL paths) | | | |
| `attack/attack_gpu_vaegan_encoded.py` | 5,693 | z-optimisation attack, VAE-GAN, z from encoded real stamp, batch 128, 2000 it, GPU | | | |
| `attack/attack_gpu_vaegan_random.py` | 5,020 | z-optimisation attack, VAE-GAN, z ~ N(0,I), batch 128, 2000 it, GPU | | | |
| `attack/attack_gpu_vanilla.py` | 6,867 | z-optimisation attack, low-KL VAE, z from encoded real stamp, batch 128, 2000 it, GPU | | | |
| `attack/attack_random_z.py` | 6,491 | z-optimisation attack, low-KL VAE, z ~ N(0,I), 4 classes only, batch 8, 300 it (WSL paths) | | | |
| `attack/attack_vae_gan.py` | 7,203 | z-optimisation attack, VAE-GAN, variants encoded_stamp/random_z, batch 64, 1000 it; writes attack_results_vaegan_<variant>/ | | | |
| `attack/build_correct_list.py` | 3,335 | lists RVL-CDIP images the frozen ResNet classifies correctly | | | |
| `attack/evaluate_attack.py` | 7,145 | evaluates attack_results/ z vectors (low-KL VAE) -> evaluation_results_full.json | | | |
| `attack/evaluate_random_z.py` | 4,677 | evaluates attack_results_random_z/ -> comparison_results.json | | | |
| `attack/evaluate_vaegan.py` | 7,590 | evaluates z-attack result dirs -> evaluation_<name>.json in repo root | | | |
| `attack/patch_utils.py` | 2,129 | preprocess_document / paste_patch helpers (shared) | | | |
| `attack/qualitative_results.py` | 8,817 | qualitative figure for attack_results/ (low-KL VAE) | | | |
| `attack/qualitative_vaegan.py` | 9,964 | qualitative figure for attack_results_vaegan_random_z/ -> qualitative_vaegan_random_z.pdf | | | |
| `attack/test_model.py` | 833 | prints one RVL-CDIP sample (dataset sanity check) | | | |
| `attack/visualize_results.py` | 4,942 | decodes and plots the optimised z of attack_results/ | | | |
| `attack_results/evaluation_results.json` | 4,490 | per-class success rates, low-KL CPU attack (short form) | | | |
| `attack_results/evaluation_results_full.json` | 31,326 | per-class success rates, low-KL CPU attack (with counts) | | | |
| `attack_results_endtoend/ablation_position/ablation_position_results.json` | 3,876 | position ablation, fixed-position generators | | | |
| `attack_results_endtoend/ablation_size.json` | 2,560 | size ablation, fixed-position generators | | | |
| `attack_results_endtoend/evaluation_endtoend.json` | 643 | ASR of end-to-end generators, fixed bottom-right | | | |
| `attack_results_endtoend_diagfinetune_final/evaluation_diagfinetune.json` | 645 | ASR of diagnostic-disc fine-tuned generators | | | |
| `attack_results_endtoend_gan_randpos/evaluation_endtoend_gan_randpos.json` | 647 | ASR of plain-GAN-initialised generators, random position | | | |
| `attack_results_endtoend_highreal_test/evaluation_highreal_test.json` | 151 | ASR, lambda_real=0.3 test, 3 classes | | | |
| `attack_results_endtoend_randpos/ablation_position_randpos/ablation_position_randpos_results.json` | 3,874 | position ablation, random-position generators | | | |
| `attack_results_endtoend_randpos/evaluation_endtoend_randpos.json` | 647 | ASR of end-to-end generators, random position | | | |
| `attack_results_random_z/comparison_results.json` | 1,410 | low-KL random-z attack, 4 classes | | | |
| `autoencoder/config.py` | 292 | low-KL VAE hyperparameters | | | |
| `autoencoder/evaluate.py` | 2,193 | low-KL VAE reconstruction plot | | | |
| `autoencoder/generate.py` | 1,515 | low-KL VAE sampling/reconstruction plot | | | |
| `autoencoder/train.py` | 5,460 | low-KL VAE training (contains an older commented-out copy) | | | |
| `autoencoder/utils.py` | 579 | vae_loss (BCE + KL) | | | |
| `autoencoder_v2/__init__.py` | 0 | empty package marker | | | |
| `autoencoder_v2/config_v2.py` | 426 | high-KL VAE hyperparameters | | | |
| `autoencoder_v2/evaluate_v2.py` | 3,669 | high-KL VAE evaluation plot | | | |
| `autoencoder_v2/train_v2.py` | 3,423 | high-KL VAE training (ResBlock model) | | | |
| `autoencoder_v3/config_v3.py` | 568 | VAE-GAN hyperparameters (300 epochs) | | | |
| `autoencoder_v3/evaluate_v3.py` | 3,052 | VAE-GAN evaluation plot (loads finetuned checkpoint) | | | |
| `autoencoder_v3/finetune_v3.py` | 7,018 | VAE-GAN fine-tuning -> vae_gan_finetune_best.pt | | | |
| `autoencoder_v3/train_v3.py` | 6,541 | VAE-GAN training -> vae_gan_best.pt, disc_gan_best.pt | | | |
| `comparison_bar_chart.pdf` | 22,335 | figure (from plot_comparison_final.py) | | | |
| `comparison_scatter.pdf` | 24,864 | figure (from plot_comparison_final.py) | | | |
| `comparison_stamps_grid.pdf` | 390,106 | figure (from plot_comparison_final.py) | | | |
| `config.py` | 292 | byte-identical copy of autoencoder/config.py; imported as `import config` by 5 attack scripts | | | |
| `correct_images.json` | 779,153 | indices of correctly classified RVL-CDIP images per class | | | |
| `dataset.py` | 1,280 | StampsDataset + get_dataloader (CRLF endings, 1,280 bytes) | | | |
| `download_rvl.py` | 123 | downloads RVL-CDIP from HF and saves rvl_cdip_full/ | | | |
| `endtoend/ablation_position.py` | 9,142 | position ablation, fixed-position generators | | | |
| `endtoend/ablation_position_randpos.py` | 9,277 | position ablation, random-position generators | | | |
| `endtoend/ablation_size.py` | 7,506 | stamp-size ablation, fixed-position generators | | | |
| `endtoend/check_diagfinetune_stamps.py` | 2,145 | sample grid from diagfinetune_test generators | | | |
| `endtoend/check_diversity.py` | 2,433 | sample grid from diversity-loss test generators | | | |
| `endtoend/check_morph_stamps.py` | 2,253 | sample grid from morph-loss test generators | | | |
| `endtoend/check_singledisc_stamps.py` | 2,140 | sample grid from single-disc fine-tuned generators | | | |
| `endtoend/config_endtoend.py` | 2,021 | shared paths + hyperparameters for end-to-end | | | |
| `endtoend/evaluate_diagfinetune_final.py` | 5,854 | evaluates ..._diagfinetune_final generators | | | |
| `endtoend/evaluate_endtoend.py` | 7,244 | evaluates fixed-position generators | | | |
| `endtoend/evaluate_endtoend_gan_randpos.py` | 6,080 | evaluates plain-GAN random-position generators | | | |
| `endtoend/evaluate_endtoend_randpos.py` | 6,398 | evaluates random-position generators | | | |
| `endtoend/evaluate_highreal_test.py` | 6,600 | evaluates lambda_real=0.3 test generators | | | |
| `endtoend/finetune_with_diagnostic_disc.py` | 8,394 | fine-tune with diagnostic disc (single-disc version) | | | |
| `endtoend/interpolate_endtoend.py` | 8,011 | latent interpolation grids for generators | | | |
| `endtoend/train_diagnostic_discriminator.py` | 6,684 | trains real-vs-generated diagnostic discriminator | | | |
| `endtoend/train_endtoend.py` | 12,929 | end-to-end generator training, fixed bottom-right | | | |
| `endtoend/train_endtoend_gan_randpos.py` | 6,871 | end-to-end training from plain GAN, random position | | | |
| `endtoend/train_endtoend_randpos.py` | 8,282 | end-to-end training, random position | | | |
| `endtoend/train_endtoend_randpos_anchor.py` | 9,466 | test: weight anchoring to Phase 1 decoder | | | |
| `endtoend/train_endtoend_randpos_diversity.py` | 8,084 | test: diversity loss | | | |
| `endtoend/train_endtoend_randpos_highreal.py` | 7,484 | test: higher lambda_real | | | |
| `endtoend/train_endtoend_randpos_morph.py` | 8,977 | test: morphological opening loss | | | |
| `endtoend/train_gan.py` | 5,437 | plain GAN (decoder only) Phase 1 baseline | | | |
| `evaluation_gpu_vaegan_encoded.json` | 24,942 | z-attack results, VAE-GAN encoded (GPU) | | | |
| `evaluation_gpu_vaegan_random.json` | 25,010 | z-attack results, VAE-GAN random z (GPU) | | | |
| `evaluation_gpu_vanilla_encoded.json` | 24,510 | z-attack results, low-KL encoded (GPU) | | | |
| `evaluation_vaegan_encoded_stamp.json` | 25,115 | z-attack results, VAE-GAN encoded (attack_vae_gan.py run) | | | |
| `evaluation_vaegan_random_z.json` | 24,961 | z-attack results, VAE-GAN random z (attack_vae_gan.py run) | | | |
| `generative_model_comparison.pdf` | 765,601 | figure (from make_interpolation_comparison_figure.py) | | | |
| `make_diagfinetune_grid.py` | 1,645 | figure: 4 diag-finetuned stamps | | | |
| `make_interpolation_comparison_figure.py` | 2,117 | figure: high-KL vs VAE-GAN interpolation | | | |
| `make_spods_examples.py` | 1,335 | figure: real SPODS stamps | | | |
| `make_stamp_grid.py` | 2,596 | figure: one stamp per class, fixed-position generators | | | |
| `models/__init__.py` | 0 | empty package marker | | | |
| `models/base_vae.py` | 1,336 | abstract BaseVAE | | | |
| `models/discriminator.py` | 1,837 | conv discriminator (128x128 grayscale) | | | |
| `models/model_resnet50.py` | 710 | ResNet-50 classifier wrapper (HF transformers) | | | |
| `models/rvl-resnet50.model` | 98,601,192 | frozen RVL-CDIP ResNet-50 weights (98.6 MB) | | | |
| `models/vanilla_vae.py` | 5,613 | VAE (low-KL model) | | | |
| `models/vanilla_vae_ssim.py` | 5,365 | VAE variant with SSIM loss (experiment) | | | |
| `models/vanilla_vae_v2.py` | 5,041 | VAE with residual blocks (high-KL, VAE-GAN, generators) | | | |
| `plot_comparison_final.py` | 11,621 | figures: bar chart, scatter, stamps grid of z-attack methods (hardcoded rates) | | | |
| `plot_scatter_comparison.py` | 4,074 | figure: scatter_comparison.pdf (hardcoded rates) | | | |
| `proves/config_ssim.py` | 279 | SSIM VAE config (experiment) | | | |
| `proves/sampling.py` | 1,363 | low-KL VAE sampling test | | | |
| `proves/train_ssim.py` | 3,159 | SSIM VAE training (experiment) | | | |
| `qualitative_vaegan_random_z.pdf` | 691,575 | figure (from qualitative_vaegan.py) | | | |
| `scatter_comparison.pdf` | 29,016 | figure (from plot_scatter_comparison.py) | | | |
