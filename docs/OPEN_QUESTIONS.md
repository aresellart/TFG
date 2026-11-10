# Open questions

Files whose role I could not settle without guessing, plus decisions I made that you may want to reverse. Unclear files were **left where they were** on `main`.

## Files left in place

1. **`attack/attack_vae_gan.py`**: phase2_zopt/ or archive/?
   It is *not* just an earlier copy of `attack_gpu_vaegan_encoded.py`. It ran both VAE-GAN variants (encoded_stamp, random_z) with batch 64 and 1000 iterations, and its outputs feed two committed figures:
   - `results/evaluation_vaegan_{encoded_stamp,random_z}.json` → the VAE-GAN series in `figures/scatter_comparison.pdf` (`plot_scatter_comparison.py`);
   - `attack_results_vaegan_random_z/` → `figures/qualitative_vaegan_random_z.pdf` (`phase2_zopt/qualitative_vaegan.py`).
   If either figure is in the thesis, this script belongs in `phase2_zopt/`. If not, both it and those figures/JSONs can go to `archive/`.
2. **`attack/attack_random_z.py`** and **`attack/evaluate_random_z.py`**: low-KL VAE with random z, 4 classes (form, memo, news article, scientific report), batch 8, 300 iterations. Result: `results/attack_results_random_z/comparison_results.json`. Is this low-KL "encoded vs random init" comparison in the thesis? If yes, phase2_zopt/; if not, archive/.
3. **Root `config.py`**: you said to archive it if unused, but it **is** used. `import config` in `phase2_zopt/evaluate_attack.py`, `qualitative_results.py`, `visualize_results.py` and the two files in item 2 resolves to it. It is byte-identical to `phase1_generative/lowkl_vae/config.py`. Options: (a) keep it; (b) change those five `import config` lines to `import phase1_generative.lowkl_vae.config as config` (same values) and archive the root copy. I did (a). Say if you want (b).
4. **`download_rvl.py`**: no script imports it, but it is the setup step that creates `rvl_cdip_full/`. Keep it in the root, or move it to `common/`? (I would not archive it.)

## Decisions I made that you may want to reverse

5. **Phase numbering clash.** The target layout calls z-optimisation `phase2_zopt/` and end-to-end `phase3_endtoend/`. But the docstrings inside the end-to-end scripts (e.g. `phase3_endtoend/train_endtoend.py` l.4 "Phase 2 — end-to-end adversarial stamp generation") and your own notes call end-to-end "Phase 2". A reader comparing the code with the thesis may be confused. Which numbering does the thesis use? I did not edit docstrings beyond the `Usage:` paths.
6. **Scripts in `phase1_generative/<variant>/` are two levels deep, not one.** Your step 4 assumed every script would be one level below the root. For those 6 scripts (highkl_vae_resblocks/{train,evaluate}_v2, vaegan/{train,finetune,evaluate}_v3, plain_gan/train_gan) I changed the existing `sys.path.append(...)` / `ROOT = ...` lines to three `dirname()` calls, so they still resolve to the repo root and the checkpoint paths do not change. The 3 `lowkl_vae` scripts never had such a line (they used `tfg_stamps.` absolute imports); they need `python -m phase1_generative.lowkl_vae.train` from the root. I did not add new lines.
7. **`test_model.py` → `archive/attack/test_model.py`.** It only prints one dataset sample. I treated it as clearly not part of the paper. Tell me if you want it back.
8. **`qualitative_vaegan.py` was moved into `phase2_zopt/`** under the "qualitative_ scripts" rule, even though it reads the output of `attack/attack_vae_gan.py` (item 1), not of Method 3. See OBSERVATIONS 12.
9. **`correct_images.json` moved to `common/`.** I updated every reader (relative `ROOT` joins, and the absolute server path in `phase3_endtoend/config_endtoend.py` → `/home/asellart/tfg_stamps/common/correct_images.json`). If the server checkout is not switched to this layout, that absolute path will not exist there.
10. **Output paths were not redirected** to `results/`/`figures/` (see OBSERVATIONS 18). Re-running a script on the server writes to the old root-level locations, next to the untracked `.pt` files.

## Import check results (step 5)

`python -m py_compile`: 83/83 files compile.
AST import check: every first-party import resolves, **except in `archive/`**, which I left untouched on purpose:

- `archive/models/vanilla_vae_ssim.py:5` `from .base_vae import BaseVAE`: `base_vae.py` stays in `models/`.
- `archive/proves/train_ssim.py:5` `from models.vanilla_vae_ssim import VanillaVAE`: the model moved to `archive/models/`.
- `archive/proves/sampling.py:6` `import tfg_stamps.autoencoder.config`: never existed in this repo.
- (`archive/proves/*` also import `datasets.stamps_dataset`, which collides with HuggingFace `datasets`.)

Should archived code be kept importable, or is "kept for the record, not runnable" fine?

## Other

11. `common/stamps_dataset.py` is probably not identical to the server's `datasets_local/stamps_dataset.py` (1,248 vs 1,188 bytes after normalising line endings). Please diff it against the server copy (OBSERVATIONS 5).
12. `requirements.txt` lists package names only, without versions, because I couldn't see the versions installed on the server. Run `pip freeze` there to pin them.
