# Uncertainty / Pseudo-Labeling Project

Summary
- Implementation of a 3-phase semi‑supervised pipeline for CIFAR-10: supervised warmup, pseudo‑label generation (global prototypes + local propagation / Nyström), and semi‑supervised fine‑tuning using agreed pseudo‑labels.

Key files (repo content)
- Scripts:
  - 1_local_global_pred.py
  - 2_local_pred.py
  - 3_global_pred.py
  - 4_local_global.py
  - 5_local_nystrom.py
- Utilities / support:
  - loader.py
  - model/ (ResNet implementations)
  - utils.py
  - metrics.py
  - loss_local_global.py
  - loss.py
  - config.yaml

Do you need a Table of Contents?
- Not necessary right now: the repo is small and file names are clear.
- Add a TOC only if you expand README with many sections (experiments, multiple examples, long troubleshooting). A short file list is sufficient.

Quick start
1. Create environment:
   python -m venv venv
   source venv/bin/activate
   pip install torch torchvision numpy scikit-learn pyyaml pillow
2. Edit `config.yaml` (paths, label_size, warmup_epochs, k_neighbors, alpha_propagation, confidence_high, n_landmarks, gpus, save_folder).
3. Prepare data as expected by loader.py (place .npy arrays in ./cifar10_data/).
4. Run a script:
   - python 1_local_global_pred.py
   - or choose variants: python 2_local_pred.py / 3_global_pred.py / 5_local_nystrom.py

What to record after a run
- Save train.log, result.log, and the used config.yaml to a folder under `experiments/<run-id>/`.
- Populate `experiments/results.md` with summarized metrics (Accuracy, AURC, E-AURC, AUPR, FPR95, ECE, NLL, Brier), config used, and commands.

Recommended minimal README additions (optional)
- Example command lines per script.
- Short explanation of loss functions (compute_kld_labeled, compute_kld_unlabeled).
- How to reproduce a run (seed, GPU env var, config snapshot).

If you want, I can:
- Add short example command lines per script into this README.
- Generate an experiments/README template for recording runs.
- Insert a one-line table of contents — tell me which option you prefer.
- pillow

Example install:
  python -m venv venv
  source venv/bin/activate
  pip install torch torchvision numpy scikit-learn pyyaml pillow

(Optionally create a requirements.txt with pinned versions.)

Configuration (config.yaml)
The project reads hyperparameters and paths from `config.yaml`. Key fields:
- label_size: number of labeled examples
- warmup_epochs: epochs to run supervised warmup
- k_neighbors: neighbors for label propagation
- alpha_propagation: propagation weight
- confidence_high: confidence value for soft targets (e.g., 0.9)
- n_landmarks: (Nyström) number of landmarks
- unlabeled_limit: optional cap for unlabeled samples
- gpus: comma-separated GPU ids (e.g., "0")
- save_folder: where to write logs & checkpoints

Provide a minimal example:
```yaml
label_size: 1000
warmup_epochs: 30
k_neighbors: 70
alpha_propagation: 0.5
confidence_high: 0.9
n_landmarks: 500
unlabeled_limit: 50000
gpus: "0"
save_folder: "./checkpoints"
```

Data preparation
- Place CIFAR-10 arrays in `./cifar10_data/` as expected by loader.py:
  - img_labeled_<label_size>.npy
  - ann_labeled_<label_size>.npy
  - img_unlabeled_<label_size>.npy
  - ann_unlabeled_<label_size>.npy
  - img_test.npy, ann_test.npy
- If you use raw CIFAR-10, write a small preprocessing script to create these .npy files in the required order and indexing.

Scripts and how to run them
General pattern:
  python <script>.py

- 1_local_global_pred.py
  - Combined pipeline (global + local agreement) using compute_kld_labeled and compute_kld_unlabeled.
  - Use for baseline 3-phase experiments.

- 2_local_pred.py
  - Uses local label propagation as the "agreed" source.
  - Useful to evaluate local-only pseudo-label accuracy.

- 3_global_pred.py
  - Uses global prototypes only (treats global predictions as agreed).
  - Useful for prototype-only experiments.

- 4_local_global.py
  - Alternate/updated combined pipeline; may dynamically import loss functions depending on filename.
  - Use when you want to swap in a specific loss file (e.g., `4_loss.py`).

- 5_local_nystrom.py
  - Local propagation via Nyström approximation to scale propagation to larger sets.

Common command:
  python 1_local_global_pred.py
  or
  python 5_local_nystrom.py

Losses and pseudo-label flow
- compute_kld_labeled(output, targets, num_classes, high)
  - Creates a soft target: if model prediction matches true label -> high confidence for that label, else uniform.
  - KLD between model log-probs and soft-targets is added to CE during Phase 1.

- compute_kld_unlabeled(unlabeled_logits, unlabeled_indices_batch, pseudo_storage, ...)
  - For unlabeled samples that have agreed pseudo-labels (local == global), generates soft-targets with high confidence and computes KLD.
  - Returns (kld_loss, used_count). Used_count is the number of agreed samples in the batch.

PseudoLabelStorage
- Maintains maps: global_pseudo[idx] and local_pseudo[idx].
- get_agreed_samples returns indices where local and global labels match (unless the script intentionally overrides behavior).

Logs, outputs and results recording
- `save_folder` contains:
  - train.log — training metrics per epoch
  - result.log — evaluation metrics per epoch
  - model checkpoints model_*.pth
- Use `experiments/results.md` to record run summaries (date, config, final metrics).
- Recommended metrics to report: Accuracy, AURC, E-AURC, AUPR, FPR95, ECE, NLL, Brier.

Experiments and recommended workflow
1. Edit config.yaml for the target experiment.
2. Run warmup (the scripts run warmup automatically).
3. For ablations:
   - Run global-only (3_global_pred.py).
   - Run local-only (2_local_pred.py).
   - Run Nyström variant (5_local_nystrom.py) for large unlabeled sets.
4. Save logs into `experiments/<run-name>/` and update `experiments/results.md`.

Reproducibility tips
- Set seed using `set_seed()` already present in scripts.
- Keep a copy of the config used for each run in the experiment folder.
- Use deterministic cudnn settings (already set) but be aware it may slow training.

Troubleshooting
- OOM errors:
  - Reduce batch_size, reduce n_landmarks, or limit unlabeled samples.
- No agreed samples:
  - Check labeled classifier accuracy (Phase 1). If too low, prototypes/local propagation will be poor.
  - Increase labeled set or warmup_epochs.
- Slow propagation:
  - Use smaller k_neighbors or Nyström variant.
  - Use num_workers in DataLoader or run accumulation on a machine with more memory.

Git workflow (quick)
  git init
  git add .
  git commit -m "Initial project"
  git branch -M main
  git remote add origin <your-remote-url>
  git push -u origin main

Where to put results
- Create a directory `experiments/<run-id>/` and copy:
  - config.yaml (used)
  - train.log, result.log
  - short README.md describing the run
  - snapshot of saved model (optional)
- Update `experiments/results.md` with summarized metrics.

Contact / License
- Add your preferred license (e.g., MIT) in LICENSE file.
- Add author/contact details in this README if desired.

Appendix: minimal example run
1. Update config.yaml
2. Run:
   python 1_local_global_pred.py
3. After run, collect metrics from `result.log` and summarize in `experiments/results.md`.
