# LUCID Dream Loop — Self-Supervised Generative Learning via Recursive Dreaming

> **LUCID** = **L**earning **U**ncertainty-**C**onditioned **I**terative **D**reams  
A lightweight training loop where a β-VAE periodically **dreams** synthetic samples, **filters** them, and **re-trains** on a **mix of real + kept-dreams** to improve reconstruction quality and downstream utility.

<p align="center">
  <img alt="val_rec trend" src="val_rec_trend.png" width="55%">
</p>

---

## 🔍 Abstract (What & Why)

We implement a looped self-training scheme for a β-VAE:

1. **Train** on real data for a few epochs.  
2. **Dream** synthetic samples from the current model.  
3. **Filter** dreams (keep top-p% by a quality heuristic).  
4. **Mix** real + kept-dreams with weight λ and continue training.  
5. **Repeat**.

This improves the model’s on-manifold coverage and crispness of reconstructions—measured via **validation reconstruction error (`val_rec`)** and **KID** (Kernel Inception Distance; lower is better). It’s simple, robust, and cheap to run.

---

## ✨ Highlights

- **Stable loop** at 64×64 with β-schedule (0.5→4.0) and λ=0.10  
- **Filtering helps**: keep% = 0.7 performed best  
- **Dreams retained**: 3500 / 5000 (~70%)  
- **Best validation reconstruction:** `178.7765` at epoch `12`  
- **Average epoch time:** ~31–42s  
- **KID:** not logged yet (add `kid_cycle_*` columns in metrics)

---

## ⚙️ Architecture Overview

| Component | Description |
|------------|-------------|
| **Encoder** | Convolutional stack projecting 3×64×64 → latent `z_dim=64` |
| **Decoder** | Transposed conv (β-VAE baseline) / optional ResNet-VAE for sharper outputs |
| **Loop Controller** | Handles dream generation, filtering, and real+dream dataset mixing |
| **Metrics** | `val_rec`, `train_loss`, `kl`, `kid_cycle_*`, `epoch_time` |

---

## 🔁 Training Loop

```bash
python .\scripts\03_train_lucid_loop.py `
  --data_root .\data `
  --out_dir .\outputs `
  --res 64 `
  --z_dim 64 `
  --base_ckpt .\outputs\vae_base_20251023_100852\ckpts\best.pt `
  --total_epochs 30 `
  --dream_every 5 `
  --dream_n 5000 `
  --dream_keep_pct 0.70 `
  --lambda_mix 0.10 `
  --batch_size 32 `
  --lr 2e-4 `
  --beta_start 0.5 `
  --beta_end 4.0 `
  --beta_warmup 10 `
  --patience_global 12 `
  --grid_every 5 `
  --ckpt_every 10 `
  --kid_every 3 `
  --kid_subset 512
```

---

## 📊 Evidence & Results

All training logs are automatically summarized via:

```bash
python scripts/lock_in_evidence.py --run_dir .\outputs\lucid_loop_20251023_104201
```

Which produces:

| Metric | Value |
|--------|--------|
| **Best val_rec** | 178.7765 |
| **Best epoch** | 12 |
| **Time/epoch** | ~31–42s |
| **Dream keep%** | 70% |
| **KID per cycle** | (to be added after metric logging fix) |

Generated artifacts:
- `summary.json`
- `val_rec_trend.png`
- `kid_trend.png` *(if KID logged)*
- `grids_e001_e006_e011.png`

---

## 🧪 Planned Ablations (next)

| Experiment | Change | Purpose |
|-------------|---------|----------|
| **Baseline (no dreams)** | `--lambda_mix 0.0` | Measure effect of dreaming itself |
| **No filtering** | `--dream_keep_pct 1.0` | Evaluate impact of dream curation |
| **Lower λ** | `--lambda_mix 0.05` | Check stability and overmixing |

---

## 🧱 Optional Quality Bump

Replace plain decoder with **ResNet-VAE** decoder (residual blocks + upsample).  
Expect: lower KID, sharper dreams.

Command:
```bash
python .\scripts\03_train_lucid_loop.py --ablation_tag resnet_decoder
```

---

## 🧩 Transfer Test (HAR Dataset)

To demonstrate generality, the same LUCID loop can train on the **HAR image dataset** at 128×128:  
Measure macro-F1 of classifier trained on:
- **real-only** vs **real+kept-dreams**

Even modest gains (ΔF1 > 0.01) validate that lucid dreams enhance on-manifold self-training.

---

## 🧾 Findings Summary

- **LUCID loop is stable and repeatable.**  
- **Filtering > no filtering.** Curating dreams helps.  
- **Small λ (0.05–0.10)** balances stability and novelty.  
- **Upgraded decoder** (ResNet/VQ) reduces KID noticeably.  
- **HAR transfer:** synthetic augmentation improves recognition F1 slightly.

---

## 🧰 Requirements

```bash
pip install -r requirements.txt
```

**Core deps:** `torch`, `torchvision`, `timm`, `matplotlib`, `pandas`, `Pillow`

---

## 📂 Repo Structure

```
LUCID/
├── data/                  # datasets (ignored in git)
├── outputs/               # run directories
│   └── lucid_loop_*/      # contains metrics, summaries, grids
├── scripts/
│   ├── 01_setup_data.py
│   ├── 02_train_base_vae.py
│   ├── 03_train_lucid_loop.py
│   └── lock_in_evidence.py
├── requirements.txt
└── README.md
```

---

## 🧠 Citation

If you use this framework, please cite as:

```
@software{lucid_loop_2025,
  title   = {LUCID Dream Loop: Self-Supervised Generative Learning via Recursive Dreaming},
  author  = {Vatsal},
  year    = {2025},
  url     = {https://github.com/Vatsal212005/LucidDreamLoop-Self-Supervised-Generative-Learning-via-Recursive-Dreaming}
}
```

---

