#!/usr/bin/env python3
# 03_train_lucid_loop.py (QoL+)
# LUCID Step 3: Self-training via "dreams" with resume, global early-stop, CSV logs,
# AMP, grad clipping, progress bars, optional KID, adaptive lambda, and robust checkpoints.

from __future__ import annotations
import argparse, math, time, json, csv, random, os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# --- Optional KID support ---
_HAS_TM = True
try:
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    _HAS_TM = False

# -------------------------
# β-VAE (same as Step 2)
# -------------------------
class BetaVAE(nn.Module):
    def __init__(self, in_ch=3, res=64, z_dim=64):
        super().__init__()
        C = 64
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, C, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 4, 2, 1), nn.BatchNorm2d(C*2), nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 4, 2, 1), nn.BatchNorm2d(C*4), nn.ReLU(inplace=True),
            nn.Conv2d(C*4, C*8, 4, 2, 1), nn.BatchNorm2d(C*8), nn.ReLU(inplace=True),
        )
        fsz = res // 16
        self.enc_out_dim = C*8*fsz*fsz
        self.mu = nn.Linear(self.enc_out_dim, z_dim)
        self.logvar = nn.Linear(self.enc_out_dim, z_dim)
        self.fc = nn.Linear(z_dim, self.enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C*8, C*4, 4, 2, 1), nn.BatchNorm2d(C*4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*4, C*2, 4, 2, 1), nn.BatchNorm2d(C*2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*2, C,   4, 2, 1), nn.BatchNorm2d(C),    nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C,   in_ch, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# -------------------------
# Helpers
# -------------------------
def set_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)

def beta_schedule(epoch, beta_start, beta_end, warmup_epochs):
    if epoch < warmup_epochs:
        t = epoch / max(1, warmup_epochs)
        return beta_start + t * (beta_end - beta_start)
    return beta_end

def time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_grid(tensor_batch, path, nrow=8, pad=2):
    vutils.save_image(tensor_batch.cpu(), str(path), nrow=nrow, padding=pad)

@torch.no_grad()
def save_recon_grid(model, device, fixed_batch, out_path, nrow=8, pad=2):
    model.eval()
    x = fixed_batch[0].to(device)
    x_hat, _, _ = model(x)
    grid = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    save_grid(grid, out_path, nrow=nrow, pad=pad)

# -------------------------
# Dream dataset mixer
# -------------------------
class MixedDreamDataset(Dataset):
    """
    Wraps a real dataset (Subset of CIFAR-10) and an optional tensor of dreams (N,C,H,W).
    On __getitem__, returns a dream with prob=lambda_mix, else a real sample.
    __len__ == len(real) for comparable epoch sizes.
    """
    def __init__(self, real_subset: Dataset, dreams: torch.Tensor | None, lambda_mix: float):
        self.real = real_subset
        self.dreams = dreams
        self.lambda_mix = float(lambda_mix)
        self._dream_n = 0 if dreams is None else dreams.size(0)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        use_dream = (self._dream_n > 0) and (random.random() < self.lambda_mix)
        if use_dream:
            j = random.randrange(self._dream_n)
            return self.dreams[j], 0
        x, _ = self.real[idx]
        return x, 0

# -------------------------
# Dream generation + filtering
# -------------------------
@torch.no_grad()
def generate_dreams(model: BetaVAE, n: int, z_dim: int, device: torch.device, batch_size: int = 256):
    model.eval()
    xs = []
    for i in range(0, n, batch_size):
        bs = min(batch_size, n - i)
        z = torch.randn(bs, z_dim, device=device)
        x_star = model.decode(z)
        xs.append(x_star.cpu())
    return torch.cat(xs, dim=0)  # [n,3,H,W] in [0,1]

@torch.no_grad()
def recon_confidence_filter(model: BetaVAE, device, dreams: torch.Tensor, keep_pct: float, batch_size: int = 128):
    """
    Score each dream by round-trip recon error:
       x* -> encode -> decode(mean) -> x** ; score = MSE(x**, x*)
    Keep lowest 'keep_pct' fraction (most self-consistent on manifold).
    """
    model.eval()
    N = dreams.size(0)
    scores = torch.empty(N)
    ptr = 0
    for i in range(0, N, batch_size):
        xb = dreams[i:i+batch_size].to(device)
        mu, logvar = model.encode(xb)
        x_hat_mu = model.decode(mu)
        mse = F.mse_loss(x_hat_mu, xb, reduction='none')  # [B,3,H,W]
        mse = mse.view(mse.size(0), -1).mean(dim=1)       # [B]
        bs = xb.size(0)
        scores[ptr:ptr+bs] = mse.cpu()
        ptr += bs
    k = max(1, int(N * keep_pct))
    # smallest scores (best) → take topk on negative
    _, idxs = torch.topk(-scores, k)
    kept = dreams[idxs]
    return kept, float(scores.mean().item()), float(scores[idxs].mean().item())

# -------------------------
# KID metric (optional) — FIXED to feed uint8 tensors
# -------------------------
@torch.no_grad()
def maybe_compute_kid(dreams: torch.Tensor, val_loader: DataLoader, device, kid_subset: int = 512, kid_bs: int = 64):
    if not _HAS_TM:
        return None

    kid = KernelInceptionDistance(subset_size=min(1000, kid_subset)).to(device)
    kid.eval()

    # Fake (dreams)
    d = dreams[:kid_subset].clamp(0, 1)
    for i in range(0, d.size(0), kid_bs):
        batch = (d[i:i+kid_bs] * 255).to(torch.uint8).to(device)
        kid.update(batch, real=False)
        torch.cuda.empty_cache()

    # Real (validation)
    seen = 0
    for xb, _ in val_loader:
        xb = xb.clamp(0, 1)
        batch = (xb * 255).to(torch.uint8).to(device)
        kid.update(batch, real=True)
        seen += xb.size(0)
        torch.cuda.empty_cache()
        if seen >= kid_subset:
            break

    mean, std = kid.compute()
    return float(mean.item()), float(std.item())

# -------------------------
# Training (global loop with periodic dreaming)
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_deterministic(args.seed)

    # IO
    data_root = Path(args.data_root).absolute()
    prep_dir = data_root / "prepared" / f"cifar10_{args.res}"
    split_dir = prep_dir / "splits"
    assert split_dir.exists(), f"Missing splits at {split_dir}. Run 01_setup_data.py first."

    out_root = Path(args.out_dir).absolute()
    run_dir = out_root / (Path(args.resume).parent.parent.name if args.resume else f"lucid_loop_{time_str()}")
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (run_dir / "grids").mkdir(parents=True, exist_ok=True)
    (run_dir / "dreams").mkdir(parents=True, exist_ok=True)

    # Data
    tfm = transforms.Compose([
        transforms.Resize((args.res, args.res), antialias=True),
        transforms.ToTensor(),
    ])
    train_full = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=tfm)
    with open(split_dir / "train_indices.json") as f:
        train_indices = json.load(f)
    with open(split_dir / "val_indices.json") as f:
        val_indices = json.load(f)
    train_subset = Subset(train_full, train_indices)
    val_subset   = Subset(train_full, val_indices)

    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model + optimizer
    model = BetaVAE(in_ch=3, res=args.res, z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Load base or resume
    start_epoch = 1
    best_overall = math.inf
    best_ckpt_path = None
    last_dream_epoch = 0
    current_lambda = args.lambda_mix

    if args.resume:
        # Resume a Step-3 run
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt and not args.reset_opt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch      = int(ckpt.get("epoch", 1)) + 1
        best_overall     = float(ckpt.get("best_overall_val_rec", math.inf))
        last_dream_epoch = int(ckpt.get("last_dream_epoch", 0))
        current_lambda   = float(ckpt.get("current_lambda", args.lambda_mix))
        print(f"🔁 Resumed Step-3 from {args.resume} @ epoch {start_epoch} | best_overall={best_overall:.6f} | λ={current_lambda:.3f}")
    else:
        # Fresh Step-3: load Step-2 base checkpoint
        base = torch.load(args.base_ckpt, map_location="cpu")
        model.load_state_dict(base["model"], strict=True)
        print(f"🔁 Loaded base checkpoint: {args.base_ckpt}")

    # CSV logger
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists() or args.fresh_metrics:
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "epoch","cycle_id","beta","train_loss","train_rec","train_kl",
                "val_rec","dreams_kept","seconds","lambda","kid_mean","kid_std"
            ])

    # Fixed val batch for recon grids
    fixed_batch = next(iter(val_loader))

    # State for cycles
    cycle_id = (last_dream_epoch // args.dream_every)
    dreams_cache: torch.Tensor | None = None  # updated each cycle

    # Global early stopping
    epochs_no_improve_global = 0

    print(f"\n==> LUCID Dream Loop on {device.type.upper()} | z_dim={args.z_dim} | λ0={args.lambda_mix} | keep={int(args.dream_keep_pct*100)}% | k={args.dream_every}\n")

    # Global epoch loop
    for epoch in range(start_epoch, args.total_epochs + 1):
        t0 = time.time()

        # Trigger new dream cycle at boundaries
        if (epoch - 1) % args.dream_every == 0 or dreams_cache is None:
            cycle_id += 1
            # Generate + filter dreams
            print(f"\n===== 🌙 Dream Cycle #{cycle_id} — generating {args.dream_n} dreams (epoch {epoch}) =====")
            dreams = generate_dreams(model, args.dream_n, args.z_dim, device, batch_size=args.dream_gen_bs)
            # Save raw preview
            save_grid(dreams[:64], run_dir / "dreams" / f"dream_raw_e{epoch:03d}.png", nrow=8, pad=2)
            kept, score_all, score_kept = recon_confidence_filter(model, device, dreams, args.dream_keep_pct, batch_size=args.dream_eval_bs)
            save_grid(kept[:64], run_dir / "dreams" / f"dream_kept_e{epoch:03d}.png", nrow=8, pad=2)
            dreams_cache = kept  # tensor on CPU
            last_dream_epoch = epoch

            # Adaptive lambda decay if requested
            if args.lambda_decay > 0.0:
                # linearly decay per cycle, floor at 10% of initial
                min_lambda = max(0.1 * args.lambda_mix, 0.0)
                current_lambda = max(min_lambda, args.lambda_mix * (1.0 - args.lambda_decay * (cycle_id - 1)))

            # Optional KID
            kid_mean = kid_std = None
            if _HAS_TM and args.kid_every > 0 and (cycle_id % args.kid_every == 0):
                km = maybe_compute_kid(dreams_cache, val_loader, device, kid_subset=args.kid_subset)
                if km is not None:
                    kid_mean, kid_std = km
                    print(f"🔬 KID (dream vs val): mean={kid_mean:.6f} ± {kid_std:.6f}")
            else:
                kid_mean = kid_std = None

            # Log a row for the dream event (epoch-1)
            with open(csv_path, "a", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([epoch-1, cycle_id, None, None, None, None, None, dreams_cache.size(0), 0.0, current_lambda, kid_mean, kid_std])

        # Build mixed dataset for this epoch
        mixed_ds = MixedDreamDataset(train_subset, dreams_cache, current_lambda)
        train_loader = DataLoader(mixed_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)

        # Train one epoch
        model.train()
        beta = beta_schedule(epoch-1, args.beta_start, args.beta_end, args.beta_warmup)

        run_rec = run_kl = run_loss = 0.0
        n_samples = 0
        pbar = tqdm(train_loader, desc=f"[Cycle {cycle_id}] Epoch {epoch}/{args.total_epochs}", leave=False)
        for xb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            bs = xb.size(0)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                x_hat, mu, logvar = model(xb)
                rec = F.mse_loss(x_hat, xb, reduction="sum") / bs
                kl  = kl_divergence(mu, logvar).mean()
                loss = rec + beta * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()

            run_rec += rec.item() * bs
            run_kl  += kl.item() * bs
            run_loss += loss.item() * bs
            n_samples += bs
            pbar.set_postfix(loss=f"{loss.item():.3f}", rec=f"{rec.item():.3f}", kl=f"{kl.item():.3f}", beta=f"{beta:.2f}", lam=f"{current_lambda:.2f}")

        # Averages
        train_rec = run_rec / n_samples
        train_kl  = run_kl  / n_samples
        train_loss = run_loss / n_samples

        # Validation recon
        model.eval()
        val_rec = 0.0
        val_n   = 0
        with torch.no_grad():
            for xv, _ in val_loader:
                xv = xv.to(device, non_blocking=True)
                xhv, _, _ = model(xv)
                recv = F.mse_loss(xhv, xv, reduction="sum")
                val_rec += recv.item()
                val_n   += xv.size(0)
        val_rec /= max(1, val_n)

        dt = time.time() - t0
        print(f"[E{epoch:03d}] β={beta:.3f} | train_loss={train_loss:.4f} rec={train_rec:.4f} kl={train_kl:.4f} | val_rec={val_rec:.4f} | dreams_kept={dreams_cache.size(0) if dreams_cache is not None else 0} | λ={current_lambda:.2f} | {dt:.1f}s")

        # Recon grid periodically
        if (epoch == 1) or (epoch % args.grid_every == 0):
            save_recon_grid(model, device, fixed_batch, run_dir / "grids" / f"recon_epoch_{epoch:03d}.png", nrow=8, pad=2)

        # Best / periodic ckpts (store optimizer for resume)
        improved = val_rec < best_overall
        if improved:
            best_overall = val_rec
            epochs_no_improve_global = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_overall_val_rec": best_overall,
                "last_dream_epoch": last_dream_epoch,
                "current_lambda": current_lambda,
                "args": vars(args),
            }, run_dir / "ckpts" / "best.pt")
            best_ckpt_path = run_dir / "ckpts" / "best.pt"
        else:
            epochs_no_improve_global += 1

        if epoch % args.ckpt_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_overall_val_rec": best_overall,
                "last_dream_epoch": last_dream_epoch,
                "current_lambda": current_lambda,
                "args": vars(args),
            }, run_dir / "ckpts" / f"epoch_{epoch:03d}.pt")

        # CSV row
        with open(csv_path, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, cycle_id, beta, train_loss, train_rec, train_kl, val_rec, dreams_cache.size(0), dt, current_lambda, None, None])

        # Global early stop
        if args.patience_global > 0 and epochs_no_improve_global >= args.patience_global:
            print(f"🛑 Global early stopping at epoch {epoch} (no improvement for {args.patience_global} epochs).")
            break

    # Final recon grid
    save_recon_grid(model, device, fixed_batch, run_dir / "grids" / f"recon_final.png", nrow=8, pad=2)

    # Loss curve
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_rec"], label="val_rec (MSE)")
        plt.xlabel("Epoch"); plt.ylabel("Loss / MSE"); plt.title("LUCID Dream Loop")
        plt.legend(); plt.tight_layout()
        plt.savefig(run_dir / "loss_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Manifest
    manifest = {
        "run_dir": str(run_dir),
        "best_overall_val_rec": best_overall,
        "z_dim": args.z_dim,
        "lambda_mix_start": args.lambda_mix,
        "lambda_mix_final": current_lambda,
        "lambda_decay": args.lambda_decay,
        "dream_keep_pct": args.dream_keep_pct,
        "dream_every": args.dream_every,
        "dream_n_per_cycle": args.dream_n,
        "total_epochs": args.total_epochs,
        "base_ckpt": args.base_ckpt if not args.resume else None,
        "resume_ckpt": args.resume if args.resume else None,
        "seed": args.seed,
    }
    with open(run_dir / "RUN.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ LUCID Dream Loop complete. Best overall val_rec: {best_overall:.6f}")
    print(f"📁 Outputs: {run_dir}")
    print(f"   • Dreams:    {run_dir / 'dreams'}")
    print(f"   • Grids:     {run_dir / 'grids'}")
    print(f"   • Ckpts:     {run_dir / 'ckpts'}")
    print(f"   • Metrics:   {csv_path}")
    print(f"   • Manifest:  {run_dir / 'RUN.json'}")
    if best_ckpt_path:
        print(f"   • Best ckpt: {best_ckpt_path}")

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LUCID Step 3 — Dream Loop trainer (QoL+)")
    # Data / IO
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--res", type=int, default=64, choices=[64, 128])
    p.add_argument("--workers", type=int, default=2)
    # Model / ckpt
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--base_ckpt", type=str, default="", help="path to Step-2 best.pt (ignored if --resume)")
    p.add_argument("--resume", type=str, default="", help="resume from Step-3 ckpt (outputs/.../ckpts/epoch_XXX.pt or best.pt)")
    p.add_argument("--reset_opt", action="store_true", help="reset optimizer state on resume")
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta_start", type=float, default=0.5)
    p.add_argument("--beta_end", type=float, default=4.0)
    p.add_argument("--beta_warmup", type=int, default=10)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    # Dream loop
    p.add_argument("--total_epochs", type=int, default=30)
    p.add_argument("--dream_every", type=int, default=5)
    p.add_argument("--dream_n", type=int, default=5000)
    p.add_argument("--dream_keep_pct", type=float, default=0.70)
    p.add_argument("--lambda_mix", type=float, default=0.10)
    p.add_argument("--lambda_decay", type=float, default=0.0, help="linear decay per cycle (0..1); 0 disables")
    p.add_argument("--dream_gen_bs", type=int, default=256)
    p.add_argument("--dream_eval_bs", type=int, default=128)
    p.add_argument("--grid_every", type=int, default=5)
    p.add_argument("--ckpt_every", type=int, default=10)
    # Early stop & logs
    p.add_argument("--patience_global", type=int, default=12, help="global early-stop on val_rec")
    p.add_argument("--fresh_metrics", action="store_true", help="overwrite metrics.csv header (useful after manual deletes)")
    # Optional KID
    p.add_argument("--kid_every", type=int, default=0, help="compute KID every N cycles (0 disables)")
    p.add_argument("--kid_subset", type=int, default=2048)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
