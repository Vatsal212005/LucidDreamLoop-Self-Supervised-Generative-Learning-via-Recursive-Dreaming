#!/usr/bin/env python3
# 02_train_base_vae.py (QoL edition)
# Base β-VAE training for LUCID with early stopping, resume, CSV logs, progress bar,
# grad clipping, and final loss-curve plot.

import argparse, math, time, json, csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Model: β-VAE
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
        fsz = res // 16  # 64->4
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
# Utilities
# -------------------------
def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)

def beta_schedule(epoch, beta_start, beta_end, warmup_epochs):
    if epoch < warmup_epochs:
        t = epoch / max(1, warmup_epochs)
        return beta_start + t * (beta_end - beta_start)
    return beta_end

def save_recon_grid(model, device, batch, out_path, nrow=8, pad=2):
    model.eval()
    with torch.no_grad():
        x = batch[0].to(device)
        x_hat, _, _ = model(x)
    grid = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    vutils.save_image(grid, str(out_path), nrow=nrow, padding=pad)

def time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Train
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_deterministic(args.seed)

    # I/O
    data_root = Path(args.data_root).absolute()
    prep_dir = data_root / "prepared" / f"cifar10_{args.res}"
    split_dir = prep_dir / "splits"
    assert split_dir.exists(), f"Missing splits at {split_dir}. Run 01_setup_data.py first."

    out_root = Path(args.out_dir).absolute()
    if args.resume:
        # if resuming, we still create a fresh run_dir to avoid overwriting old artifacts
        run_dir = out_root / f"vae_base_{time_str()}_resumed"
    else:
        run_dir = out_root / f"vae_base_{time_str()}"
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (run_dir / "grids").mkdir(parents=True, exist_ok=True)

    # Data
    tfm = transforms.Compose([
        transforms.Resize((args.res, args.res), antialias=True),
        transforms.ToTensor(),
    ])
    train_set_full = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=tfm)
    with open(split_dir / "train_indices.json") as f:
        train_indices = json.load(f)
    with open(split_dir / "val_indices.json") as f:
        val_indices = json.load(f)
    train_set = Subset(train_set_full, train_indices)
    val_set   = Subset(train_set_full,  val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model / Optim
    model = BetaVAE(in_ch=3, res=args.res, z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Resume if requested
    start_epoch = 1
    best_val = math.inf
    if args.resume:
        ckpt_path = Path(args.resume)
        assert ckpt_path.exists(), f"--resume path not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if not args.reset_opt and "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "best_val_rec" in ckpt:
            best_val = ckpt["best_val_rec"]
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"🔁 Resumed from {ckpt_path} | start_epoch={start_epoch} | best_val={best_val:.6f}")

    # Fixed batch for recon visualizations
    fixed_batch = next(iter(val_loader))

    # CSV logger
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "beta", "train_loss", "train_rec", "train_kl", "val_rec", "seconds"])

    # Trackers for plotting
    hist_epochs, hist_train_loss, hist_val_rec = [], [], []

    print(f"\n==> β-VAE on {device.type.upper()} | z_dim={args.z_dim}, β {args.beta_start}->{args.beta_end} warmup {args.beta_warmup} | patience={args.patience}\n")
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        beta = beta_schedule(epoch-1, args.beta_start, args.beta_end, args.beta_warmup)

        epoch_rec = 0.0
        epoch_kl  = 0.0
        epoch_loss = 0.0
        n_samples = 0

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            bs = x.size(0)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_hat, mu, logvar = model(x)
                rec_loss = F.mse_loss(x_hat, x, reduction="sum") / bs
                kl = kl_divergence(mu, logvar).mean()
                loss = rec_loss + beta * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()

            epoch_rec += rec_loss.item() * bs
            epoch_kl  += kl.item() * bs
            epoch_loss += loss.item() * bs
            n_samples  += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", kl=f"{kl.item():.4f}", beta=f"{beta:.2f}")

        # Averages
        epoch_rec /= n_samples
        epoch_kl  /= n_samples
        epoch_loss /= n_samples

        # Validation
        model.eval()
        val_rec = 0.0
        val_n   = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                x_hat, _, _ = model(x)
                rec = F.mse_loss(x_hat, x, reduction="sum")
                val_rec += rec.item()
                val_n   += x.size(0)
        val_rec /= max(1, val_n)

        dt = time.time() - t0
        print(f"[E{epoch:03d}/{args.epochs}] beta={beta:.3f} | train_loss={epoch_loss:.4f} rec={epoch_rec:.4f} kl={epoch_kl:.4f} | val_rec={val_rec:.4f} | {dt:.1f}s")

        # Save recon grids
        if epoch % args.grid_every == 0 or epoch == start_epoch:
            out_grid = run_dir / "grids" / f"recon_epoch_{epoch:03d}.png"
            save_recon_grid(model, device, fixed_batch, out_grid, nrow=8, pad=2)

        # Checkpoints
        if val_rec < best_val:
            best_val = val_rec
            epochs_no_improve = 0
            ckpt_path = run_dir / "ckpts" / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "best_val_rec": best_val,
            }, ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ckpt_path = run_dir / "ckpts" / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "best_val_rec": best_val,
            }, ckpt_path)

        # CSV log row
        with open(csv_path, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, beta, epoch_loss, epoch_rec, epoch_kl, val_rec, dt])

        # For plotting
        hist_epochs.append(epoch)
        hist_train_loss.append(epoch_loss)
        hist_val_rec.append(val_rec)

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"🛑 Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
            break

    # Final recon grid
    final_grid = run_dir / "grids" / f"recon_epoch_{hist_epochs[-1]:03d}_final.png"
    save_recon_grid(model, device, fixed_batch, final_grid, nrow=8, pad=2)

    # Loss curve plot
    try:
        plt.figure(figsize=(7,4.5))
        plt.plot(hist_epochs, hist_train_loss, label="train_loss")
        plt.plot(hist_epochs, hist_val_rec, label="val_rec (MSE)")
        plt.xlabel("Epoch"); plt.ylabel("Loss / MSE"); plt.title("β-VAE Training")
        plt.legend(); plt.tight_layout()
        plot_path = run_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Manifest
    manifest = {
        "run_dir": str(run_dir),
        "best_val_rec": best_val,
        "epochs_trained": hist_epochs[-1],
        "z_dim": args.z_dim,
        "beta": {"start": args.beta_start, "end": args.beta_end, "warmup": args.beta_warmup},
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": device.type,
        "early_stopped": (args.patience > 0 and epochs_no_improve >= args.patience),
    }
    with open(run_dir / "RUN.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Done. Best val recon MSE: {best_val:.6f}")
    print(f"📁 Outputs: {run_dir}")
    print(f"   • Grids:     {run_dir / 'grids'}")
    print(f"   • Ckpts:     {run_dir / 'ckpts'}")
    print(f"   • Metrics:   {csv_path}")
    print(f"   • Loss plot: {run_dir / 'loss_curve.png'}\n")

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train β-VAE on CIFAR-10 64x64 (no dreams) + QoL")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--res", type=int, default=64, choices=[64, 128])
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta_start", type=float, default=0.5)
    p.add_argument("--beta_end", type=float, default=4.0)
    p.add_argument("--beta_warmup", type=int, default=10)
    p.add_argument("--grid_every", type=int, default=5, help="save recon grid every N epochs")
    p.add_argument("--ckpt_every", type=int, default=10, help="save checkpoint every N epochs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    # QoL
    p.add_argument("--patience", type=int, default=8, help="early stop after N non-improving epochs (0 disables)")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint .pt to resume from")
    p.add_argument("--reset_opt", action="store_true", help="reset optimizer state when resuming")
    p.add_argument("--clip_grad", type=float, default=1.0, help="max grad norm (<=0 disables)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
