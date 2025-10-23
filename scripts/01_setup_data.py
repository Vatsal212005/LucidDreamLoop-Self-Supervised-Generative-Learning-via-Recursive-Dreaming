#!/usr/bin/env python3
# 01_setup_data.py
# Step 1 for Project LUCID: download CIFAR-10, upsample to 64x64,
# create reproducible train/val split, and export sanity grids + metadata.

import argparse
import json
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms, utils as vutils

def parse_args():
    p = argparse.ArgumentParser(description="Prepare CIFAR-10 for LUCID")
    p.add_argument("--data_root", type=str, default="data",
                   help="Root folder to store/download datasets")
    p.add_argument("--res", type=int, default=64, choices=[32, 64, 128],
                   help="Target resolution for upsampling")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Validation split ratio from training set")
    p.add_argument("--seed", type=int, default=123, help="Random seed for split")
    p.add_argument("--grid_n", type=int, default=64,
                   help="How many images per sanity grid (must be a square number for nice tiling)")
    p.add_argument("--grid_pad", type=int, default=2, help="Padding between grid images")
    p.add_argument("--force_redownload", action="store_true",
                   help="Force re-download of CIFAR-10")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_root).absolute()
    out_dir = root / "prepared" / f"cifar10_{args.res}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Transforms ---
    # Note: we keep pixel range in [0,1] for grids (no normalization yet).
    tfm = transforms.Compose([
        transforms.Resize((args.res, args.res), antialias=True),
        transforms.ToTensor(),
    ])

    # --- Download CIFAR-10 ---
    download_flag = True if args.force_redownload else False
    train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)

    # --- Train/Val split (from train set) ---
    n_total = len(train_set)  # 50,000
    n_val = int(n_total * args.val_ratio)
    indices = list(range(n_total))
    random.shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Save split indices for reproducibility
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits" / "train_indices.json", "w") as f:
        json.dump(train_indices, f)
    with open(out_dir / "splits" / "val_indices.json", "w") as f:
        json.dump(val_indices, f)

    # --- Save class names for reference ---
    meta = {
        "dataset": "CIFAR-10",
        "classes": train_set.classes,
        "resolution": args.res,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_set),
        }
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Build small loaders for just grabbing sample batches for grids ---
    # We’ll manually gather tensors from indices to avoid building full DataLoaders here.
    def gather_batch(dataset, idxs, n):
        chosen = idxs[:n]
        imgs = []
        for i in chosen:
            x, _ = dataset[i]
            imgs.append(x)
        return torch.stack(imgs, dim=0)  # [N, C, H, W]

    grid_n = args.grid_n
    # Make grid count a perfect square if possible (8x8 default for 64)
    tiles = int(grid_n ** 0.5)
    if tiles * tiles != grid_n:
        # round down to nearest square for a clean grid
        tiles = int((grid_n) ** 0.5)
        grid_n = tiles * tiles

    # --- Real train samples grid (for sanity) ---
    real_train_batch = gather_batch(train_set, train_indices, grid_n)
    vutils.save_image(
        real_train_batch,
        fp=str(out_dir / f"real_train_grid_{grid_n}.png"),
        nrow=tiles,
        padding=args.grid_pad
    )

    # --- Real val samples grid ---
    real_val_batch = gather_batch(train_set, val_indices, grid_n)
    vutils.save_image(
        real_val_batch,
        fp=str(out_dir / f"real_val_grid_{grid_n}.png"),
        nrow=tiles,
        padding=args.grid_pad
    )

    # --- Real test samples grid ---
    test_indices = list(range(len(test_set)))
    real_test_batch = gather_batch(test_set, test_indices, grid_n)
    vutils.save_image(
        real_test_batch,
        fp=str(out_dir / f"real_test_grid_{grid_n}.png"),
        nrow=tiles,
        padding=args.grid_pad
    )

    print("\n✅ CIFAR-10 prepared.")
    print(f"• Metadata:      {out_dir / 'metadata.json'}")
    print(f"• Train indices: {out_dir / 'splits' / 'train_indices.json'}")
    print(f"• Val indices:   {out_dir / 'splits' / 'val_indices.json'}")
    print(f"• Grids:         {out_dir / f'real_train_grid_{grid_n}.png'}")
    print(f"                 {out_dir / f'real_val_grid_{grid_n}.png'}")
    print(f"                 {out_dir / f'real_test_grid_{grid_n}.png'}\n")

if __name__ == "__main__":
    main()
