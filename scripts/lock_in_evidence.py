#!/usr/bin/env python3
"""
scripts/lock_in_evidence.py
Summarize lucid loop results and generate minimal plots.
"""

import pandas as pd
import json, re
from pathlib import Path
import matplotlib.pyplot as plt

def lock_in_evidence(run_dir: Path):
    metrics_path = run_dir / "metrics.csv"
    run_json = run_dir / "RUN.json"
    summary_path = run_dir / "summary.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")

    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} rows from {metrics_path}")

    # --- Extract stats ---
    best_val_rec = df["val_rec"].max()
    best_epoch = int(df.loc[df["val_rec"].idxmax(), "epoch"])
    time_per_epoch = df["epoch_time"].mean() if "epoch_time" in df else None

    # --- Extract KID per cycle ---
    kid_cols = [c for c in df.columns if c.startswith("kid_")]
    kid_per_cycle = {}
    for c in kid_cols:
        # Format like kid_cycle_005
        m = re.search(r"(\d+)", c)
        if m:
            cycle = int(m.group(1))
            kid_per_cycle[cycle] = float(df[c].dropna().iloc[-1])

    summary = {
        "best_val_rec": round(best_val_rec, 4),
        "best_epoch": best_epoch,
        "time_per_epoch_sec": round(time_per_epoch, 2) if time_per_epoch else None,
        "kid_per_cycle": kid_per_cycle,
    }

    # --- Merge with RUN.json metadata if exists ---
    if run_json.exists():
        with open(run_json) as f:
            meta = json.load(f)
        summary["run_meta"] = {k: meta.get(k) for k in ["ablation_tag", "lambda_mix", "dream_keep_pct", "res"] if k in meta}

    # --- Save summary ---
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")

    # --- Plots ---
    fig1 = plt.figure()
    plt.plot(df["epoch"], df["val_rec"], marker="o")
    plt.title("Validation Reconstruction vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("val_rec")
    plt.grid(True)
    plt.tight_layout()
    fig1.savefig(run_dir / "val_rec_trend.png", dpi=200)
    plt.close(fig1)

    if kid_per_cycle:
        fig2 = plt.figure()
        cycles, kids = zip(*sorted(kid_per_cycle.items()))
        plt.plot(cycles, kids, marker="s", color="tab:red")
        plt.title("KID vs Cycle")
        plt.xlabel("Cycle")
        plt.ylabel("KID (lower better)")
        plt.grid(True)
        plt.tight_layout()
        fig2.savefig(run_dir / "kid_trend.png", dpi=200)
        plt.close(fig2)
        print("Saved kid_trend.png")

    # --- Optional grids (if available) ---
    from PIL import Image

    grids = []
    for e in [1, 6, 11]:
        for kind in ["dreams", "recon"]:
            path = run_dir / f"{kind}_grid_e{e:03d}.png"
            if path.exists():
                grids.append(Image.open(path))

    if len(grids) == 6:
        import numpy as np
        widths, heights = zip(*(i.size for i in grids))
        w, h = max(widths), max(heights)
        combo = Image.new("RGB", (3 * w, 2 * h))
        for idx, img in enumerate(grids):
            x = (idx % 3) * w
            y = (idx // 3) * h
            combo.paste(img, (x, y))
        combo.save(run_dir / "grids_e001_e006_e011.png")
        print("Saved grids_e001_e006_e011.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to lucid_loop_* run folder")
    args = parser.parse_args()
    lock_in_evidence(args.run_dir)
