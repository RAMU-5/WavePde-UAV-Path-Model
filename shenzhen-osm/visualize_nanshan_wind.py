# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "pillow", "matplotlib", "tqdm"]
# ///
"""Render the decoded Nanshan wind field as a GIF + hero PNGs.

Each step's decoded tile is (1, 3, 16, 512, 512) for channels (u, v, w).
We stitch the 8x8 tile grid into a (3, 16, 4096, 4096) array per step,
take the z-slice at Z_IDX (~ 4 m * Z_IDX above ground), compute horizontal
wind speed |U_h| = sqrt(u^2 + v^2), and render with building footprint.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm

# --- paths ---
DECODE_DIR = Path(
    "/leonardo_scratch/large/userexternal/yli00001/SCALED_Project/"
    "SCALED-UrbanFlowv3/inference_output/NANSHAN_tileio/saved_decoded_tiles_dir"
)
RASTER_NPZ = Path(
    "/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/"
    "shenzhen-osm/nanshan/raster_4m_4096.npz"
)
OUT_DIR = Path(
    "/leonardo_scratch/large/userexternal/yli00001/Aura_Project/Levistone/"
    "shenzhen-osm/nanshan/wind_viz"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 100
N_TILES = 8        # 8x8 decoded tiles per step
TILE_PIX = 512     # pixel-space tile size (after VAE decode)
H = W = N_TILES * TILE_PIX    # 4096
Z_IDX = 8          # 8 * 4m = 32 m above ground (mid-rise level)
DS = 4             # downsample factor for GIF frames (4096/4 = 1024)
HERO_STEPS = [1, 25, 50, 75, 100]


def stitch_step(step: int, z_idx: int) -> np.ndarray:
    """Return (3, H, W) float32 — u, v, w at z_idx for given step."""
    out = np.zeros((3, H, W), dtype=np.float32)
    step_dir = DECODE_DIR / f"{step:04d}"
    for i in range(N_TILES):
        for j in range(N_TILES):
            p = step_dir / f"{i:04d}_{j:04d}.pt"
            if not p.exists():
                continue
            t = torch.load(p, map_location="cpu", weights_only=False)
            sl = t[0, :, z_idx].numpy()      # (3, 512, 512)
            y0, y1 = i * TILE_PIX, (i + 1) * TILE_PIX
            x0, x1 = j * TILE_PIX, (j + 1) * TILE_PIX
            out[:, y0:y1, x0:x1] = sl
    return out


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool an (..., H, W) array by factor."""
    if factor == 1:
        return arr
    *lead, h, w = arr.shape
    arr = arr.reshape(*lead, h // factor, factor, w // factor, factor)
    return arr.mean(axis=(-3, -1))


def main():
    print("[load] building mask", flush=True)
    nz = np.load(RASTER_NPZ, allow_pickle=True)
    bld = nz["buildings"].astype(np.uint8)         # (4096, 4096)
    bld_ds = (downsample(bld.astype(np.float32), DS) > 0.05).astype(np.uint8)

    # ── First pass: get global speed range for consistent colormap ──
    print("[scan] global |U_h| max over all steps (sampling every 10)", flush=True)
    vmax = 0.0
    for s in tqdm(range(1, N_STEPS + 1, 10), desc="scan"):
        uvw = stitch_step(s, Z_IDX)
        speed = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
        vmax = max(vmax, float(np.percentile(speed, 99.5)))
    vmax = max(vmax, 0.1)
    print(f"  vmax (P99.5) = {vmax:.3f}", flush=True)

    # ── Render all frames ──
    print("[render] frames", flush=True)
    cmap = plt.get_cmap("turbo")
    frames = []
    bld_alpha = bld_ds * 200   # max 200 → not fully opaque

    for s in tqdm(range(1, N_STEPS + 1), desc="render"):
        uvw = stitch_step(s, Z_IDX)
        u_ds = downsample(uvw[0], DS)
        v_ds = downsample(uvw[1], DS)
        speed = np.sqrt(u_ds ** 2 + v_ds ** 2)
        # normalize to [0,1], clip
        norm = np.clip(speed / vmax, 0.0, 1.0)
        rgba = (cmap(norm) * 255).astype(np.uint8)   # (h, w, 4)
        # overlay buildings (dark)
        mask = bld_ds.astype(bool)
        rgba[mask] = np.array([30, 30, 30, 255], dtype=np.uint8)
        frames.append(Image.fromarray(rgba, mode="RGBA").convert("RGB"))

        if s in HERO_STEPS:
            full_speed = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
            fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
            im = ax.imshow(full_speed, cmap="turbo", vmin=0, vmax=vmax,
                           origin="upper")
            # building footprint as overlay
            ax.imshow(np.where(bld > 0, 0.5, 0.0),
                      cmap="Greys", alpha=np.where(bld > 0, 0.85, 0.0),
                      vmin=0, vmax=1, origin="upper")
            ax.set_title(
                f"Nanshan wind |U_h| @ z={Z_IDX*4} m  —  step {s}/{N_STEPS}",
                fontsize=14)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                label="|U_h| (model units)")
            ax.set_xticks([]); ax.set_yticks([])
            hero_path = OUT_DIR / f"hero_step{s:04d}_z{Z_IDX*4}m.png"
            fig.savefig(hero_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  hero -> {hero_path.name}", flush=True)

    # ── Save GIF ──
    gif_path = OUT_DIR / f"nanshan_wind_z{Z_IDX*4}m.gif"
    print(f"[save] {gif_path}", flush=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,            # 10 fps
        loop=0,
        optimize=False,
    )
    print(f"  size {gif_path.stat().st_size/1e6:.1f} MB, {len(frames)} frames")

    # Also save as MP4-like animated PNG via PIL? Skip, GIF is enough.

    # ── Print summary ──
    print("\n[done] outputs:")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
