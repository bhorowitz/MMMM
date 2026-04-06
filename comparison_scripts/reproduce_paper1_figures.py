#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "tests" / "out" / "paper1"
REFDIR = OUTDIR / "reference_pages"


def read_fortran_record(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError(f"{path} is too small to be a Fortran unformatted record")
    n1 = struct.unpack("<I", raw[:4])[0]
    n2 = struct.unpack("<I", raw[-4:])[0]
    if n1 != n2:
        raise ValueError(f"{path} has mismatched Fortran record markers: {n1} != {n2}")
    payload = raw[4:-4]
    return np.frombuffer(payload, dtype=dtype)


def infer_cube_dim(flat: np.ndarray) -> int:
    n = round(flat.size ** (1.0 / 3.0))
    if n ** 3 != flat.size:
        raise ValueError(f"record length {flat.size} is not a perfect cube")
    return n


def ensure_run_outputs() -> None:
    needed = [
        ROOT / "prho.dat",
        ROOT / "def_chk.dat",
        ROOT / "state_chk.dat",
    ]
    if all(path.exists() for path in needed):
        return
    subprocess.run([str(ROOT / "tests" / "run_smoke_test.sh")], check=True)


def render_reference_pages() -> None:
    REFDIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            "6",
            "-l",
            "8",
            str(ROOT / "paper1.pdf"),
            str(REFDIR / "paper1"),
        ],
        check=True,
    )


def load_density() -> tuple[np.ndarray, int]:
    flat = read_fortran_record(ROOT / "prho.dat", np.float32)
    n = infer_cube_dim(flat)
    rho = flat.reshape((n, n, n), order="F")
    return rho, n


def load_deformation(n: int) -> np.ndarray:
    flat = read_fortran_record(ROOT / "def_chk.dat", np.float32)
    if flat.size != n ** 3:
        raise ValueError("def_chk.dat and prho.dat disagree on cube size")
    return flat.reshape((n, n, n), order="F")


def load_raw_particles(n: int) -> tuple[np.ndarray, np.ndarray]:
    flat = read_fortran_record(ROOT / "rawpart_chk.dat", np.float32)
    if flat.size % 6 != 0:
        raise ValueError("rawpart_chk.dat does not contain 6 values per particle")
    xv = flat.reshape((6, flat.size // 6), order="F")
    pmass_path = ROOT / "pmass0.dat"
    if pmass_path.exists():
        try:
            pmass = read_fortran_record(pmass_path, np.float32).astype(np.float64)
        except Exception:
            pmass = np.ones((xv.shape[1],), dtype=np.float64)
    else:
        pmass = np.ones((xv.shape[1],), dtype=np.float64)
    if pmass.size != xv.shape[1]:
        raise ValueError("pmass0.dat length does not match rawpart_chk.dat")
    return xv.astype(np.float64), pmass


def plot_density_slice(rho: np.ndarray, outpath: Path) -> None:
    midslice = rho[:, :, :].sum(axis=0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    im = ax.imshow(np.log10(np.maximum(midslice, 1e-6)).T, origin="lower", cmap="magma")
    ax.set_title("Paper Fig. 6-like Density Slice")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(density)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def compute_positions(def_grid: np.ndarray) -> np.ndarray:
    n = def_grid.shape[0]
    xpos = np.zeros((3, n, n, n), dtype=np.float64)
    indices = np.arange(1, n + 1, dtype=np.float64)
    base_x = indices[:, None, None]
    base_y = indices[None, :, None]
    base_z = indices[None, None, :]
    xpos[0] = (np.roll(def_grid, -1, axis=0) - np.roll(def_grid, 1, axis=0)) / 2.0 + base_x
    xpos[1] = (np.roll(def_grid, -1, axis=1) - np.roll(def_grid, 1, axis=1)) / 2.0 + base_y
    xpos[2] = (np.roll(def_grid, -1, axis=2) - np.roll(def_grid, 1, axis=2)) / 2.0 + base_z
    return xpos


def map_raw_particles_to_cartesian(xv: np.ndarray, def_grid: np.ndarray) -> np.ndarray:
    n = def_grid.shape[0]
    disp = np.zeros((3, n, n, n), dtype=np.float64)
    disp[0] = np.roll(def_grid, -1, axis=0) - def_grid
    disp[1] = np.roll(def_grid, -1, axis=1) - def_grid
    disp[2] = np.roll(def_grid, -1, axis=2) - def_grid
    disp[0] = 0.5 * (np.roll(disp[0], -1, axis=1) + disp[0])
    disp[0] = 0.5 * (np.roll(disp[0], -1, axis=2) + disp[0])
    disp[1] = 0.5 * (np.roll(disp[1], -1, axis=0) + disp[1])
    disp[1] = 0.5 * (np.roll(disp[1], -1, axis=2) + disp[1])
    disp[2] = 0.5 * (np.roll(disp[2], -1, axis=0) + disp[2])
    disp[2] = 0.5 * (np.roll(disp[2], -1, axis=1) + disp[2])

    pos = np.empty((xv.shape[1], 3), dtype=np.float64)
    x = xv[0]
    y = xv[1]
    z = xv[2]

    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    wx = x - ix
    wy = y - iy
    wz = z - iz

    ix0 = np.mod(ix - 1, n)
    iy0 = np.mod(iy - 1, n)
    iz0 = np.mod(iz - 1, n)
    ix1 = (ix0 + 1) % n
    iy1 = (iy0 + 1) % n
    iz1 = (iz0 + 1) % n

    w000 = (1 - wx) * (1 - wy) * (1 - wz)
    w001 = (1 - wx) * (1 - wy) * wz
    w010 = (1 - wx) * wy * (1 - wz)
    w011 = (1 - wx) * wy * wz
    w100 = wx * (1 - wy) * (1 - wz)
    w101 = wx * (1 - wy) * wz
    w110 = wx * wy * (1 - wz)
    w111 = wx * wy * wz

    for d in range(3):
        g = disp[d]
        delta = (
            g[ix0, iy0, iz0] * w000
            + g[ix0, iy0, iz1] * w001
            + g[ix0, iy1, iz0] * w010
            + g[ix0, iy1, iz1] * w011
            + g[ix1, iy0, iz0] * w100
            + g[ix1, iy0, iz1] * w101
            + g[ix1, iy1, iz0] * w110
            + g[ix1, iy1, iz1] * w111
        )
        pos[:, d] = np.mod(xv[d] + delta - 1.0, float(n))

    return pos


def cic_deposit_periodic(pos: np.ndarray, mass: np.ndarray, out_ng: int, box_size: float) -> np.ndarray:
    grid = np.zeros((out_ng, out_ng, out_ng), dtype=np.float64)
    scaled = np.mod(pos, box_size) * (float(out_ng) / float(box_size))

    ix = np.floor(scaled[:, 0]).astype(np.int64)
    iy = np.floor(scaled[:, 1]).astype(np.int64)
    iz = np.floor(scaled[:, 2]).astype(np.int64)
    wx = scaled[:, 0] - ix
    wy = scaled[:, 1] - iy
    wz = scaled[:, 2] - iz

    ix0 = np.mod(ix, out_ng)
    iy0 = np.mod(iy, out_ng)
    iz0 = np.mod(iz, out_ng)
    ix1 = (ix0 + 1) % out_ng
    iy1 = (iy0 + 1) % out_ng
    iz1 = (iz0 + 1) % out_ng

    weights = [
        (ix0, iy0, iz0, (1 - wx) * (1 - wy) * (1 - wz)),
        (ix0, iy0, iz1, (1 - wx) * (1 - wy) * wz),
        (ix0, iy1, iz0, (1 - wx) * wy * (1 - wz)),
        (ix0, iy1, iz1, (1 - wx) * wy * wz),
        (ix1, iy0, iz0, wx * (1 - wy) * (1 - wz)),
        (ix1, iy0, iz1, wx * (1 - wy) * wz),
        (ix1, iy1, iz0, wx * wy * (1 - wz)),
        (ix1, iy1, iz1, wx * wy * wz),
    ]
    for iix, iiy, iiz, ww in weights:
        np.add.at(grid, (iix, iiy, iiz), mass * ww)
    return grid


def plot_mesh_layer(def_grid: np.ndarray, outpath: Path) -> None:
    xpos = compute_positions(def_grid)
    n = def_grid.shape[0]
    layer = n // 2

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    for i in range(n):
        ax.plot(xpos[0, i, :, layer], xpos[1, i, :, layer], color="black", linewidth=0.8)
        ax.plot(xpos[0, :, i, layer], xpos[1, :, i, layer], color="black", linewidth=0.8)
    ax.set_title("Paper Fig. 7-like Mesh Layer")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_cumulative_density(rho: np.ndarray, outpath: Path, title: str) -> None:
    vals = rho.ravel().astype(np.float64)
    weights = vals / np.sum(vals)
    order = np.argsort(vals)
    vals = vals[order]
    weights = weights[order]
    mass_above = 1.0 - np.cumsum(weights) + weights

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.loglog(vals, mass_above, color="black", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("density")
    ax.set_ylabel("mass fraction above density")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def write_manifest(n: int, output_ng: int, source: str) -> None:
    text = f"""# Paper 1 Reproduction Notes

- Build target: `QZOOM/relaxing.x` via `make ARCH=GFORTRAN`
- Runtime used for regenerated outputs: `OMP_NUM_THREADS=1`, unlimited stack
- Generated mesh size: `{n}^3` from the current `dimen.fh`
- Fig. 8-like density binning used here: `{output_ng}^3`
- Fig. 8-like source used here: `{source}`

## Reproduced From This Checkout

- `paper1_fig6_like_density_slice.png`: density slice from the regenerated `prho.dat`
- `paper1_fig7_like_mesh_layer.png`: mesh layer from the regenerated `def_chk.dat`
- `paper1_fig8_like_cumulative_density.png`: cumulative mass-above-density curve from particle data binned to the requested output mesh

## Included As Reference

- Rasterized paper pages containing Figures 4-11 under `reference_pages/`

## Not Fully Rebuilt Here

- Figures 4-5: require the original FFT-vs-multigrid comparison workflow/data
- Figures 9-11: require higher-resolution comparison/top-hat-collapse setups or archival outputs not bundled in this checkout
"""
    (OUTDIR / "README.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig8-output-ng", type=int, default=256)
    ap.add_argument(
        "--fig8-source",
        type=str,
        default="fortran",
        choices=["fortran", "jaxpm_qzoom", "jaxpm_pm"],
    )
    ap.add_argument(
        "--jaxpm-state",
        type=str,
        default=str(ROOT / "tests" / "out" / "lpt_z10_compare" / "state_z10_to_z0.npz"),
    )
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ensure_run_outputs()
    render_reference_pages()

    rho, n = load_density()
    def_grid = load_deformation(n)

    plot_density_slice(rho, OUTDIR / "paper1_fig6_like_density_slice.png")
    plot_mesh_layer(def_grid, OUTDIR / "paper1_fig7_like_mesh_layer.png")

    if str(args.fig8_source) == "fortran":
        xv, pmass = load_raw_particles(n)
        pos = map_raw_particles_to_cartesian(xv, def_grid)
        rho_fig8 = cic_deposit_periodic(pos, pmass, int(args.fig8_output_ng), float(n))
        rho_fig8 /= rho_fig8.mean()
        title = f"Paper Fig. 8-like Cumulative Density ({args.fig8_output_ng}^3 binned)"
        source_note = "fortran moving-mesh particle dump"
    else:
        state = np.load(args.jaxpm_state)
        key = "pos_qzoom_phys" if str(args.fig8_source) == "jaxpm_qzoom" else "pos_jax"
        pos = np.asarray(state[key], dtype=np.float64)
        pmass = np.ones((pos.shape[0],), dtype=np.float64)
        box_size = float(np.asarray(state["box_ng"]).reshape(-1)[0]) if "box_ng" in state else float(n)
        rho_fig8 = cic_deposit_periodic(pos, pmass, int(args.fig8_output_ng), box_size)
        label = "QZOOM final field" if str(args.fig8_source) == "jaxpm_qzoom" else "JaxPM final field"
        title = f"Paper Fig. 8-like Cumulative Density from shared JaxPM ICs ({label}, {args.fig8_output_ng}^3)"
        source_note = f"{key} from {args.jaxpm_state}"

    plot_cumulative_density(rho_fig8, OUTDIR / "paper1_fig8_like_cumulative_density.png", title=title)
    write_manifest(n, int(args.fig8_output_ng), source_note)


if __name__ == "__main__":
    main()
