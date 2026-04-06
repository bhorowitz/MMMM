#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUTDIR = ROOT / "tests" / "out" / "compare_jax_fortran"


def read_fortran_record(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as f:
        raw = f.read()
    n1 = struct.unpack("<I", raw[:4])[0]
    n2 = struct.unpack("<I", raw[-4:])[0]
    if n1 != n2:
        raise ValueError(f"{path} has mismatched record markers")
    return np.frombuffer(raw[4:-4], dtype=dtype)


def infer_cube_dim(flat: np.ndarray) -> int:
    n = round(flat.size ** (1.0 / 3.0))
    if n ** 3 != flat.size:
        raise ValueError(f"Not a cube: {flat.size}")
    return n


def ensure_fortran_outputs() -> None:
    if not (ROOT / "prho.dat").exists():
        subprocess.run([str(ROOT / "tests" / "run_smoke_test.sh")], check=True)


def ensure_jax_outputs() -> Path:
    npz = ROOT / "tests" / "out" / "jax_reference" / "qzoom_fig6_fig7_jax.npz"
    if not npz.exists():
        subprocess.run([str(ROOT / "tests" / "generate_jax_reference.sh")], check=True)
    return npz


def load_fortran_density() -> tuple[np.ndarray, int]:
    flat = read_fortran_record(ROOT / "prho.dat", np.float32)
    n = infer_cube_dim(flat)
    return flat.reshape((n, n, n), order="F").astype(np.float64), n


def load_fortran_def(n: int) -> np.ndarray:
    flat = read_fortran_record(ROOT / "def_chk.dat", np.float32)
    if flat.size != n ** 3:
        raise ValueError("Fortran deformation cube size mismatch")
    return flat.reshape((n, n, n), order="F").astype(np.float64)


def compute_positions(def_grid: np.ndarray) -> np.ndarray:
    n = def_grid.shape[0]
    xpos = np.zeros((3, n, n, n), dtype=np.float64)
    idx = np.arange(1, n + 1, dtype=np.float64)
    xpos[0] = (np.roll(def_grid, -1, axis=0) - np.roll(def_grid, 1, axis=0)) / 2.0 + idx[:, None, None]
    xpos[1] = (np.roll(def_grid, -1, axis=1) - np.roll(def_grid, 1, axis=1)) / 2.0 + idx[None, :, None]
    xpos[2] = (np.roll(def_grid, -1, axis=2) - np.roll(def_grid, 1, axis=2)) / 2.0 + idx[None, None, :]
    return xpos


def cumulative_mass_above(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = rho.ravel().astype(np.float64)
    weights = vals / (np.sum(vals) + 1e-30)
    order = np.argsort(vals)
    vals = vals[order]
    weights = weights[order]
    mass_above = 1.0 - np.cumsum(weights) + weights
    return vals, mass_above


def make_plots() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ensure_fortran_outputs()
    jax_npz = ensure_jax_outputs()

    rho_f, n = load_fortran_density()
    def_f = load_fortran_def(n)
    xpos = compute_positions(def_f)
    mesh_layer_f = xpos[:, :, :, n // 2]

    ref = np.load(jax_npz)
    rho_slice_j = np.asarray(ref["rho_slice"], dtype=np.float64)
    mesh_xy_j = np.asarray(ref["mesh_xy"], dtype=np.float64)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=180)

    im0 = ax[0, 0].imshow(np.log10(np.maximum(rho_f[:, :, n // 2], 1e-6)).T, origin="lower", cmap="magma")
    ax[0, 0].set_title("Fortran QZOOM density slice")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    im1 = ax[0, 1].imshow(np.log10(np.maximum(rho_slice_j, 1e-6)).T, origin="lower", cmap="magma")
    ax[0, 1].set_title("JAX reference density slice")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    for i in range(n):
        ax[1, 0].plot(mesh_layer_f[0, i, :], mesh_layer_f[1, i, :], color="black", linewidth=0.8)
        ax[1, 0].plot(mesh_layer_f[0, :, i], mesh_layer_f[1, :, i], color="black", linewidth=0.8)
    ax[1, 0].set_title("Fortran QZOOM mesh layer")
    ax[1, 0].set_aspect("equal")

    for i in range(mesh_xy_j.shape[0]):
        ax[1, 1].plot(mesh_xy_j[i, :, 0], mesh_xy_j[i, :, 1], color="black", linewidth=0.8)
        ax[1, 1].plot(mesh_xy_j[:, i, 0], mesh_xy_j[:, i, 1], color="black", linewidth=0.8)
    ax[1, 1].set_title("JAX reference mesh layer")
    ax[1, 1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fortran_vs_jax_grid.png")
    plt.close(fig)

    xf, yf = cumulative_mass_above(rho_f)
    xj, yj = cumulative_mass_above(np.maximum(rho_slice_j, 1e-12))

    fig2, ax2 = plt.subplots(figsize=(6.5, 5), dpi=180)
    ax2.loglog(xf, yf, label="Fortran volume density")
    ax2.loglog(xj, yj, label="JAX slice proxy")
    ax2.set_xlabel("density")
    ax2.set_ylabel("mass fraction above density")
    ax2.set_title("Fortran vs JAX density-tail comparison")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "fortran_vs_jax_cumulative_density.png")
    plt.close(fig2)

    summary = {
        "fortran_ng": int(n),
        "jax_rho_slice_shape": list(rho_slice_j.shape),
        "fortran_density_min": float(np.min(rho_f)),
        "fortran_density_max": float(np.max(rho_f)),
        "jax_slice_min": float(np.min(rho_slice_j)),
        "jax_slice_max": float(np.max(rho_slice_j)),
        "notes": [
            "This comparison is qualitative, not a matched-physics regression.",
            "The JAX run is synthetic reference data produced by qzoom_fig6_fig7_demo.py.",
            "The Fortran run is the bundled QZOOM executable run at NG=16 with current COSMOPAR.DAT.",
        ],
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    make_plots()
