#!/usr/bin/env python3
"""Compare external simulation snapshot against jaxpm/static/moving-mesh evolutions.

Inputs (default: ./simulations)
-------------------------------
  - pos_z20.npy  : particle positions at z=20 in *grid coordinates* (periodic; may be out of bounds)
  - vel_z20.npy  : particle velocities/momenta at z=20 (see --vel-convention/--p-units)
  - field_z2.npy : reference density-like field at z=2 (normalization arbitrary; we compare deltas)
  - cosmo.txt    : cosmological parameters (Omega_b, Omega_c, h, n, sigma8, ...)

Runs
----
  1) jaxpm PM (Fourier kernel forces) on an ng_force grid
  2) static mesh (this repo's uniform-grid Poisson FFT) on ng_force
  3) moving mesh (this repo) on ng_force

All outputs are painted and compared on ng_field (default: field_z2.npy resolution).
We save lots of plots at z=2: projected densities, differences, P(k), r(k), scatter.

Notes
-----
This script uses the repo's cosmological stepping conventions (canonical momentum p,
integrating in scale factor a with dtau(a)=da/(a^2 E(a))).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

import jax_cosmo as jc

import qzoom_nbody_flow as qz
from organized_model import metric_from_triad, triad_from_def
from cosmology import CosmologyParams, canonical_kick_drift
from cosmo_apm import CosmoStepConfig, step_moving_mesh_a, step_static_pm_a

Array = jnp.ndarray


def _a_of_z(z: float) -> float:
    return float(1.0 / (1.0 + float(z)))


def _read_cosmo_txt(path: Path) -> dict[str, float]:
    txt = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    for line in txt.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if not m:
            continue
        out[m.group(1)] = float(m.group(2))
    if not out:
        raise SystemExit(f"failed to parse cosmology from {path}")
    return out


def _infer_ng_in(pos: np.ndarray) -> int:
    n = int(round(pos.shape[0] ** (1.0 / 3.0)))
    if n * n * n != int(pos.shape[0]):
        raise SystemExit(f"pos has {pos.shape[0]} particles which is not a perfect cube")
    # Common pattern in this repo: npart_side = 2*ng.
    if n % 2 == 0:
        ng_guess = n // 2
        # If the domain length is ng_guess, positions should mostly lie within ~[0,ng_guess).
        if float(np.nanmax(pos)) < 1.2 * float(ng_guess) + 2.0:
            return int(ng_guess)
    return int(n)


def _ensure_cubic(field: np.ndarray) -> np.ndarray:
    if field.ndim != 3 or field.shape[0] != field.shape[1] or field.shape[1] != field.shape[2]:
        raise SystemExit(f"reference field must be cubic 3D, got shape={field.shape}")
    return field


def _delta_from_rho(rho: np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64)
    return (r / (r.mean() + 1e-30) - 1.0).astype(np.float64)


def _rel_l2_centered(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    da = aa - aa.mean()
    db = bb - bb.mean()
    return float(np.linalg.norm(da - db) / (np.linalg.norm(da) + 1e-30))


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-30
    return float(np.dot(aa, bb) / den)


def _pk_logrmse(pk: np.ndarray, pk_ref: np.ndarray, *, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    a = np.asarray(pk, dtype=np.float64)[m]
    b = np.asarray(pk_ref, dtype=np.float64)[m]
    ok = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    x = np.log10(a[ok]) - np.log10(b[ok])
    return float(np.sqrt(np.mean(x * x)))


def _r_mean(r: np.ndarray, *, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    rr = np.asarray(r, dtype=np.float64)[m]
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        return float("nan")
    return float(np.mean(rr))


def _paint_rho(positions_grid: Array, pmass: Array, *, ng: int) -> Array:
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    return qz.cic_deposit_3d(rho0, jnp.mod(positions_grid, float(ng)), pmass)


def _project(rho: np.ndarray, axis: int) -> np.ndarray:
    return np.sum(rho, axis=int(axis), dtype=np.float64)


def _plot_imshow(out: Path, img: np.ndarray, *, title: str, cmap: str = "magma", vmin=None, vmax=None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2))
    im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_grid4(
    out: Path,
    *,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    titles: tuple[str, str, str, str],
    suptitle: str,
) -> None:
    import matplotlib.pyplot as plt

    vv = np.concatenate([A.ravel(), B.ravel(), C.ravel(), D.ravel()])
    vmin = float(np.percentile(vv, 1.0))
    vmax = float(np.percentile(vv, 99.0))

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 9.4))
    fig.suptitle(suptitle)
    ims = []
    ims.append(ax[0, 0].imshow(A.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
    ax[0, 0].set_title(titles[0])
    ims.append(ax[0, 1].imshow(B.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
    ax[0, 1].set_title(titles[1])
    ims.append(ax[1, 0].imshow(C.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
    ax[1, 0].set_title(titles[2])
    ims.append(ax[1, 1].imshow(D.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
    ax[1, 1].set_title(titles[3])

    for axy in ax.ravel():
        axy.set_xticks([])
        axy.set_yticks([])

    # Put a single shared colorbar in a dedicated axis to avoid overlap.
    fig.subplots_adjust(right=0.88, top=0.92, wspace=0.06, hspace=0.10)
    cax = fig.add_axes([0.90, 0.13, 0.025, 0.72])  # [left, bottom, width, height] in figure fraction
    fig.colorbar(ims[-1], cax=cax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_pk_corr(
    out: Path,
    *,
    k: np.ndarray,
    pk_ref: np.ndarray,
    pk_jax: np.ndarray,
    pk_static: np.ndarray,
    pk_mm: np.ndarray,
    r_jax: np.ndarray,
    r_static: np.ndarray,
    r_mm: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.3))
    fig.suptitle(title)

    ax[0].loglog(k, np.maximum(k * pk_ref, 1e-30), label="ref field")
    ax[0].loglog(k, np.maximum(k * pk_jax, 1e-30), label="jaxpm PM")
    ax[0].loglog(k, np.maximum(k * pk_static, 1e-30), label="static mesh")
    ax[0].loglog(k, np.maximum(k * pk_mm, 1e-30), label="moving mesh")
    ax[0].set_xlabel("k [h/Mpc] (from --box)")
    ax[0].set_ylabel("kP(k)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()

    ax[1].plot(k, r_jax, label="corr(jaxpm, ref)")
    ax[1].plot(k, r_static, label="corr(static, ref)")
    ax[1].plot(k, r_mm, label="corr(moving, ref)")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0.0, 1.05)
    ax[1].set_xlabel("k [h/Mpc] (from --box)")
    ax[1].set_ylabel("cross-correlation r(k)")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_scatter(out: Path, x: np.ndarray, y: np.ndarray, *, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    # Subsample for speed/clarity.
    n = xx.size
    if n > 300_000:
        idx = np.random.default_rng(0).choice(n, size=300_000, replace=False)
        xx = xx[idx]
        yy = yy[idx]

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.4))
    ax.scatter(xx, yy, s=1, alpha=0.08, rasterized=True)
    lo = float(np.percentile(np.concatenate([xx, yy]), 1.0))
    hi = float(np.percentile(np.concatenate([xx, yy]), 99.0))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _unwrap_periodic_line(x: np.ndarray, box: float) -> np.ndarray:
    """Return an unwrapped polyline (periodic-minimal increments)."""
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        dx = (x[i] - x[i - 1] + 0.5 * box) % box - 0.5 * box
        out[i] = out[i - 1] + dx
    return out


def _mesh_segments_xy(xg: np.ndarray, yg: np.ndarray, box: float) -> np.ndarray:
    """Convert a periodic grid (xg,yg) to line segments in R^2.

    Returns (Nseg, 2, 2) segments for a LineCollection.
    """
    segs = []
    nx, ny = xg.shape

    # Horizontal polylines
    for i in range(nx):
        xu = _unwrap_periodic_line(xg[i, :], box)
        yu = _unwrap_periodic_line(yg[i, :], box)
        segs.append(np.stack([np.stack([xu[:-1], yu[:-1]], axis=1), np.stack([xu[1:], yu[1:]], axis=1)], axis=1))

    # Vertical polylines
    for j in range(ny):
        xu = _unwrap_periodic_line(xg[:, j], box)
        yu = _unwrap_periodic_line(yg[:, j], box)
        segs.append(np.stack([np.stack([xu[:-1], yu[:-1]], axis=1), np.stack([xu[1:], yu[1:]], axis=1)], axis=1))

    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.concatenate(segs, axis=0)


def _plot_mesh_layer(
    out: Path,
    *,
    mesh_xy: np.ndarray,
    box: float,
    title: str,
    particles_xy: np.ndarray | None = None,
    uniform_stride: int = 1,
    quiver: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    xg0 = mesh_xy[..., 0]
    yg0 = mesh_xy[..., 1]
    segs_main = _mesh_segments_xy(xg0, yg0, box)

    # Uniform grid (for reference)
    nx, ny = xg0.shape
    uu = np.linspace(0.0, box, nx, endpoint=False)
    uu = uu[:: max(1, int(uniform_stride))]
    Ux, Uy = np.meshgrid(uu, uu, indexing="ij")
    segs_u = _mesh_segments_xy(Ux, Uy, box)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.6))
    ax.set_title(title)

    lc_u = LineCollection(segs_u, colors="white", linewidths=0.5, alpha=0.35)
    ax.add_collection(lc_u)

    lc = LineCollection(segs_main, colors="tab:cyan", linewidths=0.8, alpha=0.9)
    ax.add_collection(lc)

    if particles_xy is not None:
        ax.scatter(particles_xy[:, 0], particles_xy[:, 1], s=1.0, marker=".", color="tab:orange", alpha=0.20, linewidths=0.0)

    if quiver is not None:
        # quiver: (Nx,Ny,2) vectors at mesh points
        ax.quiver(
            xg0,
            yg0,
            quiver[..., 0],
            quiver[..., 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
            color="tab:green",
            alpha=0.7,
        )

    ax.set_xlim(0.0, box)
    ax.set_ylim(0.0, box)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_hist(out: Path, x: np.ndarray, *, title: str, xlabel: str, logx: bool = False) -> None:
    import matplotlib.pyplot as plt

    xx = np.asarray(x, dtype=np.float64)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4))
    ax.hist(xx, bins=100, density=True, alpha=0.85)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pdf")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_density_pdfs(
    out: Path,
    *,
    rho_ref: np.ndarray,
    rho_jax: np.ndarray,
    rho_static: np.ndarray,
    rho_moving: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    def prep(r: np.ndarray) -> np.ndarray:
        rr = np.asarray(r, dtype=np.float64).reshape(-1)
        rr = rr[np.isfinite(rr)]
        rr = rr[rr > 0]
        return rr

    R0 = prep(rho_ref)
    R1 = prep(rho_jax)
    R2 = prep(rho_static)
    R3 = prep(rho_moving)

    if min(R0.size, R1.size, R2.size, R3.size) == 0:
        return

    # Normalize by mean to make curves comparable despite arbitrary normalization.
    R0n = R0 / (R0.mean() + 1e-30)
    R1n = R1 / (R1.mean() + 1e-30)
    R2n = R2 / (R2.mean() + 1e-30)
    R3n = R3 / (R3.mean() + 1e-30)

    # Density PDF bins (in rho/mean space)
    lo = float(np.percentile(np.concatenate([R0n, R1n, R2n, R3n]), 0.1))
    hi = float(np.percentile(np.concatenate([R0n, R1n, R2n, R3n]), 99.9))
    lo = max(lo, 1e-4)
    hi = max(hi, lo * 1.01)
    bins_r = np.logspace(np.log10(lo), np.log10(hi), 140)

    # Log-density PDF bins (in log10(rho/mean))
    L0 = np.log10(R0n)
    L1 = np.log10(R1n)
    L2 = np.log10(R2n)
    L3 = np.log10(R3n)
    llo = float(np.percentile(np.concatenate([L0, L1, L2, L3]), 0.1))
    lhi = float(np.percentile(np.concatenate([L0, L1, L2, L3]), 99.9))
    bins_l = np.linspace(llo, lhi, 160)

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.6))
    fig.suptitle(title)

    def hist(ax0, x, bins, label):
        h, e = np.histogram(x, bins=bins, density=True)
        c = 0.5 * (e[:-1] + e[1:])
        ax0.plot(c, h, label=label)

    hist(ax[0], R0n, bins_r, "ref field")
    hist(ax[0], R1n, bins_r, "jaxpm PM")
    hist(ax[0], R2n, bins_r, "static mesh")
    hist(ax[0], R3n, bins_r, "moving mesh")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\rho/\bar{\rho}$")
    ax[0].set_ylabel("pdf")
    ax[0].grid(True, which="both", alpha=0.25)
    ax[0].legend()

    hist(ax[1], L0, bins_l, "ref field")
    hist(ax[1], L1, bins_l, "jaxpm PM")
    hist(ax[1], L2, bins_l, "static mesh")
    hist(ax[1], L3, bins_l, "moving mesh")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\log_{10}(\rho/\bar{\rho})$")
    ax[1].set_ylabel("pdf")
    ax[1].grid(True, which="both", alpha=0.25)
    ax[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _forces_jaxpm_pm(
    positions_grid: Array,
    pmass: Array,
    *,
    ng: int,
    box: float,
    force_scale: float,
) -> Array:
    """Return PM forces dp/dτ at particle positions using jaxpm Fourier kernels.

    positions_grid are in grid coordinates [0,ng).
    Returned forces are in physical units consistent with `dx=box/ng`.
    """
    import jaxpm.pm as jpm  # type: ignore
    from jaxpm.painting import cic_paint  # type: ignore

    dx = float(box) / float(ng)

    pos = jnp.mod(positions_grid, float(ng))
    rho = cic_paint(jnp.zeros((ng, ng, ng), dtype=jnp.float32), pos, weight=pmass)
    rho_mean = jnp.mean(rho)
    delta = rho / (rho_mean + 1e-12) - 1.0
    delta_k = jnp.fft.rfftn(delta)

    kvec = jpm.fftk((ng, ng, ng))
    pot_k = delta_k * jpm.invlaplace_kernel(kvec) * jpm.longrange_kernel(kvec, r_split=0.0)

    Fx = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 0) * pot_k, s=(ng, ng, ng)).real
    Fy = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 1) * pot_k, s=(ng, ng, ng)).real
    Fz = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 2) * pot_k, s=(ng, ng, ng)).real
    F_grid = jnp.stack([Fx, Fy, Fz], axis=-1)

    # Convert from grid-coordinate units (dx=1) to physical, and apply cosmology scaling.
    F_grid = F_grid * float(dx) * float(force_scale)
    return qz.cic_readout_3d_multi(F_grid, pos)


def _evolve_jaxpm_pm_a(
    pos0_grid: Array,
    p0: Array,
    pmass: Array,
    *,
    cosmo: CosmologyParams,
    box: float,
    ng: int,
    a0: float,
    afinal: float,
    steps: int,
    force_scale: float,
) -> Array:
    """Evolve using jaxpm PM forces, integrating in a with fixed da."""
    dx = float(box) / float(ng)
    da = (float(afinal) - float(a0)) / max(int(steps), 1)
    pos = pos0_grid
    p = p0
    for i in range(int(steps)):
        a_mid = float(a0) + (float(i) + 0.5) * da
        dtau, kick, drift_x = canonical_kick_drift(
            cosmo=cosmo, a_mid=jnp.array(a_mid, dtype=jnp.float32), da=jnp.array(da, dtype=jnp.float32)
        )
        forces = _forces_jaxpm_pm(pos, pmass, ng=int(ng), box=float(box), force_scale=float(force_scale))
        p = p + forces * float(kick)
        pos = jnp.mod(pos + (p * float(drift_x)) / float(dx), float(ng))
    return pos


def _spectra_and_corr(
    delta_ref: np.ndarray,
    delta_jax: np.ndarray,
    delta_static: np.ndarray,
    delta_mm: np.ndarray,
    *,
    box: float,
) -> dict[str, Any]:
    from ps_np import cross_correlation, power_spectrum

    kf = 2.0 * np.pi / float(box)
    kmin = kf
    dk = kf
    box3 = np.array([float(box)] * 3)

    k, pk_ref = power_spectrum(delta_ref, kmin=kmin, dk=dk, boxsize=box3)
    _, pk_jax = power_spectrum(delta_jax, kmin=kmin, dk=dk, boxsize=box3)
    _, pk_static = power_spectrum(delta_static, kmin=kmin, dk=dk, boxsize=box3)
    _, pk_mm = power_spectrum(delta_mm, kmin=kmin, dk=dk, boxsize=box3)

    k2, r_jax = cross_correlation(delta_jax, delta_ref, kmin=kmin, dk=dk, boxsize=box3)
    _, r_static = cross_correlation(delta_static, delta_ref, kmin=kmin, dk=dk, boxsize=box3)
    _, r_mm = cross_correlation(delta_mm, delta_ref, kmin=kmin, dk=dk, boxsize=box3)

    k = np.asarray(k, dtype=np.float64)
    k2 = np.asarray(k2, dtype=np.float64)
    r_jax = np.asarray(np.interp(k, k2, np.asarray(r_jax, dtype=np.float64)), dtype=np.float64)
    r_static = np.asarray(np.interp(k, k2, np.asarray(r_static, dtype=np.float64)), dtype=np.float64)
    r_mm = np.asarray(np.interp(k, k2, np.asarray(r_mm, dtype=np.float64)), dtype=np.float64)

    return {
        "k": k,
        "pk_ref": np.asarray(pk_ref, dtype=np.float64),
        "pk_jaxpm": np.asarray(pk_jax, dtype=np.float64),
        "pk_static": np.asarray(pk_static, dtype=np.float64),
        "pk_moving": np.asarray(pk_mm, dtype=np.float64),
        "r_jaxpm": r_jax,
        "r_static": r_static,
        "r_moving": r_mm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", type=str, default="./simulations")
    ap.add_argument("--outdir", type=str, default="./test_outputs/simulations_z20_to_z2_compare")

    ap.add_argument("--box", type=float, default=25.0, help="Box size in Mpc/h (used for dx and k-axis).")
    ap.add_argument("--z0", type=float, default=20.0)
    ap.add_argument("--zfinal", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--ng-force", type=int, default=0, help="Force mesh size. 0 means infer from pos file.")
    ap.add_argument("--ng-field", type=int, default=0, help="Output/compare field grid size. 0 means use ref field shape.")
    ap.add_argument("--ng-in", type=int, default=0, help="Domain grid size used by input positions. 0 means infer.")

    ap.add_argument("--vel-convention", type=str, default="p", choices=["p", "v"], help="vel file is canonical momentum p or physical velocity v=dx/dτ.")
    ap.add_argument("--p-units", type=str, default="grid", choices=["grid", "physical"], help="Units of p or v in vel file.")

    # Moving mesh config
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--smooth-steps", type=int, default=1)
    ap.add_argument("--mg-cycles", type=int, default=10)
    ap.add_argument("--mg-v1", type=int, default=2)
    ap.add_argument("--mg-v2", type=int, default=2)
    ap.add_argument("--mg-mu", type=int, default=2, help="1=V-cycle, 2=W-cycle")
    ap.add_argument("--no-limiter", action="store_true")
    ap.add_argument("--compressmax", type=float, default=80.0)
    ap.add_argument("--skewmax", type=float, default=60.0)
    ap.add_argument("--limiter-mode", type=str, default="source", choices=["both", "source", "post"])
    ap.add_argument("--xhigh", type=float, default=2.0, help="Limiter clip threshold for tmp (densinit-mass).")
    ap.add_argument("--relax-steps", type=float, default=30.0, help="Limiter relaxation time in steps (QZOOM ~30).")
    ap.add_argument("--hard-strength", type=float, default=1.0, help="Strength of hard limiter source term.")
    ap.add_argument("--limiter-smooth-tmp", type=int, default=3)
    ap.add_argument("--limiter-smooth-hard", type=int, default=1)
    ap.add_argument("--local-limit", action="store_true", help="Use local post-scaling limiter (recommended).")
    ap.add_argument("--global-limit", action="store_true", help="Force global post-scaling limiter.")

    ap.add_argument("--proj-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--mesh-stride", type=int, default=2, help="Stride for mesh layer visualization (moving mesh only).")
    ap.add_argument("--mesh-z", type=int, default=None, help="Fixed z-index for mesh diagnostic slices (0..ng_force-1).")
    ap.add_argument("--mesh-quiver", action="store_true", help="Overlay displacement vectors on the mesh layer plot.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    simdir = Path(args.simdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pos = np.load(simdir / "pos_z20.npy").astype(np.float32)
    vel = np.load(simdir / "vel_z20.npy").astype(np.float32)
    ref_rho = _ensure_cubic(np.load(simdir / "field_z2.npy").astype(np.float32))
    cosmo_d = _read_cosmo_txt(simdir / "cosmo.txt")

    if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 3:
        raise SystemExit(f"pos/vel must be (Np,3) with same shape; got pos={pos.shape} vel={vel.shape}")

    ng_in = int(args.ng_in) if int(args.ng_in) > 0 else _infer_ng_in(pos)
    ng_force = int(args.ng_force) if int(args.ng_force) > 0 else int(ng_in)
    ref_ng = int(ref_rho.shape[0])
    ng_field = int(args.ng_field) if int(args.ng_field) > 0 else int(ref_ng)

    # If requested, downsample the reference field to ng_field by block-averaging.
    # (We only implement the common integer-factor downsample used for comparisons.)
    if int(ng_field) != int(ref_ng):
        if ref_ng % ng_field != 0:
            raise SystemExit(f"--ng-field={ng_field} must divide ref field size {ref_ng} for downsampling.")
        f = ref_ng // ng_field
        x = ref_rho.astype(np.float64)
        x = x.reshape(ng_field, f, ng_field, f, ng_field, f).mean(axis=(1, 3, 5))
        ref_rho = x.astype(np.float32)

    a0 = _a_of_z(float(args.z0))
    afinal = _a_of_z(float(args.zfinal))
    if afinal <= a0:
        raise SystemExit("need zfinal < z0 (so afinal > a0)")

    # Cosmology
    cosmo = jc.Cosmology(
        Omega_c=float(cosmo_d.get("Omega_c")),
        Omega_b=float(cosmo_d.get("Omega_b")),
        h=float(cosmo_d.get("h")),
        n_s=float(cosmo_d.get("n", cosmo_d.get("n_s"))),
        sigma8=float(cosmo_d.get("sigma8")),
        Omega_k=float(cosmo_d.get("Omega_k", 0.0)),
        w0=float(cosmo_d.get("w0", -1.0)),
        wa=float(cosmo_d.get("wa", 0.0)),
    )
    cosmo_p = CosmologyParams(cosmo=cosmo)
    force_scale = 1.5 * float(cosmo.Omega_m)

    box = float(args.box)
    dx_force = box / float(ng_force)
    dx_in = box / float(ng_in)

    # --- Convert inputs to (pos_grid_on_ng_force, canonical p in physical units) ---
    pos_grid = jnp.asarray(pos, dtype=jnp.float32) * (float(ng_force) / float(ng_in))

    vraw = jnp.asarray(vel, dtype=jnp.float32)
    if str(args.vel_convention) == "v":
        praw = vraw * float(a0)
    else:
        praw = vraw

    if str(args.p_units) == "grid":
        p_phys = praw * float(dx_in)
    else:
        p_phys = praw

    # Convention used across this repo: choose pmass so that mean mass-per-cell on the *force grid* is ~1.
    npart = int(pos_grid.shape[0])
    pmass_val = float(ng_force ** 3) / float(npart)
    pmass = jnp.full((npart,), pmass_val, dtype=jnp.float32)

    # --- Run evolutions ---
    t0 = time.perf_counter()
    pos_jax = _evolve_jaxpm_pm_a(
        pos_grid,
        p_phys,
        pmass,
        cosmo=cosmo_p,
        box=box,
        ng=ng_force,
        a0=a0,
        afinal=afinal,
        steps=int(args.steps),
        force_scale=force_scale,
    )
    t1 = time.perf_counter()

    cfg = CosmoStepConfig(ng=int(ng_force), force_scale=float(force_scale))
    state_static = qz.NBodyState(xv=jnp.concatenate([jnp.mod(pos_grid, float(ng_force)), p_phys], axis=-1), pmass=pmass)
    da = (float(afinal) - float(a0)) / max(int(args.steps), 1)
    for i in range(int(args.steps)):
        a_mid = float(a0) + (float(i) + 0.5) * da
        state_static, _phi, _rho = step_static_pm_a(state_static, cosmo=cosmo_p, cfg=cfg, a_mid=a_mid, da=da, dx=float(dx_force))
    pos_static = state_static.xv[:, 0:3]
    t2 = time.perf_counter()

    levels = max(1, int(np.log2(int(ng_force)) - 1))
    mg_params = qz.MGParams(
        levels=levels,
        v1=int(args.mg_v1),
        v2=int(args.mg_v2),
        mu=int(args.mg_mu),
        cycles=int(args.mg_cycles),
    )
    use_source = str(args.limiter_mode) in ("both", "source")
    use_post = str(args.limiter_mode) in ("both", "post")
    if bool(args.global_limit) and bool(args.local_limit):
        raise SystemExit("Use at most one of --local-limit/--global-limit.")
    local_limit = True if bool(args.local_limit) else (False if bool(args.global_limit) else True)
    limiter = qz.LimiterParams(
        enabled=not bool(args.no_limiter),
        compressmax=float(args.compressmax),
        skewmax=float(args.skewmax),
        xhigh=float(args.xhigh),
        relax_steps=float(args.relax_steps),
        use_source_term=bool(use_source),
        use_post_scale=bool(use_post),
        post_only_on_fail=True,
        hard_strength=float(args.hard_strength),
        smooth_tmp=int(args.limiter_smooth_tmp),
        smooth_hard=int(args.limiter_smooth_hard),
        smooth_defp=int(args.smooth_steps),  # make --smooth-steps actually matter with limiter enabled
        local_limit=bool(local_limit),
    )
    state_mm = qz.NBodyState(xv=jnp.concatenate([jnp.mod(pos_grid, float(ng_force)), p_phys], axis=-1), pmass=pmass)
    def_field = jnp.zeros((ng_force, ng_force, ng_force), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    densinit = jnp.zeros_like(def_field)
    if limiter.enabled:
        rho0 = jnp.zeros_like(def_field)
        rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def_field, state_mm, dx=float(dx_force))
        densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))

    # dtau_prev: start with the first-step dtau for limiter clipping.
    a_mid0 = float(a0) + 0.5 * da
    dtau0, _kick0, _drift0 = canonical_kick_drift(
        cosmo=cosmo_p, a_mid=jnp.array(a_mid0, dtype=jnp.float32), da=jnp.array(da, dtype=jnp.float32)
    )
    dtau_prev = float(dtau0)

    last_phi = jnp.zeros_like(def_field)
    for i in range(int(args.steps)):
        a_mid = float(a0) + (float(i) + 0.5) * da
        state_mm, def_field, defp_field, last_phi, dtau_prev = step_moving_mesh_a(
            state_mm,
            def_field,
            defp_field,
            cosmo=cosmo_p,
            cfg=cfg,
            a_mid=a_mid,
            da=da,
            mg_params=mg_params,
            kappa=float(args.kappa),
            smooth_steps=int(args.smooth_steps),
            limiter=limiter,
            densinit=densinit,
            dtau_prev=float(dtau_prev),
            dx=float(dx_force),
        )
    # Map to physical positions x = xi + grad(def)(xi) (mesh coords).
    grad_def = qz.grad_central(def_field, float(dx_force))
    disp = qz.cic_readout_3d_multi(grad_def, state_mm.xv[:, 0:3])
    pos_mm = jnp.mod(state_mm.xv[:, 0:3] + (disp / float(dx_force)), float(ng_force))
    t3 = time.perf_counter()

    # --- Moving-mesh diagnostics on the force grid ---
    triad = triad_from_def(def_field, float(dx_force))
    _g, _ginv, sqrt_g = metric_from_triad(triad)
    lam_min_lb, lam_max_ub = qz.triad_gershgorin_bounds(triad)
    skew_ub = lam_max_ub / jnp.maximum(lam_min_lb, 1e-12)
    disp_mag = jnp.sqrt(jnp.sum(grad_def * grad_def, axis=-1))

    sqrt_g_np = np.array(sqrt_g, dtype=np.float64)
    lam_min_np = np.array(lam_min_lb, dtype=np.float64)
    lam_max_np = np.array(lam_max_ub, dtype=np.float64)
    skew_np = np.array(skew_ub, dtype=np.float64)
    disp_mag_np = np.array(disp_mag, dtype=np.float64)
    phi_np = np.array(last_phi, dtype=np.float64)

    np.savez(
        outdir / "moving_mesh_diagnostics.npz",
        sqrt_g=sqrt_g_np,
        lam_min_lb=lam_min_np,
        lam_max_ub=lam_max_np,
        skew_ub=skew_np,
        disp_mag=disp_mag_np,
        phi=phi_np,
        def_field=np.array(def_field, dtype=np.float64),
    )

    timings = {"jaxpm_pm": float(t1 - t0), "static_mesh": float(t2 - t1), "moving_mesh": float(t3 - t2), "total": float(t3 - t0)}
    (outdir / "timing.json").write_text(json.dumps(timings, indent=2, sort_keys=True), encoding="utf-8")

    # --- Paint densities onto ng_field for comparisons ---
    s_field = float(ng_field) / float(ng_force)
    pos_jax_f = jnp.mod(pos_jax * s_field, float(ng_field))
    pos_static_f = jnp.mod(pos_static * s_field, float(ng_field))
    pos_mm_f = jnp.mod(pos_mm * s_field, float(ng_field))

    rho_jax = _paint_rho(pos_jax_f, pmass, ng=int(ng_field))
    rho_static = _paint_rho(pos_static_f, pmass, ng=int(ng_field))
    rho_mm = _paint_rho(pos_mm_f, pmass, ng=int(ng_field))

    # Reference field: compare deltas (normalization cancels).
    delta_ref = _delta_from_rho(ref_rho)
    delta_jax = _delta_from_rho(np.array(rho_jax, dtype=np.float64))
    delta_static = _delta_from_rho(np.array(rho_static, dtype=np.float64))
    delta_mm = _delta_from_rho(np.array(rho_mm, dtype=np.float64))

    spec = _spectra_and_corr(delta_ref, delta_jax, delta_static, delta_mm, box=box)

    k = spec["k"]
    kf = 2.0 * np.pi / float(box)
    k_nyq = float(np.pi * float(ng_field) / float(box))
    mask_all = (k >= kf) & (k <= k_nyq) & np.isfinite(k)
    mask_hi = (k >= 0.5 * k_nyq) & (k <= k_nyq) & np.isfinite(k)

    metrics = {
        "box": float(box),
        "z0": float(args.z0),
        "zfinal": float(args.zfinal),
        "a0": float(a0),
        "afinal": float(afinal),
        "steps": int(args.steps),
        "ng_in": int(ng_in),
        "ng_force": int(ng_force),
        "ng_field": int(ng_field),
        "ref_ng_original": int(ref_ng),
        "dx_force": float(dx_force),
        "pmass_val": float(pmass_val),
        "vel_convention": str(args.vel_convention),
        "p_units": str(args.p_units),
        "kappa": float(args.kappa),
        "smooth_steps": int(args.smooth_steps),
        "mg_cycles": int(args.mg_cycles),
        "mg_v1": int(args.mg_v1),
        "mg_v2": int(args.mg_v2),
        "mg_mu": int(args.mg_mu),
        "limiter": bool(not args.no_limiter),
        "limiter_mode": str(args.limiter_mode),
        "compressmax": float(args.compressmax),
        "skewmax": float(args.skewmax),
        "xhigh": float(args.xhigh),
        "relax_steps": float(args.relax_steps),
        "hard_strength": float(args.hard_strength),
        "limiter_smooth_tmp": int(args.limiter_smooth_tmp),
        "limiter_smooth_hard": int(args.limiter_smooth_hard),
        "local_limit": bool(local_limit),
        "cosmo": {k: float(v) for k, v in cosmo_d.items()},
        "timing_sec": timings,
        # scalar comparisons vs ref
        "jaxpm_vs_ref_delta_rel_l2": _rel_l2_centered(delta_jax, delta_ref),
        "static_vs_ref_delta_rel_l2": _rel_l2_centered(delta_static, delta_ref),
        "moving_vs_ref_delta_rel_l2": _rel_l2_centered(delta_mm, delta_ref),
        "jaxpm_vs_ref_delta_corrcoef": _corrcoef(delta_jax, delta_ref),
        "static_vs_ref_delta_corrcoef": _corrcoef(delta_static, delta_ref),
        "moving_vs_ref_delta_corrcoef": _corrcoef(delta_mm, delta_ref),
        "jaxpm_vs_ref_corr_mean": _r_mean(spec["r_jaxpm"], mask=mask_all),
        "static_vs_ref_corr_mean": _r_mean(spec["r_static"], mask=mask_all),
        "moving_vs_ref_corr_mean": _r_mean(spec["r_moving"], mask=mask_all),
        "jaxpm_vs_ref_corr_hi_mean": _r_mean(spec["r_jaxpm"], mask=mask_hi),
        "static_vs_ref_corr_hi_mean": _r_mean(spec["r_static"], mask=mask_hi),
        "moving_vs_ref_corr_hi_mean": _r_mean(spec["r_moving"], mask=mask_hi),
        "jaxpm_vs_ref_pk_logrmse": _pk_logrmse(spec["pk_jaxpm"], spec["pk_ref"], mask=mask_all),
        "static_vs_ref_pk_logrmse": _pk_logrmse(spec["pk_static"], spec["pk_ref"], mask=mask_all),
        "moving_vs_ref_pk_logrmse": _pk_logrmse(spec["pk_moving"], spec["pk_ref"], mask=mask_all),
        "jaxpm_vs_ref_pk_logrmse_hi": _pk_logrmse(spec["pk_jaxpm"], spec["pk_ref"], mask=mask_hi),
        "static_vs_ref_pk_logrmse_hi": _pk_logrmse(spec["pk_static"], spec["pk_ref"], mask=mask_hi),
        "moving_vs_ref_pk_logrmse_hi": _pk_logrmse(spec["pk_moving"], spec["pk_ref"], mask=mask_hi),
        # Moving-mesh geometry diagnostics (force grid)
        "mesh_sqrt_g_min": float(np.nanmin(sqrt_g_np)),
        "mesh_sqrt_g_p1": float(np.nanpercentile(sqrt_g_np, 1.0)),
        "mesh_sqrt_g_p50": float(np.nanpercentile(sqrt_g_np, 50.0)),
        "mesh_sqrt_g_p99": float(np.nanpercentile(sqrt_g_np, 99.0)),
        "mesh_sqrt_g_max": float(np.nanmax(sqrt_g_np)),
        "mesh_sqrt_g_neg_frac": float(np.mean(sqrt_g_np < 0.0)),
        "mesh_lam_min_lb_min": float(np.nanmin(lam_min_np)),
        "mesh_lam_max_ub_max": float(np.nanmax(lam_max_np)),
        "mesh_skew_ub_p99": float(np.nanpercentile(skew_np, 99.0)),
        "mesh_skew_ub_max": float(np.nanmax(skew_np)),
        "mesh_disp_mag_p99": float(np.nanpercentile(disp_mag_np, 99.0)),
        "mesh_disp_mag_max": float(np.nanmax(disp_mag_np)),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(
        outdir / "spectra_and_corr.npz",
        k=spec["k"],
        pk_ref=spec["pk_ref"],
        pk_jaxpm=spec["pk_jaxpm"],
        pk_static=spec["pk_static"],
        pk_moving=spec["pk_moving"],
        r_jaxpm=spec["r_jaxpm"],
        r_static=spec["r_static"],
        r_moving=spec["r_moving"],
    )

    # --- Plots ---
    if not bool(args.no_plots):
        eps = 1e-6
        proj_axis = int(args.proj_axis)
        ref_proj = _project(ref_rho.astype(np.float64), proj_axis)
        jax_proj = _project(np.array(rho_jax, dtype=np.float64), proj_axis)
        static_proj = _project(np.array(rho_static, dtype=np.float64), proj_axis)
        mm_proj = _project(np.array(rho_mm, dtype=np.float64), proj_axis)

        A = np.log10(np.maximum(ref_proj/ref_proj.mean(), 0.0) + eps)
        B = np.log10(np.maximum(jax_proj/jax_proj.mean(), 0.0) + eps)
        C = np.log10(np.maximum(static_proj/static_proj.mean(), 0.0) + eps)
        D = np.log10(np.maximum(mm_proj/mm_proj.mean(), 0.0) + eps)

        _plot_grid4(
            outdir / "projected_density_grid.png",
            A=A,
            B=B,
            C=C,
            D=D,
            titles=("ref field z=2", "jaxpm PM", "static mesh", "moving mesh"),
            suptitle=f"Projected density (log10) @ z={float(args.zfinal):g} (ng_force={ng_force}, ng_field={ng_field})",
        )

        # Differences in delta projection (more interpretable).
        dref_p = _project(delta_ref, proj_axis)
        djax_p = _project(delta_jax, proj_axis)
        dsta_p = _project(delta_static, proj_axis)
        dmm_p = _project(delta_mm, proj_axis)

        _plot_grid4(
            outdir / "projected_delta_diff_grid.png",
            A=(djax_p - dref_p),
            B=(dsta_p - dref_p),
            C=(dmm_p - dref_p),
            D=dref_p,
            titles=("jaxpm - ref", "static - ref", "moving - ref", "ref delta (proj)"),
            suptitle="Projected delta differences (sum along axis)",
        )

        _plot_pk_corr(
            outdir / "pk_and_crosscorr_vs_ref.png",
            k=spec["k"],
            pk_ref=spec["pk_ref"],
            pk_jax=spec["pk_jaxpm"],
            pk_static=spec["pk_static"],
            pk_mm=spec["pk_moving"],
            r_jax=spec["r_jaxpm"],
            r_static=spec["r_static"],
            r_mm=spec["r_moving"],
            title=f"P(k) + r(k) vs ref @ z={float(args.zfinal):g}",
        )

        _plot_scatter(
            outdir / "scatter_delta_jaxpm_vs_ref.png",
            delta_ref,
            delta_jax,
            title="Voxel delta scatter: jaxpm vs ref",
            xlabel="ref delta",
            ylabel="jaxpm delta",
        )
        _plot_scatter(
            outdir / "scatter_delta_static_vs_ref.png",
            delta_ref,
            delta_static,
            title="Voxel delta scatter: static vs ref",
            xlabel="ref delta",
            ylabel="static delta",
        )
        _plot_scatter(
            outdir / "scatter_delta_moving_vs_ref.png",
            delta_ref,
            delta_mm,
            title="Voxel delta scatter: moving vs ref",
            xlabel="ref delta",
            ylabel="moving delta",
        )

        # Density PDFs (compare shape only; normalize by mean per field).
        _plot_density_pdfs(
            outdir / "density_pdfs.png",
            rho_ref=ref_rho.astype(np.float64),
            rho_jax=np.array(rho_jax, dtype=np.float64),
            rho_static=np.array(rho_static, dtype=np.float64),
            rho_moving=np.array(rho_mm, dtype=np.float64),
            title=f"Density PDFs @ z={float(args.zfinal):g} (normalized by mean; ng_field={ng_field})",
        )

        # --- Mesh layer and geometry diagnostics (moving mesh only) ---
        # Choose a z-slice: either user-specified, or through the densest force-grid cell.
        rho0 = jnp.zeros((ng_force, ng_force, ng_force), dtype=jnp.float32)
        rho_mesh_force, _sqrt_g_force = qz.pcalcrho(rho0, def_field, state_mm, dx=float(dx_force))
        rho_np_force = np.array(rho_mesh_force, dtype=np.float64)
        if args.mesh_z is None:
            z0 = int(np.unravel_index(int(np.argmax(rho_np_force)), rho_np_force.shape)[2])
        else:
            z0 = int(args.mesh_z)
        z0 = int(np.clip(z0, 0, ng_force - 1))

        # Mesh in physical coords at that layer (downsampled by stride).
        stride = max(1, int(args.mesh_stride))
        coords = (np.arange(ng_force, dtype=np.float64) + 0.5) * float(dx_force)
        uu = coords[::stride]
        Ux, Uy = np.meshgrid(uu, uu, indexing="ij")
        gx = np.array(grad_def[::stride, ::stride, z0, 0], dtype=np.float64)
        gy = np.array(grad_def[::stride, ::stride, z0, 1], dtype=np.float64)
        mesh_xy = np.stack([np.mod(Ux + gx, box), np.mod(Uy + gy, box)], axis=-1)

        # Particles (physical) for overlay
        xi = np.array(state_mm.xv[:, 0:3], dtype=np.float64)
        disp_p = np.array(qz.cic_readout_3d_multi(grad_def, state_mm.xv[:, 0:3]), dtype=np.float64)
        x_phys = np.mod(xi * float(dx_force) + disp_p, box)
        sel = (x_phys[:, 2] / float(dx_force) >= (z0 - 0.5)) & (x_phys[:, 2] / float(dx_force) < (z0 + 0.5))
        pts_xy = x_phys[sel, 0:2]
        if pts_xy.shape[0] > 200_000:
            rng = np.random.default_rng(0)
            pts_xy = pts_xy[rng.choice(pts_xy.shape[0], size=200_000, replace=False)]

        quiver = None
        if bool(args.mesh_quiver):
            quiver = np.stack([gx, gy], axis=-1)

        _plot_mesh_layer(
            outdir / "mesh_layer_xy.png",
            mesh_xy=mesh_xy,
            box=box,
            title=f"Moving mesh layer z-index {z0} (stride={stride})",
            particles_xy=pts_xy,
            uniform_stride=1,
            quiver=quiver,
        )

        # Geometry slices
        def_slice = np.array(def_field[:, :, z0], dtype=np.float64)
        sqrtg_slice = sqrt_g_np[:, :, z0]
        skew_slice = skew_np[:, :, z0]
        disp_slice = disp_mag_np[:, :, z0]

        _plot_imshow(outdir / "mesh_sqrt_g_slice.png", np.log10(np.abs(sqrtg_slice) + 1e-12), title=f"log10(|det(A)|) slice z={z0}")
        _plot_imshow(outdir / "mesh_skew_ub_slice.png", np.log10(np.maximum(skew_slice, 1e-12)), title=f"log10(skew_ub) slice z={z0}", cmap="viridis")
        _plot_imshow(outdir / "mesh_disp_mag_slice.png", disp_slice, title=f"|grad(def)| (disp mag) slice z={z0}", cmap="viridis")
        _plot_imshow(outdir / "mesh_def_slice.png", def_slice, title=f"def field slice z={z0}", cmap="viridis")

        # Histograms
        _plot_hist(outdir / "mesh_sqrt_g_hist.png", sqrt_g_np, title="det(A)=sqrt_g distribution", xlabel="sqrt_g", logx=False)
        _plot_hist(outdir / "mesh_skew_ub_hist.png", skew_np, title="skew upper-bound distribution", xlabel="skew_ub", logx=True)
        _plot_hist(outdir / "mesh_disp_mag_hist.png", disp_mag_np, title="|grad(def)| distribution", xlabel="disp magnitude", logx=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[sim-compare] wrote {outdir / 'metrics.json'}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
