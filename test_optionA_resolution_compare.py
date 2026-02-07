#!/usr/bin/env python3
"""Option A resolution test: same particles, different force mesh.

Goal
----
Compare dynamic evolution for the same band-limited GRF+LPT ICs:
  - jaxpm reference with 2x force mesh (and 2x mesh coordinates)
  - jaxpm baseline with 1x force mesh
  - this repo's moving mesh (1x)

We then compare all outputs on the 1x grid, reporting:
  - projected density (imshow)
  - power spectra P(k) of density contrast
  - cross-correlation r(k) against the jaxpm 2x reference

Option A definition used here
-----------------------------
We keep the *same particle set* and the *same domain* (mesh coordinates in [0, ng)),
but compute forces on either:
  - a 1x mesh (ng^3), or
  - a 2x mesh ((2ng)^3) with dx=1/2 in the same domain.

This is a clean "force resolution only" comparison without changing particle sampling
or adding new initial modes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp

import jax_cosmo as jc

import qzoom_nbody_flow as qz
import initial_conditions as IC

Array = jnp.ndarray


def _parse_snapshots(s: str) -> tuple[float, float]:
    toks = [float(x) for x in str(s).split(",") if x.strip()]
    if len(toks) != 2:
        raise SystemExit("--snapshots must be exactly 2 values like 0.1,1.0 for this test.")
    a0, afinal = toks
    if not np.isfinite(a0) or not np.isfinite(afinal) or a0 <= 0 or afinal <= a0:
        raise SystemExit("--snapshots must be finite with 0<a0<afinal.")
    return float(a0), float(afinal)


def _k_nyq(*, ng: int, box: float) -> float:
    return float(np.pi * float(ng) / float(box))


def _k_fundamental(*, box: float) -> float:
    return float(2.0 * np.pi / float(box))


def _bandlimit_to_lo_nyquist(field_hi: Array, ng_lo: int) -> Array:
    """Band-limit a real field on (2*ng_lo)^3 to the 1x Nyquist via FFT masking."""
    n2 = int(field_hi.shape[0])
    if field_hi.shape != (n2, n2, n2):
        raise ValueError("field_hi must be cubic")
    if n2 != 2 * int(ng_lo):
        raise ValueError("field_hi must have shape (2*ng_lo, 2*ng_lo, 2*ng_lo)")

    F = jnp.fft.fftn(field_hi)
    idx = jnp.arange(n2, dtype=jnp.int32)
    k = jnp.minimum(idx, n2 - idx)
    keep1d = k <= (int(ng_lo) // 2)
    mask = keep1d[:, None, None] & keep1d[None, :, None] & keep1d[None, None, :]
    Ff = jnp.where(mask, F, 0.0 + 0.0j)
    out = jnp.fft.ifftn(Ff).real.astype(jnp.float32)
    return out


def _forces_jaxpm_kernels_refined(
    pos_lo: Array,
    pmass: Array,
    *,
    ng: int,
    refine: int,
    omega_m: float,
) -> Array:
    """PM forces using jaxpm Fourier kernels on a refined force mesh.

    pos_lo: (Np,3) in *low-grid* mesh coordinates [0,ng).
    refine: 1 or 2. Force mesh has size (refine*ng)^3, spanning the same domain.

    Returns forces in the same coordinate units as pos_lo (i.e. physical units
    for a domain of length `ng` with low-grid dx=1).
    """
    import jaxpm.pm as jpm  # type: ignore
    from jaxpm.painting import cic_paint  # type: ignore

    refine = int(refine)
    if refine <= 0:
        raise ValueError("refine must be >= 1")

    ng_force = int(refine) * int(ng)
    dx_force = float(ng) / float(ng_force)  # = 1/refine in low-grid units

    pos_force = jnp.mod(pos_lo * float(refine), float(ng_force))

    rho = cic_paint(jnp.zeros((ng_force, ng_force, ng_force), dtype=jnp.float32), pos_force, weight=pmass)
    rho_mean = jnp.mean(rho)
    delta = rho / (rho_mean + 1e-12) - 1.0
    delta_k = jnp.fft.rfftn(delta)

    kvec = jpm.fftk((ng_force, ng_force, ng_force))
    pot_k = delta_k * jpm.invlaplace_kernel(kvec) * jpm.longrange_kernel(kvec, r_split=0.0)

    Fx = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 0) * pot_k, s=(ng_force, ng_force, ng_force)).real
    Fy = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 1) * pot_k, s=(ng_force, ng_force, ng_force)).real
    Fz = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 2) * pot_k, s=(ng_force, ng_force, ng_force)).real
    F_grid = jnp.stack([Fx, Fy, Fz], axis=-1)

    # Convert from force-grid units (dx=1) to the low-grid coordinate units.
    # See qzoom_nbody_flow.poisson_fft_uniform notes: acceleration_phys = dx * accel_grid.
    F_grid = F_grid * float(dx_force) * (1.5 * float(omega_m))

    return qz.cic_readout_3d_multi(F_grid, pos_force)


def _evolve_jaxpm_fixed_refined(
    pos0_lo: Array,
    p0_lo: Array,
    *,
    ng: int,
    refine: int,
    pmass: Array,
    cosmo: jc.core.Cosmology,
    a0: float,
    afinal: float,
    steps: int,
) -> Array:
    """Fixed-step evolution where positions remain in [0,ng) and force mesh is refined."""
    steps = int(steps)
    da = (float(afinal) - float(a0)) / max(steps, 1)
    pos = pos0_lo
    p = p0_lo
    for i in range(steps):
        a_n = float(a0) + float(i) * da
        a_mid = a_n + 0.5 * da
        E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))
        forces = _forces_jaxpm_kernels_refined(pos, pmass, ng=int(ng), refine=int(refine), omega_m=float(cosmo.Omega_m))
        kick = float(da) / (float(a_mid) ** 2 * (float(E) + 1e-12))
        drift = float(da) / (float(a_mid) ** 3 * (float(E) + 1e-12))
        p = p + forces * kick
        pos = jnp.mod(pos + p * drift, float(ng))
    return pos


def _density_on_ng_from_pos(pos_mesh: Array, pmass: Array, *, ng: int) -> Array:
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    return qz.cic_deposit_3d(rho0, jnp.mod(pos_mesh, float(ng)), pmass)


def _project(rho: Array, axis: int = 2) -> np.ndarray:
    return np.array(jnp.sum(rho, axis=int(axis)), dtype=np.float64)


def _plot_density_triptych(out: Path, *, proj_ref: np.ndarray, proj_j1: np.ndarray, proj_mm: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    eps = 1e-6
    A = np.log10(np.maximum(proj_ref, 0.0) + eps)
    B = np.log10(np.maximum(proj_j1, 0.0) + eps)
    C = np.log10(np.maximum(proj_mm, 0.0) + eps)

    vv = np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    vmin = float(np.percentile(vv, 1.0))
    vmax = float(np.percentile(vv, 99.0))

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.4))
    fig.suptitle(title)
    im0 = ax[0].imshow(A.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[0].set_title("jaxpm 2x (ref)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(B.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[1].set_title("jaxpm 1x")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(C.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[2].set_title("moving mesh 1x")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axy in ax:
        axy.set_xticks([])
        axy.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_pk_and_corr(
    out: Path,
    *,
    k: np.ndarray,
    pk_ref: np.ndarray,
    pk_j1: np.ndarray,
    pk_mm: np.ndarray,
    r_j1: np.ndarray,
    r_mm: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.2))
    fig.suptitle(title)

    ax[0].loglog(k, np.maximum(k*pk_ref, 1e-30), label="jaxpm 2x (ref)")
    ax[0].loglog(k, np.maximum(k*pk_j1, 1e-30), label="jaxpm 1x")
    ax[0].loglog(k, np.maximum(k*pk_mm, 1e-30), label="moving mesh 1x")
    ax[0].set_xlabel("k [h/Mpc] (from --box)")
    ax[0].set_ylabel("kP(k)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()

    ax[1].plot(k, r_j1, label="corr(jaxpm1x, ref)")
    ax[1].plot(k, r_mm, label="corr(moving1x, ref)")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0.85, 1.05)
    ax[1].set_xlabel("k [h/Mpc] (from --box)")
    ax[1].set_ylabel("cross-correlation r(k)")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _delta_np(rho: Array) -> np.ndarray:
    r = np.array(rho, dtype=np.float64)
    return (r / (r.mean() + 1e-30) - 1.0).astype(np.float64)


def _rel_l2_centered(a: Array | np.ndarray, b: Array | np.ndarray) -> float:
    aa = np.array(a, dtype=np.float64)
    bb = np.array(b, dtype=np.float64)
    da = aa - aa.mean()
    db = bb - bb.mean()
    num = np.linalg.norm(da - db)
    den = np.linalg.norm(da) + 1e-30
    return float(num / den)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-30
    return float(np.dot(aa, bb) / den)


def _pk_logrmse(pk: np.ndarray, pk_ref: np.ndarray, *, mask: np.ndarray) -> float:
    m = np.array(mask, dtype=bool)
    a = np.array(pk, dtype=np.float64)[m]
    b = np.array(pk_ref, dtype=np.float64)[m]
    ok = (a > 0) & (b > 0)
    if not np.any(ok):
        return float("nan")
    x = np.log10(a[ok]) - np.log10(b[ok])
    return float(np.sqrt(np.mean(x * x)))


def _r_mean(r: np.ndarray, *, mask: np.ndarray) -> float:
    m = np.array(mask, dtype=bool)
    rr = np.array(r, dtype=np.float64)[m]
    if rr.size == 0:
        return float("nan")
    return float(np.mean(rr))


def make_ics(
    *,
    ng: int,
    box: float,
    seed: int,
    omega_c: float,
    sigma8: float,
    a0: float,
) -> tuple[jc.core.Cosmology, Array, Array, Array]:
    """Create GRF+LPT ICs on the ng grid; returns (cosmo, pos0, p0, pmass)."""
    cosmo = jc.Planck15(Omega_c=float(omega_c), sigma8=float(sigma8))
    pk_fn = IC.make_linear_pk_fn(cosmo=cosmo)

    import jaxpm.pm as jpm  # type: ignore

    init_lo = jpm.linear_field((ng, ng, ng), (float(box),) * 3, pk_fn, seed=jax.random.PRNGKey(int(seed)))
    particles_lo = IC.make_lattice(npart_side=ng, ng=ng, shift=0.0)
    pmass = jnp.ones((particles_lo.shape[0],), dtype=jnp.float32)

    dx0, p0 = IC.lpt_1_fixed(cosmo, init_lo, particles_lo, a0=float(a0))
    pos0 = jnp.mod(particles_lo + dx0, float(ng))
    return cosmo, pos0, p0, pmass


def evolve_moving_mesh(
    pos0_lo: Array,
    p0_lo: Array,
    pmass: Array,
    *,
    ng: int,
    cosmo: jc.core.Cosmology,
    a0: float,
    afinal: float,
    steps: int,
    kappa: float,
    smooth_steps: int,
    mg_cycles: int,
    no_limiter: bool,
    compressmax: float,
    skewmax: float,
    limiter_mode: str,
    limiter_smooth_tmp: int | None = None,
    limiter_smooth_defp: int | None = None,
    limiter_smooth_hard: int | None = None,
) -> Array:
    """Evolve particles with this repo's moving mesh; returns particle positions (mesh coords)."""
    levels = max(1, int(np.log2(int(ng)) - 1))
    mg_params = qz.MGParams(levels=levels, v1=2, v2=2, mu=2, cycles=int(mg_cycles))

    use_source = str(limiter_mode) in ("both", "source")
    use_post = str(limiter_mode) in ("both", "post")
    limiter = qz.LimiterParams(
        enabled=not bool(no_limiter),
        compressmax=float(compressmax),
        skewmax=float(skewmax),
        use_source_term=bool(use_source),
        use_post_scale=bool(use_post),
        post_only_on_fail=True,
        smooth_tmp=int(limiter_smooth_tmp) if limiter_smooth_tmp is not None else 3,
        smooth_defp=int(limiter_smooth_defp) if limiter_smooth_defp is not None else int(smooth_steps),
        smooth_hard=int(limiter_smooth_hard) if limiter_smooth_hard is not None else 1,
    )

    state = qz.NBodyState(xv=jnp.concatenate([pos0_lo, p0_lo], axis=-1), pmass=pmass)
    def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    densinit = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    if limiter.enabled:
        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def_field, state, dx=1.0)
        densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))

    da = (float(afinal) - float(a0)) / max(int(steps), 1)
    dtold = None
    phi = jnp.zeros_like(def_field)
    for i in range(int(steps)):
        a_n = float(a0) + float(i) * da
        a_mid = a_n + 0.5 * da
        E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))
        dtau = float(da) / (float(a_mid) ** 2 * (float(E) + 1e-12))
        if dtold is None:
            dtold = float(dtau)

        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh, _ = qz.pcalcrho(rho0, def_field, state, dx=1.0)
        rhs_phi = qz.rgzerosum(rho_mesh, def_field, 1.0)
        if float(kappa) == 0.0:
            phi = qz.poisson_fft_uniform(rhs_phi, dx=1.0)
        else:
            phi = qz.multigrid(
                jnp.zeros_like(rhs_phi),
                rhs_phi,
                def_field,
                ubig=rho_mesh[None, ...],
                dx=1.0,
                nx=ng,
                ny=ng,
                nz=ng,
                nu=1,
                nred=1,
                iopt=1,
                mg_params=mg_params,
            )

        defp_field = qz.calcdefp(
            defp_field,
            tmp=jnp.zeros_like(rho_mesh),
            tmp2=jnp.zeros_like(rho_mesh),
            def_field=def_field,
            u=rho_mesh[None, ...],
            dtold1=jnp.array(float(dtold), dtype=jnp.float32),
            dtaumesh=jnp.array(float(dtau), dtype=jnp.float32),
            nfluid=1,
            dx=1.0,
            kappa=float(kappa),
            smooth_steps=int(smooth_steps),
            densinit=densinit,
            limiter=limiter,
            mg_params=mg_params,
        )

        defn = qz.zerosum(def_field + float(dtau) * defp_field)

        xvdot = qz.calcxvdot(phi, def_field, defn, float(dtau), float(1.5 * float(cosmo.Omega_m)), dx=1.0)
        xi = state.xv[:, 0:3]
        p = state.xv[:, 3:6]
        forces_part = qz.cic_readout_3d_multi(xvdot["accel"], xi)
        vgrid_part = qz.cic_readout_3d_multi(xvdot["vgrid"], xi)

        Ainv = xvdot["Ainv"]
        Ainv_flat = Ainv.reshape(Ainv.shape[0:3] + (9,))
        Ainv_p = qz.cic_readout_3d_multi(Ainv_flat, xi).reshape((-1, 3, 3))

        p_new = p + forces_part * float(dtau)
        v_phys = p_new / float(a_mid)  # dx/dτ
        xi_dot = jnp.einsum("nij,nj->ni", Ainv_p, (v_phys - vgrid_part))
        xi_new = jnp.mod(xi + float(dtau) * xi_dot, float(ng))

        state = qz.NBodyState(xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new), pmass=state.pmass)
        def_field = defn
        dtold = float(dtau)

    grad_def = qz.grad_central(def_field, 1.0)
    disp = qz.cic_readout_3d_multi(grad_def, state.xv[:, 0:3])
    pos_mm = jnp.mod(state.xv[:, 0:3] + disp, float(ng))
    return pos_mm


def compute_spectra_and_corr(
    rho_ref: Array,
    rho_j1: Array,
    rho_mm: Array,
    *,
    box: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ps_np import cross_correlation, power_spectrum

    d_ref = _delta_np(rho_ref)
    d_j1 = _delta_np(rho_j1)
    d_mm = _delta_np(rho_mm)

    kf = _k_fundamental(box=box)
    kmin = kf
    dk = kf

    k_bins, pk_ref = power_spectrum(d_ref, kmin=kmin, dk=dk, boxsize=np.array([float(box)] * 3))
    _, pk_j1 = power_spectrum(d_j1, kmin=kmin, dk=dk, boxsize=np.array([float(box)] * 3))
    _, pk_mm = power_spectrum(d_mm, kmin=kmin, dk=dk, boxsize=np.array([float(box)] * 3))

    k_corr, r_j1 = cross_correlation(d_j1, d_ref, kmin=kmin, dk=dk, boxsize=np.array([float(box)] * 3))
    _, r_mm = cross_correlation(d_mm, d_ref, kmin=kmin, dk=dk, boxsize=np.array([float(box)] * 3))

    k_plot = np.array(k_bins, dtype=np.float64)
    r_j1 = np.array(np.interp(k_plot, np.array(k_corr), np.array(r_j1)), dtype=np.float64)
    r_mm = np.array(np.interp(k_plot, np.array(k_corr), np.array(r_mm)), dtype=np.float64)
    return (
        k_plot,
        np.array(pk_ref, dtype=np.float64),
        np.array(pk_j1, dtype=np.float64),
        np.array(pk_mm, dtype=np.float64),
        r_j1,
        r_mm,
    )


def compute_scalar_metrics(
    *,
    ng: int,
    box: float,
    a0: float,
    afinal: float,
    steps: int,
    kappa: float,
    limiter_enabled: bool,
    rho_ref: Array,
    rho_j1: Array,
    rho_mm: Array,
    k: np.ndarray,
    pk_ref: np.ndarray,
    pk_j1: np.ndarray,
    pk_mm: np.ndarray,
    r_j1: np.ndarray,
    r_mm: np.ndarray,
    note: str,
) -> dict:
    k_nyq = _k_nyq(ng=int(ng), box=float(box))
    kf = _k_fundamental(box=float(box))
    mask_all = (k >= kf) & (k <= k_nyq) & np.isfinite(k)
    mask_hi = (k >= 0.5 * k_nyq) & (k <= k_nyq) & np.isfinite(k)

    d_ref = _delta_np(rho_ref)
    d_j1 = _delta_np(rho_j1)
    d_mm = _delta_np(rho_mm)

    return {
        "ng": int(ng),
        "ng_ref": int(2 * ng),
        "box": float(box),
        "a0": float(a0),
        "afinal": float(afinal),
        "steps": int(steps),
        "kappa": float(kappa),
        "limiter": bool(limiter_enabled),
        "k_fundamental": float(kf),
        "k_nyquist": float(k_nyq),
        "jaxpm1x_vs_jaxpm2x_rho_rel_l2": _rel_l2_centered(rho_j1, rho_ref),
        "moving1x_vs_jaxpm2x_rho_rel_l2": _rel_l2_centered(rho_mm, rho_ref),
        "jaxpm1x_vs_jaxpm2x_delta_rel_l2": _rel_l2_centered(d_j1, d_ref),
        "moving1x_vs_jaxpm2x_delta_rel_l2": _rel_l2_centered(d_mm, d_ref),
        "jaxpm1x_vs_jaxpm2x_delta_corrcoef": _corrcoef(d_j1, d_ref),
        "moving1x_vs_jaxpm2x_delta_corrcoef": _corrcoef(d_mm, d_ref),
        "jaxpm1x_vs_jaxpm2x_corr_mean": _r_mean(r_j1, mask=mask_all),
        "moving1x_vs_jaxpm2x_corr_mean": _r_mean(r_mm, mask=mask_all),
        "jaxpm1x_vs_jaxpm2x_corr_hi_mean": _r_mean(r_j1, mask=mask_hi),
        "moving1x_vs_jaxpm2x_corr_hi_mean": _r_mean(r_mm, mask=mask_hi),
        "jaxpm1x_vs_jaxpm2x_pk_logrmse": _pk_logrmse(pk_j1, pk_ref, mask=mask_all),
        "moving1x_vs_jaxpm2x_pk_logrmse": _pk_logrmse(pk_mm, pk_ref, mask=mask_all),
        "jaxpm1x_vs_jaxpm2x_pk_logrmse_hi": _pk_logrmse(pk_j1, pk_ref, mask=mask_hi),
        "moving1x_vs_jaxpm2x_pk_logrmse_hi": _pk_logrmse(pk_mm, pk_ref, mask=mask_hi),
        "note": str(note),
    }


def run_optionA(
    *,
    ng: int,
    box: float,
    seed: int,
    a0: float,
    afinal: float,
    steps: int,
    omega_c: float,
    sigma8: float,
    kappa: float,
    smooth_steps: int,
    mg_cycles: int,
    no_limiter: bool,
    compressmax: float,
    skewmax: float,
    limiter_mode: str,
    limiter_smooth_tmp: int | None,
    limiter_smooth_defp: int | None,
    limiter_smooth_hard: int | None,
    proj_axis: int,
    outdir: Path,
    make_plots: bool,
    save_spectra: bool,
) -> dict:
    t0 = time.perf_counter()
    cosmo, pos0_lo, p0_lo, pmass = make_ics(
        ng=int(ng),
        box=float(box),
        seed=int(seed),
        omega_c=float(omega_c),
        sigma8=float(sigma8),
        a0=float(a0),
    )
    t1 = time.perf_counter()

    pos_j2 = _evolve_jaxpm_fixed_refined(pos0_lo, p0_lo, ng=int(ng), refine=2, pmass=pmass, cosmo=cosmo, a0=float(a0), afinal=float(afinal), steps=int(steps))
    pos_j1 = _evolve_jaxpm_fixed_refined(pos0_lo, p0_lo, ng=int(ng), refine=1, pmass=pmass, cosmo=cosmo, a0=float(a0), afinal=float(afinal), steps=int(steps))
    t2 = time.perf_counter()

    pos_mm = evolve_moving_mesh(
        pos0_lo,
        p0_lo,
        pmass,
        ng=int(ng),
        cosmo=cosmo,
        a0=float(a0),
        afinal=float(afinal),
        steps=int(steps),
        kappa=float(kappa),
        smooth_steps=int(smooth_steps),
        mg_cycles=int(mg_cycles),
        no_limiter=bool(no_limiter),
        compressmax=float(compressmax),
        skewmax=float(skewmax),
        limiter_mode=str(limiter_mode),
        limiter_smooth_tmp=limiter_smooth_tmp,
        limiter_smooth_defp=limiter_smooth_defp,
        limiter_smooth_hard=limiter_smooth_hard,
    )
    t3 = time.perf_counter()

    rho_ref = _density_on_ng_from_pos(jnp.mod(pos_j2, float(ng)), pmass, ng=int(ng))
    rho_j1 = _density_on_ng_from_pos(jnp.mod(pos_j1, float(ng)), pmass, ng=int(ng))
    rho_mm = _density_on_ng_from_pos(pos_mm, pmass, ng=int(ng))

    outdir.mkdir(parents=True, exist_ok=True)

    proj_ref = _project(rho_ref, axis=int(proj_axis))
    proj_j1 = _project(rho_j1, axis=int(proj_axis))
    proj_mm = _project(rho_mm, axis=int(proj_axis))

    k_plot, pk_ref, pk_j1, pk_mm, r_j1, r_mm = compute_spectra_and_corr(rho_ref, rho_j1, rho_mm, box=float(box))

    note = "2x reference uses a 2ng force mesh on the same [0,ng) domain (positions unchanged; force mesh refined)."
    metrics = compute_scalar_metrics(
        ng=int(ng),
        box=float(box),
        a0=float(a0),
        afinal=float(afinal),
        steps=int(steps),
        kappa=float(kappa),
        limiter_enabled=bool(not no_limiter),
        rho_ref=rho_ref,
        rho_j1=rho_j1,
        rho_mm=rho_mm,
        k=k_plot,
        pk_ref=pk_ref,
        pk_j1=pk_j1,
        pk_mm=pk_mm,
        r_j1=r_j1,
        r_mm=r_mm,
        note=note,
    )
    metrics.update(
        {
            "smooth_steps": int(smooth_steps),
            "mg_cycles": int(mg_cycles),
            "limiter_mode": str(limiter_mode),
            "limiter_smooth_tmp": None if limiter_smooth_tmp is None else int(limiter_smooth_tmp),
            "limiter_smooth_defp": None if limiter_smooth_defp is None else int(limiter_smooth_defp),
            "limiter_smooth_hard": None if limiter_smooth_hard is None else int(limiter_smooth_hard),
            "compressmax": float(compressmax),
            "skewmax": float(skewmax),
            "timing_sec": {
                "ic": float(t1 - t0),
                "jaxpm": float(t2 - t1),
                "moving_mesh": float(t3 - t2),
                "total": float(time.perf_counter() - t0),
            },
        }
    )

    if save_spectra:
        np.savez(
            outdir / "spectra_and_corr.npz",
            k=k_plot,
            pk_ref=pk_ref,
            pk_j1=pk_j1,
            pk_mm=pk_mm,
            r_j1=r_j1,
            r_mm=r_mm,
        )
        np.savez(
            outdir / "projections.npz",
            proj_ref=proj_ref,
            proj_j1=proj_j1,
            proj_mm=proj_mm,
        )

    if make_plots:
        _plot_density_triptych(
            outdir / "projected_density_triptych.png",
            proj_ref=proj_ref,
            proj_j1=proj_j1,
            proj_mm=proj_mm,
            title=f"Option A (same particles): ng={ng}, steps={steps}, a={a0:g}->{afinal:g}, axis={int(proj_axis)}",
        )
        _plot_pk_and_corr(
            outdir / "pk_and_crosscorr.png",
            k=k_plot,
            pk_ref=pk_ref,
            pk_j1=pk_j1,
            pk_mm=pk_mm,
            r_j1=r_j1,
            r_mm=r_mm,
            title="Power spectra + cross-correlation (vs jaxpm 2x ref)",
        )

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ng", type=int, default=32)
    ap.add_argument("--box", type=float, default=100.0, help="Only affects IC P(k) and P(k) k-axis units.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=str, default="0.1,1.0")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--omega-c", type=float, default=0.25)
    ap.add_argument("--sigma8", type=float, default=0.8)

    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--smooth-steps", type=int, default=2)
    ap.add_argument("--mg-cycles", type=int, default=10)
    ap.add_argument("--no-limiter", action="store_true")
    ap.add_argument("--compressmax", type=float, default=80.0)
    ap.add_argument("--skewmax", type=float, default=40.0)
    ap.add_argument("--limiter-mode", type=str, default="both", choices=["both", "source", "post"])
    ap.add_argument("--limiter-smooth-tmp", type=int, default=None, help="Limiter tmp smoothing passes (only if limiter enabled).")
    ap.add_argument("--limiter-smooth-defp", type=int, default=None, help="Limiter defp smoothing passes (only if limiter enabled). Defaults to --smooth-steps.")
    ap.add_argument("--limiter-smooth-hard", type=int, default=None, help="Limiter hard-source smoothing passes (only if limiter enabled).")

    ap.add_argument("--outdir", type=str, default="./test_outputs/optionA")
    ap.add_argument("--proj-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots (still writes metrics + spectra NPZ).")
    ap.add_argument("--no-save-spectra", action="store_true", help="Skip writing spectra_and_corr.npz/projections.npz.")
    args = ap.parse_args()

    ng = int(args.ng)
    a0, afinal = _parse_snapshots(args.snapshots)
    steps = int(args.steps)

    outdir = Path(args.outdir)
    metrics = run_optionA(
        ng=int(ng),
        box=float(args.box),
        seed=int(args.seed),
        a0=float(a0),
        afinal=float(afinal),
        steps=int(steps),
        omega_c=float(args.omega_c),
        sigma8=float(args.sigma8),
        kappa=float(args.kappa),
        smooth_steps=int(args.smooth_steps),
        mg_cycles=int(args.mg_cycles),
        no_limiter=bool(args.no_limiter),
        compressmax=float(args.compressmax),
        skewmax=float(args.skewmax),
        limiter_mode=str(args.limiter_mode),
        limiter_smooth_tmp=args.limiter_smooth_tmp,
        limiter_smooth_defp=args.limiter_smooth_defp,
        limiter_smooth_hard=args.limiter_smooth_hard,
        proj_axis=int(args.proj_axis),
        outdir=outdir,
        make_plots=not bool(args.no_plots),
        save_spectra=not bool(args.no_save_spectra),
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not bool(args.no_plots):
        print(f"[optionA] wrote {outdir / 'projected_density_triptych.png'}")
        print(f"[optionA] wrote {outdir / 'pk_and_crosscorr.png'}")
    if not bool(args.no_save_spectra):
        print(f"[optionA] wrote {outdir / 'spectra_and_corr.npz'}")
        print(f"[optionA] wrote {outdir / 'projections.npz'}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
