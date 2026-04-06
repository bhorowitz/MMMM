#!/usr/bin/env python3
"""Static PM at 32^3 and 128^3 force resolution vs QZOOM moving mesh.

Force-resolution comparison starting from IDENTICAL Fortran initial conditions:
  - PM 32^3:  64^3 particles on 32^3  force mesh (same Fortran ICs, lower force res)
  - PM 128^3: 64^3 particles on 128^3 force mesh (same Fortran ICs, higher force res)
  - QZOOM MMMM: 64^3 particles on 64^3 adaptive mesh, compress=10

All three runs start from rawpart0.dat — particle positions written by the Fortran
code just after initialisation at z_init=10 (near-uniform lattice, v≈0).

All three density fields are deposited to the same 256^3 output grid.
"""

from __future__ import annotations

import os

import struct
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QZOOM_ROOT = ROOT / "QZOOM"
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm.pm as jpm

import qzoom_nbody_flow as qz

# ---------------------------------------------------------------------------
# Shared parameters (matched to COSMOPAR.DAT)
# ---------------------------------------------------------------------------
OMEGA_C    = 0.9        # Omega_m=1.0, Omega_b=0.1 -> Omega_c=0.9
SIGMA8     = 0.5
BOX        = 80.0       # Mpc/h (physical box edge)
Z_INIT     = 10.0
A0         = 1.0 / (1.0 + Z_INIT)
AFINAL     = 1.0
N_STEPS    = int(os.environ.get("QZOOM_PM_COMPARE_STEPS", "200"))
DENSITY_NG = int(os.environ.get("QZOOM_PM_COMPARE_DENSITY_NG", "256"))
NG_QZOOM   = 64         # QZOOM mesh/particle resolution (dimen.fh #define NG)
USE_ADAPTIVE_DT = os.environ.get("QZOOM_PM_COMPARE_ADAPTIVE_DT", "1") != "0"
DTAU_MIN = float(os.environ.get("QZOOM_PM_COMPARE_DTAU_MIN", "2.0e-4"))
DTAU_MAX = float(os.environ.get("QZOOM_PM_COMPARE_DTAU_MAX", "5.0e-2"))
MESH_CFL = float(os.environ.get("QZOOM_PM_COMPARE_MESH_CFL", "0.8"))
PART_CFL = float(os.environ.get("QZOOM_PM_COMPARE_PART_CFL", "0.35"))
MAX_DA_OVER_A = float(os.environ.get("QZOOM_PM_COMPARE_MAX_DA_OVER_A", "0.05"))
JAX_QZOOM_KAPPA = float(os.environ.get("QZOOM_PM_COMPARE_KAPPA", "1.0"))
JAX_QZOOM_COMPRESSMAX = float(os.environ.get("QZOOM_PM_COMPARE_COMPRESSMAX", "10.0"))
JAX_QZOOM_SKEWMAX = float(os.environ.get("QZOOM_PM_COMPARE_SKEWMAX", "10.0"))
JAX_QZOOM_MG_CYCLES = int(os.environ.get("QZOOM_PM_COMPARE_MG_CYCLES", "10"))
JAX_QZOOM_HARD_STRENGTH = float(os.environ.get("QZOOM_PM_COMPARE_HARD_STRENGTH", "1.0"))

OUTDIR     = QZOOM_ROOT / "tests" / "out" / "paper1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------
cosmo  = jc.Planck15(Omega_c=OMEGA_C, sigma8=SIGMA8)
OMEGA_M = float(cosmo.Omega_m)

# ---------------------------------------------------------------------------
# Load Fortran initial conditions from rawpart0.dat
# ---------------------------------------------------------------------------
def load_fortran_ic(ng_force: int):
    """Load initial particle positions and masses from Fortran rawpart0.dat + pmass0.dat.

    QZOOM encodes the initial density field in variable particle masses (not Zel'dovich
    displacements): particles sit on a near-uniform lattice at z=10 with v≈0, and each
    particle's mass is proportional to the local IC density.

    Returns (pos0, p0, pmass) in [0, ng_force) mesh coords at a=A0.
      - Positions scaled from QZOOM [0, NG_QZOOM) to [0, ng_force).
      - Momenta set to zero (matching QZOOM's "no Zel'dovich" initial state).
      - pmass scaled so mean(CIC rho) = 1 regardless of ng_force.
    """
    # Load positions (real*4 in Fortran)
    raw   = _read_fortran_record(QZOOM_ROOT / "rawpart0.dat", np.float32)
    npart = len(raw) // 6
    xv    = raw.reshape(6, npart, order="F")

    # QZOOM 1-indexed [1, NG_QZOOM+1) → 0-indexed [0, NG_QZOOM)
    pos_q = xv[0:3].T.astype(np.float32) - 1.0  # shape (npart, 3)

    # Scale positions to [0, ng_force)
    scale      = float(ng_force) / float(NG_QZOOM)
    pos_scaled = np.mod(pos_q * scale, float(ng_force))

    # Load variable particle masses (encode the IC density field)
    pmass_q = _read_fortran_record(QZOOM_ROOT / "pmass0.dat", np.float32)
    assert len(pmass_q) == npart, f"pmass0.dat length {len(pmass_q)} != npart {npart}"

    # Scale masses so mean(CIC rho) = 1: sum(pmass) = ng_force^3
    # With uniform masses, sum = npart * pmass_mean = npart * 1.0; need ng_force^3.
    # Scaling factor preserves relative density fluctuations.
    mass_scale = float(ng_force)**3 / float(npart)   # = (ng_force / NG_QZOOM)^3
    pmass_scaled = pmass_q * mass_scale

    # Initial momenta: zero (QZOOM "no Zel'dovich" approach: uniform lattice, v=0)
    p = np.zeros_like(pos_scaled)

    print(f"  load_fortran_ic(ng_force={ng_force}): {npart} particles "
          f"({round(npart**(1/3))}^3), mass_scale={mass_scale:.4f}", flush=True)
    print(f"  pos range [{pos_scaled.min():.3f}, {pos_scaled.max():.3f}], "
          f"pmass range [{pmass_scaled.min():.4f}, {pmass_scaled.max():.4f}]", flush=True)

    return (jnp.array(pos_scaled,  dtype=jnp.float32),
            jnp.array(p,           dtype=jnp.float32),
            jnp.array(pmass_scaled, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# Static PM leapfrog (uniform steps in scale factor a)
# ---------------------------------------------------------------------------
def run_static_pm(pos0, p0, pmass, *, ng: int, steps: int = N_STEPS):
    """Evolve from A0 to AFINAL on an ng^3 force mesh."""
    pos = pos0
    p   = p0
    da  = (AFINAL - A0) / steps

    for i in range(steps):
        a_n   = A0 + i * da
        a_mid = a_n + 0.5 * da
        a_arr = jnp.array(a_mid, dtype=jnp.float32)
        E_mid = float(jnp.sqrt(jc.background.Esqr(cosmo, a_arr)))

        # CIC deposit — with 1 particle/cell and pmass=1, mean(rho)=1 ✓
        rho0     = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh = qz.cic_deposit_3d(rho0, pos, pmass)
        delta    = rho_mesh - jnp.mean(rho_mesh)
        phi      = qz.poisson_fft_uniform(delta, dx=1.0)
        gphi     = qz.grad_central(phi, 1.0)
        forces   = -gphi * (1.5 * OMEGA_M)
        f_part   = qz.cic_readout_3d_multi(forces, pos)

        kick  = da / (a_mid**2 * E_mid)
        drift = da / (a_mid**3 * E_mid)
        p   = p + f_part * kick
        pos = jnp.mod(pos + p * drift, float(ng))

    return pos, p


# ---------------------------------------------------------------------------
# Deposit to the shared DENSITY_NG^3 grid, normalised to overdensity (mean=1)
# Positions in [0, ng_coord) are scaled to [0, DENSITY_NG)
# ---------------------------------------------------------------------------
def deposit_overdensity(pos, pmass, *, ng_coord: int) -> np.ndarray:
    """CIC deposit to DENSITY_NG^3 grid; returned array has mean=1."""
    scale   = float(DENSITY_NG) / float(ng_coord)
    pos_out = jnp.mod(pos, float(ng_coord)) * scale
    grid    = jnp.zeros((DENSITY_NG, DENSITY_NG, DENSITY_NG), dtype=jnp.float32)
    grid    = qz.cic_deposit_3d(grid, pos_out, pmass)
    arr     = np.asarray(grid, dtype=np.float64)
    return arr / (arr.mean() + 1e-30)


def qzoom_style_particles_phys(xi: jnp.ndarray, def_field: jnp.ndarray, *, box: float) -> jnp.ndarray:
    """Map mesh coords -> physical coords using the original QZOOM/xvpmap rule.

    This matches the legacy Fortran postprocessing more closely than grad(def)
    with central differences, and is therefore the right mapping for apples-to-apples
    visual comparison against the Fortran checkpoint products.
    """
    ng = int(def_field.shape[0])
    d = def_field
    disp0 = jnp.roll(d, -1, axis=0) - d
    disp1 = jnp.roll(d, -1, axis=1) - d
    disp2 = jnp.roll(d, -1, axis=2) - d
    disp0 = 0.5 * (jnp.roll(disp0, -1, axis=1) + disp0)
    disp0 = 0.5 * (jnp.roll(disp0, -1, axis=2) + disp0)
    disp1 = 0.5 * (jnp.roll(disp1, -1, axis=0) + disp1)
    disp1 = 0.5 * (jnp.roll(disp1, -1, axis=2) + disp1)
    disp2 = 0.5 * (jnp.roll(disp2, -1, axis=0) + disp2)
    disp2 = 0.5 * (jnp.roll(disp2, -1, axis=1) + disp2)
    disp = jnp.stack([disp0, disp1, disp2], axis=-1)

    delta = qz.cic_readout_3d_multi(disp, xi)
    pos = jnp.mod(xi + delta, float(ng))
    return pos * (float(box) / float(ng))


# ---------------------------------------------------------------------------
# QZOOM Fortran: load checkpoint, deformation-map particles, deposit
# ---------------------------------------------------------------------------
def _read_fortran_record(path: Path, dtype) -> np.ndarray:
    """Read the first Fortran unformatted record from a file."""
    with path.open("rb") as f:
        raw = f.read()
    n1 = struct.unpack("<I", raw[:4])[0]
    n2 = struct.unpack("<I", raw[4 + n1 : 8 + n1])[0]
    if n1 != n2:
        raise ValueError(f"{path}: Fortran record markers {n1} != {n2}")
    return np.frombuffer(raw[4:4 + n1], dtype=dtype)


def load_qzoom_overdensity() -> tuple[np.ndarray, int]:
    """Load Fortran checkpoint, map curvilinear→Cartesian, deposit to 256^3.

    QZOOM stores particle positions in curvilinear mesh coordinates [1, ng+1).
    We subtract 1 to get [0, ng), then apply the deformation-field displacement
    to convert to Cartesian (physical) coordinates — same as reproduce_paper1_figures.py.
    Returns (rho_overdensity, ng_qzoom).
    """
    def_flat = _read_fortran_record(QZOOM_ROOT / "def_chk.dat", np.float32)
    n        = round(def_flat.size ** (1.0 / 3.0))
    assert n**3 == def_flat.size
    def_grid = def_flat.reshape((n, n, n), order="F").astype(np.float64)

    raw_flat = _read_fortran_record(QZOOM_ROOT / "rawpart_chk.dat", np.float32)
    xv       = raw_flat.reshape((6, raw_flat.size // 6), order="F").astype(np.float64)

    # Deformation displacement field in each Cartesian direction
    disp = np.zeros((3, n, n, n))
    disp[0] = np.roll(def_grid, -1, axis=0) - def_grid
    disp[1] = np.roll(def_grid, -1, axis=1) - def_grid
    disp[2] = np.roll(def_grid, -1, axis=2) - def_grid
    # Transverse averaging (same as reproduce_paper1_figures.py)
    disp[0] = 0.5 * (np.roll(disp[0], -1, axis=1) + disp[0])
    disp[0] = 0.5 * (np.roll(disp[0], -1, axis=2) + disp[0])
    disp[1] = 0.5 * (np.roll(disp[1], -1, axis=0) + disp[1])
    disp[1] = 0.5 * (np.roll(disp[1], -1, axis=2) + disp[1])
    disp[2] = 0.5 * (np.roll(disp[2], -1, axis=0) + disp[2])
    disp[2] = 0.5 * (np.roll(disp[2], -1, axis=1) + disp[2])

    # CIC interpolate displacement at each particle's curvilinear position
    # xv[0:3] are in [1, ng+1) (QZOOM 1-indexed), convert to [0, ng) by -1
    x, y, z = xv[0] - 1.0, xv[1] - 1.0, xv[2] - 1.0
    ix = np.floor(x).astype(np.int64);  wx = x - ix
    iy = np.floor(y).astype(np.int64);  wy = y - iy
    iz = np.floor(z).astype(np.int64);  wz = z - iz

    ix0 = np.mod(ix, n);  ix1 = (ix0 + 1) % n
    iy0 = np.mod(iy, n);  iy1 = (iy0 + 1) % n
    iz0 = np.mod(iz, n);  iz1 = (iz0 + 1) % n

    w000=(1-wx)*(1-wy)*(1-wz); w001=(1-wx)*(1-wy)*wz
    w010=(1-wx)*wy*(1-wz);     w011=(1-wx)*wy*wz
    w100=wx*(1-wy)*(1-wz);     w101=wx*(1-wy)*wz
    w110=wx*wy*(1-wz);         w111=wx*wy*wz

    pos = np.empty((xv.shape[1], 3))
    for d in range(3):
        g = disp[d]
        delta = (g[ix0,iy0,iz0]*w000 + g[ix0,iy0,iz1]*w001 +
                 g[ix0,iy1,iz0]*w010 + g[ix0,iy1,iz1]*w011 +
                 g[ix1,iy0,iz0]*w100 + g[ix1,iy0,iz1]*w101 +
                 g[ix1,iy1,iz0]*w110 + g[ix1,iy1,iz1]*w111)
        # Physical position = curvilinear position + deformation displacement
        pos[:, d] = np.mod(x + delta if d == 0 else
                           y + delta if d == 1 else
                           z + delta, float(n))

    # Fix: use correct curvilinear coordinate for each dimension
    curv = np.stack([x, y, z], axis=1)
    pos = np.empty((xv.shape[1], 3))
    for d in range(3):
        g = disp[d]
        delta = (g[ix0,iy0,iz0]*w000 + g[ix0,iy0,iz1]*w001 +
                 g[ix0,iy1,iz0]*w010 + g[ix0,iy1,iz1]*w011 +
                 g[ix1,iy0,iz0]*w100 + g[ix1,iy0,iz1]*w101 +
                 g[ix1,iy1,iz0]*w110 + g[ix1,iy1,iz1]*w111)
        pos[:, d] = np.mod(curv[:, d] + delta, float(n))

    print(f"  QZOOM ng={n}, npart={xv.shape[1]}")
    print(f"  Particle physical coords: min={pos.min():.2f} max={pos.max():.2f}")

    pmass   = np.ones(pos.shape[0], dtype=np.float32)
    pos_jax = jnp.array(pos, dtype=jnp.float32)
    pm_jax  = jnp.array(pmass, dtype=jnp.float32)
    return deposit_overdensity(pos_jax, pm_jax, ng_coord=n), n


# ---------------------------------------------------------------------------
# Cumulative mass fraction above density (Fig. 8 style)
# ---------------------------------------------------------------------------
def cumulative_curve(rho: np.ndarray):
    vals = rho.ravel().astype(np.float64)
    w    = vals / (vals.sum() + 1e-30)
    idx  = np.argsort(vals)
    vals, w = vals[idx], w[idx]
    mass_above = 1.0 - np.cumsum(w) + w
    return np.maximum(vals, 1e-12), np.maximum(mass_above, 1e-12)


# ---------------------------------------------------------------------------
# Projected density image (z-projection, log scale)
# ---------------------------------------------------------------------------
def project_density(rho: np.ndarray, axis: int = 2) -> np.ndarray:
    return np.sum(rho, axis=axis)


# ---------------------------------------------------------------------------
# JAX-QZOOM (Python moving mesh) with Fortran initial conditions
# ---------------------------------------------------------------------------
def run_jax_qzoom() -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the JAX/Python QZOOM moving-mesh code from the same Fortran ICs.

    Matches the Fortran setup:
      - ng=64 adaptive mesh, box=80 Mpc/h
      - compressmax=10 (COSMOPAR.DAT)
      - uniform densinit (Fortran's default when densinit0.dat is absent)
      - Particles on near-uniform lattice at z=10, v≈0 (no Zel'dovich)
      - Variable particle masses encoding the IC density (from pmass0.dat)

    Returns rho overdensity on the shared DENSITY_NG^3 grid.
    """
    from cosmology import CosmologyParams
    from cosmo_apm import CosmoStepConfig, make_step_moving_mesh_a_jit_with_diag

    ng  = NG_QZOOM       # 64
    dx  = BOX / ng       # 1.25 Mpc/h per cell (physical)

    # --- Load Fortran ICs (positions in [0,64), v=0, variable masses) ------
    pos0, _p0, pmass0 = load_fortran_ic(ng_force=ng)
    # Momenta: zero (QZOOM "no Zel'dovich" approach)
    p0 = jnp.zeros_like(pos0)
    xv0   = jnp.concatenate([pos0, p0], axis=-1)
    state = qz.NBodyState(xv=xv0, pmass=pmass0)

    def_field  = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    # --- densinit: uniform = mean initial mass-per-cell (matches Fortran) --
    rho0_mesh = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0_mesh, def_field, state, dx=dx)
    mean_mass0 = float(jnp.mean(rho_mesh0 * sqrt_g0))
    densinit = jnp.full((ng, ng, ng), mean_mass0, dtype=jnp.float32)

    # --- Cosmology (EdS-like: Omega_m≈1, Omega_b=0.1) ----------------------
    cosmo_p   = CosmologyParams(cosmo=cosmo)
    force_scale = 1.5 * OMEGA_M

    # --- Limiter: compressmax=10 matching COSMOPAR.DAT ----------------------
    # The Fortran ramps compressmax from 5→10 via  max(5, min(10, 1.1*10*a)).
    # We use 10 throughout (modest over-compression at early times is harmless
    # since the mesh is nearly undeformed at z=10).
    limiter = qz.LimiterParams(
        enabled       = True,
        compressmax   = JAX_QZOOM_COMPRESSMAX,
        skewmax       = JAX_QZOOM_SKEWMAX,
        xhigh         = 2.0,
        relax_steps   = 30.0,
        smooth_tmp    = 3,
        smooth_defp   = 2,
        smooth_hard   = 1,
        use_source_term  = True,
        use_post_scale   = True,
        post_only_on_fail= True,
        hard_strength    = JAX_QZOOM_HARD_STRENGTH,
        local_limit      = True,
    )

    cfg       = CosmoStepConfig(ng=ng, force_scale=force_scale)
    mg_params = qz.MGParams(levels=int(np.log2(ng) - 1), cycles=JAX_QZOOM_MG_CYCLES, v1=2, v2=2, mu=2)

    # kappa=1.0: closest Fortran-matched default; env override allows compression sweeps.
    step_fn = make_step_moving_mesh_a_jit_with_diag(
        cosmo=cosmo_p,
        cfg=cfg,
        mg_params=mg_params,
        kappa=JAX_QZOOM_KAPPA,
        smooth_steps=2,
        limiter=limiter,
        dx=dx,
        gravity_solver="mg",
    )

    # --- Initial dtau_prev / nominal step ----------------------------------
    da_nom = (AFINAL - A0) / N_STEPS
    da = da_nom
    a_mid0 = A0 + 0.5 * da
    from cosmology import canonical_kick_drift as _ckd
    dtau0, _, _ = _ckd(cosmo=cosmo_p,
                        a_mid=jnp.array(a_mid0, dtype=jnp.float32),
                        da=jnp.array(da,    dtype=jnp.float32))
    dtau_prev = float(dtau0)

    # --- Time loop with dynamic compressmax ramp ----------------------------
    # Fortran drivers.fpp line 1030: compressmax = max(5, min(gcmpmax, 1.1*gcmpmax*a))
    gcmpmax = limiter.compressmax  # = 10.0
    diag_last = None
    print(f"  JAX-QZOOM: ng={ng}, dx={dx:.3f} Mpc/h, nominal_steps={N_STEPS}, "
          f"gcmpmax={gcmpmax}, kappa={JAX_QZOOM_KAPPA:.3g}, "
          f"mg_cycles={JAX_QZOOM_MG_CYCLES}, gravity=MG, "
          f"adaptive_dt={'on' if USE_ADAPTIVE_DT else 'off'}", flush=True)
    step_count = 0
    a_left = A0
    max_steps = max(4 * N_STEPS, N_STEPS + 16)
    while a_left < AFINAL - 1e-8 and step_count < max_steps:
        da = min(da, AFINAL - a_left)
        a_mid = a_left + 0.5 * da
        # Dynamic compressmax schedule matching Fortran
        cmpmax_t = float(np.clip(max(5.0, min(gcmpmax, 1.1 * gcmpmax * a_mid)), 5.0, gcmpmax))
        state, def_field, defp_field, _phi, dtau_j, diag = step_fn(
            state,
            def_field,
            defp_field,
            jnp.array(a_mid,     dtype=jnp.float32),
            jnp.array(da,        dtype=jnp.float32),
            jnp.array(dtau_prev, dtype=jnp.float32),
            densinit,
            jnp.array(cmpmax_t,  dtype=jnp.float32),
        )
        step_count += 1
        diag_last = diag
        dtau_this = float(diag.dtau)
        dtau_prev = dtau_this
        a_left += da

        if USE_ADAPTIVE_DT and a_left < AFINAL - 1e-8:
            dt_mesh, _max_hess = qz.mesh_dt_from_defp(defp_field, dx, safety=MESH_CFL)
            if not np.isfinite(float(dt_mesh)) or float(dt_mesh) <= 0.0:
                dt_mesh = float("inf")
            vmax_phys = float(diag.vmax)
            amax_phys = float(diag.amax)
            dt_vel = float("inf") if (not np.isfinite(vmax_phys) or vmax_phys <= 0.0) else PART_CFL * dx / vmax_phys
            dt_acc = float("inf") if (not np.isfinite(amax_phys) or amax_phys <= 0.0) else np.sqrt(2.0 * PART_CFL * dx / amax_phys)
            dtau_next = min(DTAU_MAX, float(dt_mesh), float(dt_vel), float(dt_acc))
            dtau_next = max(DTAU_MIN, dtau_next)
            dtau_da_mid = float(cosmo_p.dtau_da(jnp.array(a_mid, dtype=jnp.float32)))
            da_from_dtau = dtau_next / max(dtau_da_mid, 1e-12)
            da_growth_cap = MAX_DA_OVER_A * max(a_left, 1e-6)
            da = min(da_from_dtau, da_growth_cap, AFINAL - a_left)
            da = max(1.0e-5, da)

        if step_count % 40 == 0 or a_left >= AFINAL - 1e-8:
            print(f"  step {step_count}: a={a_mid:.3f} cmpmax={cmpmax_t:.1f} "
                  f"sqrt_g_min={float(diag.sqrt_g_new_min):.4f} "
                  f"phi_res={float(diag.phi_rel_res):.3e} "
                  f"disp_max={float(diag.disp_max):.3f} "
                  f"vmax={float(diag.vmax):.3f} "
                  f"dtau={float(diag.dtau):.3e} "
                  f"da={da:.3e} E={float(diag.E):.2f}", flush=True)

    if step_count >= max_steps and a_left < AFINAL - 1e-8:
        raise RuntimeError(
            f"Adaptive JAX-QZOOM hit max_steps={max_steps} before reaching a=1 "
            f"(stopped at a={a_left:.6f})."
        )

    # --- Convert curvilinear → physical → deposit ---------------------------
    import helper as H
    from organized_model import triad_from_def, metric_from_triad
    _, _, sqrt_g_final = metric_from_triad(triad_from_def(def_field, dx))
    sg = np.asarray(sqrt_g_final)
    print(f"  JAX sqrt_g final: min={sg.min():.4f} p1={np.percentile(sg,1):.4f} "
          f"median={np.median(sg):.4f} p99={np.percentile(sg,99):.4f} max={sg.max():.4f}", flush=True)
    print(f"  cells with sqrt_g<0.1: {(sg<0.1).sum()} / {sg.size}", flush=True)

    x_phys_central = H.particles_phys(state, def_field, dx, BOX)                 # [0, BOX)
    x_phys_qzoom   = qzoom_style_particles_phys(state.xv[:, 0:3], def_field, box=BOX)
    pos_mesh = jnp.mod(x_phys_qzoom / dx, float(ng))                             # [0, ng)
    rho_out = deposit_overdensity(pos_mesh, state.pmass, ng_coord=ng)

    pos_diff = np.asarray(((x_phys_central - x_phys_qzoom + 0.5 * BOX) % BOX) - 0.5 * BOX, dtype=np.float64)
    map_r = np.linalg.norm(pos_diff, axis=-1)
    print(f"  mapping diff central-vs-qzoom: rms={np.sqrt(np.mean(np.sum(pos_diff**2, axis=-1))):.3f} Mpc/h "
          f"p99={np.quantile(map_r, 0.99):.3f} Mpc/h", flush=True)
    summary = {
        "steps": int(step_count),
        "nominal_steps": int(N_STEPS),
        "adaptive_dt": bool(USE_ADAPTIVE_DT),
        "density_ng": int(DENSITY_NG),
        "kappa": float(JAX_QZOOM_KAPPA),
        "compressmax": float(JAX_QZOOM_COMPRESSMAX),
        "skewmax": float(JAX_QZOOM_SKEWMAX),
        "mg_cycles": int(JAX_QZOOM_MG_CYCLES),
        "hard_strength": float(JAX_QZOOM_HARD_STRENGTH),
        "sqrt_g_min": float(sg.min()),
        "sqrt_g_p1": float(np.percentile(sg, 1)),
        "sqrt_g_median": float(np.median(sg)),
        "sqrt_g_p99": float(np.percentile(sg, 99)),
        "sqrt_g_max": float(sg.max()),
        "sqrt_g_lt_0p1_count": int((sg < 0.1).sum()),
        "mapping_rms_mpc_h": float(np.sqrt(np.mean(np.sum(pos_diff**2, axis=-1)))),
        "mapping_p99_mpc_h": float(np.quantile(map_r, 0.99)),
    }
    if diag_last is not None:
        summary.update(
            {
                "phi_rel_res_last": float(diag_last.phi_rel_res),
                "disp_max_last": float(diag_last.disp_max),
                "sqrt_g_new_min_last": float(diag_last.sqrt_g_new_min),
            }
        )
    return rho_out, np.asarray(def_field), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load Fortran ICs and run PM at 32^3 force mesh -------------------
    print("=== Fortran ICs → PM 32^3 force mesh ===", flush=True)
    pos32, p32, pm32 = load_fortran_ic(ng_force=32)
    pos32_z0, _ = run_static_pm(pos32, p32, pm32, ng=32)

    # --- Same Fortran ICs → PM 128^3 force mesh ---------------------------
    print("=== Fortran ICs → PM 128^3 force mesh ===", flush=True)
    pos128, p128, pm128 = load_fortran_ic(ng_force=128)
    pos128_z0, _ = run_static_pm(pos128, p128, pm128, ng=128)

    # --- QZOOM Fortran result ---------------------------------------------
    print("=== Loading QZOOM Fortran result ===", flush=True)
    rho_qzoom, ng_qzoom = load_qzoom_overdensity()

    # --- JAX-QZOOM moving mesh -------------------------------------------
    print("=== JAX-QZOOM (Python moving mesh) ===", flush=True)
    rho_jaxqzoom, def_jaxqzoom, jax_summary = run_jax_qzoom()

    # --- Deposit PM results to 256^3 --------------------------------------
    print(f"=== Depositing PM results to {DENSITY_NG}^3 ===", flush=True)
    rho32  = deposit_overdensity(pos32_z0,  pm32,  ng_coord=32)
    rho128 = deposit_overdensity(pos128_z0, pm128, ng_coord=128)

    print(f"  PM 32^3:      max rho/rhobar = {rho32.max():.1f}", flush=True)
    print(f"  PM 128^3:     max rho/rhobar = {rho128.max():.1f}", flush=True)
    print(f"  QZOOM Fortran:max rho/rhobar = {rho_qzoom.max():.1f}", flush=True)
    print(f"  JAX-QZOOM:    max rho/rhobar = {rho_jaxqzoom.max():.1f}", flush=True)
    print(f"  All grids:  {rho32.shape} == {rho128.shape} == {rho_qzoom.shape} == {rho_jaxqzoom.shape}", flush=True)

    delta_q = rho_jaxqzoom - rho_qzoom
    compare_summary = {
        "steps": int(N_STEPS),
        "density_ng": int(DENSITY_NG),
        "fortran_qzoom_max_rho": float(rho_qzoom.max()),
        "jax_qzoom_max_rho": float(rho_jaxqzoom.max()),
        "rho_rel_l2_vs_fortran": float(np.linalg.norm(delta_q.ravel()) / (np.linalg.norm(rho_qzoom.ravel()) + 1e-12)),
        "rho_corrcoef_vs_fortran": float(np.corrcoef(rho_qzoom.ravel(), rho_jaxqzoom.ravel())[0, 1]),
    }
    compare_summary.update({f"jax_{k}": v for k, v in jax_summary.items()})
    (OUTDIR / "paper1_jaxqzoom_debug_summary.json").write_text(json.dumps(compare_summary, indent=2, sort_keys=True))

    # --- Cumulative density curves -----------------------------------------
    xq,   yq   = cumulative_curve(rho_qzoom)
    x32,  y32  = cumulative_curve(rho32)
    x128, y128 = cumulative_curve(rho128)
    xjq,  yjq  = cumulative_curve(rho_jaxqzoom)

    # -----------------------------------------------------------------------
    # Figure 1: Cumulative density (4 curves)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=180)
    ax.loglog(xq,   yq,   color="#c25b2a", lw=2.5,
              label=f"Fortran QZOOM MMMM ({ng_qzoom}³ mesh, compress×10)")
    ax.loglog(xjq,  yjq,  color="#9b59b6", lw=2.0, linestyle="-.",
              label=f"JAX-QZOOM MMMM {NG_QZOOM}³ (kappa={JAX_QZOOM_KAPPA:.3g}, compress×10, MG gravity)")
    ax.loglog(x128, y128, color="#205f8f", lw=2.0,
              label=f"Static PM 128³ force ({NG_QZOOM}³ ptcl, Fortran ICs)")
    ax.loglog(x32,  y32,  color="#4a9f50", lw=2.0, linestyle="--",
              label=f"Static PM 32³ force ({NG_QZOOM}³ ptcl, Fortran ICs)")
    ax.axvline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlim(1e1, 1e4)
    ax.set_xlabel(r"density $\rho/\bar\rho$", fontsize=12)
    ax.set_ylabel("mass fraction above density", fontsize=12)
    ax.set_title(
        f"Fig. 8–style: MMMM vs static PM — same Fortran ICs  (box={BOX:.0f} Mpc/h, "
        rf"$\Omega_m\approx{OMEGA_M:.2f}$, $\sigma_8={SIGMA8}$, $z_i={Z_INIT:.0f}$)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out_cumul = OUTDIR / "paper1_fig8_pm_resolution_compare.png"
    fig.savefig(out_cumul)
    plt.close(fig)
    print(f"Saved cumulative: {out_cumul}")

    # -----------------------------------------------------------------------
    # Figure 2: Projected density (log), all four on same grid (1×4)
    # -----------------------------------------------------------------------
    proj_qzoom     = project_density(rho_qzoom,     axis=2)
    proj_jaxqzoom  = project_density(rho_jaxqzoom,  axis=2)
    proj32         = project_density(rho32,          axis=2)
    proj128        = project_density(rho128,         axis=2)

    # Common colour scale (10th–99.9th percentile of all four)
    all_vals = np.concatenate([proj_qzoom.ravel(), proj_jaxqzoom.ravel(),
                               proj32.ravel(), proj128.ravel()])
    vmin = float(np.log10(np.percentile(all_vals[all_vals > 0], 10) + 1e-3))
    vmax = float(np.log10(np.percentile(all_vals, 99.9) + 1e-3))

    def log_proj(a):
        return np.log10(np.maximum(a, 1e-3))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150,
                             gridspec_kw={"wspace": 0.05})

    panels = [
        (proj32,        f"Static PM 32³ force\n({NG_QZOOM}³ ptcl, 256³ deposit)"),
        (proj128,       f"Static PM 128³ force\n({NG_QZOOM}³ ptcl, 256³ deposit)"),
        (proj_qzoom,    f"Fortran QZOOM MMMM {ng_qzoom}³\n({NG_QZOOM}³ ptcl, 256³ deposit)"),
        (proj_jaxqzoom, f"JAX-QZOOM MMMM {NG_QZOOM}³\n(kappa={JAX_QZOOM_KAPPA:.3g}, MG gravity, xvpmap)"),
    ]
    for ax, (proj, title) in zip(axes, panels):
        im = ax.imshow(log_proj(proj).T, origin="lower", cmap="magma",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.02,
                 label=r"$\log_{10}(\Sigma\,\rho/\bar\rho)$")
    fig.suptitle(
        f"Projected density (z-axis, 256³ deposit) — same Fortran ICs  |  "
        rf"box={BOX:.0f} Mpc/h, $\sigma_8={SIGMA8}$, $z_i={Z_INIT:.0f}$",
        fontsize=11,
    )
    out_proj = OUTDIR / "paper1_projected_density_compare.png"
    fig.savefig(out_proj, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved projection: {out_proj}")

    # -----------------------------------------------------------------------
    # Figure 3: Mesh distortion — sqrt(g) = det(A) slices (z=ng//2 plane)
    # Shows local volume compression/expansion of the adaptive mesh.
    # -----------------------------------------------------------------------
    from organized_model import triad_from_def, metric_from_triad

    # Fortran def: load def_chk.dat (already read inside load_qzoom_overdensity,
    # but we need the array itself here)
    def_flat_f = _read_fortran_record(QZOOM_ROOT / "def_chk.dat", np.float32)
    n_f = round(def_flat_f.size ** (1.0 / 3.0))
    def_fortran = def_flat_f.reshape((n_f, n_f, n_f), order="F").astype(np.float32)

    dx_q = BOX / NG_QZOOM

    def compute_sqrt_g(def_arr: np.ndarray, dx: float) -> np.ndarray:
        """Compute sqrt(g) = det(I + Hess(def)) on the full grid."""
        def_jax = jnp.array(def_arr, dtype=jnp.float32)
        triad = triad_from_def(def_jax, dx)
        _, _, sqrt_g = metric_from_triad(triad)
        return np.asarray(sqrt_g, dtype=np.float64)

    sqrt_g_fort = compute_sqrt_g(def_fortran, dx_q)
    sqrt_g_jax  = compute_sqrt_g(def_jaxqzoom, dx_q)

    # z-slice at midplane
    zsl = NG_QZOOM // 2
    sg_f_sl = sqrt_g_fort[:, :, zsl]
    sg_j_sl = sqrt_g_jax[:, :, zsl]

    # Common log scale
    all_sg = np.concatenate([sg_f_sl.ravel(), sg_j_sl.ravel()])
    all_sg = all_sg[all_sg > 0]
    vmin_sg = float(np.log10(np.percentile(all_sg, 1) + 1e-6))
    vmax_sg = float(np.log10(np.percentile(all_sg, 99) + 1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150,
                             gridspec_kw={"wspace": 0.08})
    panels_sg = [
        (sg_f_sl, f"Fortran QZOOM MMMM\n"
                  f"min={sg_f_sl.min():.3f}  max={sg_f_sl.max():.2f}"),
        (sg_j_sl, f"JAX-QZOOM MMMM\n"
                  f"min={sg_j_sl.min():.3f}  max={sg_j_sl.max():.2f}"),
    ]
    for ax, (sg, title) in zip(axes, panels_sg):
        im = ax.imshow(np.log10(np.maximum(sg, 1e-6)).T, origin="lower",
                       cmap="RdBu_r", vmin=vmin_sg, vmax=vmax_sg,
                       interpolation="bilinear")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$x$ [mesh cells]")
        ax.set_ylabel(r"$y$ [mesh cells]")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.02,
                 label=r"$\log_{10}\,\sqrt{g}$ (local volume ratio)")
    fig.suptitle(
        rf"Mesh distortion $\sqrt{{g}}$ at $z\approx0$ — midplane slice (z={zsl})  |  "
        rf"{NG_QZOOM}³ mesh, box={BOX:.0f} Mpc/h",
        fontsize=11,
    )
    out_mesh = OUTDIR / "paper1_mesh_distortion_compare.png"
    fig.savefig(out_mesh, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mesh distortion: {out_mesh}")


if __name__ == "__main__":
    main()
