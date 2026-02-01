"""QZOOM-style APM N-body stepping in JAX (hydro ignored).

Goal
----
Provide a small, readable JAX implementation that follows the same *logical flow*
as the QZOOM APM N-body code:

  - deposit particles -> mesh density
  - solve curvilinear Poisson for gravity (multigrid)
  - solve curvilinear Poisson-like equation for mesh deformation-rate potential
  - advance deformation potential
  - push particles using gravity + mesh motion (leapfrog-like)

This mirrors the QZOOM call chain described in:
  - QZOOM/relaxing.fpp: stepgh, matpp
  - QZOOM/stepghp.fpp: pcalcrho, stepxv, calcxvdot, pushparticle, limitx
  - QZOOM/limiter.fpp: calcdefp
  - QZOOM/multigrid.fpp + QZOOM/gauss1.fpp: multigrid/relax (curvilinear operator)

Notes / differences vs Fortran QZOOM
-----------------------------------
1) QZOOM uses COMMON blocks (nbody.fip) for particle storage; here particle state is
   explicit in a dataclass and passed through pure functions.
2) QZOOM stores particle positions in mesh coordinates in [1, ng+1). Here we store
   them in [0, ng) (periodic).
3) The full limiter logic in QZOOM/limiter.fpp is complex. We provide a *minimal*
   Pen (1995) eq. (10)-style update for defp (density-contrast diffusion), with
   optional smoothing/clipping hooks.
4) The stepgh split-step details in QZOOM/relaxing.fpp include hydro and some
   careful time centering. Here we keep the same conceptual ordering but only
   implement the N-body+mesh pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

# Allow importing this module from outside the diffAPM_new folder.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Local, QZOOM-aligned geometry helpers (triad A = I + Hess(def), sqrt_g = det(A)).
from organized_model import triad_from_def, metric_from_triad  # noqa: E402

# Curvilinear multigrid solver (Jacobi smoother).
import multigrid_mesh2 as mg  # noqa: E402

Array = jnp.ndarray


@dataclass(frozen=True)
class NBodyState:
    """Particle state (explicit replacement for QZOOM nbody.fip COMMON blocks).

    xv: (Np, 6) with:
      xv[:,0:3] = particle mesh coordinates ξ (periodic in [0, ng))
      xv[:,3:6] = particle Cartesian velocities v (same as QZOOM's v^i slots)
    pmass: (Np,) particle masses
    """

    xv: Array
    pmass: Array


@dataclass(frozen=True)
class MGParams:
    """Multigrid parameters (roughly corresponds to QZOOM's internal defaults)."""

    levels: int = 4
    v1: int = 2
    v2: int = 2
    mu: int = 1
    cycles: int = 10


@dataclass(frozen=True)
class APMParams:
    """APM configuration (grid + cosmology scaling knob)."""

    ng: int = 32
    box: float = 1.0
    a: float = 1.0  # cosmology scale factor multiplier used in QZOOM/stepghp.fpp

    def dx(self) -> float:
        return self.box / self.ng


@dataclass(frozen=True)
class LimiterParams:
    """Mesh limiter controls (QZOOM/Pen 1995 style; hydro limiter pieces omitted).

    Paper reference: Pen (1995) ApJS 100, 269, Section 5.1 "Grid Limiters".
    QZOOM reference: QZOOM/limiter.fpp: subroutine calcdefp1(...)

    This limiter has two parts:
      1) RHS limiting for the deformation-rate solve (clip + smooth).
      2) A "hard" compression/skewness limiter applied to the def update by
         scaling the defp step so the triad eigenvalues remain in-bounds.

    Notes:
      - QZOOM also contains a more elaborate gradient limiter involving neighbor
        comparisons along the eigenvector of maximum compression; we do not
        implement that here (N-body focus).
    """

    enabled: bool = False

    # --- Part 1: RHS limiting for defp (QZOOM calcdefp1) ---
    xhigh: float = 2.0          # clip |densinit-rho| to at most xhigh
    relax_steps: float = 30.0   # approach densinit over ~30 steps (see limiter.fpp)
    smooth_tmp: int = 3         # tmp smoothing passes (nsmooth(tmp) called 3x)
    smooth_defp: int = 2        # post-smoothing for defp

    # --- Part 2: hard limiter for def update (Pen 1995 Sec. 5.1) ---
    compressmax: float = 20.0   # maximal linear compression factor (paper suggests ~20)
    skewmax: float = 10.0       # max eigenvalue ratio (lambda_max/lambda_min)
    backtrack_iters: int = 12   # bisection steps to find maximal safe scaling in [0,1]
    local_limit: bool = True    # QZOOM-style post-limiter: limit only where triggered (preferred)

    # Prefer QZOOM-like behavior: add a local "repulsive" source term T(def) to the
    # deformation-rate solve (Pen 1995 eq. 13) so the mesh avoids singular states
    # naturally, rather than freezing the mesh after the solve.
    use_source_term: bool = True
    use_post_scale: bool = True        # safety backstop if the source-term limiter isn't enough
    post_only_on_fail: bool = True     # only apply post scaling if the candidate def update violates bounds

    hard_strength: float = 1.0  # strength of hard limiter source term T(def) (eq. 13 spirit)
    smooth_hard: int = 1        # smoothing passes for the hard limiter source


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def zerosum(x: Array) -> Array:
    """Subtract the (unweighted) mean. QZOOM equivalent: zerosum(u,nx,ny,nz)."""
    return x - jnp.mean(x)


def rgzerosum(rhs: Array, defpot: Array, dx: float) -> Array:
    """Weighted solvability projection for periodic curvilinear Poisson.

    QZOOM equivalent: rgzerosum(rhs, defpot, dx, nx, ny, nz) called in multigrid.fpp.
    In QZOOM the nullspace is controlled by subtracting a constant multiple of sqrt(g).
    Here we enforce:
        sum(rhs * sqrt_g) = 0
    """
    triad = triad_from_def(defpot, dx)
    _, _, sqrt_g = metric_from_triad(triad)
    mean_w = jnp.sum(rhs * sqrt_g) / (jnp.sum(sqrt_g) + 1e-12)
    return rhs - mean_w


def grad_central(phi: Array, d: float) -> Array:
    """Central gradient ∂_i phi at cell centers."""
    gx = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * d)
    gy = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * d)
    gz = (jnp.roll(phi, -1, axis=2) - jnp.roll(phi, 1, axis=2)) / (2 * d)
    return jnp.stack([gx, gy, gz], axis=-1)


def smooth7(x: Array, steps: int) -> Array:
    """Isotropic-ish local smoothing: (self + 6 neighbors)/7 repeated."""
    for _ in range(int(steps)):
        x = (
            x
            + jnp.roll(x, 1, 0) + jnp.roll(x, -1, 0)
            + jnp.roll(x, 1, 1) + jnp.roll(x, -1, 1)
            + jnp.roll(x, 1, 2) + jnp.roll(x, -1, 2)
        ) / 7.0
    return x


def _qzoom_clip_tmp(tmp: Array, *, xhigh: float, relax_steps: float, dtold: float) -> Array:
    """QZOOM-style clipping/scaling of densinit-rho (limiter.fpp).

    QZOOM calcdefp1 does (after setting tmp=densinit-rho):
      - clip to |tmp|<=xhigh (with xhigh ~2)
      - tmp <- tmp/(relax_steps*dtold)   (relax_steps ~30)
      - nsmooth(tmp) a few times

    In the released QZOOM source, a low-threshold xlow is computed but then
    immediately set to 0, so we omit it here.
    """
    t = jnp.where(jnp.abs(tmp) > xhigh, jnp.sign(tmp) * xhigh, tmp)
    t = t / (relax_steps * jnp.maximum(dtold, 1e-12))
    return t


def poisson_fft_uniform(rhs: Array, dx: float) -> Array:
    """Uniform-grid periodic Poisson solve using the standard 7-pt Laplacian.

    Solves:
        Laplacian(u) = rhs
    on a periodic grid, with zero-mean gauge (u_k=0 at k=0).

    QZOOM reference: QZOOM/limiter.fpp (compression limiter): FFT inversion with
    kernel `2*cos(kx)+2*cos(ky)+2*cos(kz)-6`.

    Notes
    -----
    Our discrete Laplacian is:
        (sum 6 neighbors - 6*u) / dx^2
    so in Fourier space the eigenvalue is akernel/dx^2 where:
        akernel = 2*cos(kx)+2*cos(ky)+2*cos(kz)-6  <= 0.
    """
    ng = int(rhs.shape[0])
    if rhs.shape != (ng, ng, ng):
        raise ValueError("poisson_fft_uniform expects a cubic grid.")

    # Dimensionless Fourier wave numbers (same as QZOOM; dx handled separately).
    k1 = 2.0 * jnp.pi * jnp.fft.fftfreq(ng)
    kx, ky, kz = jnp.meshgrid(k1, k1, k1, indexing="ij")
    akernel = 2.0 * jnp.cos(kx) + 2.0 * jnp.cos(ky) + 2.0 * jnp.cos(kz) - 6.0

    rhs_k = jnp.fft.fftn(rhs)
    # u_k = rhs_k / (akernel/dx^2) = rhs_k * dx^2 / akernel
    denom = jnp.where(jnp.abs(akernel) < 1e-12, jnp.ones_like(akernel), akernel)
    u_k = rhs_k * (dx * dx) / denom
    u_k = u_k.at[0, 0, 0].set(0.0 + 0.0j)
    u = jnp.fft.ifftn(u_k).real
    return zerosum(u)


def triad_gershgorin_bounds(triad: Array) -> Tuple[Array, Array]:
    """Cheap bounds on eigenvalues of a symmetric 3x3 matrix field via Gershgorin.

    Returns:
      (lam_min_lower_bound, lam_max_upper_bound) as arrays with the grid shape.
    """
    # Row 0
    a00 = triad[..., 0, 0]
    a01 = triad[..., 0, 1]
    a02 = triad[..., 0, 2]
    r0 = jnp.abs(a01) + jnp.abs(a02)
    lo0 = a00 - r0
    hi0 = a00 + r0

    # Row 1
    a11 = triad[..., 1, 1]
    a10 = triad[..., 1, 0]
    a12 = triad[..., 1, 2]
    r1 = jnp.abs(a10) + jnp.abs(a12)
    lo1 = a11 - r1
    hi1 = a11 + r1

    # Row 2
    a22 = triad[..., 2, 2]
    a20 = triad[..., 2, 0]
    a21 = triad[..., 2, 1]
    r2 = jnp.abs(a20) + jnp.abs(a21)
    lo2 = a22 - r2
    hi2 = a22 + r2

    lam_min_lb = jnp.minimum(jnp.minimum(lo0, lo1), lo2)
    lam_max_ub = jnp.maximum(jnp.maximum(hi0, hi1), hi2)
    return lam_min_lb, lam_max_ub


def limiter_scale_defp(
    def_field: Array,
    defp_field: Array,
    dt: float,
    *,
    dx: float,
    compressmax: float,
    skewmax: float,
    iters: int,
) -> Tuple[Array, Dict[str, float]]:
    """Scale defp so that def_new = def + dt*defp stays non-singular/limited.

    Paper reference: Pen (1995) Sec. 5.1 "Grid Limiters".

    We apply a conservative bisection scaling s in [0,1] so that for:
        def_new = def + (s*dt)*defp
    the triad A = I + Hess(def_new) stays bounded:
      - min eigenvalue >= 1/compressmax   (hard compression limiter)
      - max eigenvalue <= compressmax     (hard expansion limiter)
      - skewness ratio <= skewmax

    This captures the hard compression/expansion limiting described in the
    paper; it is not a byte-for-byte port of QZOOM's additional gradient
    limiter heuristics.
    """
    if compressmax <= 1.0:
        raise ValueError("compressmax must be > 1.0")
    if skewmax <= 1.0:
        raise ValueError("skewmax must be > 1.0")

    lam_min_req = 1.0 / float(compressmax)
    lam_max_req = float(compressmax)

    def ok(scale: float) -> Tuple[bool, Dict[str, float]]:
        defn = def_field + (dt * scale) * defp_field
        triad = triad_from_def(defn, dx)  # A = I + Hess(def)
        eigs = jnp.linalg.eigvalsh(triad)  # (...,3) sorted
        lam_min = jnp.min(eigs[..., 0])
        lam_max = jnp.max(eigs[..., 2])
        skew = lam_max / jnp.maximum(lam_min, 1e-12)
        # If we ever hit NaNs/Infs (e.g., due to a bad defp solve), fail closed.
        finite = jnp.isfinite(lam_min) & jnp.isfinite(lam_max) & jnp.isfinite(skew)
        ok0 = finite & (lam_min >= lam_min_req) & (lam_max <= lam_max_req) & (skew <= skewmax)
        return bool(ok0), {
            "lam_min": float(lam_min),
            "lam_max": float(lam_max),
            "skew_max": float(skew),
        }

    # Start from scale=0 (always safe unless def_field itself is bad).
    ok0, diag0 = ok(0.0)
    ok1, diag1 = ok(1.0)
    if ok1:
        return defp_field, {"limiter_scale": 1.0, **diag1}

    lo = 0.0
    hi = 1.0
    best = 0.0
    best_diag: Dict[str, float] = diag0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        okm, diagm = ok(mid)
        if okm:
            best = mid
            best_diag = diagm
            lo = mid
        else:
            hi = mid

    # Important: if best==0 and defp_field contains NaNs, defp_field*0 is still NaN.
    # Fail closed by explicitly returning zeros when the limiter cannot find a safe scale.
    if best == 0.0:
        return jnp.zeros_like(defp_field), {"limiter_scale": 0.0, **best_diag}
    return defp_field * best, {"limiter_scale": float(best), **best_diag}


def limiter_scale_defp_local(
    def_field: Array,
    defp_field: Array,
    dt: float,
    *,
    dx: float,
    compressmax: float,
    skewmax: float,
    iters: int = 12,
) -> Tuple[Array, Dict[str, float]]:
    """Local (per-cell) limiter for the mesh update (QZOOM-style behavior).

    QZOOM behavior: limiter acts where triggered (locally), rather than
    globally freezing mesh motion.

    We compute a per-cell scaling field s(x) in [0,1] such that the updated
    deformation potential:
        def_new = def + dt * (s * defp)
    yields a triad A = I + Hess(def_new) that stays well-conditioned:
      - min eigenvalue >= 1/compressmax
      - max eigenvalue <= compressmax
      - skewness ratio <= skewmax

    Implementation:
      Vectorized bisection per cell using the *actual* triad_from_def(def_new),
      not linearized bounds. This is more expensive than a single global scale,
      but avoids the "freeze entire mesh" behavior and matches the intended
      locality in Pen (1995) Sec. 5.1 / QZOOM limiter logic.
    """
    if compressmax <= 1.0:
        raise ValueError("compressmax must be > 1.0")
    if skewmax <= 1.0:
        raise ValueError("skewmax must be > 1.0")
    if dt <= 0.0:
        return defp_field, {"limiter_scale_min": 1.0, "limiter_frac_limited": 0.0}

    lam_min_req = 1.0 / float(compressmax)
    lam_max_req = float(compressmax)

    def ok(scale: Array) -> Array:
        defn = def_field + (dt * scale) * defp_field
        triad = triad_from_def(defn, dx)
        eigs = jnp.linalg.eigvalsh(triad)
        lam_min = eigs[..., 0]
        lam_max = eigs[..., 2]
        skew = lam_max / jnp.maximum(lam_min, 1e-12)
        finite = jnp.isfinite(lam_min) & jnp.isfinite(lam_max) & jnp.isfinite(skew)
        return finite & (lam_min >= lam_min_req) & (lam_max <= lam_max_req) & (skew <= float(skewmax))

    lo = jnp.zeros(def_field.shape, dtype=def_field.dtype)
    hi = jnp.ones(def_field.shape, dtype=def_field.dtype)

    # If def_field is already invalid, ok(0) will be false and lo stays 0 there.
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        okm = ok(mid)
        lo = jnp.where(okm, mid, lo)
        hi = jnp.where(okm, hi, mid)

    s = jnp.clip(lo, 0.0, 1.0)
    s = jnp.where(jnp.isfinite(s), s, 0.0)
    s = smooth7(s, steps=1)

    defp_limited = defp_field * s
    frac_limited = jnp.mean((s < 0.999).astype(jnp.float32))
    return defp_limited, {
        "limiter_scale_min": float(jnp.min(s)),
        "limiter_frac_limited": float(frac_limited),
    }


def mesh_dt_from_defp(defp_field: Array, dx: float, *, safety: float = 0.8) -> Tuple[float, float]:
    """Estimate a stable dt from the mesh deformation-rate field defp.

    Paper reference: Pen (1995) eq. (12) discussion (grid self-intersection limit).
    QZOOM reference: QZOOM/stepghp.fpp comments around the mesh deformation timestep.

    In our notation:
      - triad A = I + Hess(def)
      - A_dot ~ Hess(defp) because defp = d(def)/dt

    The paper motivates a constraint of the form:
        dt < safety / (5 * max |lambda(A_dot)|)

    To avoid an expensive eigen-decomposition on the full grid, we use a cheap
    bound based on the Gershgorin row-sum (spectral radius upper bound).
    """
    if safety <= 0.0:
        raise ValueError("safety must be > 0")

    d2 = dx * dx

    hxx = (jnp.roll(defp_field, -1, 0) - 2 * defp_field + jnp.roll(defp_field, 1, 0)) / d2
    hyy = (jnp.roll(defp_field, -1, 1) - 2 * defp_field + jnp.roll(defp_field, 1, 1)) / d2
    hzz = (jnp.roll(defp_field, -1, 2) - 2 * defp_field + jnp.roll(defp_field, 1, 2)) / d2

    hxy = (
        jnp.roll(jnp.roll(defp_field, -1, 0), -1, 1)
        - jnp.roll(jnp.roll(defp_field, -1, 0), 1, 1)
        - jnp.roll(jnp.roll(defp_field, 1, 0), -1, 1)
        + jnp.roll(jnp.roll(defp_field, 1, 0), 1, 1)
    ) / (4 * d2)
    hxz = (
        jnp.roll(jnp.roll(defp_field, -1, 0), -1, 2)
        - jnp.roll(jnp.roll(defp_field, -1, 0), 1, 2)
        - jnp.roll(jnp.roll(defp_field, 1, 0), -1, 2)
        + jnp.roll(jnp.roll(defp_field, 1, 0), 1, 2)
    ) / (4 * d2)
    hyz = (
        jnp.roll(jnp.roll(defp_field, -1, 1), -1, 2)
        - jnp.roll(jnp.roll(defp_field, -1, 1), 1, 2)
        - jnp.roll(jnp.roll(defp_field, 1, 1), -1, 2)
        + jnp.roll(jnp.roll(defp_field, 1, 1), 1, 2)
    ) / (4 * d2)

    # Gershgorin row-sum bound on max |eigenvalue|.
    s1 = jnp.abs(hxx) + jnp.abs(hxy) + jnp.abs(hxz)
    s2 = jnp.abs(hxy) + jnp.abs(hyy) + jnp.abs(hyz)
    s3 = jnp.abs(hxz) + jnp.abs(hyz) + jnp.abs(hzz)
    bound = jnp.maximum(jnp.maximum(s1, s2), s3)
    max_abs = jnp.max(bound)

    max_abs_f = float(max_abs)
    if not np.isfinite(max_abs_f) or max_abs_f <= 0.0:
        return float("inf"), max_abs_f
    dt = float(safety) / (5.0 * max_abs_f)
    return dt, max_abs_f


# -----------------------------------------------------------------------------
# Particle-mesh (CIC) in mesh coordinates (periodic)
# -----------------------------------------------------------------------------

def cic_deposit_3d(mesh: Array, pos: Array, mass: Array) -> Array:
    """Periodic CIC deposit to mesh (positions in mesh coords [0, ng))."""
    ng = mesh.shape[0]
    i0 = jnp.floor(pos).astype(jnp.int32)
    d = pos - i0
    i1 = i0 + 1

    i0 = jnp.mod(i0, ng)
    i1 = jnp.mod(i1, ng)

    wx0 = 1.0 - d[:, 0]
    wy0 = 1.0 - d[:, 1]
    wz0 = 1.0 - d[:, 2]
    wx1 = d[:, 0]
    wy1 = d[:, 1]
    wz1 = d[:, 2]

    def add(ix, iy, iz, w, out):
        return out.at[ix, iy, iz].add(mass * w)

    out = mesh
    out = add(i0[:, 0], i0[:, 1], i0[:, 2], wx0 * wy0 * wz0, out)
    out = add(i0[:, 0], i0[:, 1], i1[:, 2], wx0 * wy0 * wz1, out)
    out = add(i0[:, 0], i1[:, 1], i0[:, 2], wx0 * wy1 * wz0, out)
    out = add(i0[:, 0], i1[:, 1], i1[:, 2], wx0 * wy1 * wz1, out)
    out = add(i1[:, 0], i0[:, 1], i0[:, 2], wx1 * wy0 * wz0, out)
    out = add(i1[:, 0], i0[:, 1], i1[:, 2], wx1 * wy0 * wz1, out)
    out = add(i1[:, 0], i1[:, 1], i0[:, 2], wx1 * wy1 * wz0, out)
    out = add(i1[:, 0], i1[:, 1], i1[:, 2], wx1 * wy1 * wz1, out)
    return out


def cic_readout_3d(field: Array, pos: Array) -> Array:
    """Periodic CIC readout from mesh (positions in mesh coords [0, ng))."""
    ng = field.shape[0]
    i0 = jnp.floor(pos).astype(jnp.int32)
    d = pos - i0
    i1 = i0 + 1

    i0 = jnp.mod(i0, ng)
    i1 = jnp.mod(i1, ng)

    wx0 = 1.0 - d[:, 0]
    wy0 = 1.0 - d[:, 1]
    wz0 = 1.0 - d[:, 2]
    wx1 = d[:, 0]
    wy1 = d[:, 1]
    wz1 = d[:, 2]

    def gather(ix, iy, iz):
        return field[ix, iy, iz]

    v000 = gather(i0[:, 0], i0[:, 1], i0[:, 2])
    v001 = gather(i0[:, 0], i0[:, 1], i1[:, 2])
    v010 = gather(i0[:, 0], i1[:, 1], i0[:, 2])
    v011 = gather(i0[:, 0], i1[:, 1], i1[:, 2])
    v100 = gather(i1[:, 0], i0[:, 1], i0[:, 2])
    v101 = gather(i1[:, 0], i0[:, 1], i1[:, 2])
    v110 = gather(i1[:, 0], i1[:, 1], i0[:, 2])
    v111 = gather(i1[:, 0], i1[:, 1], i1[:, 2])

    w000 = wx0 * wy0 * wz0
    w001 = wx0 * wy0 * wz1
    w010 = wx0 * wy1 * wz0
    w011 = wx0 * wy1 * wz1
    w100 = wx1 * wy0 * wz0
    w101 = wx1 * wy0 * wz1
    w110 = wx1 * wy1 * wz0
    w111 = wx1 * wy1 * wz1

    return (
        v000 * w000
        + v001 * w001
        + v010 * w010
        + v011 * w011
        + v100 * w100
        + v101 * w101
        + v110 * w110
        + v111 * w111
    )


# -----------------------------------------------------------------------------
# QZOOM-named routines (Python equivalents)
# -----------------------------------------------------------------------------

def pcalcrho(rho: Array, def_field: Array, state: NBodyState, *, dx: float) -> Tuple[Array, Array]:
    """Deposit particle mass to the mesh (curvilinear mass per physical volume).

    QZOOM reference: QZOOM/stepghp.fpp: subroutine pcalcrho(rho,def)

    In QZOOM, pcalcrho reads particles from COMMON blocks and writes `rho` in place.
    Here we take an explicit `state` and return:
      (rho_mesh, sqrt_g)

    Convention here:
      - particles live in mesh coords ξ in [0, ng)
      - we deposit mass to a mesh mass field m(ξ) via CIC
      - we convert to a density per physical volume using sqrt_g:
            rho_mesh = m / sqrt_g
        so that sum(rho_mesh * sqrt_g) = total mass.
    """
    triad = triad_from_def(def_field, dx)
    _, _, sqrt_g = metric_from_triad(triad)
    mass_mesh = cic_deposit_3d(rho, state.xv[:, 0:3], state.pmass)
    rho_mesh = mass_mesh / (sqrt_g + 1e-12)
    return rho_mesh, sqrt_g


def multigrid(
    u: Array,
    rhs: Array,
    defpot: Array,
    ubig: Array,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nred: int,
    iopt: int,
    *,
    mg_params: MGParams,
) -> Array:
    """Solve a curvilinear Poisson system with periodic BCs.

    QZOOM reference: QZOOM/multigrid.fpp: subroutine multigrid(u,rhs,defpot,ubig,dx,nx,ny,nz,nu,nred,iopt)
    QZOOM reference: QZOOM/gauss1.fpp: relax/cbajraw (metric construction + smoothing)

    This is a thin wrapper around diffAPM_new/multigrid_mesh2.py that keeps the
    QZOOM-like signature. Only cubic grids are supported here.

    Notes:
      - We enforce the periodic solvability condition by subtracting the
        sqrt(g)-weighted mean from rhs (rgzerosum).
      - If iopt includes the "+32" convention (use current u as initial guess),
        just pass u through. Otherwise we start from zeros.
    """
    if (nx, ny, nz) != u.shape:
        raise ValueError(f"u has shape {u.shape} but expected {(nx, ny, nz)}")
    if nred != 1:
        raise ValueError("Only nred=1 supported (QZOOM restriction).")
    if nx != ny or ny != nz:
        raise ValueError("Only perfect cubes supported (QZOOM restriction).")

    rhs0 = rgzerosum(rhs, defpot, dx)

    use_guess = (iopt & 32) != 0
    U0 = u if use_guess else jnp.zeros_like(u)
    U0 = zerosum(U0)

    # iopt mod 8 selects operator in QZOOM; we always use the Laplace–Beltrami operator.
    out = mg.poisson_multigrid(
        F=rhs0,
        U=U0,
        l=int(mg_params.levels),
        v1=int(mg_params.v1),
        v2=int(mg_params.v2),
        mu=int(mg_params.mu),
        iter_cycle=int(mg_params.cycles),
        def_field=defpot,
        h=dx,
    )
    return zerosum(out)


def calcdefp(
    defp: Array,
    tmp: Array,
    tmp2: Array,
    def_field: Array,
    u: Array,
    dtold1: float,
    dtaumesh: float,
    nfluid: int,
    *,
    dx: float,
    kappa: float = 0.1,
    smooth_steps: int = 2,
    densinit: Optional[Array] = None,
    limiter: Optional[LimiterParams] = None,
    mg_params: MGParams,
) -> Array:
    """Compute deformation-rate potential defp from density contrast (minimal).

    QZOOM reference: QZOOM/limiter.fpp: subroutine calcdefp(defp,tmp,tmp2,def,u,dtold1,dtaumesh,nfluid)

    QZOOM does:
      - builds a limiter RHS based on desired constant mass-per-cell
      - applies smoothing/limiters
      - solves an elliptic equation for defp via multigrid(defp, tmp, def, u, ..., iopt=2)
      - smooths defp again and optionally applies additional limiter logic

    Here (N-body only):
      - we treat u[0] as the mesh *density* field rho_mesh (as returned by pcalcrho)
      - if limiter.enabled:
          mimic QZOOM calcdefp1: build tmp = densinit - u(1) in *normalized*
          units, clip it, divide by ~30*dtold, smooth, then solve for defp.
          Here u(1) corresponds to rho*sqrt(g), i.e. the per-cell mass in mesh
          coordinates (Pen 1995 Sec. 5.1).
      - otherwise:
          solve a minimal Pen (1995) eq. (10)-style update driven by density
          contrast, using the sign convention aligned with QZOOM.
    """
    del tmp2, dtaumesh, nfluid  # ignored in this minimal port

    if limiter is None:
        limiter = LimiterParams(enabled=False)

    triad = triad_from_def(def_field, dx)
    _, _, sqrt_g = metric_from_triad(triad)

    rho = u[0]
    if limiter.enabled:
        # QZOOM equalizes u(1)=rho*sqrt(g), i.e. mass per mesh cell (Pen 1995 Sec. 5.1).
        # Our pcalcrho returns rho_mesh = mass_mesh/sqrt_g, so:
        mass_mesh = rho * sqrt_g
        mean_mass = jnp.mean(mass_mesh)

        # Use a uniform densinit by default (QZOOM uses densinit=1 if unset).
        if densinit is None:
            densinit = jnp.full_like(mass_mesh, mean_mass)

        # Normalize by the mean so the clip scale (xhigh ~ 2) is meaningful.
        densinit_n = densinit / (mean_mass + 1e-12)
        mass_n = mass_mesh / (mean_mass + 1e-12)
        tmp_n = densinit_n - mass_n
        tmp_n = _qzoom_clip_tmp(tmp_n, xhigh=float(limiter.xhigh), relax_steps=float(limiter.relax_steps), dtold=float(dtold1))
        tmp_n = smooth7(tmp_n, steps=int(limiter.smooth_tmp))

        rhs = float(kappa) * tmp_n
    else:
        mean_rho = jnp.sum(rho * sqrt_g) / (jnp.sum(sqrt_g) + 1e-12)
        # QZOOM/limiter.fpp uses tmp = densinit - u(1); overdensities should drive
        # a negative RHS and contract the mesh for x = xi + grad(def).
        Ap = mean_rho - rho
        Ap = smooth7(Ap, steps=int(smooth_steps))
        rhs = float(kappa) * Ap

    out = multigrid(
        defp,
        rhs,
        def_field,
        ubig=u,
        dx=dx,
        nx=rho.shape[0],
        ny=rho.shape[1],
        nz=rho.shape[2],
        nu=u.shape[0],
        nred=1,
        iopt=2 | 32,  # iopt=2 (defp flow solver), +32 use defp as initial guess
        mg_params=mg_params,
    )

    # QZOOM smooths the raw defp before applying compression limiter corrections.
    out = smooth7(out, steps=int(limiter.smooth_defp if limiter.enabled else smooth_steps))

    # QZOOM-style "repulsive" limiter: modify defp by adding a potential correction
    # that counteracts imminent cell collapse, rather than globally scaling defp.
    #
    # QZOOM reference: QZOOM/limiter.fpp (after multigrid + nsmooth(defp)):
    #   - form candidate triad from def + dtold*defp
    #   - compute tlim = 3*(1/cmpmax - 1/compression)/dtold = 3*(1/cmpmax - lam_min)/dtold
    #   - solve Laplacian(defplim) = tlim via FFT and add: defp += defplim
    if (
        limiter.enabled
        and bool(limiter.use_source_term)
        and float(limiter.hard_strength) != 0.0
        and dtold1 > 0.0
    ):
        # Candidate update uses dtold (QZOOM uses dtold1 here, not dtaumesh).
        def_cand = def_field + float(dtold1) * out
        triad_cand = triad_from_def(def_cand, dx)
        lam_min_lb, _ = triad_gershgorin_bounds(triad_cand)

        # Local compression cap: cmpmax * densinit^{-1/3} (QZOOM limiter.fpp).
        # densinit_n is normalized so densinit_n~1 for uniform initial mass-per-cell.
        cmpmax_local = float(limiter.compressmax) * jnp.power(jnp.maximum(densinit_n, 1e-12), -1.0 / 3.0)
        lam_min_req = 1.0 / jnp.maximum(cmpmax_local, 1.0 + 1e-12)

        # tlim > 0 when lam_min would fall below the minimum allowed value.
        tlim = 3.0 * (lam_min_req - lam_min_lb) / jnp.maximum(float(dtold1), 1e-12)
        tlim = jnp.maximum(tlim, 0.0)
        tlim = smooth7(tlim, steps=int(limiter.smooth_hard))

        # Solve Laplacian(corr) = tlim on the *uniform* mesh stencil (QZOOM FFT kernel).
        corr = poisson_fft_uniform(tlim, dx=dx)
        out = out + float(limiter.hard_strength) * corr
        out = smooth7(out, steps=1)

    return zerosum(out)


def matpp(a: Array, b: Array, s: float) -> Array:
    """a <- a + s*b

    QZOOM reference: QZOOM/relaxing.fpp: subroutine matpp(a,b,s)
    """
    return a + b * s


def calcxvdot(
    phi: Array,
    def_field: Array,
    defn_field: Array,
    dt: float,
    a: float,
    *,
    dx: float,
) -> Dict[str, Array]:
    """Compute grid fields needed to push particles.

    QZOOM reference: QZOOM/stepghp.fpp: subroutine calcxvdot(xvdot,phi,def,defn,dt,a,k,iflip)

    QZOOM does this on z-planes and stores many coefficients (xvdot(1..12))
    for CIC interpolation during pushparticle.

    Here we compute full 3D fields:
      - Ainv(ξ): inverse triad at cell centers (A = I + Hess(def))
      - accel_v(ξ): dv/dt at cell centers in Cartesian components
      - vgrid(ξ): mesh physical velocity at cell centers, approximated as
            v_grid = ∂_t (∇def) = ∇((defn-def)/dt)
        (this is the "mesh motion" term in Pen 1995 eq. 2 / QZOOM stepxv comments)
    """
    triad = triad_from_def(def_field, dx)
    Ainv = jnp.linalg.inv(triad)

    gphi = grad_central(phi, dx)  # ∂_α phi in mesh coords
    # QZOOM uses minus central differences, so acceleration is -Ainv·grad(phi) * a
    accel = -jnp.einsum("...ij,...j->...i", Ainv, gphi) * a

    ddef_dt = (defn_field - def_field) / max(dt, 1e-12)
    vgrid = grad_central(ddef_dt, dx)  # physical mesh velocity (approx)

    return {"Ainv": Ainv, "accel": accel, "vgrid": vgrid}


def pushparticle(
    state: NBodyState,
    xvdot_fields: Dict[str, Array],
    odt: float,
    tdt: float,
    *,
    dx: float,
) -> NBodyState:
    """Advance particles by one step using CIC interpolation of grid fields.

    QZOOM reference: QZOOM/stepghp.fpp: subroutine pushparticle(xvdot,odt,tdt,k,ik0,ik1)

    This is a vectorized (all-particles-at-once) version:
      - interpolate accel and Ainv and vgrid at particle positions ξ
      - update velocities (leapfrog kick)
      - update mesh coordinates via ξ_dot = Ainv · (v - vgrid)
    """
    xi = state.xv[:, 0:3]
    v = state.xv[:, 3:6]

    # Readout grid fields at particle positions in mesh coordinates.
    ax = cic_readout_3d(xvdot_fields["accel"][..., 0], xi)
    ay = cic_readout_3d(xvdot_fields["accel"][..., 1], xi)
    az = cic_readout_3d(xvdot_fields["accel"][..., 2], xi)
    a_part = jnp.stack([ax, ay, az], axis=-1)

    # Read Ainv as 9 scalar fields (CIC each), then reshape.
    Ainv = xvdot_fields["Ainv"]
    Ainv_ij = []
    for i in range(3):
        for j in range(3):
            Ainv_ij.append(cic_readout_3d(Ainv[..., i, j], xi))
    Ainv_p = jnp.stack(Ainv_ij, axis=-1).reshape((-1, 3, 3))

    vg = xvdot_fields["vgrid"]
    vg_part = jnp.stack(
        [cic_readout_3d(vg[..., 0], xi), cic_readout_3d(vg[..., 1], xi), cic_readout_3d(vg[..., 2], xi)],
        axis=-1,
    )

    # Velocity update: v_{n+1/2} = v_{n-1/2} + a_n * dt
    v_new = v + a_part * tdt

    # Mesh-coordinate drift: xi_dot = Ainv · (v - v_grid)
    xi_dot = jnp.einsum("nij,nj->ni", Ainv_p, (v_new - vg_part))
    xi_new = xi + (tdt / dx) * xi_dot  # convert physical to mesh-coordinate units

    xv_new = jnp.concatenate([xi_new, v_new], axis=-1)
    return NBodyState(xv=xv_new, pmass=state.pmass)


def limitx(state: NBodyState, ng: int) -> NBodyState:
    """Enforce periodic bounds on particle mesh coordinates.

    QZOOM reference: QZOOM/stepghp.fpp: subroutine limitx
    """
    xi = jnp.mod(state.xv[:, 0:3], float(ng))
    return NBodyState(xv=state.xv.at[:, 0:3].set(xi), pmass=state.pmass)


def stepxv(
    phi: Array,
    def_field: Array,
    defn_field: Array,
    tdt: float,
    odt: float,
    a: float,
    state: NBodyState,
    *,
    dx: float,
) -> NBodyState:
    """Advance particle positions/velocities by one time step.

    QZOOM reference: QZOOM/stepghp.fpp: subroutine stepxv(phi,def,defn,tdt,odt,a)
    """
    xvdot_fields = calcxvdot(phi, def_field, defn_field, tdt, a, dx=dx)
    out = pushparticle(state, xvdot_fields, odt, tdt, dx=dx)
    return limitx(out, ng=phi.shape[0])


def step_nbody_apm(
    state: NBodyState,
    def_field: Array,
    defp_field: Array,
    *,
    params: APMParams,
    mg_params: MGParams,
    kappa: float = 0.1,
    smooth_steps: int = 2,
    limiter: Optional[LimiterParams] = None,
    densinit: Optional[Array] = None,
    dt: float,
    dtold: float,
) -> Tuple[NBodyState, Array, Array, Array, Dict[str, Any]]:
    """One APM N-body step: rho->phi, rho->defp, def update, particle push.

    This is the minimal Python equivalent of the QZOOM per-step logic in:
      - QZOOM/relaxing.fpp: stepgh (N-body + mesh parts)
      - QZOOM/stepghp.fpp: pcalcrho, stepxv
      - QZOOM/limiter.fpp: calcdefp

    Returns:
      (state_new, def_new, defp_new, phi_new, diagnostics)
    """
    dx = params.dx()
    ng = params.ng
    if limiter is None:
        limiter = LimiterParams(enabled=False)

    # 1) Particle -> rho (mesh)
    rho0 = jnp.zeros((ng, ng, ng), dtype=def_field.dtype)
    rho_mesh, sqrt_g = pcalcrho(rho0, def_field, state, dx=dx)

    # 2) Gravity solve: multigrid(phi, rhs, def, ...)
    rhs_phi = rgzerosum(rho_mesh, def_field, dx)  # mean-subtracted (weighted)
    phi0 = jnp.zeros_like(rhs_phi)
    phi = multigrid(
        phi0,
        rhs_phi,
        def_field,
        ubig=rho_mesh[None, ...],
        dx=dx,
        nx=ng,
        ny=ng,
        nz=ng,
        nu=1,
        nred=1,
        iopt=1,  # gravity
        mg_params=mg_params,
    )

    # 3) Mesh deformation-rate solve (defp) using density contrast.
    #    We follow Pen (1995) eq. (10) spirit (QZOOM/limiter.fpp), but minimal.
    u = rho_mesh[None, ...]
    defp = calcdefp(
        defp_field,
        tmp=jnp.zeros_like(rho_mesh),
        tmp2=jnp.zeros_like(rho_mesh),
        def_field=def_field,
        u=u,
        dtold1=dtold,
        dtaumesh=dt,
        nfluid=1,
        dx=dx,
        kappa=kappa,
        smooth_steps=smooth_steps,
        densinit=densinit,
        limiter=limiter,
        mg_params=mg_params,
    )

    # 4) Update deformation potential (matpp)
    limiter_diag: Dict[str, float] = {}
    post_scale_applied = False
    if limiter.enabled and bool(limiter.use_post_scale):
        need_scale = True

        # If requested, only apply the post-solve scaling if the candidate update
        # looks unsafe. We use a conservative Gershgorin bound (cheap); if it
        # suggests possible violation, we fall back to the expensive local limiter.
        if bool(limiter.post_only_on_fail):
            defn_cand = matpp(def_field, defp, dt)
            triad_cand = triad_from_def(defn_cand, dx)
            lam_min_lb, lam_max_ub = triad_gershgorin_bounds(triad_cand)
            lam_min_lb_f = float(jnp.min(lam_min_lb))
            lam_max_ub_f = float(jnp.max(lam_max_ub))
            skew_ub_f = float(lam_max_ub_f / max(lam_min_lb_f, 1e-12))

            lam_min_req = 1.0 / float(limiter.compressmax)
            lam_max_req = float(limiter.compressmax)
            finite = np.isfinite(lam_min_lb_f) and np.isfinite(lam_max_ub_f) and np.isfinite(skew_ub_f)
            need_scale = not (
                finite
                and (lam_min_lb_f >= lam_min_req)
                and (lam_max_ub_f <= lam_max_req)
                and (skew_ub_f <= float(limiter.skewmax))
            )
            limiter_diag.update(
                {
                    "cand_lam_min_lb": lam_min_lb_f,
                    "cand_lam_max_ub": lam_max_ub_f,
                    "cand_skew_ub": skew_ub_f,
                }
            )

        if need_scale:
            post_scale_applied = True
            if limiter.local_limit:
                defp, limiter_diag2 = limiter_scale_defp_local(
                    def_field,
                    defp,
                    float(dt),
                    dx=dx,
                    compressmax=float(limiter.compressmax),
                    skewmax=float(limiter.skewmax),
                    iters=int(limiter.backtrack_iters),
                )
            else:
                # Fallback: global scaling (can freeze the whole mesh if it fails).
                defp, limiter_diag2 = limiter_scale_defp(
                    def_field,
                    defp,
                    float(dt),
                    dx=dx,
                    compressmax=float(limiter.compressmax),
                    skewmax=float(limiter.skewmax),
                    iters=int(limiter.backtrack_iters),
                )
            limiter_diag.update(limiter_diag2)
        else:
            # Make it explicit in diagnostics that we skipped the post-scaling.
            limiter_diag.update({"limiter_scale_min": 1.0, "limiter_frac_limited": 0.0})

    defn = matpp(def_field, defp, dt)
    defn = zerosum(defn)

    # 5) Push particles (stepxv)
    state_new = stepxv(phi, def_field, defn, dt, dtold, params.a, state, dx=dx)

    # Diagnostics: characterize mesh motion using the *updated* deformation.
    triad_new = triad_from_def(defn, dx)
    _, _, sqrt_g_new = metric_from_triad(triad_new)
    disp = grad_central(defn, dx)
    disp_mag = jnp.linalg.norm(disp, axis=-1)

    # Particle/force-based timestep diagnostics (QZOOM has several dt criteria; this helps avoid blow-ups
    # when dt_max is large but mesh_dt is permissive).
    triad_old = triad_from_def(def_field, dx)
    Ainv_old = jnp.linalg.inv(triad_old)
    gphi = grad_central(phi, dx)
    accel_grid = -jnp.einsum("...ij,...j->...i", Ainv_old, gphi) * params.a
    amax = jnp.max(jnp.linalg.norm(accel_grid, axis=-1))
    vmax = jnp.max(jnp.linalg.norm(state_new.xv[:, 3:6], axis=-1))

    # Cheap energy diagnostics (grid-based potential energy).
    # This is not a symplectic energy monitor, but is useful for comparing runs.
    K = 0.5 * jnp.sum(state_new.pmass * jnp.sum(state_new.xv[:, 3:6] ** 2, axis=-1))
    U = 0.5 * (dx ** 3) * jnp.sum(rhs_phi * phi * sqrt_g)
    E = K + U

    diagnostics = {
        "rho_mean_phys": float(jnp.sum(rho_mesh * sqrt_g) / (jnp.sum(sqrt_g) + 1e-12)),
        "sqrt_g_min": float(jnp.min(sqrt_g)),
        "sqrt_g_max": float(jnp.max(sqrt_g)),
        "sqrt_g_new_min": float(jnp.min(sqrt_g_new)),
        "sqrt_g_new_max": float(jnp.max(sqrt_g_new)),
        "disp_max": float(jnp.max(disp_mag)),
        "disp_rms": float(jnp.sqrt(jnp.mean(disp_mag * disp_mag))),
        "phi_rms": float(jnp.sqrt(jnp.mean(phi * phi))),
        "vmax": float(vmax),
        "amax": float(amax),
        "K": float(K),
        "U": float(U),
        "E": float(E),
        "post_scale_applied": float(post_scale_applied),
    }
    diagnostics.update(limiter_diag)
    return state_new, defn, defp, phi, diagnostics
