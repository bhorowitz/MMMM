"""Shared helpers for diffAPM_new test problems and demos.

This file is intentionally *not* a pytest module. It's a small toolbox for
generating ICs, running moving-mesh vs static-mesh comparisons, and producing
consistent plots/animations.

Key model code lives in:
  - diffAPM_new/qzoom_nbody_flow.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import time

import jax
import jax.numpy as jnp

# Allow running as scripts without installing diffAPM_new as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import qzoom_nbody_flow as qz  # noqa: E402


Array = jnp.ndarray


@dataclass(frozen=True)
class RunConfig:
    ng: int = 64
    box: float = 1.0
    ntotal: int = 200

    # dt scheme
    dt_max: float = 2e-3
    dt_min: float = 1e-5
    cfl_mesh: float = 0.8
    cfl_part: float = 0.25  # particle CFL: max displacement per step in units of dx

    # solvers / model knobs
    mg_cycles: int = 4
    kappa: float = 20.0
    smooth_steps: int = 2
    a: float = 1.0

    # limiter
    limiter: bool = True
    limiter_mode: str = "both"  # {"both","source","post"}
    compressmax: float = 80.0
    skewmax: float = 40.0
    xhigh: float = 2.0
    relax_steps: float = 30.0
    hard_strength: float = 1.0

    # plotting
    mesh_stride: int = 1
    mesh_exaggerate: float = 5.0
    mesh_z: Optional[int] = None  # fixed z index for all plots
    max_plot: int = 40000

    # animation
    animate: bool = True
    anim_every: int = 10
    anim_fps: int = 8

    def dx(self) -> float:
        return float(self.box) / int(self.ng)


@dataclass
class RunResult:
    name: str
    mode: str  # "moving" or "static"

    state: qz.NBodyState
    def_field: Array
    defp_field: Array
    phi: Array

    dt_hist: np.ndarray
    sqrt_g_min_hist: np.ndarray
    disp_max_hist: np.ndarray
    K_hist: np.ndarray
    U_hist: np.ndarray
    E_hist: np.ndarray
    com_hist: np.ndarray  # (Nt, 3) physical COM (periodic, centered)
    walltime_s: float

    # Frames for GIF: list of dicts with x_phys_xy, x_mesh_xy, mesh_xy, step
    frames: List[Dict[str, Any]]

    # Diagnostics from the last step
    last_diag: Dict[str, Any]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def unwrap_periodic_line(x: np.ndarray, box: float) -> np.ndarray:
    """Return an unwrapped polyline (periodic-minimal increments)."""
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        dx = (x[i] - x[i - 1] + 0.5 * box) % box - 0.5 * box
        out[i] = out[i - 1] + dx
    return out


def mesh_segments_xy(xg: np.ndarray, yg: np.ndarray, box: float) -> np.ndarray:
    """Convert a periodic grid (xg,yg) to line segments in R^2 (LineCollection)."""
    segs = []
    nx, ny = xg.shape
    for i in range(nx):
        xu = unwrap_periodic_line(xg[i, :], box)
        yu = unwrap_periodic_line(yg[i, :], box)
        segs.append(np.stack([np.stack([xu[:-1], yu[:-1]], axis=1), np.stack([xu[1:], yu[1:]], axis=1)], axis=1))
    for j in range(ny):
        xu = unwrap_periodic_line(xg[:, j], box)
        yu = unwrap_periodic_line(yg[:, j], box)
        segs.append(np.stack([np.stack([xu[:-1], yu[:-1]], axis=1), np.stack([xu[1:], yu[1:]], axis=1)], axis=1))
    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.concatenate(segs, axis=0)


def minimal_image(dx: np.ndarray, box: float) -> np.ndarray:
    """Map displacements to [-box/2, box/2) for periodic distance computations."""
    return (dx + 0.5 * box) % box - 0.5 * box


def radii_from_center(x_phys: np.ndarray, *, center: Tuple[float, float, float], box: float) -> np.ndarray:
    """Periodic radii |x-center| with minimal-image convention."""
    c = np.array(center, dtype=np.float64)
    d = minimal_image(x_phys - c[None, :], float(box))
    return np.sqrt(np.sum(d * d, axis=1))


def radius_quantiles(x_phys: np.ndarray, *, center: Tuple[float, float, float], box: float, qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    r = radii_from_center(x_phys, center=center, box=box)
    out = {}
    for q in qs:
        out[f"r{int(100*q):02d}"] = float(np.quantile(r, q))
    return out


def percentile_width_1d(x: np.ndarray, *, center: float, box: float, p_lo: float = 10.0, p_hi: float = 90.0) -> float:
    """Percentile width of a periodic 1D distribution around a center."""
    d = minimal_image(x - float(center), float(box))
    return float(np.percentile(d, p_hi) - np.percentile(d, p_lo))


def peak_separation_1d(x: np.ndarray, *, box: float, nbins: int = 64, smooth: int = 2) -> float:
    """Crude 1D peak separation (periodic) from a histogram.

    Returns the periodic distance between the top-2 histogram peaks.
    """
    hist, edges = np.histogram(x % box, bins=int(nbins), range=(0.0, float(box)))
    h = hist.astype(np.float64)
    for _ in range(int(smooth)):
        h = 0.25 * np.roll(h, 1) + 0.5 * h + 0.25 * np.roll(h, -1)
    i1 = int(np.argmax(h))
    h2 = h.copy()
    # suppress a neighborhood around the strongest peak
    w = max(1, nbins // 16)
    for di in range(-w, w + 1):
        h2[(i1 + di) % nbins] = -np.inf
    i2 = int(np.argmax(h2))
    c1 = 0.5 * (edges[i1] + edges[i1 + 1])
    c2 = 0.5 * (edges[i2] + edges[i2 + 1])
    sep = abs(((c2 - c1 + 0.5 * box) % box) - 0.5 * box)
    return float(sep)


def make_lattice_positions(
    nside: int,
    box: float,
    *,
    seed: int = 0,
    random_shift: bool = True,
    jitter: float = 0.0,
) -> np.ndarray:
    """nside^3 lattice in physical units, with optional random shift and jitter."""
    rng = np.random.default_rng(int(seed))
    shift = rng.random(3) if random_shift else np.zeros(3)
    ii, jj, kk = np.meshgrid(np.arange(nside), np.arange(nside), np.arange(nside), indexing="ij")
    q = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
    if jitter > 0.0:
        q = q + float(jitter) * rng.normal(size=q.shape)
    q = (q + 0.5 + shift) * (float(box) / float(nside))
    return np.mod(q, box).astype(np.float32)


def _make_power_spectrum(kmag: np.ndarray, *, n: float, k0: float, kcut: float) -> np.ndarray:
    """Simple CDM-ish toy power spectrum (same as qzoom_fig6_fig7_demo.py)."""
    k = np.maximum(kmag, 1e-12)
    return (k / float(k0)) ** float(n) * np.exp(-(k / float(kcut)) ** 2)


def gaussian_ic_displacement(
    ng: int,
    box: float,
    *,
    seed: int,
    n: float = -1.0,
    k0: float = 6.0,
    kcut: float = 30.0,
    disp_rms: float = 0.03,
) -> np.ndarray:
    """Return a displacement field s(x) on the ng^3 mesh in *physical* units.

    Matches the implementation used in diffAPM_new/qzoom_fig6_fig7_demo.py.
    """
    rng = np.random.default_rng(int(seed))
    g = rng.normal(size=(ng, ng, ng)).astype(np.float64)

    dx = float(box) / int(ng)
    k1 = 2.0 * np.pi * np.fft.fftfreq(ng, d=dx)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    kmag = np.sqrt(k2)

    Pk = _make_power_spectrum(kmag, n=float(n), k0=float(k0), kcut=float(kcut))
    Pk[k2 == 0.0] = 0.0

    gk = np.fft.fftn(g)
    deltak = gk * np.sqrt(Pk)
    phik = np.zeros_like(deltak)
    phik[k2 > 0.0] = -deltak[k2 > 0.0] / k2[k2 > 0.0]

    sx = np.fft.ifftn(1j * kx * phik).real
    sy = np.fft.ifftn(1j * ky * phik).real
    sz = np.fft.ifftn(1j * kz * phik).real
    s = np.stack([sx, sy, sz], axis=-1)

    rms = float(np.sqrt(np.mean(np.sum(s * s, axis=-1))))
    if rms > 0:
        s = s * (float(disp_rms) / rms)
    return s.astype(np.float32)


def make_state_grf_zeldovich(
    *,
    ng: int,
    npart_side: int,
    box: float,
    seed: int,
    pk_n: float,
    pk_k0: float,
    pk_kcut: float,
    disp_rms: float,
    vel_fac: float,
    pmass: float,
    lattice_random_shift: bool = True,
    lattice_jitter: float = 0.0,
) -> Tuple[qz.NBodyState, Tuple[float, float, float]]:
    """GRF displacement on a lattice, Zel'dovich-like IC (paper Fig6-style)."""
    dx = float(box) / int(ng)

    s = gaussian_ic_displacement(
        ng,
        box,
        seed=int(seed),
        n=float(pk_n),
        k0=float(pk_k0),
        kcut=float(pk_kcut),
        disp_rms=float(disp_rms) * float(box),
    )

    q = make_lattice_positions(
        int(npart_side),
        float(box),
        seed=int(seed) + 12345,
        random_shift=bool(lattice_random_shift),
        jitter=float(lattice_jitter),
    )

    xi_q = (q / dx) % float(ng)
    s_j = jnp.array(s)
    xi_q_j = jnp.array(xi_q)
    disp_q = jnp.stack(
        [
            qz.cic_readout_3d(s_j[..., 0], xi_q_j),
            qz.cic_readout_3d(s_j[..., 1], xi_q_j),
            qz.cic_readout_3d(s_j[..., 2], xi_q_j),
        ],
        axis=-1,
    )  # physical units

    xi0 = (xi_q_j + disp_q / dx) % float(ng)
    v0 = (float(vel_fac) * disp_q).astype(jnp.float32)
    xv0 = jnp.concatenate([xi0, v0], axis=1)
    pm = jnp.full((xv0.shape[0],), float(pmass), dtype=jnp.float32)
    state0 = qz.NBodyState(xv=xv0, pmass=pm)

    center = (0.5 * float(box), 0.5 * float(box), 0.5 * float(box))
    return state0, center


def make_state_single_halo(
    *,
    ng: int,
    box: float,
    npart: int,
    seed: int,
    clump_frac: float,
    clump_sigma: float,
    v_infall: float,
    pmass: float,
) -> Tuple[qz.NBodyState, Tuple[float, float, float]]:
    """Single Gaussian halo + background (centered) in physical coordinates."""
    rng = np.random.default_rng(int(seed))
    center = (0.5 * float(box), 0.5 * float(box), 0.5 * float(box))

    ncl = int(round(float(clump_frac) * int(npart)))
    nbg = int(npart) - ncl
    x_bg = rng.uniform(0.0, float(box), size=(nbg, 3)).astype(np.float32)
    x_cl = sample_gaussian_positions(ncl, float(box), center=center, sigma=float(clump_sigma) * float(box), seed=int(seed) + 1)
    x = np.concatenate([x_bg, x_cl], axis=0)

    v = np.zeros_like(x, dtype=np.float32)
    if float(v_infall) != 0.0:
        d = minimal_image(x - np.array(center, dtype=np.float64)[None, :], float(box))
        r = np.sqrt(np.sum(d * d, axis=1, keepdims=True)) + 1e-12
        v = (-float(v_infall) * d / r).astype(np.float32)

    state0 = make_state_from_phys(x, v, ng=int(ng), box=float(box), pmass=float(pmass))
    return state0, center


def make_state_two_halo_merger(
    *,
    ng: int,
    box: float,
    npart: int,
    seed: int,
    clump_frac: float,
    clump_sigma: float,
    sep: float,
    v_merge: float,
    pmass: float,
) -> Tuple[qz.NBodyState, Tuple[float, float, float]]:
    """Two Gaussian halos along x, centered in y/z."""
    rng = np.random.default_rng(int(seed))
    cx = 0.5 * float(box)
    cy = 0.5 * float(box)
    cz = 0.5 * float(box)
    c1 = (cx - 0.5 * float(sep) * float(box), cy, cz)
    c2 = (cx + 0.5 * float(sep) * float(box), cy, cz)

    ncl = int(round(float(clump_frac) * int(npart)))
    nbg = int(npart) - ncl
    n1 = ncl // 2
    n2 = ncl - n1

    x_bg = rng.uniform(0.0, float(box), size=(nbg, 3)).astype(np.float32)
    x1 = sample_gaussian_positions(n1, float(box), center=c1, sigma=float(clump_sigma) * float(box), seed=int(seed) + 1)
    x2 = sample_gaussian_positions(n2, float(box), center=c2, sigma=float(clump_sigma) * float(box), seed=int(seed) + 2)
    x = np.concatenate([x_bg, x1, x2], axis=0)

    v = np.zeros_like(x, dtype=np.float32)
    if float(v_merge) != 0.0:
        v[nbg : nbg + n1, 0] = +float(v_merge)
        v[nbg + n1 :, 0] = -float(v_merge)

    state0 = make_state_from_phys(x, v, ng=int(ng), box=float(box), pmass=float(pmass))
    center = (cx, cy, cz)
    return state0, center


def make_state_pancake_filament(
    *,
    ng: int,
    box: float,
    npart_side: int,
    seed: int,
    amp_x: float,
    amp_y: float,
    vel_fac: float,
    lattice_jitter: float,
    lattice_random_shift: bool,
    pmass: float,
) -> Tuple[qz.NBodyState, Tuple[float, float, float]]:
    """Lattice IC with sinusoidal displacement along x (pancake) and optionally y (filament)."""
    q = make_lattice_positions(
        int(npart_side),
        float(box),
        seed=int(seed) + 12345,
        random_shift=bool(lattice_random_shift),
        jitter=float(lattice_jitter),
    )

    L = float(box)
    cx = 0.5 * L
    cy = 0.5 * L
    cz = 0.5 * L

    sx = -float(amp_x) * float(box) * np.sin(2.0 * np.pi * (q[:, 0] - cx) / L)
    sy = -float(amp_y) * float(box) * np.sin(2.0 * np.pi * (q[:, 1] - cy) / L)

    x = q.astype(np.float64)
    x[:, 0] = x[:, 0] + sx
    x[:, 1] = x[:, 1] + sy
    x = np.mod(x, L).astype(np.float32)

    v = np.zeros_like(x, dtype=np.float32)
    v[:, 0] = (float(vel_fac) * sx).astype(np.float32)
    v[:, 1] = (float(vel_fac) * sy).astype(np.float32)

    state0 = make_state_from_phys(x, v, ng=int(ng), box=float(box), pmass=float(pmass))
    center = (cx, cy, cz)
    return state0, center


def sample_gaussian_positions(
    n: int,
    box: float,
    *,
    center: Tuple[float, float, float],
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    """Sample positions from a periodic Gaussian blob centered at `center`."""
    rng = np.random.default_rng(int(seed))
    x = rng.normal(loc=np.array(center, dtype=np.float64), scale=float(sigma), size=(int(n), 3))
    return np.mod(x, box).astype(np.float32)


def make_state_from_phys(
    x_phys: np.ndarray,
    v_phys: Optional[np.ndarray],
    *,
    ng: int,
    box: float,
    pmass: float = 1.0,
) -> qz.NBodyState:
    """Convert physical positions/velocities to the qzoom_nbody_flow state."""
    dx = float(box) / int(ng)
    xi = np.mod(x_phys / dx, float(ng)).astype(np.float32)
    if v_phys is None:
        v = np.zeros_like(xi, dtype=np.float32)
    else:
        v = v_phys.astype(np.float32)
    xv = np.concatenate([xi, v], axis=1)
    pm = np.full((xv.shape[0],), float(pmass), dtype=np.float32)
    return qz.NBodyState(xv=jnp.array(xv), pmass=jnp.array(pm))


def particles_phys(state: qz.NBodyState, def_field: Array, dx: float, box: float) -> Array:
    """Map particle mesh coords xi -> physical coords x = xi*dx + grad(def)(xi)."""
    xi = state.xv[:, 0:3]
    grad_def = qz.grad_central(def_field, dx)
    disp = jnp.stack(
        [
            qz.cic_readout_3d(grad_def[..., 0], xi),
            qz.cic_readout_3d(grad_def[..., 1], xi),
            qz.cic_readout_3d(grad_def[..., 2], xi),
        ],
        axis=-1,
    )
    x_phys = xi * dx + disp
    return jnp.mod(x_phys, box)


def particles_mesh_frame(state: qz.NBodyState, dx: float, box: float) -> Array:
    """Particle locations in moving-mesh (comoving) frame, in physical units: x_mesh = xi*dx."""
    xi = state.xv[:, 0:3]
    return jnp.mod(xi * dx, box)


def mesh_centers_phys(def_field: Array, dx: float, box: float) -> Array:
    """Physical coordinates of mesh cell centers: x = xi + grad(def)."""
    ng = int(def_field.shape[0])
    coords = (jnp.arange(ng, dtype=def_field.dtype) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing="ij")
    grad_def = qz.grad_central(def_field, dx)
    x = jnp.stack([X, Y, Z], axis=-1) + grad_def
    return jnp.mod(x, box)


def build_limiter(cfg: RunConfig, densinit_mass_mesh: Optional[Array]) -> Tuple[qz.LimiterParams, Optional[Array]]:
    """Create LimiterParams and densinit field for qzoom_nbody_flow."""
    if not bool(cfg.limiter):
        return qz.LimiterParams(enabled=False), None

    use_source = str(cfg.limiter_mode) in ("both", "source")
    use_post = str(cfg.limiter_mode) in ("both", "post")
    lim = qz.LimiterParams(
        enabled=True,
        compressmax=float(cfg.compressmax),
        skewmax=float(cfg.skewmax),
        xhigh=float(cfg.xhigh),
        relax_steps=float(cfg.relax_steps),
        use_source_term=bool(use_source),
        use_post_scale=bool(use_post),
        post_only_on_fail=True,
        hard_strength=float(cfg.hard_strength),
    )
    return lim, densinit_mass_mesh


def _capture_frame(
    frames: List[Dict[str, Any]],
    *,
    step: int,
    state: qz.NBodyState,
    def_field: Array,
    cfg: RunConfig,
    z_anim: int,
) -> None:
    """Capture a lightweight frame: particle x-y (phys & mesh) and mesh layer."""
    ng = int(cfg.ng)
    dx = cfg.dx()
    box = float(cfg.box)
    stride = int(cfg.mesh_stride)

    x_phys_f = np.array(particles_phys(state, def_field, dx, box), dtype=np.float64)
    x_mesh_f = np.array(particles_mesh_frame(state, dx, box), dtype=np.float64)
    mesh_phys_f = np.array(mesh_centers_phys(def_field, dx, box), dtype=np.float64)
    mesh_xy_f = mesh_phys_f[::stride, ::stride, z_anim, 0:2]

    frames.append(
        {
            "step": int(step),
            "x_phys_xy": x_phys_f[:, 0:2],
            "x_mesh_xy": x_mesh_f[:, 0:2],
            "mesh_xy": mesh_xy_f,
            "ng": ng,
        }
    )


def run_moving_mesh(
    name: str,
    state0: qz.NBodyState,
    *,
    cfg: RunConfig,
    verbose_every: int = 25,
) -> RunResult:
    """Run the moving-mesh APM flow (qzoom_nbody_flow.step_nbody_apm)."""
    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()

    # Multigrid setup
    levels = int(np.log2(ng) - 1)
    mg_params = qz.MGParams(levels=levels, v1=2, v2=2, mu=1, cycles=int(cfg.mg_cycles))
    params = qz.APMParams(ng=ng, box=box, a=float(cfg.a))

    def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    # densinit is uniform mass-per-cell at t=0 (QZOOM default is 1; we use our units).
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def_field, state0, dx=dx)
    densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))
    limiter, densinit = build_limiter(cfg, densinit)

    # time stepping
    dt = float(cfg.dt_max)
    dtold = dt
    dt_hist: List[float] = []
    sgmin_hist: List[float] = []
    dispmax_hist: List[float] = []
    K_hist: List[float] = []
    U_hist: List[float] = []
    E_hist: List[float] = []
    com_hist: List[np.ndarray] = []
    frames: List[Dict[str, Any]] = []

    z_anim = int(cfg.mesh_z) if cfg.mesh_z is not None else (ng // 2)
    z_anim = int(np.clip(z_anim, 0, ng - 1))
    if bool(cfg.animate):
        _capture_frame(frames, step=0, state=state0, def_field=def_field, cfg=cfg, z_anim=z_anim)

    state = state0
    last_diag: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for it in range(1, int(cfg.ntotal) + 1):
        state, def_field, defp_field, phi, diag = qz.step_nbody_apm(
            state,
            def_field,
            defp_field,
            params=params,
            mg_params=mg_params,
            kappa=float(cfg.kappa),
            smooth_steps=int(cfg.smooth_steps),
            limiter=limiter,
            densinit=densinit,
            dt=dt,
            dtold=dtold,
        )
        last_diag = dict(diag)

        # Fail-fast if anything goes non-finite; keep partial outputs for debugging.
        sg_min = float(diag.get("sqrt_g_new_min", diag.get("sqrt_g_min", np.nan)))
        sg_ok = np.isfinite(sg_min) and (sg_min > 0.0)
        if not sg_ok or not np.isfinite(float(diag.get("phi_rms", np.nan))) or not np.isfinite(float(diag.get("disp_max", np.nan))):
            print(f"[{name} moving] STOP: non-finite or invalid mesh detected at step {it} (sqrt_g_min={sg_min}).")
            break

        dt_hist.append(dt)
        sgmin_hist.append(float(diag.get("sqrt_g_new_min", diag.get("sqrt_g_min", np.nan))))
        dispmax_hist.append(float(diag.get("disp_max", np.nan)))
        K_hist.append(float(diag.get("K", np.nan)))
        U_hist.append(float(diag.get("U", np.nan)))
        E_hist.append(float(diag.get("E", np.nan)))

        # Physical COM each step (periodic, minimal-image around box center).
        x_phys = np.array(particles_phys(state, def_field, dx, box), dtype=np.float64)
        c0 = np.array([0.5 * box, 0.5 * box, 0.5 * box], dtype=np.float64)
        dcom = minimal_image(x_phys - c0[None, :], box)
        com = (c0 + np.mean(dcom, axis=0)) % box
        com_hist.append(com.astype(np.float64))

        # Adaptive dt: mesh-based limit + particle CFL limits.
        dt_mesh, max_hess = qz.mesh_dt_from_defp(defp_field, dx, safety=float(cfg.cfl_mesh))
        if not np.isfinite(float(dt_mesh)) or float(dt_mesh) <= 0.0:
            dt_mesh = float("inf")

        vmax = float(diag.get("vmax", np.nan))
        amax = float(diag.get("amax", np.nan))
        dt_vel = float("inf") if (not np.isfinite(vmax) or vmax <= 0.0) else float(cfg.cfl_part) * dx / vmax
        dt_acc = float("inf") if (not np.isfinite(amax) or amax <= 0.0) else np.sqrt(2.0 * float(cfg.cfl_part) * dx / amax)

        dt_next = min(float(cfg.dt_max), float(dt_mesh), float(dt_vel), float(dt_acc))
        dt_next = max(float(cfg.dt_min), dt_next)
        dtold = dt
        dt = dt_next

        if verbose_every and (it % int(verbose_every) == 0 or it == 1 or it == int(cfg.ntotal)):
            meshlim_str = f"{float(dt_mesh):.3e}" if np.isfinite(float(dt_mesh)) else "inf(defp~const)"
            print(
                f"[{name} moving][step {it:04d}/{int(cfg.ntotal)}] "
                f"dt={dt:.3e} meshlim={meshlim_str} max|H|~{float(max_hess):.3e} "
                f"dt_vel={dt_vel:.3e} dt_acc={dt_acc:.3e} "
                f"sqrt_g_min={sgmin_hist[-1]:.3e} disp_max/dx={dispmax_hist[-1]/dx:.2f}"
            )

        if bool(cfg.animate) and (it % int(cfg.anim_every) == 0 or it == int(cfg.ntotal)):
            _capture_frame(frames, step=it, state=state, def_field=def_field, cfg=cfg, z_anim=z_anim)

    walltime_s = float(time.perf_counter() - t0)

    return RunResult(
        name=name,
        mode="moving",
        state=state,
        def_field=def_field,
        defp_field=defp_field,
        phi=phi,
        dt_hist=np.array(dt_hist, dtype=np.float64),
        sqrt_g_min_hist=np.array(sgmin_hist, dtype=np.float64),
        disp_max_hist=np.array(dispmax_hist, dtype=np.float64),
        K_hist=np.array(K_hist, dtype=np.float64),
        U_hist=np.array(U_hist, dtype=np.float64),
        E_hist=np.array(E_hist, dtype=np.float64),
        com_hist=np.stack(com_hist, axis=0) if com_hist else np.zeros((0, 3), dtype=np.float64),
        walltime_s=walltime_s,
        frames=frames,
        last_diag=last_diag,
    )


def step_static_pm(
    state: qz.NBodyState,
    *,
    params: qz.APMParams,
    dt: float,
    dtold: float,
) -> Tuple[qz.NBodyState, Array, Array]:
    """One step on a static (uniform) mesh using FFT Poisson (def=0, defp=0).

    This is the baseline fixed-mesh PM to compare against.
    """
    ng = int(params.ng)
    dx = float(params.box) / ng
    def0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)

    # Deposit on uniform mesh: rho_mesh is mass-per-cell in mesh coords.
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh, _ = qz.pcalcrho(rho0, def0, state, dx=dx)
    rhs = rho_mesh - jnp.mean(rho_mesh)
    phi = qz.poisson_fft_uniform(rhs, dx=dx)

    # No mesh motion: def=defn=0.
    state_new = qz.stepxv(phi, def0, def0, dt, dtold, float(params.a), state, dx=dx)
    return state_new, phi, rho_mesh


def run_static_mesh(
    name: str,
    state0: qz.NBodyState,
    *,
    cfg: RunConfig,
    verbose_every: int = 25,
) -> RunResult:
    """Run the same ICs with a static mesh (fixed-mesh PM via FFT Poisson)."""
    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()
    params = qz.APMParams(ng=ng, box=box, a=float(cfg.a))

    def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    dt = float(cfg.dt_max)
    dtold = dt
    dt_hist: List[float] = []
    sgmin_hist: List[float] = []
    dispmax_hist: List[float] = []
    K_hist: List[float] = []
    U_hist: List[float] = []
    E_hist: List[float] = []
    com_hist: List[np.ndarray] = []
    frames: List[Dict[str, Any]] = []

    z_anim = int(cfg.mesh_z) if cfg.mesh_z is not None else (ng // 2)
    z_anim = int(np.clip(z_anim, 0, ng - 1))
    if bool(cfg.animate):
        _capture_frame(frames, step=0, state=state0, def_field=def_field, cfg=cfg, z_anim=z_anim)

    state = state0
    last_diag: Dict[str, Any] = {}
    phi = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    t0 = time.perf_counter()
    for it in range(1, int(cfg.ntotal) + 1):
        state, phi, rho_mesh = step_static_pm(state, params=params, dt=dt, dtold=dtold)
        dt_hist.append(dt)
        sgmin_hist.append(1.0)
        dispmax_hist.append(0.0)
        # Particle CFL limits for fair comparison against moving mesh.
        gphi = qz.grad_central(phi, dx)  # def=0 => accel = -grad(phi)*a
        accel_grid = -gphi * float(params.a)
        amax = float(jnp.max(jnp.linalg.norm(accel_grid, axis=-1)))
        vmax = float(jnp.max(jnp.linalg.norm(state.xv[:, 3:6], axis=-1)))
        dt_vel = float("inf") if (not np.isfinite(vmax) or vmax <= 0.0) else float(cfg.cfl_part) * dx / vmax
        dt_acc = float("inf") if (not np.isfinite(amax) or amax <= 0.0) else float(np.sqrt(2.0 * float(cfg.cfl_part) * dx / amax))
        dt_next = min(float(cfg.dt_max), float(dt_vel), float(dt_acc))
        dt_next = max(float(cfg.dt_min), dt_next)
        dtold = dt
        dt = dt_next

        # Energy diagnostics
        rhs = rho_mesh - jnp.mean(rho_mesh)
        K = 0.5 * jnp.sum(state.pmass * jnp.sum(state.xv[:, 3:6] ** 2, axis=-1))
        U = 0.5 * (dx ** 3) * jnp.sum(rhs * phi)
        E = K + U
        K_hist.append(float(K))
        U_hist.append(float(U))
        E_hist.append(float(E))

        # COM history in physical coordinates (def=0 => x=xi*dx)
        x_phys = np.array(state.xv[:, 0:3] * dx, dtype=np.float64) % box
        c0 = np.array([0.5 * box, 0.5 * box, 0.5 * box], dtype=np.float64)
        dcom = minimal_image(x_phys - c0[None, :], box)
        com = (c0 + np.mean(dcom, axis=0)) % box
        com_hist.append(com.astype(np.float64))

        last_diag = {
            "rho_mean": float(jnp.mean(rho_mesh)),
            "phi_rms": float(jnp.sqrt(jnp.mean(phi * phi))),
            "vmax": vmax,
            "amax": amax,
            "K": float(K),
            "U": float(U),
            "E": float(E),
        }

        if verbose_every and (it % int(verbose_every) == 0 or it == 1 or it == int(cfg.ntotal)):
            print(f"[{name} static][step {it:04d}/{int(cfg.ntotal)}] dt={dt:.3e} dt_vel={dt_vel:.3e} dt_acc={dt_acc:.3e}")

        if bool(cfg.animate) and (it % int(cfg.anim_every) == 0 or it == int(cfg.ntotal)):
            _capture_frame(frames, step=it, state=state, def_field=def_field, cfg=cfg, z_anim=z_anim)

    walltime_s = float(time.perf_counter() - t0)

    return RunResult(
        name=name,
        mode="static",
        state=state,
        def_field=def_field,
        defp_field=defp_field,
        phi=phi,
        dt_hist=np.array(dt_hist, dtype=np.float64),
        sqrt_g_min_hist=np.array(sgmin_hist, dtype=np.float64),
        disp_max_hist=np.array(dispmax_hist, dtype=np.float64),
        K_hist=np.array(K_hist, dtype=np.float64),
        U_hist=np.array(U_hist, dtype=np.float64),
        E_hist=np.array(E_hist, dtype=np.float64),
        com_hist=np.stack(com_hist, axis=0) if com_hist else np.zeros((0, 3), dtype=np.float64),
        walltime_s=walltime_s,
        frames=frames,
        last_diag=last_diag,
    )


def save_animation_gif(
    result: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    title_prefix: str = "",
) -> None:
    """Save an animation GIF for a RunResult (frames must be populated)."""
    if len(result.frames) < 2:
        return

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import animation

    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()
    stride = int(cfg.mesh_stride)

    coords = (np.arange(ng) + 0.5) * dx
    uu = coords[::stride]
    Ux, Uy = np.meshgrid(uu, uu, indexing="ij")

    # consistent particle downsampling
    nplot = min(int(cfg.max_plot), result.frames[0]["x_phys_xy"].shape[0])
    if nplot < result.frames[0]["x_phys_xy"].shape[0]:
        sel = np.random.default_rng(0).choice(result.frames[0]["x_phys_xy"].shape[0], size=nplot, replace=False)
    else:
        sel = None

    fig, (axM, axP) = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5.8))
    axM.set_title("Mesh layer (physical x-y)")
    axM.set_aspect("equal")
    axM.set_xlim(0, box)
    axM.set_ylim(0, box)
    axM.set_xticks([])
    axM.set_yticks([])

    axP.set_title("Particles: physical vs mesh")
    axP.set_aspect("equal")
    axP.set_xlim(0, box)
    axP.set_ylim(0, box)
    axP.set_xticks([])
    axP.set_yticks([])

    for i in range(Ux.shape[0]):
        axM.plot(Ux[i, :], Uy[i, :], color="0.88", lw=0.6, alpha=0.9, zorder=1)
        axM.plot(Ux[:, i], Uy[:, i], color="0.88", lw=0.6, alpha=0.9, zorder=1)

    # Use a single consistent style for the deformed mesh, even for periodic wrap-around
    # images. (Thin "periodic reflections" can be visually misleading in diagnostics.)
    lc_main = LineCollection([], colors="k", linewidths=0.9, alpha=1.0, zorder=3)
    lc_ref = LineCollection([], colors="k", linewidths=0.9, alpha=1.0, zorder=2)
    axM.add_collection(lc_ref)
    axM.add_collection(lc_main)

    scatM = axM.scatter([], [], s=1.0, marker=".", color="k", alpha=0.35, linewidths=0.0, zorder=4)
    scatP_phys = axP.scatter([], [], s=1.0, marker=".", color="k", alpha=0.75, linewidths=0.0, label="physical")
    scatP_mesh = axP.scatter([], [], s=1.0, marker=".", color="tab:orange", alpha=0.75, linewidths=0.0, label="mesh (xi*dx)")
    axP.legend(frameon=False, markerscale=6, loc="upper right")

    # Draw periodic images so wrap-around looks continuous, but keep a *single*
    # consistent style so nothing appears "lighter" when it crosses the boundary.
    shifts = [(0.0, 0.0), (-box, 0.0), (box, 0.0), (0.0, -box), (0.0, box)]

    def update(frame_idx: int):
        fr = result.frames[frame_idx]
        step = int(fr["step"])
        x_phys_xy = fr["x_phys_xy"]
        x_mesh_xy = fr["x_mesh_xy"]
        mesh_xy = fr["mesh_xy"]

        if sel is not None:
            x_phys_xy = x_phys_xy[sel]
            x_mesh_xy = x_mesh_xy[sel]

        scatM.set_offsets(x_phys_xy)
        scatP_phys.set_offsets(x_phys_xy)
        scatP_mesh.set_offsets(x_mesh_xy)

        ex = float(cfg.mesh_exaggerate)
        xg0 = mesh_xy[..., 0]
        yg0 = mesh_xy[..., 1]
        dxy = np.stack([xg0, yg0], axis=-1) - np.stack([Ux, Uy], axis=-1)
        dxy = (dxy + 0.5 * box) % box - 0.5 * box
        xg = (Ux + ex * dxy[..., 0]) % box
        yg = (Uy + ex * dxy[..., 1]) % box

        segs_main = mesh_segments_xy(xg, yg, box)
        segs_ref_all = []
        for sx, sy in shifts:
            if (sx, sy) == (0.0, 0.0):
                continue
            segs_ref_all.append(segs_main + np.array([sx, sy], dtype=np.float64)[None, None, :])
        segs_ref = np.concatenate(segs_ref_all, axis=0) if segs_ref_all else np.zeros((0, 2, 2), dtype=np.float64)

        lc_main.set_segments(segs_main)
        lc_ref.set_segments(segs_ref)

        fig.suptitle(
            f"{title_prefix}{result.name} ({result.mode}) step={step} (mesh_exaggerate={float(cfg.mesh_exaggerate):g})",
            y=0.98,
        )
        return lc_main, lc_ref, scatM, scatP_phys, scatP_mesh

    ani = animation.FuncAnimation(fig, update, frames=len(result.frames), interval=1000 // max(1, int(cfg.anim_fps)), blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=int(cfg.anim_fps)))
    plt.close(fig)


def _density_on_uniform_grid(x_phys: np.ndarray, *, ng: int, box: float, pmass: float = 1.0) -> np.ndarray:
    """CIC mass-per-cell on a uniform ng^3 grid from physical coordinates."""
    dx = float(box) / int(ng)
    xi = (x_phys / dx) % float(ng)
    mesh = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho = qz.cic_deposit_3d(mesh, jnp.array(xi, dtype=jnp.float32), jnp.full((xi.shape[0],), float(pmass), dtype=jnp.float32))
    return np.array(rho, dtype=np.float64)


def compute_metrics(
    result: RunResult,
    *,
    cfg: RunConfig,
    center: Tuple[float, float, float],
    tag: str,
) -> Dict[str, float]:
    """Basic metrics to compare moving vs static runs."""
    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()

    x_phys = np.array(particles_phys(result.state, result.def_field, dx, box), dtype=np.float64)
    x_mesh = np.array(particles_mesh_frame(result.state, dx, box), dtype=np.float64)

    if not np.all(np.isfinite(x_phys)) or not np.all(np.isfinite(x_mesh)):
        return {
            f"{tag}_com_dist": float("nan"),
            f"{tag}_rho_phys_p99": float("nan"),
            f"{tag}_rho_phys_max": float("nan"),
            f"{tag}_rho_phys_logstd": float("nan"),
            f"{tag}_rho_mesh_logstd": float("nan"),
            f"{tag}_mass_mesh_logstd": float("nan"),
            f"{tag}_mass_mesh_over_rho_mesh_logstd_ratio": float("nan"),
            f"{tag}_sqrt_g_min_final": float(result.sqrt_g_min_hist[-1]) if result.sqrt_g_min_hist.size else float("nan"),
            f"{tag}_disp_max_over_dx_final": float((result.disp_max_hist[-1] / dx) if result.disp_max_hist.size else float("nan")),
        }

    # Physical-frame density (uniform grid CIC). This is the standard fixed-mesh
    # PM "what would FFT see" diagnostic.
    rho_phys = _density_on_uniform_grid(x_phys, ng=ng, box=box, pmass=1.0)

    # Mesh-space diagnostics closer to QZOOM's intent:
    #  - mass_mesh(ξ) is mass-per-mesh-cell (what the def/defp tries to equalize)
    #  - rho_mesh(ξ) is mass per *physical* volume element (mass_mesh/sqrt_g)
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh_j, sqrt_g_j = qz.pcalcrho(rho0, result.def_field, result.state, dx=dx)
    rho_mesh = np.array(rho_mesh_j, dtype=np.float64)
    sqrt_g = np.array(sqrt_g_j, dtype=np.float64)
    mass_mesh = rho_mesh * sqrt_g

    def _log_std(rho):
        return float(np.std(np.log10(np.maximum(rho, 1e-12))))

    # CoM distance to desired center (periodic).
    c = np.array(center, dtype=np.float64)
    d = minimal_image(x_phys - c[None, :], box)
    com = (c + np.mean(d, axis=0)) % box
    com_dist = float(np.linalg.norm(minimal_image(com - c, box)))

    # Particle-level displacement between mesh-frame and physical coordinates.
    # This is exactly grad(def)(xi) (modulo periodic wrapping). A mesh can look
    # very distorted due to large *derivatives* (Hess(def)) even when this
    # absolute displacement is modest.
    disp_pm = minimal_image(x_phys - x_mesh, box)
    disp_mag = np.sqrt(np.sum(disp_pm * disp_pm, axis=1))
    disp_p50 = float(np.percentile(disp_mag, 50.0))
    disp_p90 = float(np.percentile(disp_mag, 90.0))
    disp_p99 = float(np.percentile(disp_mag, 99.0))
    disp_max = float(np.max(disp_mag))

    # A crude "diffuseness" measure: log-std(mesh) / log-std(phys).
    logstd_phys = _log_std(rho_phys)
    logstd_rho_mesh = _log_std(rho_mesh)
    logstd_mass_mesh = _log_std(mass_mesh)
    diff_ratio = float(logstd_mass_mesh / max(logstd_rho_mesh, 1e-12))

    out = {
        f"{tag}_com_dist": com_dist,
        f"{tag}_rho_phys_p99": float(np.percentile(rho_phys, 99.0)),
        f"{tag}_rho_phys_max": float(np.max(rho_phys)),
        f"{tag}_rho_phys_logstd": logstd_phys,
        f"{tag}_rho_mesh_logstd": logstd_rho_mesh,
        f"{tag}_mass_mesh_logstd": logstd_mass_mesh,
        f"{tag}_mass_mesh_over_rho_mesh_logstd_ratio": diff_ratio,
        f"{tag}_sqrt_g_min_final": float(result.sqrt_g_min_hist[-1]) if result.sqrt_g_min_hist.size else 1.0,
        f"{tag}_disp_max_over_dx_final": float((result.disp_max_hist[-1] / dx) if result.disp_max_hist.size else 0.0),
        f"{tag}_particle_disp_p50": disp_p50,
        f"{tag}_particle_disp_p90": disp_p90,
        f"{tag}_particle_disp_p99": disp_p99,
        f"{tag}_particle_disp_max": disp_max,
    }
    return out


def save_summary_figure(
    result: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    title: str,
) -> None:
    """Fig6/7-like summary figure for a run (single output image)."""
    import matplotlib.pyplot as plt

    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()
    z0 = int(cfg.mesh_z) if cfg.mesh_z is not None else (ng // 2)
    z0 = int(np.clip(z0, 0, ng - 1))

    x_phys = np.array(particles_phys(result.state, result.def_field, dx, box), dtype=np.float64)
    if not np.all(np.isfinite(x_phys)):
        x_phys = np.nan_to_num(x_phys, nan=0.0, posinf=0.0, neginf=0.0)

    # density slice: deposit in mesh coordinates of the run (so moving mesh shows "rho_mesh").
    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh, _ = qz.pcalcrho(rho0, result.def_field, result.state, dx=dx)
    rho_np = np.array(rho_mesh, dtype=np.float64)
    if not np.all(np.isfinite(rho_np)):
        rho_np = np.nan_to_num(rho_np, nan=0.0, posinf=0.0, neginf=0.0)

    mesh_phys = np.array(mesh_centers_phys(result.def_field, dx, box), dtype=np.float64)
    if not np.all(np.isfinite(mesh_phys)):
        mesh_phys = np.nan_to_num(mesh_phys, nan=0.0, posinf=0.0, neginf=0.0)
    mesh_xy = mesh_phys[:: int(cfg.mesh_stride), :: int(cfg.mesh_stride), z0, 0:2]

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    nplot = min(int(cfg.max_plot), x_phys.shape[0])
    if nplot < x_phys.shape[0]:
        sel = np.random.default_rng(0).choice(x_phys.shape[0], size=nplot, replace=False)
        pts = x_phys[sel]
    else:
        pts = x_phys
    ax0.scatter(pts[:, 0], pts[:, 1], s=1.0, marker=".", color="k", alpha=0.8, linewidths=0.0)
    ax0.set_title(f"{title}: particle projection (physical x-y)")
    ax0.set_aspect("equal")
    ax0.set_xlim(0, box)
    ax0.set_ylim(0, box)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    t = np.arange(1, int(cfg.ntotal) + 1)
    if result.dt_hist.size:
        ax1.plot(t[: result.dt_hist.size], result.dt_hist, color="tab:blue", label="dt")
        ax1.set_yscale("log")
    ax1.set_xlabel("step")
    ax1.set_ylabel("dt")
    ax1b = ax1.twinx()
    if result.sqrt_g_min_hist.size:
        ax1b.plot(t[: result.sqrt_g_min_hist.size], result.sqrt_g_min_hist, color="tab:red", alpha=0.8, label="sqrt_g_min")
        ax1b.set_yscale("log")
    ax1b.set_ylabel("sqrt_g_min")
    ax1.set_title("dt + mesh compression history")
    ax1.grid(True, alpha=0.2)

    # mesh panel
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title(f"Mesh layer at z-index {z0} (exaggerate x{cfg.mesh_exaggerate:g})")
    ax2.set_aspect("equal")
    coords = (np.arange(ng) + 0.5) * dx
    uu = coords[:: int(cfg.mesh_stride)]
    Ux, Uy = np.meshgrid(uu, uu, indexing="ij")
    for i in range(Ux.shape[0]):
        ax2.plot(Ux[i, :], Uy[i, :], color="0.80", lw=0.6, alpha=0.8, zorder=1)
        ax2.plot(Ux[:, i], Uy[:, i], color="0.80", lw=0.6, alpha=0.8, zorder=1)

    xg0 = mesh_xy[..., 0]
    yg0 = mesh_xy[..., 1]
    dxy = np.stack([xg0, yg0], axis=-1) - np.stack([Ux, Uy], axis=-1)
    dxy = (dxy + 0.5 * box) % box - 0.5 * box
    ex = float(cfg.mesh_exaggerate)
    xg = (Ux + ex * dxy[..., 0]) % box
    yg = (Uy + ex * dxy[..., 1]) % box

    shifts = [(0.0, 0.0), (-box, 0.0), (box, 0.0), (0.0, -box), (0.0, box)]
    for i in range(xg.shape[0]):
        xu = unwrap_periodic_line(xg[i, :], box)
        yu = unwrap_periodic_line(yg[i, :], box)
        for sx, sy in shifts:
            ax2.plot(xu + sx, yu + sy, color="k", lw=0.9, alpha=1.0, zorder=3)
    for j in range(xg.shape[1]):
        xu = unwrap_periodic_line(xg[:, j], box)
        yu = unwrap_periodic_line(yg[:, j], box)
        for sx, sy in shifts:
            ax2.plot(xu + sx, yu + sy, color="k", lw=0.9, alpha=1.0, zorder=3)

    ax2.set_xlim(0, box)
    ax2.set_ylim(0, box)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # density slice (log10)
    ax3 = fig.add_subplot(gs[1, 1])
    sl = rho_np[:, :, z0]+1
    sl = np.maximum(sl, 1e-12)
    im = ax3.imshow(np.log10(sl).T, origin="lower", cmap="magma")
    ax3.set_title("log10(rho_mesh) slice")
    ax3.set_xticks([])
    ax3.set_yticks([])
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.28])
    fig.colorbar(im, cax=cax, label="log10(rho)")

    fig.subplots_adjust(left=0.04, right=0.90, top=0.95, bottom=0.05, wspace=0.15, hspace=0.22)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_mesh_vs_phys_figure(
    result: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    title: str,
) -> None:
    """3-panel scatter: physical, mesh-frame, overlay."""
    import matplotlib.pyplot as plt

    box = float(cfg.box)
    dx = cfg.dx()
    x_phys = np.array(particles_phys(result.state, result.def_field, dx, box), dtype=np.float64)
    x_mesh = np.array(particles_mesh_frame(result.state, dx, box), dtype=np.float64)

    nplot = min(int(cfg.max_plot), x_phys.shape[0])
    if nplot < x_phys.shape[0]:
        sel = np.random.default_rng(1).choice(x_phys.shape[0], size=nplot, replace=False)
        pts_phys = x_phys[sel]
        pts_mesh = x_mesh[sel]
    else:
        pts_phys = x_phys
        pts_mesh = x_mesh

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14.0, 4.8))
    axes[0].scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.0, marker=".", color="k", alpha=0.75, linewidths=0.0)
    axes[0].set_title(f"{title}: physical x-y")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(0, box)
    axes[0].set_ylim(0, box)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(pts_mesh[:, 0], pts_mesh[:, 1], s=1.0, marker=".", color="tab:green", alpha=0.75, linewidths=0.0)
    axes[1].set_title(f"{title}: moving-mesh frame (xi*dx)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(0, box)
    axes[1].set_ylim(0, box)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.0, marker=".", color="k", alpha=0.35, linewidths=0.0, label="physical")
    axes[2].scatter(pts_mesh[:, 0], pts_mesh[:, 1], s=1.0, marker=".", color="tab:orange", alpha=0.35, linewidths=0.0, label="mesh")
    axes[2].set_title(f"{title}: overlay")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(0, box)
    axes[2].set_ylim(0, box)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].legend(frameon=False, markerscale=5, loc="upper right")

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.06, wspace=0.10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_compare_figure(
    moving: RunResult,
    static: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    title: str,
) -> None:
    """Compare moving vs static on the same physical grid (particles + density slices)."""
    import matplotlib.pyplot as plt

    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()
    z0 = int(cfg.mesh_z) if cfg.mesh_z is not None else (ng // 2)
    z0 = int(np.clip(z0, 0, ng - 1))

    x_mov = np.array(particles_phys(moving.state, moving.def_field, dx, box), dtype=np.float64)
    x_sta = np.array(particles_phys(static.state, static.def_field, dx, box), dtype=np.float64)

    # Uniform-grid CIC densities in physical coordinates.
    rho_mov = _density_on_uniform_grid(x_mov, ng=ng, box=box, pmass=1.0)
    rho_sta = _density_on_uniform_grid(x_sta, ng=ng, box=box, pmass=1.0)

    sl_mov = np.maximum(rho_mov[:, :, z0], 1e-12)
    sl_sta = np.maximum(rho_sta[:, :, z0], 1e-12)
    lmov = np.log10(sl_mov+1)
    lsta = np.log10(sl_sta+1)
    vmin = float(np.min([lmov.min(), lsta.min()]))
    vmax = float(np.max([lmov.max(), lsta.max()]))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.0, 10.0))

    # Particle projections
    nplot = min(int(cfg.max_plot), x_mov.shape[0])
    if nplot < x_mov.shape[0]:
        sel = np.random.default_rng(0).choice(x_mov.shape[0], size=nplot, replace=False)
        xm = x_mov[sel]
        xs = x_sta[sel]
    else:
        xm = x_mov
        xs = x_sta

    axes[0, 0].scatter(xm[:, 0], xm[:, 1], s=1.0, marker=".", color="k", alpha=0.75, linewidths=0.0)
    axes[0, 0].set_title("Particles (moving mesh, physical x-y)")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlim(0, box)
    axes[0, 0].set_ylim(0, box)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].scatter(xs[:, 0], xs[:, 1], s=1.0, marker=".", color="k", alpha=0.75, linewidths=0.0)
    axes[0, 1].set_title("Particles (static mesh, physical x-y)")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_xlim(0, box)
    axes[0, 1].set_ylim(0, box)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Density slices
    im0 = axes[1, 0].imshow(lmov.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"log10(rho_phys) slice z={z0} (moving)")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    im1 = axes[1, 1].imshow(lsta.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"log10(rho_phys) slice z={z0} (static)")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    fig.suptitle(title, y=0.98)
    cax = fig.add_axes([0.92, 0.14, 0.015, 0.25])
    fig.colorbar(im1, cax=cax, label="log10(rho_phys)")
    fig.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05, wspace=0.18, hspace=0.22)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_density_mesh_vs_phys(
    result: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    title: str,
    z0: Optional[int] = None,
) -> None:
    """imshow density on a physical uniform grid vs a mesh-coordinate grid.

    - "physical": CIC deposit of particle positions x_phys onto a uniform grid in physical coordinates.
    - "mesh": mass per mesh cell m(ξ) = rho_mesh(ξ) * sqrt_g(ξ) on the mesh-coordinate grid.

    These are not the same object; QZOOM's mesh motion attempts to make m(ξ)
    closer to uniform.
    """
    import matplotlib.pyplot as plt

    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()

    x_phys = np.array(particles_phys(result.state, result.def_field, dx, box), dtype=np.float64)
    rho_phys = _density_on_uniform_grid(x_phys, ng=ng, box=box, pmass=1.0)  # mass-per-cell on physical grid

    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh_j, sqrt_g_j = qz.pcalcrho(rho0, result.def_field, result.state, dx=dx)
    rho_mesh = np.array(rho_mesh_j, dtype=np.float64)
    sqrt_g = np.array(sqrt_g_j, dtype=np.float64)
    mass_mesh = rho_mesh * sqrt_g  # mass per mesh cell (in mesh coordinates)

    if z0 is None:
        idx = np.unravel_index(int(np.argmax(rho_phys)), rho_phys.shape)
        z0 = int(idx[2])
    z0 = int(np.clip(int(z0), 0, ng - 1))

    sl_phys = np.maximum(rho_phys[:, :, z0], 1e-12)
    sl_mesh = np.maximum(mass_mesh[:, :, z0], 1e-12)
    l_phys = np.log10(sl_phys+1)
    l_mesh = np.log10(sl_mesh+1)

    vmin = float(np.min([l_phys.min(), l_mesh.min()]))
    vmax = float(np.max([l_phys.max(), l_mesh.max()]))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.0, 5.2))
    im0 = axes[0].imshow(l_phys.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"log10(mass per cell) on physical grid (z={z0})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(l_mesh.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"log10(mass per cell) in mesh coords (z={z0})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title, y=0.98)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im1, cax=cax, label="log10(mass per cell)")
    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08, wspace=0.14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_state_npz(
    result: RunResult,
    *,
    out_path: Path,
    cfg: RunConfig,
    center: Tuple[float, float, float],
    z0: Optional[int] = None,
) -> None:
    """Save a compact analysis bundle (npz) with particles, mesh, and fields."""
    ng = int(cfg.ng)
    box = float(cfg.box)
    dx = cfg.dx()

    xv = np.array(result.state.xv, dtype=np.float32)
    pmass = np.array(result.state.pmass, dtype=np.float32)
    xi = xv[:, 0:3]
    v = xv[:, 3:6]

    x_phys = np.array(particles_phys(result.state, result.def_field, dx, box), dtype=np.float32)
    x_mesh = np.array(particles_mesh_frame(result.state, dx, box), dtype=np.float32)

    # Mesh and density fields
    def_field = np.array(result.def_field, dtype=np.float32)
    defp_field = np.array(result.defp_field, dtype=np.float32)
    phi = np.array(result.phi, dtype=np.float32)

    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh_j, sqrt_g_j = qz.pcalcrho(rho0, result.def_field, result.state, dx=dx)
    rho_mesh = np.array(rho_mesh_j, dtype=np.float32)
    sqrt_g = np.array(sqrt_g_j, dtype=np.float32)
    mass_mesh = (rho_mesh * sqrt_g).astype(np.float32)

    rho_phys = _density_on_uniform_grid(np.array(x_phys, dtype=np.float64), ng=ng, box=box, pmass=1.0).astype(np.float32)

    if z0 is None:
        idx = np.unravel_index(int(np.argmax(rho_phys)), rho_phys.shape)
        z0 = int(idx[2])
    z0 = int(np.clip(int(z0), 0, ng - 1))

    meta = {
        "name": result.name,
        "mode": result.mode,
        "ng": ng,
        "box": box,
        "dx": dx,
        "center": tuple(float(c) for c in center),
        "z0": int(z0),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()},
        "last_diag": result.last_diag,
        "walltime_s": float(result.walltime_s),
    }

    np.savez_compressed(
        out_path,
        meta_json=np.array(json.dumps(meta), dtype=np.object_),
        xi=xi,
        v=v,
        pmass=pmass,
        x_phys=x_phys,
        x_mesh=x_mesh,
        def_field=def_field,
        defp_field=defp_field,
        phi=phi,
        rho_phys=rho_phys,
        rho_mesh=rho_mesh,
        mass_mesh=mass_mesh,
        sqrt_g=sqrt_g,
        dt_hist=np.array(result.dt_hist, dtype=np.float32),
        sqrt_g_min_hist=np.array(result.sqrt_g_min_hist, dtype=np.float32),
        disp_max_hist=np.array(result.disp_max_hist, dtype=np.float32),
        K_hist=np.array(result.K_hist, dtype=np.float32),
        U_hist=np.array(result.U_hist, dtype=np.float32),
        E_hist=np.array(result.E_hist, dtype=np.float32),
        com_hist=np.array(result.com_hist, dtype=np.float32),
    )
