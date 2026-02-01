#!/usr/bin/env python3
"""Fig.6/Fig.7-style APM demo (Pen 1995) using the QZOOM-like JAX flow.

Goal
----
Produce plots qualitatively similar to Pen (1995) Fig. 6 and Fig. 7:
  - Fig.6-like: 3D particle distribution projected to (x,y) showing clumping.
  - Fig.7-like: a deformed mesh layer (x,y plane through a dense clump),
    showing clear mesh movement compared to the uniform grid and periodic
    reflections.

This uses:
  - diffAPM_new/qzoom_nbody_flow.py (QZOOM logical flow, N-body only)
  - Pen (1995) adaptive mesh deformation via def/defp

Notes
-----
This is not intended to match the paper's exact cosmological normalization.
It simply generates a CDM-like clumpy field via a Gaussian random field and a
Zel'dovich-like initial displacement, then evolves with an adaptive dt limiter
based on the mesh deformation rate (Pen 1995 eq. 12 style).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

import jax
import jax.numpy as jnp

# Allow running as a script without installing diffAPM_new as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import qzoom_nbody_flow as qz  # noqa: E402


def _make_power_spectrum(kmag: np.ndarray, *, n: float, k0: float, kcut: float) -> np.ndarray:
    """Simple CDM-ish toy power spectrum."""
    k = np.maximum(kmag, 1e-12)
    return (k / k0) ** n * np.exp(-(k / kcut) ** 2)


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

    Method:
      - draw a white-noise field g(x)
      - FFT -> g(k)
      - delta(k) = g(k) * sqrt(P(k))
      - phi(k) = -delta(k) / k^2
      - s(k) = i k phi(k)
      - inverse FFT -> s(x)
      - rescale to target disp_rms
    """
    rng = np.random.default_rng(int(seed))
    g = rng.normal(size=(ng, ng, ng)).astype(np.float64)

    dx = box / ng
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

    # Displacement in Fourier space: s = grad(phi)
    sx = np.fft.ifftn(1j * kx * phik).real
    sy = np.fft.ifftn(1j * ky * phik).real
    sz = np.fft.ifftn(1j * kz * phik).real
    s = np.stack([sx, sy, sz], axis=-1)

    # Rescale to a desired RMS (physical units).
    rms = float(np.sqrt(np.mean(np.sum(s * s, axis=-1))))
    if rms > 0:
        s = s * (float(disp_rms) / rms)
    return s.astype(np.float32)


def _particles_phys(state: qz.NBodyState, def_field: jnp.ndarray, dx: float, box: float) -> jnp.ndarray:
    """Map particle mesh coords xi -> physical coords x = xi*dx + grad(def)(xi)."""
    xi = state.xv[:, 0:3]  # [0,ng) mesh units
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


def _mesh_centers_phys(def_field: jnp.ndarray, dx: float, box: float) -> jnp.ndarray:
    """Physical coordinates of mesh cell centers: x = xi + grad(def)."""
    ng = def_field.shape[0]
    coords = (jnp.arange(ng, dtype=def_field.dtype) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing="ij")
    grad_def = qz.grad_central(def_field, dx)
    x = jnp.stack([X, Y, Z], axis=-1) + grad_def
    return jnp.mod(x, box)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ng", type=int, default=64, help="mesh size (paper uses 64)")
    ap.add_argument("--npart-side", type=int, default=32, help="particles per side (paper uses 32)")
    ap.add_argument("--box", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lattice-random-shift",
        action="store_true",
        help="Randomly shift the initial particle lattice by a fraction of a cell to avoid grid-aligned artifacts.",
    )
    ap.add_argument(
        "--lattice-jitter",
        type=float,
        default=0.0,
        help="Add random jitter to the initial lattice, as a fraction of the lattice spacing (helps remove grid pattern in projections).",
    )

    # IC (GRF -> Zel'dovich-like displacement)
    ap.add_argument("--pk-n", type=float, default=-1.0)
    ap.add_argument("--pk-k0", type=float, default=6.0)
    ap.add_argument("--pk-kcut", type=float, default=30.0)
    ap.add_argument("--disp-rms", type=float, default=0.03, help="RMS initial displacement (box units)")
    ap.add_argument("--vel-fac", type=float, default=0.0, help="velocity = vel_fac * displacement (toy)")
    ap.add_argument(
        "--pmass",
        type=float,
        default=1.0,
        help="Particle mass (QZOOM-like scaling is O(1); do not normalize by Np).",
    )

    # Evolution
    ap.add_argument("--ntotal", type=int, default=200, help="total steps (paper uses 400)")
    ap.add_argument("--dt-max", type=float, default=2e-3)
    ap.add_argument("--dt-min", type=float, default=1e-5)
    ap.add_argument("--cfl-mesh", type=float, default=0.8, help="safety factor in eq. 12 style mesh dt limit")
    ap.add_argument("--mg-cycles", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--smooth-steps", type=int, default=2)
    ap.add_argument("--a", type=float, default=1.0, help="Acceleration scale factor (toy; multiplies dv/dt).")

    # Limiter (recommended)
    ap.add_argument("--limiter", action="store_true", help="enable mesh limiter")
    ap.add_argument(
        "--limiter-mode",
        type=str,
        default="both",
        choices=["both", "source", "post"],
        help="Limiter strategy: 'source' adds a QZOOM-style repulsive correction inside defp; "
        "'post' only does post-solve scaling; 'both' uses source-term with post-solve backstop.",
    )
    ap.add_argument("--compressmax", type=float, default=100.0)
    ap.add_argument("--skewmax", type=float, default=50.0)
    ap.add_argument("--xhigh", type=float, default=2.0)
    ap.add_argument("--relax-steps", type=float, default=30.0)
    ap.add_argument("--hard-strength", type=float, default=1.0, help="Strength of the source-term compression limiter.")

    # Plotting
    ap.add_argument("--mesh-stride", type=int, default=1)
    ap.add_argument("--mesh-z", type=int, default=None, help="Fixed z-index for the mesh layer plots (0..ng-1).")
    ap.add_argument(
        "--mesh-exaggerate",
        type=float,
        default=1.0,
        help="Visual-only exaggeration factor for mesh displacement in the Fig.7-like panel.",
    )
    ap.add_argument("--mesh-quiver", action="store_true", help="Overlay displacement vectors in the mesh panel.")
    ap.add_argument("--max-plot", type=int, default=40000, help="max particles to scatter in projection")
    ap.add_argument("--out", type=str, default="qzoom_fig6_fig7.png")
    ap.add_argument(
        "--mesh-phys-plot",
        action="store_true",
        help="Also write a second figure comparing particle positions in physical vs moving-mesh coordinates.",
    )
    ap.add_argument("--animate", action="store_true", help="Write an animated GIF showing mesh distortion + particles.")
    ap.add_argument("--anim-every", type=int, default=10, help="Capture a frame every N steps.")
    ap.add_argument("--anim-fps", type=int, default=8, help="Frames per second for the GIF.")
    ap.add_argument(
        "--anim-out",
        type=str,
        default=None,
        help="Output path for animation (GIF). Default: <out_stem>_anim.gif",
    )
    args = ap.parse_args()

    ng = int(args.ng)
    nps = int(args.npart_side)
    npart = nps**3
    box = float(args.box)
    dx = box / ng

    # --- Initial conditions ---
    s = gaussian_ic_displacement(
        ng,
        box,
        seed=int(args.seed),
        n=float(args.pk_n),
        k0=float(args.pk_k0),
        kcut=float(args.pk_kcut),
        disp_rms=float(args.disp_rms) * box,
    )

    # Lagrangian particle positions q on an nps^3 lattice.
    #
    # Important for "Fig.6-like" projections: if ng is commensurate with nps,
    # the lattice points can land exactly on mesh nodes (e.g. ng=32, nps=16 ->
    # xi are odd integers). Then the GRF displacement is sampled at grid points
    # and the x-y projection can show a faint grid/moire structure at early
    # times. A random lattice shift and/or sub-cell jitter helps.
    rng_lat = np.random.default_rng(int(args.seed) + 12345)
    shift = rng_lat.random(3) if bool(args.lattice_random_shift) else np.zeros(3)
    jitter = float(args.lattice_jitter)
    if jitter < 0.0:
        raise ValueError("--lattice-jitter must be >= 0")

    ii, jj, kk = np.meshgrid(np.arange(nps), np.arange(nps), np.arange(nps), indexing="ij")
    q = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
    if jitter > 0.0:
        q = q + jitter * rng_lat.normal(size=q.shape)
    q = (q + 0.5 + shift) * (box / nps)
    q = np.mod(q, box).astype(np.float32)

    # Convert q to mesh coords and read out displacement s(q) via CIC.
    xi_q = (q / dx) % ng
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

    xi0 = (xi_q_j + disp_q / dx) % ng
    v0 = (float(args.vel_fac) * disp_q).astype(jnp.float32)
    xv0 = jnp.concatenate([xi0, v0], axis=1)
    # Important: for QZOOM-like scaling, do NOT normalize pmass by Np.
    # QZOOM typically uses O(1) masses and achieves <rho>~O(0.1..1) depending on Np/ng^3.
    pmass = jnp.full((npart,), float(args.pmass), dtype=jnp.float32)
    state = qz.NBodyState(xv=xv0, pmass=pmass)

    def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    levels = int(np.log2(ng) - 1)
    mg_params = qz.MGParams(levels=levels, v1=2, v2=2, mu=1, cycles=int(args.mg_cycles))
    params = qz.APMParams(ng=ng, box=box, a=float(args.a))

    limiter = qz.LimiterParams(enabled=False)
    densinit = None
    if bool(args.limiter):
        use_source = str(args.limiter_mode) in ("both", "source")
        use_post = str(args.limiter_mode) in ("both", "post")
        limiter = qz.LimiterParams(
            enabled=True,
            compressmax=float(args.compressmax),
            skewmax=float(args.skewmax),
            xhigh=float(args.xhigh),
            relax_steps=float(args.relax_steps),
            use_source_term=bool(use_source),
            use_post_scale=bool(use_post),
            post_only_on_fail=True,
            hard_strength=float(args.hard_strength),
        )
        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def_field, state, dx=dx)
        densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))

    # --- Time stepping ---
    dt = float(args.dt_max)
    dtold = dt
    dt_hist = []
    sgmin_hist = []
    dispmax_hist = []
    defp_rms_hist = []
    limscale_hist = []

    # Animation snapshots (optional): store lightweight 2D products per frame.
    anim_every = max(1, int(args.anim_every))
    anim_frames = []
    if bool(args.animate):
        z_anim = int(args.mesh_z) if args.mesh_z is not None else (ng // 2)
        z_anim = int(np.clip(z_anim, 0, ng - 1))

        def _capture_frame(step: int):
            x_phys_f = np.array(_particles_phys(state, def_field, dx, box), dtype=np.float64)
            xi_f = np.array(state.xv[:, 0:3], dtype=np.float64)
            x_mesh_f = np.mod(xi_f * dx, box)
            mesh_phys_f = np.array(_mesh_centers_phys(def_field, dx, box), dtype=np.float64)
            mesh_xy_f = mesh_phys_f[:: int(args.mesh_stride), :: int(args.mesh_stride), z_anim, 0:2]
            anim_frames.append(
                {
                    "step": int(step),
                    "x_phys_xy": x_phys_f[:, 0:2],
                    "x_mesh_xy": x_mesh_f[:, 0:2],
                    "mesh_xy": mesh_xy_f,
                }
            )

        _capture_frame(0)

    for it in range(1, int(args.ntotal) + 1):
        state, def_field, defp_field, phi, diag = qz.step_nbody_apm(
            state,
            def_field,
            defp_field,
            params=params,
            mg_params=mg_params,
            kappa=float(args.kappa),
            smooth_steps=int(args.smooth_steps),
            limiter=limiter,
            densinit=densinit,
            dt=dt,
            dtold=dtold,
        )

        dt_hist.append(dt)
        sgmin_hist.append(float(diag.get("sqrt_g_new_min", diag["sqrt_g_min"])))
        dispmax_hist.append(float(diag.get("disp_max", np.nan)))
        defp_rms = float(jnp.sqrt(jnp.mean(defp_field * defp_field)))
        defp_rms_hist.append(defp_rms)
        # Global limiter reports limiter_scale; local limiter reports limiter_scale_min.
        limscale_hist.append(float(diag.get("limiter_scale_min", diag.get("limiter_scale", np.nan))))

        # Mesh dt limit (Pen 1995 eq. 12 style): dt < cfl/(5*max|Hess(defp)|).
        # Use a cheap bound (Gershgorin) implemented in qzoom_nbody_flow.
        dt_mesh, max_hess = qz.mesh_dt_from_defp(defp_field, dx, safety=float(args.cfl_mesh))
        dt_next = min(float(args.dt_max), float(dt_mesh))
        dt_next = max(float(args.dt_min), dt_next)
        dtold = dt
        dt = dt_next

        if it % 25 == 0 or it == 1:
            meshlim_str = f"{float(dt_mesh):.3e}"
            if not np.isfinite(float(dt_mesh)):
                meshlim_str = "inf(defp~const)"
            print(
                f"[step {it:04d}/{int(args.ntotal)}] "
                f"dt={dt:.3e} (meshlim={meshlim_str}, max|H|~{float(max_hess):.3e}) "
                f"sqrt_g_min={sgmin_hist[-1]:.3e} disp_max={dispmax_hist[-1]:.3e} "
                f"(disp_max/dx={dispmax_hist[-1]/dx:.2f}) defp_rms={defp_rms:.3e} "
                f"limscale_min={limscale_hist[-1]:.3e} frac_limited={float(diag.get('limiter_frac_limited', np.nan)):.3e}"
            )

        if bool(args.animate) and (it % anim_every == 0 or it == int(args.ntotal)):
            _capture_frame(it)

    # Ensure the last progress line is printed even for small ntotal.
    if int(args.ntotal) % 25 != 0:
        print(
            f"[step {int(args.ntotal):04d}/{int(args.ntotal)}] "
            f"dt={dt:.3e} sqrt_g_min={sgmin_hist[-1]:.3e} disp_max={dispmax_hist[-1]:.3e} "
            f"(disp_max/dx={dispmax_hist[-1]/dx:.2f}) defp_rms={defp_rms_hist[-1]:.3e} limscale={limscale_hist[-1]:.3e}"
        )

    # --- Final fields for plotting ---
    x_phys = np.array(_particles_phys(state, def_field, dx, box), dtype=np.float64)
    xi_final = np.array(state.xv[:, 0:3], dtype=np.float64)
    x_mesh = np.mod(xi_final * dx, box)

    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh, sqrt_g = qz.pcalcrho(rho0, def_field, state, dx=dx)
    rho_np = np.array(rho_mesh, dtype=np.float64)

    # Choose a plane through a clump: z index of densest mesh cell.
    idx = np.unravel_index(int(np.argmax(rho_np)), rho_np.shape)
    z0 = int(args.mesh_z) if args.mesh_z is not None else int(idx[2])
    z0 = int(np.clip(z0, 0, ng - 1))

    mesh_phys = np.array(_mesh_centers_phys(def_field, dx, box), dtype=np.float64)
    mesh_xy = mesh_phys[:: int(args.mesh_stride), :: int(args.mesh_stride), z0, 0:2]  # (Nx,Ny,2)

    # --- Plot ---
    import matplotlib.pyplot as plt

    out_path = Path(args.out)

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])

    # Fig.6-like: particle projection (x,y)
    ax0 = fig.add_subplot(gs[0, 0])
    nplot = min(int(args.max_plot), x_phys.shape[0])
    if nplot < x_phys.shape[0]:
        sel = np.random.default_rng(0).choice(x_phys.shape[0], size=nplot, replace=False)
        pts = x_phys[sel]
    else:
        pts = x_phys
    ax0.scatter(pts[:, 0], pts[:, 1], s=1.0, marker=".", color="k", alpha=0.8, linewidths=0.0)
    ax0.set_title("Fig.6-like: particle projection (physical x-y)")
    ax0.set_aspect("equal")
    ax0.set_xlim(0, box)
    ax0.set_ylim(0, box)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Diagnostic: time step + compression history
    ax1 = fig.add_subplot(gs[0, 1])
    t = np.arange(1, int(args.ntotal) + 1)
    ax1.plot(t, dt_hist, color="tab:blue", label="dt")
    ax1.set_yscale("log")
    ax1.set_xlabel("step")
    ax1.set_ylabel("dt")
    ax1b = ax1.twinx()
    ax1b.plot(t, sgmin_hist, color="tab:red", alpha=0.8, label="sqrt_g_min")
    ax1b.set_yscale("log")
    ax1b.set_ylabel("sqrt_g_min")
    ax1.set_title("Adaptive dt + mesh compression history")
    ax1.grid(True, alpha=0.2)

    # Fig.7-like: mesh layer through a clump
    ax2 = fig.add_subplot(gs[1, 0])
    ex = float(args.mesh_exaggerate)
    ax2.set_title(f"Fig.7-like: mesh layer at z-index {z0} (exaggerate x{ex:g})")
    ax2.set_aspect("equal")

    # Uniform mesh (thin)
    coords = (np.arange(ng) + 0.5) * dx
    uu = coords[:: int(args.mesh_stride)]
    Ux, Uy = np.meshgrid(uu, uu, indexing="ij")

    # Plot uniform grid lines (light)
    for i in range(Ux.shape[0]):
        ax2.plot(Ux[i, :], Uy[i, :], color="0.80", lw=0.6, alpha=0.8, zorder=1)
        ax2.plot(Ux[:, i], Uy[:, i], color="0.80", lw=0.6, alpha=0.8, zorder=1)

    # Plot deformed mesh lines + periodic wrap-around images.
    # Keep a single consistent style so nothing appears "lighter" at the domain edges.
    # The true displacement can be <~ dx (visually subtle). Allow a plot-only
    # exaggeration factor using periodic-minimal displacements.
    xg0 = mesh_xy[..., 0]
    yg0 = mesh_xy[..., 1]
    dxy = np.stack([xg0, yg0], axis=-1) - np.stack([Ux, Uy], axis=-1)
    dxy = (dxy + 0.5 * box) % box - 0.5 * box
    xg = (Ux + ex * dxy[..., 0]) % box
    yg = (Uy + ex * dxy[..., 1]) % box

    shifts = [(0.0, 0.0), (-box, 0.0), (box, 0.0), (0.0, -box), (0.0, box)]
    for i in range(xg.shape[0]):
        xu = _unwrap_periodic_line(xg[i, :], box)
        yu = _unwrap_periodic_line(yg[i, :], box)
        for sx, sy in shifts:
            ax2.plot(
                xu + sx,
                yu + sy,
                color="k",
                lw=0.9,
                alpha=1.0,
                zorder=3,
            )
    for j in range(xg.shape[1]):
        xu = _unwrap_periodic_line(xg[:, j], box)
        yu = _unwrap_periodic_line(yg[:, j], box)
        for sx, sy in shifts:
            ax2.plot(
                xu + sx,
                yu + sy,
                color="k",
                lw=0.9,
                alpha=1.0,
                zorder=3,
            )

    if bool(args.mesh_quiver):
        ax2.quiver(
            Ux,
            Uy,
            ex * dxy[..., 0],
            ex * dxy[..., 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
            color="tab:red",
            alpha=0.5,
        )

    ax2.set_xlim(0, box)
    ax2.set_ylim(0, box)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Diagnostic: density slice (log10)
    ax3 = fig.add_subplot(gs[1, 1])
    sl = rho_np[:, :, z0]
    sl = np.maximum(sl, 1e-12)
    im = ax3.imshow(np.log10(sl).T, origin="lower", cmap="magma",vmin=0)
    ax3.set_title("log10(rho_mesh) slice at same z")
    ax3.set_xticks([])
    ax3.set_yticks([])
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.28])
    fig.colorbar(im, cax=cax, label="log10(rho)")

    fig.subplots_adjust(left=0.04, right=0.90, top=0.95, bottom=0.05, wspace=0.15, hspace=0.22)
    fig.savefig(out_path, dpi=180)
    print(f"[fig6/7 demo] wrote {out_path}")

    # Animation: mesh distortion + particle motion (fixed z slice for mesh).
    if bool(args.animate) and len(anim_frames) >= 2:
        from matplotlib.collections import LineCollection
        from matplotlib import animation

        out_anim = Path(args.anim_out) if args.anim_out is not None else out_path.with_name(out_path.stem + "_anim.gif")

        # Fixed uniform grid for the animated mesh layer (same stride as mesh_xy).
        coords = (np.arange(ng) + 0.5) * dx
        uu = coords[:: int(args.mesh_stride)]
        Ux, Uy = np.meshgrid(uu, uu, indexing="ij")

        # For consistent subsampling across frames.
        nplot_anim = min(int(args.max_plot), anim_frames[0]["x_phys_xy"].shape[0])
        if nplot_anim < anim_frames[0]["x_phys_xy"].shape[0]:
            sel_anim = 19 #np.random.default_rng(2).choice(anim_frames[0]["x_phys_xy"].shape[0], size=nplot_anim, replace=False)
        else:
            sel_anim = None

        figA, (axM, axP) = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5.8))

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

        # Uniform grid background (static, light).
        for i in range(Ux.shape[0]):
            axM.plot(Ux[i, :], Uy[i, :], color="0.88", lw=0.6, alpha=0.9, zorder=1)
            axM.plot(Ux[:, i], Uy[:, i], color="0.88", lw=0.6, alpha=0.9, zorder=1)

        # Deformed grid line collections (updated per-frame).
        # Keep a single consistent style for the deformed mesh, even for periodic images.
        lc_main = LineCollection([], colors="k", linewidths=0.9, alpha=1.0, zorder=3)
        lc_ref = LineCollection([], colors="k", linewidths=0.9, alpha=1.0, zorder=2)
        axM.add_collection(lc_ref)
        axM.add_collection(lc_main)

        # Particles on mesh panel (physical only, faint, to connect to clumps).
        scatM = axM.scatter([], [], s=1.0, marker=".", color="k", alpha=0.35, linewidths=0.0, zorder=4)

        # Particle comparison panel: physical vs mesh (two point clouds).
        scatP_phys = axP.scatter([], [], s=1.0, marker=".", color="k", alpha=0.75, linewidths=0.0, label="physical")
        scatP_mesh = axP.scatter([], [], s=1.0, marker=".", color="tab:orange", alpha=0.75, linewidths=0.0, label="mesh (xi*dx)")
        axP.legend(frameon=False, markerscale=6, loc="upper right")

        # Periodic reflections (same as static plot).
        shifts = [(0.0, 0.0), (-box, 0.0), (box, 0.0), (0.0, -box), (0.0, box)]

        def _update(frame_idx: int):
            fr = anim_frames[frame_idx]
            step = int(fr["step"])
            x_phys_xy = fr["x_phys_xy"]
            x_mesh_xy = fr["x_mesh_xy"]
            mesh_xy_f = fr["mesh_xy"]

            if sel_anim is not None:
                x_phys_xy = x_phys_xy[sel_anim]
                x_mesh_xy = x_mesh_xy[sel_anim]

            # Update particles
            scatM.set_offsets(x_phys_xy)
            scatP_phys.set_offsets(x_phys_xy)
            scatP_mesh.set_offsets(x_mesh_xy)

            # Update mesh lines (with optional exaggeration).
            ex = float(args.mesh_exaggerate)
            xg0 = mesh_xy_f[..., 0]
            yg0 = mesh_xy_f[..., 1]
            dxy = np.stack([xg0, yg0], axis=-1) - np.stack([Ux, Uy], axis=-1)
            dxy = (dxy + 0.5 * box) % box - 0.5 * box
            xg = (Ux + ex * dxy[..., 0]) % box
            yg = (Uy + ex * dxy[..., 1]) % box

            segs_main = _mesh_segments_xy(xg, yg, box)
            segs_ref_all = []
            for sx, sy in shifts:
                if (sx, sy) == (0.0, 0.0):
                    continue
                segs_ref_all.append(segs_main + np.array([sx, sy], dtype=np.float64)[None, None, :])
            segs_ref = np.concatenate(segs_ref_all, axis=0) if segs_ref_all else np.zeros((0, 2, 2), dtype=np.float64)

            lc_main.set_segments(segs_main)
            lc_ref.set_segments(segs_ref)

            figA.suptitle(f"QZOOM-like moving mesh: frame {frame_idx+1}/{len(anim_frames)} (step={step})", y=0.98)
            return lc_main, lc_ref, scatM, scatP_phys, scatP_mesh

        ani = animation.FuncAnimation(figA, _update, frames=len(anim_frames), interval=1000 // max(1, int(args.anim_fps)), blit=False)
        ani.save(out_anim, writer=animation.PillowWriter(fps=int(args.anim_fps)))
        plt.close(figA)
        print(f"[fig6/7 demo] wrote {out_anim}")

    # Optional: a second figure to show how the moving-mesh frame "diffuses" clumps.
    if bool(args.mesh_phys_plot):
        fig2, axes = plt.subplots(nrows=1, ncols=3, figsize=(14.0, 4.8))

        nplot2 = min(int(args.max_plot), x_phys.shape[0])
        if nplot2 < x_phys.shape[0]:
            sel2 = np.random.default_rng(1).choice(x_phys.shape[0], size=nplot2, replace=False)
            pts_phys = x_phys[sel2]
            pts_mesh = x_mesh[sel2]
        else:
            pts_phys = x_phys
            pts_mesh = x_mesh

        axes[0].scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.5, marker=".", color="k", alpha=1.0, linewidths=0.0)
        axes[0].set_title("Particles (physical x-y)")
        axes[0].set_aspect("equal")
        axes[0].set_xlim(0, box)
        axes[0].set_ylim(0, box)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].scatter(pts_mesh[:, 0], pts_mesh[:, 1], s=1.5, marker=".", color="tab:green",  alpha=1.0, linewidths=0.0)
        axes[1].set_title("Particles (moving-mesh x-y = xi*dx)")
        axes[1].set_aspect("equal")
        axes[1].set_xlim(0, box)
        axes[1].set_ylim(0, box)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].scatter(pts_phys[:, 0], pts_phys[:, 1], s=1.5, marker=".", color="k",  alpha=1.0, linewidths=0.0, label="physical")
        axes[2].scatter(pts_mesh[:, 0], pts_mesh[:, 1], s=1.5, marker=".", color="tab:orange",  alpha=1.0, linewidths=0.0, label="mesh")
        axes[2].set_title("Overlay (physical vs mesh)")
        axes[2].set_aspect("equal")
        axes[2].set_xlim(0, box)
        axes[2].set_ylim(0, box)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].legend(frameon=False, markerscale=5, loc="upper right")

        fig2.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.06, wspace=0.10)
        out2 = out_path.with_name(out_path.stem + "_mesh_vs_phys.png")
        fig2.savefig(out2, dpi=180)
        plt.close(fig2)
        print(f"[fig6/7 demo] wrote {out2}")

    # Extra: store a small NPZ with final state for quick followup
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        x_phys=x_phys.astype(np.float32),
        x_mesh=x_mesh.astype(np.float32),
        xi=xi_final.astype(np.float32),
        rho_slice=sl.astype(np.float32),
        mesh_xy=mesh_xy.astype(np.float32),
        dt=np.array(dt_hist, dtype=np.float32),
        sqrt_g_min=np.array(sgmin_hist, dtype=np.float32),
        disp_max=np.array(dispmax_hist, dtype=np.float32),
        defp_rms=np.array(defp_rms_hist, dtype=np.float32),
        limiter_scale=np.array(limscale_hist, dtype=np.float32),
        z0=np.int32(z0),
    )
    print(f"[fig6/7 demo] wrote {npz_path}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
