#!/usr/bin/env python3
"""Compare jaxpm GRF+LPT+N-body evolution against this repo's mesh PM solvers.

What it does
------------
1) Uses jaxpm==0.0.2 + jax_cosmo to build a Gaussian random field (linear_field).
2) Computes 1LPT displacements and momenta (with a local fix for jaxpm's broken cic_read).
3) Evolves particles with jaxpm to late time.
   - default: fixed-step kick-drift in scale factor `a`
   - optional: `jax.experimental.ode.odeint`
4) Evolves the same LPT-initialized particles with this repo:
   - static mesh PM (def=0, defp=0)
   - moving mesh APM (kappa/limiter),
   then compares both to the jaxpm final snapshot.

Expectation
-----------
The repo static mesh should track the jaxpm result reasonably well, while the
moving mesh should deviate (by design). In the kappa->0 limit the moving-mesh
run should converge back to the static-mesh run.

This is a standalone test script (not pytest) to keep dependencies optional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional

import numpy as np

import jax
import jax.numpy as jnp

import jax_cosmo as jc
from jax.experimental.ode import odeint

import qzoom_nbody_flow as qz


Array = jnp.ndarray


def _parse_snapshots(s: str) -> Array:
    out = [float(x) for x in s.split(",") if x.strip()]
    if len(out) < 2:
        raise SystemExit("--snapshots must contain at least 2 comma-separated values (e.g. 0.1,0.5,1.0)")
    if any(not np.isfinite(x) or x <= 0.0 for x in out):
        raise SystemExit("--snapshots must be finite and >0")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise SystemExit("--snapshots must be strictly increasing")
    return jnp.array(out, dtype=jnp.float32)


def pm_forces_jaxpm_fixed(
    positions: Array,
    *,
    mesh_shape: Tuple[int, int, int],
    delta_k: Array | None = None,
    delta: Array | None = None,
    r_split: float = 0.0,
) -> Array:
    """jaxpm.pm.pm_forces replacement that avoids jaxpm==0.0.2 cic_read bug."""
    import jaxpm.pm as jpm  # type: ignore
    from jaxpm.painting import cic_paint  # type: ignore

    if delta_k is None:
        if delta is None:
            delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape, dtype=jnp.float32), positions))
        elif jnp.isrealobj(delta):
            delta_k = jnp.fft.rfftn(delta)
        else:
            delta_k = delta

    kvec = jpm.fftk(mesh_shape)
    pot_k = delta_k * jpm.invlaplace_kernel(kvec) * jpm.longrange_kernel(kvec, r_split=r_split)

    Fx = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 0) * pot_k, s=mesh_shape).real
    Fy = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 1) * pot_k, s=mesh_shape).real
    Fz = jnp.fft.irfftn(-jpm.gradient_kernel(kvec, 2) * pot_k, s=mesh_shape).real
    F_grid = jnp.stack([Fx, Fy, Fz], axis=-1)

    return qz.cic_readout_3d_multi(F_grid, positions)


def lpt_1_fixed(
    cosmo: jc.core.Cosmology,
    init_mesh: Array,
    positions: Array,
    *,
    a0: float,
) -> Tuple[Array, Array]:
    """1LPT displacement + momentum, mirroring jaxpm.pm.lpt(order=1), but fixed."""
    import jaxpm.pm as jpm  # type: ignore

    a = jnp.atleast_1d(jnp.array(a0, dtype=jnp.float32))
    E = jnp.sqrt(jc.background.Esqr(cosmo, a))
    delta_k = jnp.fft.rfftn(init_mesh)
    mesh_shape = init_mesh.shape

    init_force = pm_forces_jaxpm_fixed(positions, mesh_shape=mesh_shape, delta_k=delta_k)
    dx = jpm.growth_factor(cosmo, a) * init_force
    p = a**2 * jpm.growth_rate(cosmo, a) * E * dx

    # For scalar a0, the (1,) time axis broadcasts and dx/p come out as (Np, 3).
    return dx, p


def _make_lattice(npart_side: int, ng: int, *, shift: float = 0.0) -> Array:
    """Uniform lattice in mesh coordinates [0,ng)."""
    s = int(npart_side)
    if s <= 0:
        raise ValueError("npart_side must be >0")
    spacing = float(ng) / float(s)
    coords = (jnp.arange(s, dtype=jnp.float32) + float(shift)) * spacing
    grid = jnp.stack(jnp.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    return grid.reshape((-1, 3))


def _density_mesh_from_positions(pos: Array, pmass: Array, *, ng: int) -> Array:
    mesh = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    return qz.cic_deposit_3d(mesh, jnp.mod(pos, float(ng)), pmass)


def _density_phys_from_xphys(x_phys: Array, pmass: Array, *, ng: int, box: float) -> Array:
    """CIC mass-per-cell on a uniform physical grid from physical coordinates."""
    dx = float(box) / int(ng)
    xi = jnp.mod(x_phys / dx, float(ng))
    return _density_mesh_from_positions(xi, pmass, ng=ng)


def _moving_particles_phys(state: qz.NBodyState, def_field: Array, *, ng: int, box: float) -> Array:
    """Map particle mesh coords xi -> physical coords x = xi*dx + grad(def)(xi)."""
    dx = float(box) / int(ng)
    xi = state.xv[:, 0:3]
    grad_def = qz.grad_central(def_field, dx)
    disp = qz.cic_readout_3d_multi(grad_def, xi)
    x_phys = xi * dx + disp
    return jnp.mod(x_phys, float(box))


def _project_density(rho: Array, *, axis: int) -> np.ndarray:
    proj = jnp.sum(rho, axis=int(axis))
    return np.array(proj, dtype=np.float64)


def _save_projected_density_compare(
    *,
    out_path: Path,
    title: str,
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    label_a: str,
    label_b: str,
    eps: float = 1e-6,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plots") from e

    A = np.log10(np.maximum(proj_a, 0.0) + float(eps))
    B = np.log10(np.maximum(proj_b, 0.0) + float(eps))
    D = A - B

    vv = np.concatenate([A.reshape(-1), B.reshape(-1)])
    vmin = float(np.percentile(vv, 1.0))
    vmax = float(np.percentile(vv, 99.0))
    dv = float(np.percentile(np.abs(D).reshape(-1), 99.0))
    dv = max(dv, 1e-6)

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.4))
    fig.suptitle(title)

    im0 = ax[0].imshow(A.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[0].set_title(label_a)
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(B.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[1].set_title(label_b)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(D.T, origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv)
    ax[2].set_title(f"{label_a} - {label_b}")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axy in ax:
        axy.set_xticks([])
        axy.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_projected_density_triptych(
    *,
    out_path: Path,
    title: str,
    proj_ref: np.ndarray,
    proj_static: np.ndarray,
    proj_moving: np.ndarray,
    label_ref: str,
    label_static: str,
    label_moving: str,
    eps: float = 1e-6,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plots") from e

    R = np.log10(np.maximum(proj_ref, 0.0) + float(eps))
    S = np.log10(np.maximum(proj_static, 0.0) + float(eps))
    M = np.log10(np.maximum(proj_moving, 0.0) + float(eps))

    vv = np.concatenate([R.reshape(-1), S.reshape(-1), M.reshape(-1)])
    vmin = float(np.percentile(vv, 1.0))
    vmax = float(np.percentile(vv, 99.0))

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.4))
    fig.suptitle(title)

    im0 = ax[0].imshow(R.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[0].set_title(label_ref)
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(S.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[1].set_title(label_static)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(M.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[2].set_title(label_moving)
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axy in ax:
        axy.set_xticks([])
        axy.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _poisson_discrete(rhs: Array) -> Array:
    # Discrete 7-pt Laplacian inversion on mesh units (dx=1).
    phi = qz.poisson_fft_uniform(rhs, dx=1.0)
    return phi - jnp.mean(phi)


def forces_mesh_discrete(pos: Array, pmass: Array, *, ng: int, a_scale: float) -> Array:
    rho = _density_mesh_from_positions(pos, pmass, ng=ng)
    rhs = rho - jnp.mean(rho)
    phi = _poisson_discrete(rhs)
    gphi = qz.grad_central(phi, 1.0)  # d/dxi
    accel_grid = -gphi * float(a_scale)
    return qz.cic_readout_3d_multi(accel_grid, jnp.mod(pos, float(ng)))


def forces_jaxpm_kernels(pos: Array, pmass: Array, *, ng: int, omega_m: float) -> Array:
    # Build delta_k from CIC density; use jaxpm kernels; sample using our CIC readout.
    rho = _density_mesh_from_positions(pos, pmass, ng=ng)
    rhs = rho - jnp.mean(rho)
    delta_k = jnp.fft.rfftn(rhs)
    return pm_forces_jaxpm_fixed(pos, mesh_shape=(ng, ng, ng), delta_k=delta_k) * (1.5 * float(omega_m))


def make_nbody_ode(
    force_fn: Callable[[Array, Array, float, float], Array],
    *,
    pmass: Array,
    ng: int,
    cosmo: jc.core.Cosmology,
) -> Callable[[Tuple[Array, Array], Array], Tuple[Array, Array]]:
    """Return ODE RHS: dpos/da, dvel/da in the jaxpm conventions."""

    def rhs(state, a):
        pos, vel = state
        E = jnp.sqrt(jc.background.Esqr(cosmo, a))
        forces = force_fn(pos, pmass, float(ng), float(cosmo.Omega_m))  # already includes 1.5*Omega_m when desired
        dpos = vel / (a**3 * E)
        dvel = forces / (a**2 * E)
        return dpos, dvel

    return rhs


def _force_fn_jaxpm(pos: Array, pmass: Array, ng: float, omega_m: float) -> Array:
    return forces_jaxpm_kernels(pos, pmass, ng=int(ng), omega_m=float(omega_m))


def _force_fn_mesh(pos: Array, pmass: Array, ng: float, omega_m: float) -> Array:
    # Our discrete PM forces, with the same cosmology scaling as jaxpm's ODE.
    return forces_mesh_discrete(pos, pmass, ng=int(ng), a_scale=1.5 * float(omega_m))


def run_repo_static_mesh_a(
    state: qz.NBodyState,
    *,
    ng: int,
    steps: int,
    a0: float,
    afinal: float,
    cosmo: jc.core.Cosmology,
    force_scale: float,
) -> Tuple[qz.NBodyState, Array, Array]:
    """Evolve particles to `afinal` with a fixed-step kick-drift scheme in scale factor `a`.

    This uses the repo's static-mesh force solve (FFT Poisson + central gradient),
    but evolves particles in the same (pos, p) variable definitions as jaxpm's
    `make_ode_fn`: p is the canonical momentum appearing in:
        dpos/da = p/(a^3 E(a)),   dp/da = forces/(a^2 E(a)).
    """
    def0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    phi = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh = jnp.zeros((ng, ng, ng), dtype=jnp.float32)

    da = (float(afinal) - float(a0)) / max(int(steps), 1)
    if da <= 0.0:
        raise ValueError("afinal must be > a0")

    for i in range(int(steps)):
        a_n = float(a0) + float(i) * da
        a_mid = a_n + 0.5 * da
        E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))

        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh, _ = qz.pcalcrho(rho0, def0, state, dx=1.0)
        rhs = rho_mesh - jnp.mean(rho_mesh)
        phi = qz.poisson_fft_uniform(rhs, dx=1.0)

        # Forces in mesh units (d/dxi), scaled like jaxpm (~1.5 * Omega_m by default).
        gphi = qz.grad_central(phi, 1.0)
        forces_grid = -gphi * float(force_scale)
        forces_part = qz.cic_readout_3d_multi(forces_grid, state.xv[:, 0:3])

        p = state.xv[:, 3:6]
        kick = float(da) / (float(a_mid) ** 2 * (E + 1e-12))
        drift = float(da) / (float(a_mid) ** 3 * (E + 1e-12))

        p_new = p + forces_part * kick
        xi_new = state.xv[:, 0:3] + p_new * drift
        xi_new = jnp.mod(xi_new, float(ng))
        state = qz.NBodyState(xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new), pmass=state.pmass)

    return state, phi, rho_mesh


def _parse_kappa_sweep(s: str) -> list[float]:
    vals: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise SystemExit("--kappa-sweep must contain at least one value, e.g. 0,0.1,1,20")
    return vals


def run_repo_moving_mesh_a(
    state: qz.NBodyState,
    *,
    ng: int,
    steps: int,
    a0: float,
    afinal: float,
    cosmo: jc.core.Cosmology,
    mg_params: qz.MGParams,
    kappa: float,
    smooth_steps: int,
    limiter: qz.LimiterParams,
    densinit: Array,
    force_scale: float,
    def0: Optional[Array] = None,
    defp0: Optional[Array] = None,
) -> Tuple[qz.NBodyState, Array, Array, Array]:
    """Evolve particles + mesh to `afinal` with fixed steps in scale factor `a`.

    Particle state stores canonical momentum `p` in `xv[:,3:6]` (jaxpm convention).
    Mesh evolution uses dt=da so defp is interpreted as d(def)/da.
    """
    if def0 is None:
        def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    else:
        def_field = def0
    if defp0 is None:
        defp_field = jnp.zeros_like(def_field)
    else:
        defp_field = defp0

    da = (float(afinal) - float(a0)) / max(int(steps), 1)
    if da <= 0.0:
        raise ValueError("afinal must be > a0")

    phi = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    # For limiter logic, dtold1 participates in the first-step clipping scale.
    # Use the first mid-point dτ to avoid overly aggressive defp updates.
    a_mid0 = float(a0) + 0.5 * da
    E0 = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid0, dtype=jnp.float32)))
    dtold = jnp.array(float(da) / (a_mid0**2 * (float(E0) + 1e-12)), dtype=jnp.float32)

    for i in range(int(steps)):
        a_n = float(a0) + float(i) * da
        a_mid = a_n + 0.5 * da
        E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))
        dtau = float(da) / (float(a_mid) ** 2 * (float(E) + 1e-12))  # dτ = da/(a^2 E)

        # --- Fields: rho -> phi, rho -> defp, def -> defn ---
        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh, _sqrt_g = qz.pcalcrho(rho0, def_field, state, dx=1.0)

        rhs_phi = qz.rgzerosum(rho_mesh, def_field, 1.0)
        if float(kappa) == 0.0:
            # In the kappa->0 limit the mesh stays uniform (def=0, sqrt_g=1), so the
            # standard uniform-mesh FFT Poisson solve matches the static-mesh run.
            phi = qz.poisson_fft_uniform(rhs_phi, dx=1.0)
        else:
            phi0 = jnp.zeros_like(rhs_phi)
            phi = qz.multigrid(
                phi0,
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

        defp = qz.calcdefp(
            defp_field,
            tmp=jnp.zeros_like(rho_mesh),
            tmp2=jnp.zeros_like(rho_mesh),
            def_field=def_field,
            u=rho_mesh[None, ...],
            dtold1=dtold,
            dtaumesh=jnp.array(dtau, dtype=jnp.float32),
            nfluid=1,
            dx=1.0,
            kappa=float(kappa),
            smooth_steps=int(smooth_steps),
            densinit=densinit,
            limiter=limiter,
            mg_params=mg_params,
        )

        # Evolve def in the same "time" variable used by calcxvdot/pushparticle, i.e.
        # interpret defp as d(def)/dτ and use dτ = da/(a^2 E).
        defn = qz.zerosum(def_field + float(dtau) * defp)

        # --- Particle update in (xi, p) with kick-drift factors ---
        xvdot = qz.calcxvdot(phi, def_field, defn, float(dtau), float(force_scale), dx=1.0)
        xi = state.xv[:, 0:3]
        p = state.xv[:, 3:6]

        # Acceleration/force at particles (mesh units).
        forces_part = qz.cic_readout_3d_multi(xvdot["accel"], xi)
        vgrid_part = qz.cic_readout_3d_multi(xvdot["vgrid"], xi)

        Ainv = xvdot["Ainv"]
        Ainv_flat = Ainv.reshape(Ainv.shape[0:3] + (9,))
        Ainv_p = qz.cic_readout_3d_multi(Ainv_flat, xi).reshape((-1, 3, 3))

        kick = dtau
        drift = float(dtau) / float(a_mid)  # da/(a^3 E) = (da/(a^2 E))/a = dtau/a

        p_new = p + forces_part * float(kick)
        # Mesh-coordinate drift in τ with physical velocity v = dx/dτ = p/a.
        v_phys = p_new / float(a_mid)
        xi_dot = jnp.einsum("nij,nj->ni", Ainv_p, (v_phys - vgrid_part))
        xi_new = jnp.mod(xi + float(kick) * xi_dot, float(ng))

        state = qz.NBodyState(xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new), pmass=state.pmass)

        def_field = defn
        defp_field = defp
        dtold = jnp.array(dtau, dtype=jnp.float32)

    return state, def_field, defp_field, phi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ng", type=int, default=32)
    ap.add_argument("--npart-side", type=int, default=None, help="defaults to ng (one particle per cell)")
    ap.add_argument("--box", type=float, default=1.0, help="only affects the GRF power spectrum scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=str, default="0.1,1.0")
    ap.add_argument("--omega-c", type=float, default=0.25)
    ap.add_argument("--sigma8", type=float, default=0.8)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--jaxpm-mode", type=str, default="fixed", choices=["fixed", "odeint"], help="How to evolve jaxpm to late time.")
    ap.add_argument("--jaxpm-steps", type=int, default=None, help="Fixed-step count for --jaxpm-mode fixed (defaults to max(repo steps)).")
    ap.add_argument(
        "--vel-mode",
        type=str,
        default="p",
        choices=["p", "v_from_p"],
        help="How to feed LPT momentum `p` into the repo integrators: raw `p`, or convert to physical v ~ dx*p/(a^3 E) at a0.",
    )
    # Moving-mesh APM run (this repo)
    ap.add_argument("--moving-steps", type=int, default=256, help="number of steps in scale factor a for the repo moving-mesh run")
    ap.add_argument("--moving-dt", type=float, default=2e-3, help="(deprecated in this script) kept for compatibility; a-step integrator is used.")
    ap.add_argument("--static-steps", type=int, default=None, help="static-mesh a-steps; defaults to --moving-steps")
    ap.add_argument("--static-dt", type=float, default=None, help="(deprecated in this script) kept for compatibility.")
    ap.add_argument(
        "--kappa",
        type=float,
        default=0.1,
        help="moving-mesh defp strength (Pen/QZOOM-style). Large values (e.g. 20) may require many more steps and --limiter.",
    )
    ap.add_argument("--kappa-sweep", type=str, default=None, help="comma-separated kappa values; runs multiple moving-mesh comparisons")
    ap.add_argument("--smooth-steps", type=int, default=2, help="moving-mesh smoothing passes for defp RHS/field")
    ap.add_argument("--mg-cycles", type=int, default=10, help="moving-mesh multigrid V-cycle count per solve (use higher for convergence tests)")
    g_lim = ap.add_mutually_exclusive_group()
    g_lim.add_argument("--limiter", dest="limiter", action="store_true", help="Enable moving-mesh limiter.")
    g_lim.add_argument("--no-limiter", dest="limiter", action="store_false", help="Disable moving-mesh limiter.")
    ap.add_argument("--limiter-mode", type=str, default="both", choices=["both", "source", "post"])
    ap.add_argument("--compressmax", type=float, default=80.0)
    ap.add_argument("--skewmax", type=float, default=40.0)
    ap.add_argument("--xhigh", type=float, default=2.0)
    ap.add_argument("--relax-steps", type=float, default=30.0)
    ap.add_argument("--hard-strength", type=float, default=1.0)
    ap.add_argument("--a-scale", type=float, default=None, help="gravity scaling passed into the repo integrators (defaults to 1.5*Omega_m)")
    ap.add_argument("--outdir", type=str, default="./test_outputs")
    ap.add_argument("--plots", action="store_true", help="Write imshow comparison plots (projected density).")
    ap.add_argument("--proj-axis", type=int, default=2, choices=[0, 1, 2], help="Axis to project over for density plots.")
    ap.add_argument("--save-npz", action="store_true", help="Also write an .npz bundle with final fields/particles.")
    # For moving-mesh runs, the limiter is the primary guard against cell inversion
    # (NaNs from triad inversion). Keep it on by default for this comparison script.
    ap.set_defaults(limiter=True)
    args = ap.parse_args()

    ng = int(args.ng)
    npart_side = int(args.npart_side) if args.npart_side is not None else ng
    snapshots = _parse_snapshots(str(args.snapshots))
    a0 = float(snapshots[0])
    afinal = float(snapshots[-1])
    if afinal <= a0:
        raise SystemExit("Final snapshot must be > initial snapshot (e.g. --snapshots 0.1,1.0).")

    # jax_cosmo power spectrum function
    cosmo = jc.Planck15(Omega_c=float(args.omega_c), sigma8=float(args.sigma8))
    k = jnp.logspace(-4, 1, 128, dtype=jnp.float32)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x: Array) -> Array:
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Initial conditions: GRF on ng^3 mesh
    import jaxpm.pm as jpm  # type: ignore

    mesh_shape = (ng, ng, ng)
    box_size = (float(args.box), float(args.box), float(args.box))
    init_mesh = jpm.linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(int(args.seed)))

    # Particle lattice in mesh coordinates
    particles = _make_lattice(npart_side, ng, shift=0.0)
    pmass = jnp.ones((particles.shape[0],), dtype=jnp.float32)

    # LPT (1st order) at a0: positions + momenta
    dx0, p0 = lpt_1_fixed(cosmo, init_mesh, particles, a0=a0)
    pos0 = jnp.mod(particles + dx0, float(ng))

    if str(args.vel_mode) != "p":
        raise SystemExit("--vel-mode v_from_p is not supported in the linked-a integrator; use --vel-mode p.")

    # Evolve with ODEINT in scale factor a
    ode_jaxpm = make_nbody_ode(_force_fn_jaxpm, pmass=pmass, ng=ng, cosmo=cosmo)

    if str(args.jaxpm_mode) == "odeint":
        (pos_jaxpm, vel_jaxpm) = odeint(ode_jaxpm, (pos0, p0), snapshots, rtol=float(args.rtol), atol=float(args.atol))
    else:
        # Fixed-step integrator aligned with the repo runs for easier late-time comparison.
        nsteps = int(args.jaxpm_steps) if args.jaxpm_steps is not None else max(int(args.moving_steps), int(args.static_steps or 0) or int(args.moving_steps))

        snap_list = [float(x) for x in list(np.array(snapshots, dtype=np.float64))]
        total_da = float(snap_list[-1] - snap_list[0])
        if total_da <= 0.0:
            raise SystemExit("--snapshots must be strictly increasing.")

        # Proportional allocation with exact total step count (fix rounding on last interval).
        steps_by_interval = [
            max(1, int(round(nsteps * (snap_list[i + 1] - snap_list[i]) / total_da))) for i in range(len(snap_list) - 1)
        ]
        steps_by_interval[-1] = max(1, int(steps_by_interval[-1] + (nsteps - sum(steps_by_interval))))

        pos = pos0
        p = p0
        pos_out = [pos0]
        p_out = [p0]

        for (a_start, a_end, n_int) in zip(snap_list[:-1], snap_list[1:], steps_by_interval):
            da = (float(a_end) - float(a_start)) / max(int(n_int), 1)
            for i in range(int(n_int)):
                a_n = float(a_start) + float(i) * da
                a_mid = a_n + 0.5 * da
                E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))
                forces = _force_fn_jaxpm(pos, pmass, float(ng), float(cosmo.Omega_m))
                kick = float(da) / (float(a_mid) ** 2 * (float(E) + 1e-12))
                drift = float(da) / (float(a_mid) ** 3 * (float(E) + 1e-12))
                p = p + forces * kick
                pos = jnp.mod(pos + p * drift, float(ng))
            pos_out.append(pos)
            p_out.append(p)

        pos_jaxpm = jnp.stack(pos_out, axis=0)
        vel_jaxpm = jnp.stack(p_out, axis=0)

    # Comparisons
    def rel_l2(a: Array, b: Array) -> float:
        da = a - jnp.mean(a)
        db = b - jnp.mean(b)
        return float(jnp.linalg.norm(da - db) / (jnp.linalg.norm(da) + 1e-12))

    comp: Dict[str, object] = {}
    comp["_ng"] = float(ng)
    comp["_npart"] = float(pmass.shape[0])
    comp["_a0"] = float(a0)
    comp["_afinal"] = float(afinal)
    comp["_jaxpm_mode"] = float({"fixed": 0, "odeint": 1}[str(args.jaxpm_mode)])

    state_init = qz.NBodyState(xv=jnp.concatenate([pos0, p0], axis=-1), pmass=pmass)
    levels = max(1, int(np.log2(ng) - 1))
    mg_params = qz.MGParams(levels=levels, v1=2, v2=2, mu=2, cycles=int(args.mg_cycles))
    force_scale = float(args.a_scale) if args.a_scale is not None else float(1.5 * float(cosmo.Omega_m))

    use_source = str(args.limiter_mode) in ("both", "source")
    use_post = str(args.limiter_mode) in ("both", "post")
    limiter = qz.LimiterParams(
        enabled=bool(args.limiter),
        compressmax=float(args.compressmax),
        skewmax=float(args.skewmax),
        xhigh=float(args.xhigh),
        relax_steps=float(args.relax_steps),
        use_source_term=bool(use_source),
        use_post_scale=bool(use_post),
        post_only_on_fail=True,
        hard_strength=float(args.hard_strength),
    )

    # densinit is uniform mass-per-cell at the start (needed only if limiter is enabled)
    if limiter.enabled:
        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        def0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def0, state_init, dx=1.0)
        densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))
    else:
        densinit = jnp.zeros((ng, ng, ng), dtype=jnp.float32)

    # (3) Repo static mesh evolution (def=0; no mesh motion)
    static_steps = int(args.static_steps) if args.static_steps is not None else int(args.moving_steps)
    state_static, phi_static, rho_mesh_static = run_repo_static_mesh_a(
        state_init,
        ng=ng,
        steps=static_steps,
        a0=a0,
        afinal=afinal,
        cosmo=cosmo,
        force_scale=float(force_scale),
    )

    # (4) Repo moving mesh evolution (single kappa run, or sweep)
    kappa_values = [float(args.kappa)]
    if args.kappa_sweep is not None:
        kappa_values = _parse_kappa_sweep(args.kappa_sweep)

    moving_runs: Dict[float, Tuple[qz.NBodyState, Array, Array, Array]] = {}
    for kappa in kappa_values:
        state_k, def_k, defp_k, phi_k = run_repo_moving_mesh_a(
            state_init,
            ng=ng,
            steps=int(args.moving_steps),
            a0=a0,
            afinal=afinal,
            cosmo=cosmo,
            mg_params=mg_params,
            kappa=float(kappa),
            smooth_steps=int(args.smooth_steps),
            limiter=limiter,
            densinit=densinit,
            force_scale=float(force_scale),
        )
        moving_runs[float(kappa)] = (state_k, def_k, defp_k, phi_k)

    # Final comparisons vs jaxpm final snapshot in physical coordinates (mesh units; dx=1).
    x_j = jnp.mod(pos_jaxpm[-1], float(ng))
    x_s = jnp.mod(state_static.xv[:, 0:3], float(ng))
    rho_j = _density_mesh_from_positions(x_j, pmass, ng=ng)
    rho_s = _density_mesh_from_positions(x_s, pmass, ng=ng)

    d_s = (x_s - x_j + 0.5 * ng) % ng - 0.5 * ng
    comp["_a_scale_repo"] = float(force_scale)
    comp["_static_steps"] = float(static_steps)
    comp["repo_static_vs_jaxpm_pos_rms_over_dx"] = float(jnp.sqrt(jnp.mean(jnp.sum(d_s * d_s, axis=-1))))
    comp["repo_static_vs_jaxpm_rho_mesh_rel_l2"] = rel_l2(rho_s, rho_j)

    for kappa, (state_k, def_k, _defp_k, _phi_k) in moving_runs.items():
        x_m = _moving_particles_phys(state_k, def_k, ng=ng, box=float(ng))
        rho_m = _density_mesh_from_positions(x_m, pmass, ng=ng)
        d_m = (x_m - x_j + 0.5 * ng) % ng - 0.5 * ng
        comp[f"repo_moving_kappa{kappa:g}_vs_jaxpm_pos_rms_over_dx"] = float(jnp.sqrt(jnp.mean(jnp.sum(d_m * d_m, axis=-1))))
        comp[f"repo_moving_kappa{kappa:g}_vs_jaxpm_rho_mesh_rel_l2"] = rel_l2(rho_m, rho_j)
        comp[f"repo_moving_kappa{kappa:g}_vs_static_pos_rms_over_dx"] = float(jnp.sqrt(jnp.mean(jnp.sum(((x_m - x_s + 0.5 * ng) % ng - 0.5 * ng) ** 2, axis=-1))))
        comp[f"repo_moving_kappa{kappa:g}_vs_static_rho_mesh_rel_l2"] = rel_l2(rho_m, rho_s)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"jaxpm_grf_compare_ng{ng}_nps{npart_side}.json"
    out_path.write_text(json.dumps(comp, indent=2, sort_keys=True))

    if bool(args.plots):
        proj_j = _project_density(rho_j, axis=int(args.proj_axis))
        proj_s = _project_density(rho_s, axis=int(args.proj_axis))
        _save_projected_density_compare(
            out_path=outdir / "repo_static_vs_jaxpm_proj_final.png",
            title=f"Projected density (axis={int(args.proj_axis)}): repo static vs jaxpm final (a={afinal:g})",
            proj_a=proj_s,
            proj_b=proj_j,
            label_a=f"repo static (steps={static_steps})",
            label_b=f"jaxpm ({str(args.jaxpm_mode)})",
        )

        for kappa, (state_k, def_k, _defp_k, _phi_k) in moving_runs.items():
            x_m = _moving_particles_phys(state_k, def_k, ng=ng, box=float(ng))
            rho_m = _density_mesh_from_positions(x_m, pmass, ng=ng)
            proj_m = _project_density(rho_m, axis=int(args.proj_axis))
            out_name = (
                f"repo_moving_vs_jaxpm_proj_kappa{kappa:g}.png" if len(moving_runs) > 1 else "repo_moving_vs_jaxpm_proj_final.png"
            )
            _save_projected_density_compare(
                out_path=outdir / out_name,
                title=f"Projected density (axis={int(args.proj_axis)}): repo moving vs jaxpm final (kappa={kappa:g})",
                proj_a=proj_m,
                proj_b=proj_j,
                label_a=f"repo moving (steps={int(args.moving_steps)})",
                label_b=f"jaxpm ({str(args.jaxpm_mode)})",
            )

            if len(moving_runs) == 1:
                _save_projected_density_triptych(
                    out_path=outdir / "triptych_proj_final.png",
                    title=f"Projected density (axis={int(args.proj_axis)}), final (a={afinal:g})",
                    proj_ref=proj_j,
                    proj_static=proj_s,
                    proj_moving=proj_m,
                    label_ref=f"jaxpm ({str(args.jaxpm_mode)})",
                    label_static="repo static",
                    label_moving=f"repo moving (kappa={kappa:g})",
                )

    if bool(args.save_npz):
        rho_j_final = _density_mesh_from_positions(pos_jaxpm[-1], pmass, ng=ng)
        rho_static_final = _density_mesh_from_positions(state_static.xv[:, 0:3], pmass, ng=ng)
        k0 = sorted(moving_runs.keys())[0]
        state0, def_field, defp_field, phi = moving_runs[k0]
        x_m = _moving_particles_phys(state0, def_field, ng=ng, box=float(ng))
        rho_mov_final = _density_mesh_from_positions(x_m, pmass, ng=ng)
        npz_path = outdir / f"jaxpm_grf_compare_ng{ng}_nps{npart_side}.npz"
        np.savez_compressed(
            npz_path,
            init_mesh=np.array(init_mesh, dtype=np.float32),
            particles=np.array(particles, dtype=np.float32),
            pos0=np.array(pos0, dtype=np.float32),
            p0=np.array(p0, dtype=np.float32),
            snapshots=np.array(snapshots, dtype=np.float32),
            pos_jaxpm=np.array(pos_jaxpm, dtype=np.float32),
            vel_jaxpm=np.array(vel_jaxpm, dtype=np.float32),
            pos_static=np.array(state_static.xv[:, 0:3], dtype=np.float32),
            vel_static=np.array(state_static.xv[:, 3:6], dtype=np.float32),
            phi_static=np.array(phi_static, dtype=np.float32),
            pos_moving=np.array(state0.xv[:, 0:3], dtype=np.float32),
            vel_moving=np.array(state0.xv[:, 3:6], dtype=np.float32),
            def_field=np.array(def_field, dtype=np.float32),
            defp_field=np.array(defp_field, dtype=np.float32),
            phi_moving=np.array(phi, dtype=np.float32),
            rho_jaxpm_final=np.array(rho_j_final, dtype=np.float32),
            rho_static_final=np.array(rho_static_final, dtype=np.float32),
            rho_moving_final=np.array(rho_mov_final, dtype=np.float32),
        )
        print(f"[jaxpm_grf_compare] wrote {npz_path}")

    print(f"[jaxpm_grf_compare] wrote {out_path}")
    print(json.dumps(comp, indent=2, sort_keys=True))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
