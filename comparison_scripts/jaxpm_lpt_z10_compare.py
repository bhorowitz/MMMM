#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

import qzoom_nbody_flow as qz
from test_jaxpm_grf_compare import (
    _density_mesh_from_positions,
    _make_lattice,
    _moving_particles_phys,
    lpt_1_fixed,
    pm_forces_jaxpm_fixed,
    run_repo_moving_mesh_a,
)


Array = jnp.ndarray


def _project_density(rho: Array, axis: int) -> np.ndarray:
    return np.asarray(jnp.sum(rho, axis=int(axis)), dtype=np.float64)


def _deposit_to_density_grid(pos: Array, pmass: Array, *, mesh_ng: int, box_ng: float) -> Array:
    xi = jnp.mod(pos, float(box_ng)) * (float(mesh_ng) / float(box_ng))
    mesh = jnp.zeros((mesh_ng, mesh_ng, mesh_ng), dtype=jnp.float32)
    return qz.cic_deposit_3d(mesh, xi, pmass)


def _save_density_panel(
    out_path: Path,
    *,
    proj_qzoom: np.ndarray,
    proj_jax: np.ndarray,
    axis: int,
    title: str,
    eps: float = 1e-6,
) -> None:
    q = np.log10(np.maximum(proj_qzoom, 0.0) + float(eps))
    j = np.log10(np.maximum(proj_jax, 0.0) + float(eps))
    d = q - j

    vv = np.concatenate([q.reshape(-1), j.reshape(-1)])
    vmin = float(np.percentile(vv, 1.0))
    vmax = float(np.percentile(vv, 99.0))
    dv = max(float(np.percentile(np.abs(d).reshape(-1), 99.0)), 1e-6)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle(title)

    im0 = ax[0].imshow(q.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[0].set_title("QZOOM moving mesh")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(j.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax[1].set_title("JaxPM PM")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(d.T, origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv)
    ax[2].set_title("QZOOM - JaxPM")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    for axy in ax:
        axy.set_xticks([])
        axy.set_yticks([])
        axy.set_xlabel(f"project axis {axis}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_particle_panel(
    out_path: Path,
    *,
    pos_qzoom: np.ndarray,
    pos_jax: np.ndarray,
    ng: int,
    axis: int,
    title: str,
    max_points: int = 12000,
) -> None:
    keep = [0, 1, 2]
    keep.pop(int(axis))
    rng = np.random.default_rng(0)

    def sample(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx]

    q = sample(np.asarray(pos_qzoom, dtype=np.float64))
    j = sample(np.asarray(pos_jax, dtype=np.float64))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.0), sharex=True, sharey=True)
    fig.suptitle(title)

    ax[0].scatter(q[:, keep[0]], q[:, keep[1]], s=1.0, alpha=0.35, linewidths=0, color="#c25b2a")
    ax[0].set_title("QZOOM moving mesh")
    ax[1].scatter(j[:, keep[0]], j[:, keep[1]], s=1.0, alpha=0.35, linewidths=0, color="#205f8f")
    ax[1].set_title("JaxPM PM")

    for axy in ax:
        axy.set_xlim(0.0, float(ng))
        axy.set_ylim(0.0, float(ng))
        axy.set_aspect("equal")
        axy.set_xlabel(f"axis {keep[0]}")
        axy.set_ylabel(f"axis {keep[1]}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_cumulative_density_compare(
    out_path: Path,
    *,
    rho_qzoom: np.ndarray,
    rho_jax: np.ndarray,
    title: str,
) -> None:
    def curve(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = rho.reshape(-1).astype(np.float64)
        weights = vals / max(np.sum(vals), 1e-30)
        order = np.argsort(vals)
        vals = vals[order]
        weights = weights[order]
        mass_above = 1.0 - np.cumsum(weights) + weights
        return np.maximum(vals, 1e-12), np.maximum(mass_above, 1e-12)

    xq, yq = curve(rho_qzoom)
    xj, yj = curve(rho_jax)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.loglog(xq, yq, color="#c25b2a", linewidth=2.0, label="QZOOM moving mesh")
    ax.loglog(xj, yj, color="#205f8f", linewidth=2.0, label="JaxPM PM")
    ax.set_title(title)
    ax.set_xlabel("density")
    ax.set_ylabel("mass fraction above density")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _evolve_jaxpm_fixed(
    *,
    pos0: Array,
    p0: Array,
    pmass: Array,
    ng: int,
    a0: float,
    afinal: float,
    steps: int,
    cosmo: jc.core.Cosmology,
) -> tuple[Array, Array]:
    pos = pos0
    p = p0
    da = (float(afinal) - float(a0)) / max(int(steps), 1)
    if da <= 0.0:
        raise ValueError("afinal must be > a0")

    for i in range(int(steps)):
        a_n = float(a0) + float(i) * da
        a_mid = a_n + 0.5 * da
        E = jnp.sqrt(jc.background.Esqr(cosmo, jnp.array(a_mid, dtype=jnp.float32)))

        rho = _density_mesh_from_positions(pos, pmass, ng=ng)
        delta_k = jnp.fft.rfftn(rho - jnp.mean(rho))
        forces = pm_forces_jaxpm_fixed(pos, mesh_shape=(ng, ng, ng), delta_k=delta_k) * (1.5 * float(cosmo.Omega_m))

        kick = float(da) / (float(a_mid) ** 2 * (float(E) + 1e-12))
        drift = float(da) / (float(a_mid) ** 3 * (float(E) + 1e-12))
        p = p + forces * kick
        pos = jnp.mod(pos + p * drift, float(ng))

    return pos, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ng", type=int, default=32)
    ap.add_argument("--npart-side", type=int, default=None)
    ap.add_argument("--box", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-init", type=float, default=10.0)
    ap.add_argument("--z-final", type=float, default=0.0)
    ap.add_argument("--omega-c", type=float, default=0.25)
    ap.add_argument("--sigma8", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--smooth-steps", type=int, default=2)
    ap.add_argument("--mg-cycles", type=int, default=10)
    ap.add_argument("--density-ng", type=int, default=256)
    ap.add_argument("--proj-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--outdir", type=str, default=str(ROOT / "QZOOM" / "tests" / "out" / "lpt_z10_compare"))
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    ng = int(args.ng)
    npart_side = int(args.npart_side) if args.npart_side is not None else ng
    a0 = 1.0 / (1.0 + float(args.z_init))
    afinal = 1.0 / (1.0 + float(args.z_final))
    if afinal <= a0:
        raise SystemExit("Need z-final < z-init so the scale factor grows forward in time.")

    cosmo = jc.Planck15(Omega_c=float(args.omega_c), sigma8=float(args.sigma8))
    kvals = jnp.logspace(-4, 1, 128, dtype=jnp.float32)
    pkvals = jc.power.linear_matter_power(cosmo, kvals)

    def pk_fn(x: Array) -> Array:
        return jnp.interp(x.reshape([-1]), kvals, pkvals).reshape(x.shape)

    import jaxpm.pm as jpm  # type: ignore

    mesh_shape = (ng, ng, ng)
    box_size = (float(args.box), float(args.box), float(args.box))
    init_mesh = jpm.linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(int(args.seed)))

    particles = _make_lattice(npart_side, ng, shift=0.0)
    pmass = jnp.ones((particles.shape[0],), dtype=jnp.float32)
    dx0, p0 = lpt_1_fixed(cosmo, init_mesh, particles, a0=a0)
    pos0 = jnp.mod(particles + dx0, float(ng))

    pos_jax, p_jax = _evolve_jaxpm_fixed(
        pos0=pos0,
        p0=p0,
        pmass=pmass,
        ng=ng,
        a0=a0,
        afinal=afinal,
        steps=int(args.steps),
        cosmo=cosmo,
    )

    state_init = qz.NBodyState(xv=jnp.concatenate([pos0, p0], axis=-1), pmass=pmass)
    levels = max(1, int(np.log2(ng) - 1))
    mg_params = qz.MGParams(levels=levels, v1=2, v2=2, mu=2, cycles=int(args.mg_cycles))
    limiter = qz.LimiterParams(
        enabled=True,
        compressmax=80.0,
        skewmax=40.0,
        xhigh=2.0,
        relax_steps=30.0,
        use_source_term=True,
        use_post_scale=True,
        post_only_on_fail=True,
        hard_strength=1.0,
    )

    rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    def0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def0, state_init, dx=1.0)
    densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))

    state_qzoom, def_field, defp_field, phi_field = run_repo_moving_mesh_a(
        state_init,
        ng=ng,
        steps=int(args.steps),
        a0=a0,
        afinal=afinal,
        cosmo=cosmo,
        mg_params=mg_params,
        kappa=float(args.kappa),
        smooth_steps=int(args.smooth_steps),
        limiter=limiter,
        densinit=densinit,
        force_scale=float(1.5 * float(cosmo.Omega_m)),
    )

    pos_qzoom_phys = _moving_particles_phys(state_qzoom, def_field, ng=ng, box=float(ng))
    density_ng = int(args.density_ng)
    rho_jax = _deposit_to_density_grid(pos_jax, pmass, mesh_ng=density_ng, box_ng=float(ng))
    rho_qzoom = _deposit_to_density_grid(pos_qzoom_phys, pmass, mesh_ng=density_ng, box_ng=float(ng))

    if not bool(jnp.all(jnp.isfinite(rho_jax))) or not bool(jnp.all(jnp.isfinite(rho_qzoom))):
        raise RuntimeError("Non-finite values detected in final binned density fields.")

    proj_jax = _project_density(rho_jax, axis=int(args.proj_axis))
    proj_qzoom = _project_density(rho_qzoom, axis=int(args.proj_axis))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _save_density_panel(
        out_path=outdir / "density_binned_z10_to_z0.png",
        proj_qzoom=proj_qzoom,
        proj_jax=proj_jax,
        axis=int(args.proj_axis),
        title=f"LPT ICs from native JaxPM: z={args.z_init:g} to z={args.z_final:g}, binned {density_ng}^3",
    )
    _save_particle_panel(
        out_path=outdir / "particles_raw_z10_to_z0.png",
        pos_qzoom=np.asarray(pos_qzoom_phys, dtype=np.float64),
        pos_jax=np.asarray(pos_jax, dtype=np.float64),
        ng=ng,
        axis=int(args.proj_axis),
        title=f"Raw particles at z={args.z_final:g} from shared LPT z={args.z_init:g} ICs",
    )
    _save_cumulative_density_compare(
        out_path=outdir / "fig8_style_cumulative_density_256.png",
        rho_qzoom=np.asarray(rho_qzoom, dtype=np.float64),
        rho_jax=np.asarray(rho_jax, dtype=np.float64),
        title=f"Fig. 8-style cumulative density, binned {density_ng}^3",
    )

    dq = (pos_qzoom_phys - np.asarray(pos_jax, dtype=np.float64) + 0.5 * ng) % ng - 0.5 * ng
    metrics = {
        "box": float(args.box),
        "seed": int(args.seed),
        "density_ng": density_ng,
        "ng": ng,
        "npart": int(particles.shape[0]),
        "z_init": float(args.z_init),
        "z_final": float(args.z_final),
        "a_init": float(a0),
        "a_final": float(afinal),
        "steps": int(args.steps),
        "kappa": float(args.kappa),
        "proj_axis": int(args.proj_axis),
        "rho_rel_l2": float(np.linalg.norm((proj_qzoom - proj_jax).ravel()) / (np.linalg.norm(proj_jax.ravel()) + 1e-12)),
        "particle_rms_over_dx": float(np.sqrt(np.mean(np.sum(dq * dq, axis=-1)))),
    }
    (outdir / "summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    if bool(args.save_npz):
        np.savez_compressed(
            outdir / "state_z10_to_z0.npz",
            box_ng=np.asarray([ng], dtype=np.float32),
            init_mesh=np.asarray(init_mesh, dtype=np.float32),
            particles=np.asarray(particles, dtype=np.float32),
            pos0=np.asarray(pos0, dtype=np.float32),
            p0=np.asarray(p0, dtype=np.float32),
            pos_jax=np.asarray(pos_jax, dtype=np.float32),
            p_jax=np.asarray(p_jax, dtype=np.float32),
            pos_qzoom_mesh=np.asarray(state_qzoom.xv[:, 0:3], dtype=np.float32),
            p_qzoom=np.asarray(state_qzoom.xv[:, 3:6], dtype=np.float32),
            pos_qzoom_phys=np.asarray(pos_qzoom_phys, dtype=np.float32),
            def_field=np.asarray(def_field, dtype=np.float32),
            defp_field=np.asarray(defp_field, dtype=np.float32),
            phi_field=np.asarray(phi_field, dtype=np.float32),
            rho_jax=np.asarray(rho_jax, dtype=np.float32),
            rho_qzoom=np.asarray(rho_qzoom, dtype=np.float32),
        )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[jaxpm_lpt_z10_compare] wrote {outdir}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
