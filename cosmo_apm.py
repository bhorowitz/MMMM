from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import jax_cosmo as jc

import qzoom_nbody_flow as qz
import multigrid_mesh2 as mg

from cosmology import CosmologyParams, canonical_kick_drift

Array = jnp.ndarray


@dataclass(frozen=True)
class CosmoStepConfig:
    """Configuration for cosmological stepping in mesh coordinates."""

    ng: int = 32
    force_scale: float = 0.0  # typically 1.5*Omega_m


class CosmoStepDiagnostics(NamedTuple):
    dtau: Array
    sqrt_g_new_min: Array
    phi_rms: Array
    phi_rel_res: Array
    disp_max: Array
    vmax: Array
    amax: Array
    K: Array
    U: Array
    E: Array


def _moving_mesh_push_cosmo_fortranish(
    state: qz.NBodyState,
    *,
    phi: Array,
    def_field_cur: Array,
    defn_field: Array,
    dtau: Array,
    tdt: Array,
    a_mid: Array,
    force_scale: float,
    dx: float,
) -> tuple[qz.NBodyState, Array, Array]:
    """Second-order moving-mesh push consistent with the JAX cosmology state.

    We keep the useful second-order moving-mesh structure from QZOOM, but avoid
    importing the full Fortran `gdt/gdtold` transport literally. In this JAX path
    the particle state stores canonical momentum `p=a*v`, so using the full
    combined interval everywhere tends to over-drag particles with the mesh.

    Current choices:
      - `dtau` controls the actual KDK momentum update.
      - `ddef_dt` is computed over the actual current step `dtau`, because
        `defn - def_field_cur` is the current-step deformation change.
      - mesh transport uses a moderated interval: current step plus half of the
        previous step, with a startup guard so the first step does not double.
      - we omit the explicit acceleration-drift term from `const_disp` because
        the canonical momentum kicks already account for that transport.
    """
    triad = qz.triad_from_def(def_field_cur, dx)
    Ainv = jnp.linalg.inv(triad)

    gphi = qz.grad_central(phi, dx)
    force_p_grid = -jnp.einsum("...ij,...j->...i", Ainv, gphi) * float(force_scale)

    prev_dt = jnp.maximum(tdt - dtau, 0.0)
    startup = jnp.max(jnp.abs(def_field_cur)) < 1e-12
    mesh_transport_dt = jnp.where(startup, dtau, dtau + 0.5 * prev_dt)

    # The deformation update defn-def_cur is a current-step quantity, so use dtau
    # rather than the full combined interval when estimating the mesh-rate field.
    ddef_dt = (defn_field - def_field_cur) / jnp.maximum(dtau, 1e-12)
    adot = qz.triad_from_def(ddef_dt, dx) - jnp.eye(3, dtype=def_field_cur.dtype)
    bdot = -jnp.einsum("...ik,...kl,...lj->...ij", Ainv, adot, Ainv)

    mesh_grad = -qz.grad_central(ddef_dt, dx)
    const_disp = (
        jnp.einsum("...ij,...j->...i", Ainv, mesh_grad) * mesh_transport_dt
        + 0.5 * jnp.einsum("...ij,...j->...i", bdot, mesh_grad) * (mesh_transport_dt ** 2)
    )
    vm = Ainv * mesh_transport_dt + 0.5 * bdot * (mesh_transport_dt ** 2)

    xi = state.xv[:, 0:3]
    p = state.xv[:, 3:6]

    force_p_part = qz.cic_readout_3d_multi(force_p_grid, xi)
    const_part = qz.cic_readout_3d_multi(const_disp, xi)
    vm_flat = vm.reshape(vm.shape[0:3] + (9,))
    vm_part = qz.cic_readout_3d_multi(vm_flat, xi).reshape((-1, 3, 3))

    # KDK kicks use dtau (individual step) — consistent with JAX p=a*v convention.
    p_half = p + force_p_part * (dtau / 2.0)
    u_half = p_half / (a_mid + 1e-12)
    xi_new = jnp.mod(xi + (const_part + jnp.einsum("nij,nj->ni", vm_part, u_half)) / dx, float(def_field_cur.shape[0]))
    p_new = p_half + force_p_part * (dtau / 2.0)

    out_state = qz.NBodyState(
        xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new),
        pmass=state.pmass,
    )
    return out_state, force_p_grid, Ainv


def _density_mesh(state: qz.NBodyState, def_field: Array, *, dx: float) -> tuple[Array, Array]:
    rho0 = jnp.zeros((def_field.shape[0],) * 3, dtype=jnp.float32)
    return qz.pcalcrho(rho0, def_field, state, dx=dx)


def step_static_pm_a(
    state: qz.NBodyState,
    *,
    cosmo: CosmologyParams,
    cfg: CosmoStepConfig,
    a_mid: float,
    da: float,
    dx: float = 1.0,
) -> tuple[qz.NBodyState, Array, Array]:
    """Static-mesh PM step integrating in scale factor `a` (jaxpm conventions).

    State convention:
      xv[:,0:3] = particle mesh coordinates ξ in [0,ng)
      xv[:,3:6] = canonical momentum p (jaxpm's "vel" state)
    """
    ng = int(cfg.ng)
    def0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    rho_mesh, _sqrt_g = _density_mesh(state, def0, dx=dx)
    rhs = rho_mesh - jnp.mean(rho_mesh)
    phi = qz.poisson_fft_uniform(rhs, dx=dx)

    gphi = qz.grad_central(phi, dx)
    forces_grid = -gphi * float(cfg.force_scale)
    forces_part = qz.cic_readout_3d_multi(forces_grid, state.xv[:, 0:3])

    dtau, kick, drift_x = canonical_kick_drift(cosmo=cosmo, a_mid=jnp.array(a_mid, dtype=jnp.float32), da=jnp.array(da, dtype=jnp.float32))
    dtau = float(dtau)
    kick = float(kick)
    drift_x = float(drift_x)

    p = state.xv[:, 3:6]
    xi = state.xv[:, 0:3]
    p_new = p + forces_part * kick
    xi_new = jnp.mod(xi + (p_new * drift_x) / float(dx), float(ng))
    out = qz.NBodyState(xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new), pmass=state.pmass)
    return out, phi, rho_mesh


def step_moving_mesh_a(
    state: qz.NBodyState,
    def_field: Array,
    defp_field: Array,
    *,
    cosmo: CosmologyParams,
    cfg: CosmoStepConfig,
    a_mid: float,
    da: float,
    mg_params: qz.MGParams,
    kappa: float,
    smooth_steps: int,
    limiter: qz.LimiterParams,
    densinit: Array,
    dtau_prev: float,
    dx: float = 1.0,
) -> tuple[qz.NBodyState, Array, Array, Array, float]:
    """Moving-mesh APM step integrating in scale factor `a` via conformal time τ.

    `dtau_prev`: previous individual conformal step for the limiter solve.
    """
    ng = int(cfg.ng)
    dtau, kick, _drift_x = canonical_kick_drift(cosmo=cosmo, a_mid=jnp.array(a_mid, dtype=jnp.float32), da=jnp.array(da, dtype=jnp.float32))
    dtau = float(dtau)

    rho_mesh, _sqrt_g = _density_mesh(state, def_field, dx=dx)
    rhs_phi = qz.rgzerosum(rho_mesh, def_field, dx)

    if float(kappa) == 0.0:
        phi = qz.poisson_fft_uniform(rhs_phi, dx=dx)
    else:
        phi0 = jnp.zeros_like(rhs_phi)
        phi = qz.multigrid(
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
            iopt=1,
            mg_params=mg_params,
        )

    defp = qz.calcdefp(
        defp_field,
        tmp=jnp.zeros_like(rho_mesh),
        tmp2=jnp.zeros_like(rho_mesh),
        def_field=def_field,
        u=rho_mesh[None, ...],
        dtold1=jnp.array(float(dtau_prev), dtype=jnp.float32),
        dtaumesh=jnp.array(float(dtau), dtype=jnp.float32),
        nfluid=1,
        dx=dx,
        kappa=float(kappa),
        smooth_steps=int(smooth_steps),
        densinit=densinit,
        limiter=limiter,
        mg_params=mg_params,
    )

    defn = qz.zerosum(def_field + float(dtau) * defp)
    tdt = dtau + dtau_prev  # Fortran's gdt = dt + dtold

    out_state, _force_p_grid, _ainv = _moving_mesh_push_cosmo_fortranish(
        state,
        phi=phi,
        def_field_cur=def_field,
        defn_field=defn,
        dtau=jnp.asarray(dtau, dtype=def_field.dtype),
        tdt=jnp.asarray(tdt, dtype=def_field.dtype),
        a_mid=jnp.asarray(a_mid, dtype=def_field.dtype),
        force_scale=float(cfg.force_scale),
        dx=dx,
    )
    return out_state, defn, defp, phi, float(dtau)


def make_step_moving_mesh_a_jit_with_diag(
    *,
    cosmo: CosmologyParams,
    cfg: CosmoStepConfig,
    mg_params: qz.MGParams,
    kappa: float,
    smooth_steps: int,
    limiter: qz.LimiterParams,
    dx: float = 1.0,
    gravity_solver: str = "fft",
) -> Any:
    """Return a jitted cosmological moving-mesh step with scalar diagnostics.

    This keeps the outer time loop in Python, but compiles the heavy MMMM step
    itself so callers can still do frame capture / logging / fail-fast checks.
    """
    ng = int(cfg.ng)
    dx_f = float(dx)
    force_scale_f = float(cfg.force_scale)
    kappa_f = float(kappa)
    smooth_steps_i = int(smooth_steps)
    limiter = limiter if limiter is not None else qz.LimiterParams(enabled=False)
    solve_poisson = qz._get_poisson_mg_solver(mg_params=mg_params, dx=dx_f)
    gravity_solver = str(gravity_solver).lower()
    if gravity_solver not in ("fft", "mg"):
        raise ValueError(f"Unsupported gravity_solver={gravity_solver!r}; expected 'fft' or 'mg'.")

    @jax.jit
    def step(
        state: qz.NBodyState,
        def_field: Array,
        defp_field: Array,
        a_mid: Array,
        da: Array,
        dtau_prev: Array,
        densinit: Array,
        compressmax: Array,
    ):
        """One moving-mesh step.

        `dtau_prev`: previous individual conformal step for the limiter solve.
        """
        a_mid = jnp.asarray(a_mid, dtype=def_field.dtype)
        da = jnp.asarray(da, dtype=def_field.dtype)
        dtau_prev = jnp.asarray(dtau_prev, dtype=def_field.dtype)

        dtau, kick, _drift_x = canonical_kick_drift(cosmo=cosmo, a_mid=a_mid, da=da)
        dtau = jnp.asarray(dtau, dtype=def_field.dtype)
        kick = jnp.asarray(kick, dtype=def_field.dtype)

        rho0 = jnp.zeros((ng, ng, ng), dtype=def_field.dtype)
        rho_mesh, sqrt_g = qz.pcalcrho(rho0, def_field, state, dx=dx_f)
        rhs_phi = qz.rgzerosum(rho_mesh, def_field, dx_f)

        def solve_mg(_: None) -> Array:
            phi0 = jnp.zeros_like(rhs_phi)
            return qz.zerosum(solve_poisson(rhs_phi, phi0, def_field))

        def solve_fft(_: None) -> Array:
            return qz.poisson_fft_uniform(rhs_phi, dx=dx_f)

        phi = solve_mg(None) if gravity_solver == "mg" else solve_fft(None)

        defp = qz.calcdefp(
            defp_field,
            tmp=jnp.zeros_like(rho_mesh),
            tmp2=jnp.zeros_like(rho_mesh),
            def_field=def_field,
            u=rho_mesh[None, ...],
            dtold1=dtau_prev,
            dtaumesh=dtau,
            nfluid=1,
            dx=dx_f,
            kappa=kappa_f,
            smooth_steps=smooth_steps_i,
            densinit=densinit,
            limiter=limiter,
            mg_params=mg_params,
            compressmax_override=compressmax,
        )

        defn = qz.zerosum(def_field + dtau * defp)
        tdt = dtau + dtau_prev  # Fortran's gdt = dt + dtold; used for mesh coupling
        out_state, force_p_grid, Ainv_old = _moving_mesh_push_cosmo_fortranish(
            state,
            phi=phi,
            def_field_cur=def_field,
            defn_field=defn,
            dtau=dtau,
            tdt=tdt,
            a_mid=a_mid,
            force_scale=force_scale_f,
            dx=dx_f,
        )

        triad_new = qz.triad_from_def(defn, dx_f)
        _, _, sqrt_g_new = qz.metric_from_triad(triad_new)
        sqrt_g_new_min = jnp.min(sqrt_g_new)

        disp = qz.grad_central(defn, dx_f)
        disp_mag = jnp.linalg.norm(disp, axis=-1)
        disp_max = jnp.max(disp_mag)

        phi_rms = jnp.sqrt(jnp.mean(phi * phi))
        phi_res = mg.apply_poisson(phi, dx_f, def_field) - rhs_phi
        phi_rel_res = jnp.linalg.norm(phi_res) / (jnp.linalg.norm(rhs_phi) + 1e-12)
        vmax = jnp.max(jnp.linalg.norm(out_state.xv[:, 3:6], axis=-1) / (a_mid + 1e-12))
        amax = jnp.max(jnp.linalg.norm(force_p_grid, axis=-1) / (a_mid + 1e-12))

        K = 0.5 * jnp.sum(out_state.pmass * jnp.sum(out_state.xv[:, 3:6] ** 2, axis=-1))
        U = 0.5 * (dx_f ** 3) * jnp.sum(rhs_phi * phi * sqrt_g)
        E = K + U

        diag = CosmoStepDiagnostics(
            dtau=dtau,
            sqrt_g_new_min=sqrt_g_new_min,
            phi_rms=phi_rms,
            phi_rel_res=phi_rel_res,
            disp_max=disp_max,
            vmax=vmax,
            amax=amax,
            K=K,
            U=U,
            E=E,
        )
        return out_state, defn, defp, phi, dtau, diag

    return step
