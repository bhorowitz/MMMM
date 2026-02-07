from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp

import jax_cosmo as jc

import qzoom_nbody_flow as qz

from cosmology import CosmologyParams, canonical_kick_drift

Array = jnp.ndarray


@dataclass(frozen=True)
class CosmoStepConfig:
    """Configuration for cosmological stepping in mesh coordinates."""

    ng: int = 32
    force_scale: float = 0.0  # typically 1.5*Omega_m


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
    """Moving-mesh APM step integrating in scale factor `a` via conformal time τ."""
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

    xvdot = qz.calcxvdot(phi, def_field, defn, float(dtau), float(cfg.force_scale), dx=dx)
    xi = state.xv[:, 0:3]
    p = state.xv[:, 3:6]

    forces_part = qz.cic_readout_3d_multi(xvdot["accel"], xi)  # dp/dτ
    vgrid_part = qz.cic_readout_3d_multi(xvdot["vgrid"], xi)  # d/dτ (∇def)

    Ainv = xvdot["Ainv"]
    Ainv_flat = Ainv.reshape(Ainv.shape[0:3] + (9,))
    Ainv_p = qz.cic_readout_3d_multi(Ainv_flat, xi).reshape((-1, 3, 3))

    p_new = p + forces_part * float(kick)
    u_new = p_new / float(a_mid)  # dx/dτ
    xi_dot = jnp.einsum("nij,nj->ni", Ainv_p, (u_new - vgrid_part))
    xi_new = jnp.mod(xi + (float(dtau) / float(dx)) * xi_dot, float(ng))

    out_state = qz.NBodyState(xv=state.xv.at[:, 0:3].set(xi_new).at[:, 3:6].set(p_new), pmass=state.pmass)
    return out_state, defn, defp, phi, float(dtau)

