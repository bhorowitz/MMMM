"""Curvilinear multigrid Poisson solver with Jacobi smoother.

Mirrors the API/logic of diffAPM/multigrid_jacobi.py but uses a
Laplace–Beltrami operator built from the mesh deformation (def_field).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from organized_model import triad_from_def, metric_from_triad

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Core discrete operators (curvilinear, periodic)
# -----------------------------------------------------------------------------

class PoissonCoeffs(NamedTuple):
    sqrt_g: Array
    A_xx_xf: Array
    A_xy_xf: Array
    A_xz_xf: Array
    A_yy_yf: Array
    A_xy_yf: Array
    A_yz_yf: Array
    A_zz_zf: Array
    A_xz_zf: Array
    A_yz_zf: Array
    diag: Array


def build_poisson_coeffs(def_field: Array, h: float) -> PoissonCoeffs:
    """Precompute metric-weighted coefficients for repeated Poisson applications.

    This avoids recomputing triad/metric (and face-interpolated coefficients)
    on every Jacobi sweep, which is a large win on GPU.
    """
    triad = triad_from_def(def_field, h)
    _, ginv, sqrt_g = metric_from_triad(triad)

    A_xx = sqrt_g * ginv[..., 0, 0]
    A_xy = sqrt_g * ginv[..., 0, 1]
    A_xz = sqrt_g * ginv[..., 0, 2]
    A_yy = sqrt_g * ginv[..., 1, 1]
    A_yz = sqrt_g * ginv[..., 1, 2]
    A_zz = sqrt_g * ginv[..., 2, 2]

    A_xx_xf = interp_center_to_face(A_xx, axis=0)
    A_xy_xf = interp_center_to_face(A_xy, axis=0)
    A_xz_xf = interp_center_to_face(A_xz, axis=0)

    A_yy_yf = interp_center_to_face(A_yy, axis=1)
    A_xy_yf = interp_center_to_face(A_xy, axis=1)
    A_yz_yf = interp_center_to_face(A_yz, axis=1)

    A_zz_zf = interp_center_to_face(A_zz, axis=2)
    A_xz_zf = interp_center_to_face(A_xz, axis=2)
    A_yz_zf = interp_center_to_face(A_yz, axis=2)

    diag = -(
        A_xx_xf + jnp.roll(A_xx_xf, 1, axis=0)
        + A_yy_yf + jnp.roll(A_yy_yf, 1, axis=1)
        + A_zz_zf + jnp.roll(A_zz_zf, 1, axis=2)
    ) / ((sqrt_g + 1e-12) * (h * h))
    diag = jnp.where(jnp.abs(diag) < 1e-12, -1e-12, diag)

    return PoissonCoeffs(
        sqrt_g=sqrt_g,
        A_xx_xf=A_xx_xf,
        A_xy_xf=A_xy_xf,
        A_xz_xf=A_xz_xf,
        A_yy_yf=A_yy_yf,
        A_xy_yf=A_xy_yf,
        A_yz_yf=A_yz_yf,
        A_zz_zf=A_zz_zf,
        A_xz_zf=A_xz_zf,
        A_yz_zf=A_yz_zf,
        diag=diag,
    )

def grad_central(phi: Array, d: float, axis: int) -> Array:
    return (jnp.roll(phi, -1, axis=axis) - jnp.roll(phi, 1, axis=axis)) / (2 * d)


def interp_center_to_face(arr: Array, axis: int) -> Array:
    return 0.5 * (arr + jnp.roll(arr, -1, axis=axis))


def div_faces(Fx: Array, Fy: Array, Fz: Array, dx: float) -> Array:
    return (
        (Fx - jnp.roll(Fx, 1, axis=0)) / dx
        + (Fy - jnp.roll(Fy, 1, axis=1)) / dx
        + (Fz - jnp.roll(Fz, 1, axis=2)) / dx
    )


def apply_poisson(u: Array, h: float, def_field: Array) -> Array:
    """Curvilinear Laplace–Beltrami operator.

    Applies: L[u] = ∇·(√g g^{ij} ∂_j u) / √g
    where g^{ij} is from def_field.
    """
    triad = triad_from_def(def_field, h)
    _, ginv, sqrt_g = metric_from_triad(triad)

    # Build A^{ij} = sqrt(g) * g^{ij} at cell centers, then average to faces.
    A_xx = sqrt_g * ginv[..., 0, 0]
    A_xy = sqrt_g * ginv[..., 0, 1]
    A_xz = sqrt_g * ginv[..., 0, 2]
    A_yy = sqrt_g * ginv[..., 1, 1]
    A_yz = sqrt_g * ginv[..., 1, 2]
    A_zz = sqrt_g * ginv[..., 2, 2]

    # Face-centered normal derivatives: (phi_{i+1}-phi_i)/h (2nd order in div form).
    dphi_dx_xf = (jnp.roll(u, -1, axis=0) - u) / h
    dphi_dy_yf = (jnp.roll(u, -1, axis=1) - u) / h
    dphi_dz_zf = (jnp.roll(u, -1, axis=2) - u) / h

    # Tangential derivatives: central at centers, then averaged to the face.
    dphi_dy_c = grad_central(u, h, axis=1)
    dphi_dz_c = grad_central(u, h, axis=2)
    dphi_dx_c = grad_central(u, h, axis=0)

    dphi_dy_xf = interp_center_to_face(dphi_dy_c, axis=0)
    dphi_dz_xf = interp_center_to_face(dphi_dz_c, axis=0)
    dphi_dx_yf = interp_center_to_face(dphi_dx_c, axis=1)
    dphi_dz_yf = interp_center_to_face(dphi_dz_c, axis=1)
    dphi_dx_zf = interp_center_to_face(dphi_dx_c, axis=2)
    dphi_dy_zf = interp_center_to_face(dphi_dy_c, axis=2)

    # Metric coefficients to faces
    A_xx_xf = interp_center_to_face(A_xx, axis=0)
    A_xy_xf = interp_center_to_face(A_xy, axis=0)
    A_xz_xf = interp_center_to_face(A_xz, axis=0)

    A_yy_yf = interp_center_to_face(A_yy, axis=1)
    A_xy_yf = interp_center_to_face(A_xy, axis=1)
    A_yz_yf = interp_center_to_face(A_yz, axis=1)

    A_zz_zf = interp_center_to_face(A_zz, axis=2)
    A_xz_zf = interp_center_to_face(A_xz, axis=2)
    A_yz_zf = interp_center_to_face(A_yz, axis=2)

    # Fluxes at faces
    Fx = A_xx_xf * dphi_dx_xf + A_xy_xf * dphi_dy_xf + A_xz_xf * dphi_dz_xf
    Fy = A_yy_yf * dphi_dy_yf + A_xy_yf * dphi_dx_yf + A_yz_yf * dphi_dz_yf
    Fz = A_zz_zf * dphi_dz_zf + A_xz_zf * dphi_dx_zf + A_yz_zf * dphi_dy_zf

    divF = div_faces(Fx, Fy, Fz, h)
    return divF / (sqrt_g + 1e-12)


def apply_poisson_precomputed(u: Array, h: float, coeffs: PoissonCoeffs) -> Array:
    # Face-centered normal derivatives: (phi_{i+1}-phi_i)/h.
    dphi_dx_xf = (jnp.roll(u, -1, axis=0) - u) / h
    dphi_dy_yf = (jnp.roll(u, -1, axis=1) - u) / h
    dphi_dz_zf = (jnp.roll(u, -1, axis=2) - u) / h

    # Tangential derivatives: central at centers, then averaged to the face.
    dphi_dy_c = grad_central(u, h, axis=1)
    dphi_dz_c = grad_central(u, h, axis=2)
    dphi_dx_c = grad_central(u, h, axis=0)

    dphi_dy_xf = interp_center_to_face(dphi_dy_c, axis=0)
    dphi_dz_xf = interp_center_to_face(dphi_dz_c, axis=0)
    dphi_dx_yf = interp_center_to_face(dphi_dx_c, axis=1)
    dphi_dz_yf = interp_center_to_face(dphi_dz_c, axis=1)
    dphi_dx_zf = interp_center_to_face(dphi_dx_c, axis=2)
    dphi_dy_zf = interp_center_to_face(dphi_dy_c, axis=2)

    Fx = coeffs.A_xx_xf * dphi_dx_xf + coeffs.A_xy_xf * dphi_dy_xf + coeffs.A_xz_xf * dphi_dz_xf
    Fy = coeffs.A_yy_yf * dphi_dy_yf + coeffs.A_xy_yf * dphi_dx_yf + coeffs.A_yz_yf * dphi_dz_yf
    Fz = coeffs.A_zz_zf * dphi_dz_zf + coeffs.A_xz_zf * dphi_dx_zf + coeffs.A_yz_zf * dphi_dy_zf

    divF = div_faces(Fx, Fy, Fz, h)
    return divF / (coeffs.sqrt_g + 1e-12)


def residual(F: Array, u: Array, h: float, def_field: Array) -> Array:
    return F - apply_poisson(u, h, def_field)


def residual_precomputed(F: Array, u: Array, h: float, coeffs: PoissonCoeffs) -> Array:
    return F - apply_poisson_precomputed(u, h, coeffs)


# -----------------------------------------------------------------------------
# Restriction / Prolongation (same as multigrid_jacobi)
# -----------------------------------------------------------------------------

def restrict_full_weighting(fine: Array) -> Array:
    x = fine
    nd = x.ndim
    for ax in range(nd):
        x = 0.25 * jnp.roll(x, +1, ax) + 0.5 * x + 0.25 * jnp.roll(x, -1, ax)
    sl = tuple(slice(None, None, 2) for _ in range(nd))
    return x[sl]


def _upsample1d_linear(a: Array, axis: int) -> Array:
    n = a.shape[axis]
    new_n = 2 * n
    new_shape = tuple(a.shape[i] if i != axis else new_n for i in range(a.ndim))
    out = jnp.zeros(new_shape, dtype=a.dtype)

    idx_even = [slice(None)] * a.ndim
    idx_even[axis] = slice(0, new_n, 2)
    out = out.at[tuple(idx_even)].set(a)

    a_avg = 0.5 * (a + jnp.roll(a, -1, axis))
    idx_odd = [slice(None)] * a.ndim
    idx_odd[axis] = slice(1, new_n, 2)
    out = out.at[tuple(idx_odd)].set(a_avg)

    return out


def prolong_trilinear(coarse: Array, fine_shape: Tuple[int, ...]) -> Array:
    x = coarse
    for ax in range(coarse.ndim):
        x = _upsample1d_linear(x, axis=ax)
    if x.shape != fine_shape:
        raise ValueError(f"prolong_trilinear: shape mismatch {x.shape} != {fine_shape}")
    return x


# -----------------------------------------------------------------------------
# Smoother: Weighted Jacobi (variable coefficients)
# -----------------------------------------------------------------------------

def jacobi_sweep(u: Array, F: Array, h: float, def_field: Array, omega: float = 2.0 / 3.0) -> Array:
    triad = triad_from_def(def_field, h)
    _, ginv, sqrt_g = metric_from_triad(triad)

    # Approximate diagonal (negative) from face-averaged diagonal coefficients.
    A_xx = sqrt_g * ginv[..., 0, 0]
    A_yy = sqrt_g * ginv[..., 1, 1]
    A_zz = sqrt_g * ginv[..., 2, 2]
    A_xx_xf = interp_center_to_face(A_xx, axis=0)
    A_yy_yf = interp_center_to_face(A_yy, axis=1)
    A_zz_zf = interp_center_to_face(A_zz, axis=2)
    diag = -(
        A_xx_xf + jnp.roll(A_xx_xf, 1, axis=0)
        + A_yy_yf + jnp.roll(A_yy_yf, 1, axis=1)
        + A_zz_zf + jnp.roll(A_zz_zf, 1, axis=2)
    ) / ((sqrt_g + 1e-12) * (h * h))
    diag = jnp.where(jnp.abs(diag) < 1e-12, -1e-12, diag)
    Au = apply_poisson(u, h, def_field)
    u_star = u + (F - Au) / diag
    return (1.0 - omega) * u + omega * u_star


def jacobi_sweep_precomputed(u: Array, F: Array, h: float, coeffs: PoissonCoeffs, omega: float = 2.0 / 3.0) -> Array:
    Au = apply_poisson_precomputed(u, h, coeffs)
    u_star = u + (F - Au) / coeffs.diag
    return (1.0 - omega) * u + omega * u_star


@partial(jax.jit, static_argnums=(3,))
def smooth_weighted_jacobi(u0, F, h, iters, def_field, omega=2.0 / 3.0):
    iters = int(iters)

    def body(k, u):
        return jacobi_sweep(u, F, h, def_field, omega=omega)

    return jax.lax.fori_loop(0, iters, body, u0)


@partial(jax.jit, static_argnums=(3,))
def smooth_weighted_jacobi_precomputed(u0, F, h, iters, coeffs, omega=2.0 / 3.0):
    iters = int(iters)

    def body(k, u):
        return jacobi_sweep_precomputed(u, F, h, coeffs, omega=omega)

    return jax.lax.fori_loop(0, iters, body, u0)


def smooth_weighted_jacobi_old(
    u0: Array, F: Array, h: float, iters: int, def_field: Array, omega: float = 2.0 / 3.0
) -> Array:
    """Run iters sweeps of weighted Jacobi (plain Python loop)."""
    u = u0
    for _ in range(int(iters)):
        u = jacobi_sweep(u, F, h, def_field, omega=omega)
    return u


# -----------------------------------------------------------------------------
# One multigrid V-cycle (periodic), same structure as multigrid_jacobi
# -----------------------------------------------------------------------------

@dataclass
class PoissonCycle:
    F: Array
    def_field: Array
    v1: int = 2
    v2: int = 2
    # mu=1 -> V-cycle, mu=2 -> W-cycle (more generally a "mu-cycle").
    mu: int = 2
    l: int = 1
    eps: float = 1e-6
    h: float = 1.0
    laplace: Optional[Callable[[Array, float, Array], Array]] = None  # kept for API symmetry

    def norm(self, U: Array) -> Array:
        coeffs = build_poisson_coeffs(self.def_field, self.h)
        r = residual_precomputed(self.F, U, self.h, coeffs)
        return jnp.sqrt(jnp.mean(r * r))

    def __call__(self, U: Array) -> Array:
        return self.do_cycle(self.F, U, self.def_field, self.l, self.h)

    def do_cycle(self, F: Array, U: Array, def_field: Array, level: int, h: float) -> Array:
        coeffs = build_poisson_coeffs(def_field, h)
        if level <= 0 or min(U.shape) <= 2:
            return smooth_weighted_jacobi_precomputed(U, F, h, iters=16, coeffs=coeffs, omega=2.0 / 3.0)

        # Pre-smooth
        U = smooth_weighted_jacobi_precomputed(U, F, h, iters=self.v1, coeffs=coeffs, omega=2.0 / 3.0)

        # Residual and restrict to coarse
        r = residual_precomputed(F, U, h, coeffs)
        Rc = restrict_full_weighting(r)
        def_c = restrict_full_weighting(def_field)

        # Coarse-grid correction
        hc = 2.0 * h
        Ec = jnp.zeros_like(Rc, dtype=U.dtype)
        for _ in range(int(self.mu)):
            Ec = self.do_cycle(Rc, Ec, def_c, level - 1, hc)

        # Prolongate and correct
        Ef = prolong_trilinear(Ec, U.shape)
        U = U + Ef

        # Post-smooth
        U = smooth_weighted_jacobi_precomputed(U, F, h, iters=self.v2, coeffs=coeffs, omega=2.0 / 3.0)
        return U


# -----------------------------------------------------------------------------
# Public driver (IO mirrors multigrid_jacobi)
# -----------------------------------------------------------------------------

def multigrid(cycle: PoissonCycle, U: Array, eps: float, iter_cycle: int) -> Array:
    U_out = U
    for _ in range(int(iter_cycle)):
        U_out = cycle(U_out)
    return U_out


def poisson_multigrid(
    F: Array,
    U: Array,
    l: int,
    v1: int,
    v2: int,
    mu: int,
    iter_cycle: int,
    *,
    def_field: Array,
    eps: float = 1e-6,
    h: Optional[float] = None,
    laplace: Optional[Callable[[Array, float, Array], Array]] = apply_poisson,
) -> Array:
    """Solve L phi = F using multigrid V-/W-cycles (curvilinear operator)."""
    if h is None:
        h = 1.0
    cycle = PoissonCycle(F=F, def_field=def_field, v1=v1, v2=v2, mu=mu, l=l, eps=eps, h=h, laplace=laplace)
    return multigrid(cycle, U, eps, iter_cycle)


# -----------------------------------------------------------------------------
# Custom-VJP wrapper: treat multigrid as an implicit linear solve
# -----------------------------------------------------------------------------

def make_poisson_mg_solver(
    l: int,
    v1: int,
    v2: int,
    mu: int,
    iter_cycle: int,
    *,
    eps: float = 1e-6,
    h: float = 1.0,
):
    """Build a custom-VJP MG solver with fixed parameters; def_field is runtime input.

    Usage:
        solve = make_poisson_mg_solver(l=3, v1=2, v2=2, mu=2, iter_cycle=3, h=dx)
        phi = solve(F, U0, def_field)
    """

    @jax.custom_vjp
    def solve(F: Array, U0: Array, def_field: Array) -> Array:
        return poisson_multigrid(
            F=F,
            U=U0,
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            def_field=def_field,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )

    def solve_fwd(F: Array, U0: Array, def_field: Array):
        phi = poisson_multigrid(
            F=F,
            U=U0,
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            def_field=def_field,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )
        residuals = (def_field,)
        return phi, residuals

    def solve_bwd(residuals, g_phi: Array):
        (def_field,) = residuals
        g_F = poisson_multigrid(
            F=g_phi,
            U=jnp.zeros_like(g_phi),
            l=l,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iter_cycle,
            def_field=def_field,
            eps=eps,
            h=h,
            laplace=apply_poisson,
        )
        g_U0 = jnp.zeros_like(g_phi)
        g_def = jnp.zeros_like(def_field)
        return (g_F, g_U0, g_def)

    solve.defvjp(solve_fwd, solve_bwd)
    return jax.jit(solve)
