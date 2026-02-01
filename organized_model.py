"""Organized, corrected moving-mesh components based on diffAPM notebook.

This module wraps the core math into a clean API and fixes several
issues found in the original diffAPM notebook/helpers, using QZOOM
as the correctness baseline.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp

try:
    # Optional dependency. We only need CIC painting for particle->mesh.
    from jaxpm.painting import cic_paint  # type: ignore
except Exception:  # pragma: no cover
    cic_paint = None


def _cic_paint_fallback(mesh, positions, weights=None):
    """Minimal periodic CIC deposit (fallback when jaxpm isn't installed).

    `positions` are assumed to be in *grid coordinates* [0, N) for each axis.
    """
    nx, ny, nz = mesh.shape
    if weights is None:
        weights = jnp.ones((positions.shape[0],), dtype=mesh.dtype)

    # (Np, 3)
    pos = positions
    i0 = jnp.floor(pos).astype(jnp.int32)
    d = pos - i0
    i1 = i0 + 1

    i0x = jnp.mod(i0[:, 0], nx)
    i0y = jnp.mod(i0[:, 1], ny)
    i0z = jnp.mod(i0[:, 2], nz)
    i1x = jnp.mod(i1[:, 0], nx)
    i1y = jnp.mod(i1[:, 1], ny)
    i1z = jnp.mod(i1[:, 2], nz)

    wx0 = 1.0 - d[:, 0]
    wy0 = 1.0 - d[:, 1]
    wz0 = 1.0 - d[:, 2]
    wx1 = d[:, 0]
    wy1 = d[:, 1]
    wz1 = d[:, 2]

    def add(ix, iy, iz, w):
        return mesh.at[ix, iy, iz].add(weights * w)

    out = mesh
    out = add(i0x, i0y, i0z, wx0 * wy0 * wz0)
    out = add(i0x, i0y, i1z, wx0 * wy0 * wz1)
    out = add(i0x, i1y, i0z, wx0 * wy1 * wz0)
    out = add(i0x, i1y, i1z, wx0 * wy1 * wz1)
    out = add(i1x, i0y, i0z, wx1 * wy0 * wz0)
    out = add(i1x, i0y, i1z, wx1 * wy0 * wz1)
    out = add(i1x, i1y, i0z, wx1 * wy1 * wz0)
    out = add(i1x, i1y, i1z, wx1 * wy1 * wz1)
    return out


@dataclass(frozen=True)
class ModelConfig:
    mesh_shape: tuple
    box_size: float
    G: float = 1.0
    dx: float | None = None

    def grid_spacing(self):
        if self.dx is not None:
            return self.dx
        return self.box_size / self.mesh_shape[0]


# ---------------------- basic ops ----------------------

def grad_central(phi, d, axis):
    return (jnp.roll(phi, -1, axis=axis) - jnp.roll(phi, 1, axis=axis)) / (2 * d)


def interp_center_to_face(arr, axis):
    return 0.5 * (arr + jnp.roll(arr, -1, axis=axis))


def div_faces(Fx, Fy, Fz, dx):
    return (
        (Fx - jnp.roll(Fx, 1, axis=0)) / dx
        + (Fy - jnp.roll(Fy, 1, axis=1)) / dx
        + (Fz - jnp.roll(Fz, 1, axis=2)) / dx
    )


# ---------------------- geometry ----------------------

def hessian_from_def(def_field, dx):
    phixx = (jnp.roll(def_field, -1, 0) - 2 * def_field + jnp.roll(def_field, 1, 0)) / (
        dx * dx
    )
    phiyy = (jnp.roll(def_field, -1, 1) - 2 * def_field + jnp.roll(def_field, 1, 1)) / (
        dx * dx
    )
    phizz = (jnp.roll(def_field, -1, 2) - 2 * def_field + jnp.roll(def_field, 1, 2)) / (
        dx * dx
    )
    phixy = (
        jnp.roll(jnp.roll(def_field, -1, 0), -1, 1)
        - jnp.roll(jnp.roll(def_field, -1, 0), 1, 1)
        - jnp.roll(jnp.roll(def_field, 1, 0), -1, 1)
        + jnp.roll(jnp.roll(def_field, 1, 0), 1, 1)
    ) / (4 * dx * dx)
    phixz = (
        jnp.roll(jnp.roll(def_field, -1, 0), -1, 2)
        - jnp.roll(jnp.roll(def_field, -1, 0), 1, 2)
        - jnp.roll(jnp.roll(def_field, 1, 0), -1, 2)
        + jnp.roll(jnp.roll(def_field, 1, 0), 1, 2)
    ) / (4 * dx * dx)
    phiyz = (
        jnp.roll(jnp.roll(def_field, -1, 1), -1, 2)
        - jnp.roll(jnp.roll(def_field, -1, 1), 1, 2)
        - jnp.roll(jnp.roll(def_field, 1, 1), -1, 2)
        + jnp.roll(jnp.roll(def_field, 1, 1), 1, 2)
    ) / (4 * dx * dx)
    return phixx, phiyy, phizz, phixy, phixz, phiyz


def triad_from_def(def_field, dx):
    """Triad A = I + Hess(def). QZOOM eqs. 5-6 in 3D."""
    phixx, phiyy, phizz, phixy, phixz, phiyz = hessian_from_def(def_field, dx)
    a = jnp.zeros(def_field.shape + (3, 3))
    a = a.at[..., 0, 0].set(1.0 + phixx)
    a = a.at[..., 1, 1].set(1.0 + phiyy)
    a = a.at[..., 2, 2].set(1.0 + phizz)
    a = a.at[..., 0, 1].set(phixy)
    a = a.at[..., 1, 0].set(phixy)
    a = a.at[..., 0, 2].set(phixz)
    a = a.at[..., 2, 0].set(phixz)
    a = a.at[..., 1, 2].set(phiyz)
    a = a.at[..., 2, 1].set(phiyz)
    return a


def metric_from_triad(triad):
    """g = A^T A, ginv = g^{-1}, sqrt_g = det(A).

    FIX: diffAPM used det(ginv) as sqrt(g); correct is det(A).
    """
    g = jnp.einsum("...ia,...ja->...ij", triad, triad)
    ginv = jnp.linalg.inv(g)
    sqrt_g = jnp.linalg.det(triad)
    return g, ginv, sqrt_g


# ---------------------- operators ----------------------

def laplace_beltrami(phi, ginv, sqrt_g, dx):
    """∇·(√g g^{ij} ∂_j phi) / √g.

    FIX: diffAPM mixed triad and metric and used det(ginv) for √g.
    Here we use ginv directly with proper √g weighting.
    """
    g_xx = ginv[..., 0, 0]
    g_xy = ginv[..., 0, 1]
    g_xz = ginv[..., 0, 2]
    g_yy = ginv[..., 1, 1]
    g_yz = ginv[..., 1, 2]
    g_zz = ginv[..., 2, 2]

    dphi_dx = grad_central(phi, dx, axis=0)
    dphi_dy = grad_central(phi, dx, axis=1)
    dphi_dz = grad_central(phi, dx, axis=2)

    Fx = g_xx * dphi_dx + g_xy * dphi_dy + g_xz * dphi_dz
    Fy = g_yy * dphi_dy + g_xy * dphi_dx + g_yz * dphi_dz
    Fz = g_zz * dphi_dz + g_xz * dphi_dx + g_yz * dphi_dy

    Fx *= sqrt_g
    Fy *= sqrt_g
    Fz *= sqrt_g

    Fx_faces = interp_center_to_face(Fx, axis=0)
    Fy_faces = interp_center_to_face(Fy, axis=1)
    Fz_faces = interp_center_to_face(Fz, axis=2)

    divF = div_faces(Fx_faces, Fy_faces, Fz_faces, dx)
    return divF / (sqrt_g + 1e-12)


# ---------------------- CIC readout ----------------------

def cic_readout(mesh, part):
    """Read mesh values at particle positions (periodic CIC).

    FIX: diffAPM used nx for all axes; we use per-axis shapes.
    """
    nx, ny, nz = mesh.shape
    part = jnp.expand_dims(part, 2)
    floor = jnp.floor(part)
    connection = jnp.array(
        [[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0],
          [1.0, 0, 1.0], [0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    neigh = floor + connection
    kernel = 1.0 - jnp.abs(part - neigh)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    neigh = jnp.array(neigh, dtype=jnp.int16)

    # FIX: per-axis modulo
    neigh = neigh.at[..., 0].set(jnp.mod(neigh[..., 0], nx))
    neigh = neigh.at[..., 1].set(jnp.mod(neigh[..., 1], ny))
    neigh = neigh.at[..., 2].set(jnp.mod(neigh[..., 2], nz))

    vals = mesh[tuple(neigh[0, :, :].T)]
    weighted = vals.T * kernel[0]
    return jnp.sum(weighted, axis=-1)


# ---------------------- top-level model ----------------------

class DiffAPMModel:
    """Organized wrapper of diffAPM notebook components.

    This class does NOT fully reproduce QZOOM yet (needs curvilinear multigrid),
    but fixes several math bugs and exposes a clean API.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def paint_density(self, positions):
        mesh = jnp.zeros(self.cfg.mesh_shape)
        if cic_paint is not None:
            return cic_paint(mesh, positions)
        # Fallback (no jaxpm dependency)
        return _cic_paint_fallback(mesh, positions)

    def build_geometry(self, def_field):
        dx = self.cfg.grid_spacing()
        triad = triad_from_def(def_field, dx)
        g, ginv, sqrt_g = metric_from_triad(triad)
        return triad, g, ginv, sqrt_g

    def poisson_operator(self, phi, ginv, sqrt_g):
        dx = self.cfg.grid_spacing()
        return laplace_beltrami(phi, ginv, sqrt_g, dx)

    def solve_poisson_gs(self, rhs, ginv, sqrt_g, n_iter=50, omega=1.0):
        """Red-black Gauss-Seidel on variable-coefficient operator.

        Note: this is a simplified smoother, not full multigrid.
        """
        ng = rhs.shape[0]
        dx = self.cfg.grid_spacing()
        red = ((jnp.arange(ng)[:, None, None] + jnp.arange(ng)[None, :, None] + jnp.arange(ng)[None, None, :]) % 2) == 0
        black = ~red

        def sweep(phi, mask):
            # Jacobi-style update for variable coefficients (approximate)
            lap = laplace_beltrami(phi, ginv, sqrt_g, dx)
            phi_new = phi + (rhs - lap) * (dx * dx) / 6.0
            return jnp.where(mask, (1 - omega) * phi + omega * phi_new, phi)

        phi = jnp.zeros_like(rhs)
        for _ in range(n_iter):
            phi = sweep(phi, red)
            phi = sweep(phi, black)
        return phi

    def solve_poisson_multigrid(self, rhs, def_field, levels=4, v1=2, v2=2, mu=1, iters=3):
        """Multigrid solve using curvilinear Laplace–Beltrami operator.

        FIX: diffAPM notebook used Euclidean MG; this uses geometry from def_field.
        """
        from diffAPM.multigrid_mesh import poisson_multigrid

        dx = self.cfg.grid_spacing()
        U0 = jnp.zeros_like(rhs)
        return poisson_multigrid(
            F=rhs,
            U=U0,
            l=levels,
            v1=v1,
            v2=v2,
            mu=mu,
            iter_cycle=iters,
            def_field=def_field,
            h=dx,
        )
