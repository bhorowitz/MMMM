from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

import jax_cosmo as jc

import qzoom_nbody_flow as qz

Array = jnp.ndarray


def make_lattice(*, npart_side: int, ng: int, shift: float = 0.0) -> Array:
    """Uniform lattice in mesh coordinates [0,ng)."""
    s = int(npart_side)
    if s <= 0:
        raise ValueError("npart_side must be >0")
    spacing = float(ng) / float(s)
    coords = (jnp.arange(s, dtype=jnp.float32) + float(shift)) * spacing
    grid = jnp.stack(jnp.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    return grid.reshape((-1, 3))


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
) -> tuple[Array, Array]:
    """1LPT displacement + canonical momentum (jaxpm convention)."""
    import jaxpm.pm as jpm  # type: ignore

    a = jnp.atleast_1d(jnp.array(a0, dtype=jnp.float32))
    E = jnp.sqrt(jc.background.Esqr(cosmo, a))
    delta_k = jnp.fft.rfftn(init_mesh)
    mesh_shape = init_mesh.shape

    init_force = pm_forces_jaxpm_fixed(positions, mesh_shape=mesh_shape, delta_k=delta_k)
    dx = jpm.growth_factor(cosmo, a) * init_force
    p = a**2 * jpm.growth_rate(cosmo, a) * E * dx
    return dx, p


def make_linear_pk_fn(*, cosmo: jc.core.Cosmology, nk: int = 128) -> Callable[[Array], Array]:
    k = jnp.logspace(-4, 1, int(nk), dtype=jnp.float32)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x: Array) -> Array:
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    return pk_fn


@dataclass(frozen=True)
class LPTIC:
    init_mesh: Array
    particles_lattice: Array
    pos0: Array
    p0: Array
    pmass: Array


def make_grf_lpt_ic(
    *,
    ng: int,
    box: float,
    seed: int,
    omega_c: float,
    sigma8: float,
    a0: float,
    npart_side: int | None = None,
    lattice_shift: float = 0.0,
) -> LPTIC:
    """Generate GRF + 1LPT ICs using jaxpm (GRF + kernels) with a cic_read fix."""
    import jaxpm.pm as jpm  # type: ignore

    ng = int(ng)
    npart_side = int(npart_side) if npart_side is not None else ng

    cosmo = jc.Planck15(Omega_c=float(omega_c), sigma8=float(sigma8))
    pk_fn = make_linear_pk_fn(cosmo=cosmo)

    mesh_shape = (ng, ng, ng)
    box_size = (float(box), float(box), float(box))
    init_mesh = jpm.linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(int(seed)))

    particles = make_lattice(npart_side=npart_side, ng=ng, shift=float(lattice_shift))
    pmass = jnp.ones((particles.shape[0],), dtype=jnp.float32)

    dx0, p0 = lpt_1_fixed(cosmo, init_mesh, particles, a0=float(a0))
    pos0 = jnp.mod(particles + dx0, float(ng))

    return LPTIC(init_mesh=init_mesh, particles_lattice=particles, pos0=pos0, p0=p0, pmass=pmass)

