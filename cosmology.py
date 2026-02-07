from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

import jax_cosmo as jc

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CosmologyParams:
    """Thin JAX-friendly wrapper around a `jax_cosmo` cosmology.

    Notes
    -----
    - This repo's particle-mesh solvers are written in code units; for cosmological
      time-stepping we typically only need the dimensionless expansion function:
          E(a) = H(a)/H0 = sqrt(Esqr(cosmo, a)).
    - jaxpm's N-body ODE uses the canonical momentum `p` and integrates in `a`:
          d x / d a = p / (a^3 E(a))
          d p / d a = F(x) / (a^2 E(a)),   with F = pm_forces(x) * 1.5*Omega_m
      which is equivalent to integrating in conformal time τ with:
          dτ = da / (a^2 E(a)),   d p / dτ = F,   d x / dτ = p / a.
    """

    cosmo: jc.core.Cosmology

    def tree_flatten(self):
        return (self.cosmo,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (cosmo,) = children
        return cls(cosmo=cosmo)

    def E(self, a: Array) -> Array:
        """Dimensionless Hubble function E(a)=H(a)/H0."""
        return jnp.sqrt(jc.background.Esqr(self.cosmo, a))

    def dt_da(self, a: Array) -> Array:
        """d t / d a in units of H0^{-1}: dt/da = 1/(a E(a))."""
        return 1.0 / (a * self.E(a) + 1e-12)

    def dtau_da(self, a: Array) -> Array:
        """d τ / d a in units of H0^{-1}: dτ/da = 1/(a^2 E(a))."""
        return 1.0 / (a * a * self.E(a) + 1e-12)

    def omega_m0(self) -> float:
        return float(self.cosmo.Omega_m)

    def growth_factor(self, a: Array) -> Array:
        """Linear growth factor D(a) (requires jaxpm)."""
        try:
            import jaxpm.pm as jpm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("growth_factor requires jaxpm (pip install jaxpm==0.0.2).") from e
        return jpm.growth_factor(self.cosmo, a)

    def growth_rate(self, a: Array) -> Array:
        """Linear growth rate f(a)=d ln D/d ln a (requires jaxpm)."""
        try:
            import jaxpm.pm as jpm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("growth_rate requires jaxpm (pip install jaxpm==0.0.2).") from e
        return jpm.growth_rate(self.cosmo, a)


def planck15_like(*, omega_c: float = 0.25, sigma8: float = 0.8) -> CosmologyParams:
    """Convenience constructor used by the roadmap scripts."""
    return CosmologyParams(cosmo=jc.Planck15(Omega_c=float(omega_c), sigma8=float(sigma8)))


def canonical_kick_drift(*, cosmo: CosmologyParams, a_mid: Array, da: Array) -> tuple[Array, Array, Array]:
    """Return (dtau, kick, drift_x) for one step at midpoint a.

    Conventions:
      - kick updates canonical momentum p in conformal time: p += F * dtau
      - drift updates positions in comoving coordinates: x += (p/a_mid) * dtau

    Returns:
      dtau: conformal time step (H0^{-1})
      kick: dtau
      drift_x: dtau / a_mid
    """
    dtau = da * cosmo.dtau_da(a_mid)
    return dtau, dtau, dtau / (a_mid + 1e-12)

