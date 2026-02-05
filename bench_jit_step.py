"""Micro-benchmark for GPU/JIT-friendly stepping.

This script is optional; it requires `jax` (and a GPU build if you want to
benchmark on GPU).

Example:
  python bench_jit_step.py --ng 64 --npart 200000 --steps 5 --mg-cycles 4
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp

import qzoom_nbody_flow as qz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ng", type=int, default=64)
    p.add_argument("--npart", type=int, default=200_000)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--mg-cycles", type=int, default=4)
    p.add_argument("--no-limiter", action="store_true")
    args = p.parse_args()

    ng = int(args.ng)
    npart = int(args.npart)
    dx = 1.0 / ng

    levels = max(1, int(math.log2(ng)) - 1)
    mg_params = qz.MGParams(levels=int(levels), v1=2, v2=2, mu=1, cycles=int(args.mg_cycles))
    params = qz.APMParams(ng=ng, box=1.0, a=1.0)

    limiter = qz.LimiterParams(enabled=not bool(args.no_limiter))

    key = jax.random.key(0)
    xi = jax.random.uniform(key, (npart, 3), minval=0.0, maxval=float(ng), dtype=jnp.float32)
    v = jnp.zeros((npart, 3), dtype=jnp.float32)
    xv = jnp.concatenate([xi, v], axis=-1)
    pmass = jnp.ones((npart,), dtype=jnp.float32) / float(npart)
    state = qz.NBodyState(xv=xv, pmass=pmass)

    def_field = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
    defp_field = jnp.zeros_like(def_field)

    densinit = jnp.zeros_like(def_field)
    if limiter.enabled:
        rho0 = jnp.zeros((ng, ng, ng), dtype=jnp.float32)
        rho_mesh0, sqrt_g0 = qz.pcalcrho(rho0, def_field, state, dx=dx)
        densinit = jnp.full_like(rho_mesh0 * sqrt_g0, jnp.mean(rho_mesh0 * sqrt_g0))

    step = qz.make_step_nbody_apm_jit(params=params, mg_params=mg_params, kappa=20.0, smooth_steps=2, limiter=limiter)

    dt = jnp.array(float(args.dt), dtype=jnp.float32)
    dtold = dt

    # Compile + warm up
    state, def_field, defp_field, phi = step(state, def_field, defp_field, dt, dtold, densinit)
    phi.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(int(args.steps)):
        state, def_field, defp_field, phi = step(state, def_field, defp_field, dt, dtold, densinit)
    phi.block_until_ready()
    t1 = time.perf_counter()

    print(f"backend={jax.default_backend()} ng={ng} npart={npart} steps={int(args.steps)}")
    print(f"elapsed_s={t1 - t0:.3f} per_step_s={(t1 - t0) / max(int(args.steps), 1):.3f}")


if __name__ == "__main__":
    main()
