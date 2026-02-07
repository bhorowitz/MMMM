#!/usr/bin/env python3
"""Unified test driver for several configurations (moving vs static mesh).

This script intentionally mirrors the CLI of diffAPM_new/qzoom_fig6_fig7_demo.py
for shared knobs (ng, npart-side, GRF parameters, limiter controls, plotting),
but adds scenario switches:
  - --single-halo
  - --merging-halos
  - --pancake

Outputs
-------
Writes a directory per scenario under --outdir:
  <outdir>/<scenario>/<outstem>/
containing:
  - moving_summary.png / static_summary.png
  - moving_mesh_vs_phys.png / static_mesh_vs_phys.png
  - compare.png
  - moving_anim.gif / static_anim.gif (if --animate)
  - metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import jax

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import helper as H  # noqa: E402
import config_io as C  # noqa: E402


def _pick_scenario(args: argparse.Namespace) -> str:
    if getattr(args, "scenario", None):
        return str(args.scenario)
    flags = {
        "single_halo": bool(args.single_halo),
        "two_halo_merger": bool(args.merging_halos),
        "pancake_filament": bool(args.pancake),
        "grf": bool(args.grf),
    }
    chosen = [k for k, v in flags.items() if v]
    if len(chosen) == 0:
        return "grf"  # default matches qzoom_fig6_fig7_demo.py
    if len(chosen) > 1:
        raise SystemExit(f"Choose only one scenario flag, got {chosen}")
    return chosen[0]


def main() -> None:
    # Pre-parse --config so we can apply file defaults before full parsing.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="TOML/JSON parameter file (optional).")
    pre_args, _ = pre.parse_known_args()
    cfg_raw = C.load_config_file(pre_args.config) if pre_args.config else {}
    # Support two common layouts:
    #   1) flat keys at top-level
    #   2) a dedicated [args] table (TOML) / {"args": {...}} (JSON), optionally with
    #      top-level metadata like scenario.
    cfg_args = {}
    if isinstance(cfg_raw, dict):
        for k, v in cfg_raw.items():
            if k == "args":
                continue
            cfg_args[str(k)] = v
        if isinstance(cfg_raw.get("args", None), dict):
            # args-table takes precedence over top-level defaults.
            cfg_args.update(cfg_raw["args"])

    ap = argparse.ArgumentParser()

    # Config file / scenario
    ap.add_argument("--config", type=str, default=None, help="TOML/JSON parameter file (optional).")
    ap.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["grf", "single_halo", "two_halo_merger", "pancake_filament"],
        help="Choose scenario directly (recommended for config files).",
    )

    # Scenario switches
    ap.add_argument("--grf", action="store_true", help="Use GRF Zel'dovich ICs (default if no scenario flag).")
    ap.add_argument("--single-halo", action="store_true", help="Single collapsing halo IC.")
    ap.add_argument("--merging-halos", action="store_true", help="Two-halo merger IC.")
    ap.add_argument("--pancake", action="store_true", help="Pancake/filament IC.")

    # --- Shared args (mirrors qzoom_fig6_fig7_demo.py) ---
    ap.add_argument("--ng", type=int, default=64, help="mesh size")
    ap.add_argument("--npart-side", type=int, default=32, help="particles per side (for lattice-based ICs)")
    ap.add_argument("--box", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pmass", type=float, default=1.0, help="particle mass (keep O(1) for QZOOM-like scaling)")

    # Lattice de-aliasing (for GRF/lattice based ICs)
    g_shift = ap.add_mutually_exclusive_group()
    g_shift.add_argument("--lattice-random-shift", dest="lattice_random_shift", action="store_true", help="randomly shift the initial lattice")
    g_shift.add_argument("--no-lattice-random-shift", dest="lattice_random_shift", action="store_false", help="disable random lattice shift")
    ap.add_argument("--lattice-jitter", type=float, default=0.0, help="random jitter (fraction of spacing)")

    # GRF IC params
    ap.add_argument("--pk-n", type=float, default=-1.0)
    ap.add_argument("--pk-k0", type=float, default=6.0)
    ap.add_argument("--pk-kcut", type=float, default=30.0)
    ap.add_argument("--disp-rms", type=float, default=0.03, help="RMS initial displacement (box units)")
    ap.add_argument("--vel-fac", type=float, default=0.0, help="velocity = vel_fac * displacement (toy)")

    # Evolution
    ap.add_argument("--ntotal", type=int, default=200, help="total steps")
    ap.add_argument("--dt-max", type=float, default=2e-3)
    ap.add_argument("--dt-min", type=float, default=1e-5)
    ap.add_argument("--cfl-mesh", type=float, default=0.8)
    ap.add_argument("--cfl-part", type=float, default=0.25, help="particle CFL: limit max particle displacement per step to cfl_part*dx")
    ap.add_argument("--mg-cycles", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--smooth-steps", type=int, default=2)
    ap.add_argument("--a", type=float, default=1.0)

    # Performance / execution
    ap.add_argument(
        "--jit-moving",
        action="store_true",
        help="Use the GPU/JIT-friendly moving-mesh step (no per-step Python diagnostics).",
    )
    ap.add_argument(
        "--compare-jaxpm",
        action="store_true",
        help="Also run a static-mesh PM using jaxpm and compare against our static/moving runs.",
    )

    # Limiter
    g_lim = ap.add_mutually_exclusive_group()
    g_lim.add_argument("--limiter", dest="limiter", action="store_true", help="Enable the mesh limiter (recommended).")
    g_lim.add_argument("--no-limiter", dest="limiter", action="store_false", help="Disable the mesh limiter.")
    ap.add_argument("--limiter-mode", type=str, default="both", choices=["both", "source", "post"])
    ap.add_argument("--compressmax", type=float, default=200.0)
    ap.add_argument("--skewmax", type=float, default=100.0)
    ap.add_argument("--xhigh", type=float, default=2.0)
    ap.add_argument("--relax-steps", type=float, default=30.0)
    ap.add_argument("--hard-strength", type=float, default=1.0)

    # Plotting
    ap.add_argument("--mesh-stride", type=int, default=1)
    ap.add_argument("--mesh-exaggerate", type=float, default=1.0)
    ap.add_argument("--mesh-z", type=int, default=None, help="fixed z-index for mesh layer plots")
    ap.add_argument("--mesh-quiver", action="store_true", help="Accepted for CLI parity; currently unused.")
    ap.add_argument("--max-plot", type=int, default=40000)
    ap.add_argument("--mesh-phys-plot", action="store_true", help="Accepted for CLI parity; output is always generated here.")

    # Animation
    g_anim = ap.add_mutually_exclusive_group()
    g_anim.add_argument("--animate", dest="animate", action="store_true", help="Write GIF animations (moving and static).")
    g_anim.add_argument("--no-animate", dest="animate", action="store_false", help="Disable GIF animations.")
    ap.add_argument("--anim-every", type=int, default=10)
    ap.add_argument("--anim-fps", type=int, default=8)
    ap.add_argument("--anim-out", type=str, default=None, help="Optional base filename for animations (suffixes added).")

    # Outputs
    ap.add_argument("--outdir", type=str, default="./test_outputs")
    ap.add_argument("--out", type=str, default="run.png", help="used only for naming; extension ignored")

    # --- Scenario-specific knobs ---
    ap.add_argument("--npart", type=int, default=None, help="total particles for non-lattice ICs (single/merger)")

    # Single halo
    ap.add_argument("--clump-frac", type=float, default=0.25)
    ap.add_argument("--clump-sigma", type=float, default=0.07, help="halo sigma (box units)")
    ap.add_argument("--v-infall", type=float, default=0.0, help="radial infall speed (box units/time)")

    # Two-halo merger
    ap.add_argument("--sep", type=float, default=0.35, help="initial separation (box units)")
    ap.add_argument("--v-merge", type=float, default=0.0, help="bulk approach speed along x (box units/time)")

    # Pancake/filament
    ap.add_argument("--amp-x", type=float, default=0.06, help="pancake displacement amplitude (box units)")
    ap.add_argument("--amp-y", type=float, default=0.0, help="filament displacement amplitude (box units)")

    # Keep defaults stable even with config/override groups.
    ap.set_defaults(lattice_random_shift=False, limiter=False, animate=False)

    # Apply config defaults (CLI always overrides config).
    if cfg_args:
        known_dests = {a.dest for a in ap._actions}
        ap.set_defaults(**{k: v for k, v in cfg_args.items() if k in known_dests})

    args = ap.parse_args()

    scenario = _pick_scenario(args)

    outstem = Path(args.out).stem
    root = H.ensure_dir(Path(args.outdir) / scenario / outstem)

    cfg = H.RunConfig(
        ng=int(args.ng),
        box=float(args.box),
        ntotal=int(args.ntotal),
        dt_max=float(args.dt_max),
        dt_min=float(args.dt_min),
        cfl_mesh=float(args.cfl_mesh),
        cfl_part=float(args.cfl_part),
        mg_cycles=int(args.mg_cycles),
        kappa=float(args.kappa),
        smooth_steps=int(args.smooth_steps),
        a=float(args.a),
        limiter=bool(args.limiter),
        limiter_mode=str(args.limiter_mode),
        compressmax=float(args.compressmax),
        skewmax=float(args.skewmax),
        xhigh=float(args.xhigh),
        relax_steps=float(args.relax_steps),
        hard_strength=float(args.hard_strength),
        mesh_stride=int(args.mesh_stride),
        mesh_exaggerate=float(args.mesh_exaggerate),
        mesh_z=int(args.mesh_z) if args.mesh_z is not None else None,
        max_plot=int(args.max_plot),
        animate=bool(args.animate),
        anim_every=int(args.anim_every),
        anim_fps=int(args.anim_fps),
    )

    # Persist run parameters for reproducibility.
    C.write_repro_bundle(
        root,
        argv=sys.argv,
        resolved_args=vars(args),
        scenario=scenario,
        cfg_obj=cfg,
        config_path=args.config,
        config_dict=cfg_args if cfg_args else None,
    )

    # IC selection
    npart_default = int(args.npart_side) ** 3
    npart = int(args.npart) if args.npart is not None else npart_default

    if scenario == "grf":
        state0, center = H.make_state_grf_zeldovich(
            ng=int(args.ng),
            npart_side=int(args.npart_side),
            box=float(args.box),
            seed=int(args.seed),
            pk_n=float(args.pk_n),
            pk_k0=float(args.pk_k0),
            pk_kcut=float(args.pk_kcut),
            disp_rms=float(args.disp_rms),
            vel_fac=float(args.vel_fac),
            pmass=float(args.pmass),
            lattice_random_shift=bool(args.lattice_random_shift),
            lattice_jitter=float(args.lattice_jitter),
        )
    elif scenario == "single_halo":
        state0, center = H.make_state_single_halo(
            ng=int(args.ng),
            box=float(args.box),
            npart=int(npart),
            seed=int(args.seed),
            clump_frac=float(args.clump_frac),
            clump_sigma=float(args.clump_sigma),
            v_infall=float(args.v_infall),
            pmass=float(args.pmass),
        )
    elif scenario == "two_halo_merger":
        state0, center = H.make_state_two_halo_merger(
            ng=int(args.ng),
            box=float(args.box),
            npart=int(npart),
            seed=int(args.seed),
            clump_frac=float(args.clump_frac),
            clump_sigma=float(args.clump_sigma),
            sep=float(args.sep),
            v_merge=float(args.v_merge),
            pmass=float(args.pmass),
        )
    elif scenario == "pancake_filament":
        state0, center = H.make_state_pancake_filament(
            ng=int(args.ng),
            box=float(args.box),
            npart_side=int(args.npart_side),
            seed=int(args.seed),
            amp_x=float(args.amp_x),
            amp_y=float(args.amp_y),
            vel_fac=float(args.vel_fac),
            lattice_jitter=float(args.lattice_jitter),
            lattice_random_shift=bool(args.lattice_random_shift),
            pmass=float(args.pmass),
        )
    else:
        raise SystemExit(f"Unknown scenario {scenario}")

    # Run both
    moving = H.run_moving_mesh(scenario, state0, cfg=cfg, jit_step=bool(args.jit_moving))
    static = H.run_static_mesh(scenario, state0, cfg=cfg)
    jaxpm_static = None
    if bool(args.compare_jaxpm):
        jaxpm_static = H.run_static_mesh_jaxpm(scenario, state0, cfg=cfg)

    # Metrics (shared)
    metrics = {}
    metrics.update(H.compute_metrics(moving, cfg=cfg, center=center, tag="moving"))
    metrics.update(H.compute_metrics(static, cfg=cfg, center=center, tag="static"))
    if jaxpm_static is not None:
        metrics.update(H.compute_metrics(jaxpm_static, cfg=cfg, center=center, tag="jaxpm_static"))
        metrics.update(H.compare_static_fields_to_jaxpm(static, jaxpm_static, cfg=cfg))
        metrics.update(H.compare_jaxpm_to_mesh_runs(moving, static, jaxpm_static, cfg=cfg, center=np.array(center, dtype=np.float64)))

    # Scenario-specific metrics
    dx = cfg.dx()
    box = cfg.box
    if scenario == "single_halo":
        x_mov = np.array(H.particles_phys(moving.state, moving.def_field, dx, box))
        x_sta = np.array(H.particles_phys(static.state, static.def_field, dx, box))
        metrics.update({f"moving_{k}": v for k, v in H.radius_quantiles(x_mov, center=center, box=box).items()})
        metrics.update({f"static_{k}": v for k, v in H.radius_quantiles(x_sta, center=center, box=box).items()})
    elif scenario == "two_halo_merger":
        x_mov = np.array(H.particles_phys(moving.state, moving.def_field, dx, box))
        x_sta = np.array(H.particles_phys(static.state, static.def_field, dx, box))
        metrics["moving_peak_sep_x"] = float(H.peak_separation_1d(x_mov[:, 0], box=box, nbins=max(32, cfg.ng), smooth=2))
        metrics["static_peak_sep_x"] = float(H.peak_separation_1d(x_sta[:, 0], box=box, nbins=max(32, cfg.ng), smooth=2))
    elif scenario == "pancake_filament":
        cx, cy, cz = center
        x_mov = np.array(H.particles_phys(moving.state, moving.def_field, dx, box))
        x_sta = np.array(H.particles_phys(static.state, static.def_field, dx, box))
        metrics["moving_width_x_p10_p90"] = float(H.percentile_width_1d(x_mov[:, 0], center=cx, box=box, p_lo=10.0, p_hi=90.0))
        metrics["static_width_x_p10_p90"] = float(H.percentile_width_1d(x_sta[:, 0], center=cx, box=box, p_lo=10.0, p_hi=90.0))
        metrics["moving_width_y_p10_p90"] = float(H.percentile_width_1d(x_mov[:, 1], center=cy, box=box, p_lo=10.0, p_hi=90.0))
        metrics["static_width_y_p10_p90"] = float(H.percentile_width_1d(x_sta[:, 1], center=cy, box=box, p_lo=10.0, p_hi=90.0))

    # Save plots
    H.save_summary_figure(moving, out_path=root / "moving_summary.png", cfg=cfg, title=f"{scenario} (moving mesh)")
    H.save_summary_figure(static, out_path=root / "static_summary.png", cfg=cfg, title=f"{scenario} (static mesh)")
    if jaxpm_static is not None:
        H.save_summary_figure(jaxpm_static, out_path=root / "jaxpm_static_summary.png", cfg=cfg, title=f"{scenario} (jaxpm static)")
        H.save_jaxpm_compare_figure(
            static,
            jaxpm_static,
            out_path=root / "jaxpm_compare.png",
            cfg=cfg,
            title=f"{scenario}: static PM vs jaxpm static PM",
        )
    H.save_mesh_vs_phys_figure(moving, out_path=root / "moving_mesh_vs_phys.png", cfg=cfg, title=f"{scenario} (moving mesh)")
    H.save_mesh_vs_phys_figure(static, out_path=root / "static_mesh_vs_phys.png", cfg=cfg, title=f"{scenario} (static mesh)")
    H.save_compare_figure(moving, static, out_path=root / "compare.png", cfg=cfg, title=f"{scenario}: moving vs static (physical grid)")
    H.save_density_mesh_vs_phys(moving, out_path=root / "moving_density_mesh_vs_phys.png", cfg=cfg, title=f"{scenario} (moving): density in physical vs mesh coords")
    H.save_density_mesh_vs_phys(static, out_path=root / "static_density_mesh_vs_phys.png", cfg=cfg, title=f"{scenario} (static): density in physical vs mesh coords")

    if cfg.animate:
        if args.anim_out is None:
            mov_gif = root / "moving_anim.gif"
            sta_gif = root / "static_anim.gif"
        else:
            base = Path(args.anim_out)
            mov_gif = root / f"{base.stem}_moving.gif"
            sta_gif = root / f"{base.stem}_static.gif"
        H.save_animation_gif(moving, out_path=mov_gif, cfg=cfg, title_prefix="")
        H.save_animation_gif(static, out_path=sta_gif, cfg=cfg, title_prefix="")

    # Save analysis bundles for post-processing
    H.save_state_npz(moving, out_path=root / "moving_state.npz", cfg=cfg, center=center)
    H.save_state_npz(static, out_path=root / "static_state.npz", cfg=cfg, center=center)
    if jaxpm_static is not None:
        H.save_state_npz(jaxpm_static, out_path=root / "jaxpm_static_state.npz", cfg=cfg, center=center)

    (root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print("\n[metrics]")
    print(metrics)
    print(f"\n[test_configurations] wrote outputs to {root}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
