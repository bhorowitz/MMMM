#!/usr/bin/env python3
"""Overnight parameter sweep for Option A moving-mesh tuning.

Sweeps (kappa, smooth_steps, mg_cycles) for the moving-mesh evolution while keeping
the same GRF+LPT initial conditions and the same jaxpm references:
  - jaxpm 2x force-mesh reference (refine=2)
  - jaxpm 1x baseline (refine=1)

For each configuration, writes a per-run folder containing:
  - metrics.json
  - projections.npz
  - spectra_and_corr.npz

Optionally, generates plots for the top-K configurations according to a chosen metric.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import jax
import jax.numpy as jnp

import test_optionA_resolution_compare as optA


def _parse_floats_csv(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise SystemExit("empty float list")
    return out


def _parse_ints_csv(s: str) -> list[int]:
    out: list[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise SystemExit("empty int list")
    return out


def _fmt_float_for_path(x: float) -> str:
    s = f"{float(x):.4g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _iter_product(*iters: Iterable[object]) -> Iterable[tuple[object, ...]]:
    return itertools.product(*iters)


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _rank(results: list[dict], *, key: str, mode: str, topk: int) -> list[dict]:
    def val(r: dict) -> float:
        x = r.get(key, np.nan)
        try:
            return float(x)
        except Exception:
            return float("nan")

    clean = [r for r in results if np.isfinite(val(r))]
    reverse = mode == "max"
    clean.sort(key=val, reverse=reverse)
    return clean[: max(0, int(topk))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=str, default="./test_outputs/optionA_param_sweep")

    ap.add_argument("--ng", type=int, default=64)
    ap.add_argument("--box", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=str, default="0.1,1.0")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--omega-c", type=float, default=0.25)
    ap.add_argument("--sigma8", type=float, default=0.8)

    ap.add_argument("--kappas", type=str, default="0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--smooth-steps-list", type=str, default="0,1,2,3")
    ap.add_argument("--mg-cycles-list", type=str, default="6,8,10,12")

    ap.add_argument("--no-limiter", action="store_true")
    ap.add_argument("--compressmax", type=float, default=80.0)
    ap.add_argument("--skewmax", type=float, default=60.0)
    ap.add_argument("--limiter-mode", type=str, default="source", choices=["both", "source", "post"])
    ap.add_argument("--limiter-smooth-tmp", type=int, default=None)
    ap.add_argument("--limiter-smooth-defp", type=int, default=None, help="If unset, uses smooth_steps from the sweep.")
    ap.add_argument("--limiter-smooth-hard", type=int, default=None)

    ap.add_argument("--proj-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="0 means no limit (run all).")

    ap.add_argument("--plots-topk", type=int, default=5)
    ap.add_argument("--plots-metric", type=str, default="moving1x_vs_jaxpm2x_rho_rel_l2")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    a0, afinal = optA._parse_snapshots(args.snapshots)
    ng = int(args.ng)
    steps = int(args.steps)

    sweep_config = {
        "ng": ng,
        "box": float(args.box),
        "seed": int(args.seed),
        "a0": float(a0),
        "afinal": float(afinal),
        "steps": steps,
        "omega_c": float(args.omega_c),
        "sigma8": float(args.sigma8),
        "kappas": _parse_floats_csv(args.kappas),
        "smooth_steps_list": _parse_ints_csv(args.smooth_steps_list),
        "mg_cycles_list": _parse_ints_csv(args.mg_cycles_list),
        "limiter": bool(not args.no_limiter),
        "compressmax": float(args.compressmax),
        "skewmax": float(args.skewmax),
        "limiter_mode": str(args.limiter_mode),
        "limiter_smooth_tmp": None if args.limiter_smooth_tmp is None else int(args.limiter_smooth_tmp),
        "limiter_smooth_defp": None if args.limiter_smooth_defp is None else int(args.limiter_smooth_defp),
        "limiter_smooth_hard": None if args.limiter_smooth_hard is None else int(args.limiter_smooth_hard),
        "proj_axis": int(args.proj_axis),
        "device": str(jax.devices()[0]),
    }
    _save_json(outroot / "sweep_config.json", sweep_config)

    # --- Precompute shared ICs and jaxpm reference/baseline once ---
    print("[sweep] building ICs + jaxpm references...")
    cosmo, pos0_lo, p0_lo, pmass = optA.make_ics(
        ng=ng,
        box=float(args.box),
        seed=int(args.seed),
        omega_c=float(args.omega_c),
        sigma8=float(args.sigma8),
        a0=float(a0),
    )
    pos_j2 = optA._evolve_jaxpm_fixed_refined(pos0_lo, p0_lo, ng=ng, refine=2, pmass=pmass, cosmo=cosmo, a0=float(a0), afinal=float(afinal), steps=steps)
    pos_j1 = optA._evolve_jaxpm_fixed_refined(pos0_lo, p0_lo, ng=ng, refine=1, pmass=pmass, cosmo=cosmo, a0=float(a0), afinal=float(afinal), steps=steps)

    rho_ref = optA._density_on_ng_from_pos(jnp.mod(pos_j2, float(ng)), pmass, ng=ng)
    rho_j1 = optA._density_on_ng_from_pos(jnp.mod(pos_j1, float(ng)), pmass, ng=ng)
    proj_ref = optA._project(rho_ref, axis=int(args.proj_axis))
    proj_j1 = optA._project(rho_j1, axis=int(args.proj_axis))

    np.savez(outroot / "reference_fields.npz", proj_ref=proj_ref, proj_j1=proj_j1)

    # --- Sweep ---
    results_jsonl = outroot / "results.jsonl"
    results: list[dict] = []
    note = "2x reference uses a 2ng force mesh on the same [0,ng) domain (positions unchanged; force mesh refined)."

    combos = list(
        _iter_product(
            sweep_config["kappas"],
            sweep_config["smooth_steps_list"],
            sweep_config["mg_cycles_list"],
        )
    )
    max_runs = int(args.max_runs)
    if max_runs > 0:
        combos = combos[:max_runs]

    print(f"[sweep] running {len(combos)} configs into {outroot} ...")
    for idx, (kappa, smooth_steps, mg_cycles) in enumerate(combos, start=1):
        run_name = f"k{_fmt_float_for_path(float(kappa))}_s{int(smooth_steps)}_mg{int(mg_cycles)}"
        run_dir = outroot / run_name
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists() and not bool(args.overwrite):
            m = _load_json(metrics_path)
            results.append(m)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        t_case0 = time.perf_counter()
        pos_mm = optA.evolve_moving_mesh(
            pos0_lo,
            p0_lo,
            pmass,
            ng=ng,
            cosmo=cosmo,
            a0=float(a0),
            afinal=float(afinal),
            steps=steps,
            kappa=float(kappa),
            smooth_steps=int(smooth_steps),
            mg_cycles=int(mg_cycles),
            no_limiter=bool(args.no_limiter),
            compressmax=float(args.compressmax),
            skewmax=float(args.skewmax),
            limiter_mode=str(args.limiter_mode),
            limiter_smooth_tmp=args.limiter_smooth_tmp,
            limiter_smooth_defp=args.limiter_smooth_defp,
            limiter_smooth_hard=args.limiter_smooth_hard,
        )
        rho_mm = optA._density_on_ng_from_pos(pos_mm, pmass, ng=ng)
        proj_mm = optA._project(rho_mm, axis=int(args.proj_axis))

        k_plot, pk_ref, pk_j1, pk_mm, r_j1, r_mm = optA.compute_spectra_and_corr(rho_ref, rho_j1, rho_mm, box=float(args.box))
        metrics = optA.compute_scalar_metrics(
            ng=ng,
            box=float(args.box),
            a0=float(a0),
            afinal=float(afinal),
            steps=steps,
            kappa=float(kappa),
            limiter_enabled=bool(not args.no_limiter),
            rho_ref=rho_ref,
            rho_j1=rho_j1,
            rho_mm=rho_mm,
            k=k_plot,
            pk_ref=pk_ref,
            pk_j1=pk_j1,
            pk_mm=pk_mm,
            r_j1=r_j1,
            r_mm=r_mm,
            note=note,
        )
        metrics.update(
            {
                "smooth_steps": int(smooth_steps),
                "mg_cycles": int(mg_cycles),
                "limiter_mode": str(args.limiter_mode),
                "limiter_smooth_tmp": None if args.limiter_smooth_tmp is None else int(args.limiter_smooth_tmp),
                "limiter_smooth_defp": None if args.limiter_smooth_defp is None else int(args.limiter_smooth_defp),
                "limiter_smooth_hard": None if args.limiter_smooth_hard is None else int(args.limiter_smooth_hard),
                "compressmax": float(args.compressmax),
                "skewmax": float(args.skewmax),
                "timing_sec": {
                    "moving_mesh_case": float(time.perf_counter() - t_case0),
                },
                "run_dir": str(run_dir),
            }
        )

        np.savez(
            run_dir / "spectra_and_corr.npz",
            k=k_plot,
            pk_ref=pk_ref,
            pk_j1=pk_j1,
            pk_mm=pk_mm,
            r_j1=r_j1,
            r_mm=r_mm,
        )
        np.savez(
            run_dir / "projections.npz",
            proj_ref=proj_ref,
            proj_j1=proj_j1,
            proj_mm=proj_mm,
        )
        _save_json(metrics_path, metrics)
        _append_jsonl(results_jsonl, metrics)
        results.append(metrics)

        if idx % 5 == 0 or idx == 1 or idx == len(combos):
            x = metrics.get("moving1x_vs_jaxpm2x_rho_rel_l2", None)
            print(f"[sweep] {idx:4d}/{len(combos)} done: kappa={kappa} smooth={smooth_steps} mg={mg_cycles} rel_l2={x}")

    _save_json(outroot / "results.json", results)

    # --- Summaries / best-of ---
    summary: dict = {"n_results": len(results), "rankings": {}}
    summary["rankings"]["min_moving_rho_rel_l2"] = _rank(results, key="moving1x_vs_jaxpm2x_rho_rel_l2", mode="min", topk=10)
    summary["rankings"]["min_moving_pk_logrmse_hi"] = _rank(results, key="moving1x_vs_jaxpm2x_pk_logrmse_hi", mode="min", topk=10)
    summary["rankings"]["max_moving_corr_hi_mean"] = _rank(results, key="moving1x_vs_jaxpm2x_corr_hi_mean", mode="max", topk=10)
    _save_json(outroot / "summary.json", summary)

    # --- Plots for top-K ---
    topk = int(args.plots_topk)
    if topk > 0 and str(args.plots_metric):
        ranked = _rank(results, key=str(args.plots_metric), mode="min", topk=topk)
        for r in ranked:
            run_dir = Path(str(r.get("run_dir", "")))
            if not run_dir.exists():
                continue
            proj_path = run_dir / "projections.npz"
            spec_path = run_dir / "spectra_and_corr.npz"
            if not proj_path.exists() or not spec_path.exists():
                continue

            proj = np.load(proj_path)
            spec = np.load(spec_path)
            optA._plot_density_triptych(
                run_dir / "projected_density_triptych.png",
                proj_ref=np.array(proj["proj_ref"]),
                proj_j1=np.array(proj["proj_j1"]),
                proj_mm=np.array(proj["proj_mm"]),
                title=f"Option A sweep: ng={ng}, steps={steps}, a={a0:g}->{afinal:g}, axis={int(args.proj_axis)}",
            )
            optA._plot_pk_and_corr(
                run_dir / "pk_and_crosscorr.png",
                k=np.array(spec["k"]),
                pk_ref=np.array(spec["pk_ref"]),
                pk_j1=np.array(spec["pk_j1"]),
                pk_mm=np.array(spec["pk_mm"]),
                r_j1=np.array(spec["r_j1"]),
                r_mm=np.array(spec["r_mm"]),
                title="Power spectra + cross-correlation (vs jaxpm 2x ref)",
            )

    print(f"[sweep] wrote {outroot / 'summary.json'}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    main()
