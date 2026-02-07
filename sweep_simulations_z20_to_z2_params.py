#!/usr/bin/env python3
"""Parameter sweep for `test_simulations_z20_to_z2_compare.py`.

Designed for overnight tuning around a known-good configuration.

It runs the compare script many times with `--no-plots` (fastest), collects
`metrics.json` into `results.jsonl/results.json`, and writes `summary.json`
rankings by multiple metrics (absolute vs reference and gain vs static mesh).

Optionally, re-runs the top-K configs with plots enabled to produce figures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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
    return s.replace("-", "m").replace(".", "p")


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _rank(results: list[dict], *, key: str, mode: str, topk: int) -> list[dict]:
    def val(r: dict) -> float:
        x = r.get(key, np.nan)
        try:
            return float(x)
        except Exception:
            return float("nan")

    clean = [r for r in results if _finite(val(r))]
    reverse = mode == "max"
    clean.sort(key=val, reverse=reverse)
    return clean[: max(0, int(topk))]


def _add_gain_metrics(m: dict) -> dict:
    """Augment metrics dict with 'gain over static' scalars."""
    out = dict(m)
    for base in [
        "delta_rel_l2",
        "delta_corrcoef",
        "pk_logrmse",
        "pk_logrmse_hi",
        "corr_mean",
        "corr_hi_mean",
    ]:
        ks = f"static_vs_ref_{base}"
        km = f"moving_vs_ref_{base}"
        if ks in out and km in out and _finite(out[ks]) and _finite(out[km]):
            if "corr" in base:
                # Higher is better -> gain = moving - static
                out[f"gain_moving_minus_static_{base}"] = float(out[km]) - float(out[ks])
            else:
                # Lower is better -> gain = static - moving
                out[f"gain_static_minus_moving_{base}"] = float(out[ks]) - float(out[km])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", type=str, default="./test_outputs/simulations_param_sweep")
    ap.add_argument("--script", type=str, default="test_simulations_z20_to_z2_compare.py")

    # Base run args
    ap.add_argument("--simdir", type=str, default="./simulations")
    ap.add_argument("--box", type=float, default=25.0)
    ap.add_argument("--z0", type=float, default=20.0)
    ap.add_argument("--zfinal", type=float, default=2.0)
    ap.add_argument("--ng-force", type=int, default=64)
    ap.add_argument("--ng-field", type=int, default=128)
    ap.add_argument("--ng-in", type=int, default=0)

    ap.add_argument("--mg-cycles", type=int, default=10)
    ap.add_argument("--mg-v1", type=int, default=2)
    ap.add_argument("--mg-v2", type=int, default=2)
    ap.add_argument("--mg-mu", type=int, default=2)

    ap.add_argument("--limiter-mode", type=str, default="source", choices=["both", "source", "post"])
    ap.add_argument("--local-limit", action="store_true")
    ap.add_argument("--global-limit", action="store_true")

    ap.add_argument("--no-limiter", action="store_true")

    # Sweep lists
    ap.add_argument("--steps-list", type=str, default="48,64,96")
    ap.add_argument("--kappas", type=str, default="3.0,4.0,5.0")
    ap.add_argument("--compressmax-list", type=str, default="60,80,100")
    ap.add_argument("--skewmax-list", type=str, default="40,60,80")

    ap.add_argument("--smooth-steps-list", type=str, default="0,1,2")
    ap.add_argument("--xhigh-list", type=str, default="2.0")
    ap.add_argument("--relax-steps-list", type=str, default="30")
    ap.add_argument("--hard-strength-list", type=str, default="1.0")
    ap.add_argument("--limiter-smooth-tmp-list", type=str, default="3")
    ap.add_argument("--limiter-smooth-hard-list", type=str, default="1")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="0 means run all.")

    ap.add_argument(
        "--plots-mode",
        type=str,
        default="topk",
        choices=["none", "topk", "all"],
        help="Plot generation strategy: none (fast), topk (rerun best), all (generate plots for every run).",
    )
    ap.add_argument("--rerun-topk-plots", type=int, default=5)
    ap.add_argument("--rerun-metric", type=str, default="moving_vs_ref_pk_logrmse_hi")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    steps_list = _parse_ints_csv(args.steps_list)
    kappas = _parse_floats_csv(args.kappas)
    compress_list = _parse_floats_csv(args.compressmax_list)
    skew_list = _parse_floats_csv(args.skewmax_list)
    smooth_list = _parse_ints_csv(args.smooth_steps_list)
    xhigh_list = _parse_floats_csv(args.xhigh_list)
    relax_list = _parse_floats_csv(args.relax_steps_list)
    hard_list = _parse_floats_csv(args.hard_strength_list)
    smooth_tmp_list = _parse_ints_csv(args.limiter_smooth_tmp_list)
    smooth_hard_list = _parse_ints_csv(args.limiter_smooth_hard_list)

    sweep_config = {
        "script": str(args.script),
        "simdir": str(args.simdir),
        "box": float(args.box),
        "z0": float(args.z0),
        "zfinal": float(args.zfinal),
        "ng_force": int(args.ng_force),
        "ng_field": int(args.ng_field),
        "ng_in": int(args.ng_in),
        "mg_cycles": int(args.mg_cycles),
        "mg_v1": int(args.mg_v1),
        "mg_v2": int(args.mg_v2),
        "mg_mu": int(args.mg_mu),
        "limiter_mode": str(args.limiter_mode),
        "local_limit": bool(args.local_limit),
        "global_limit": bool(args.global_limit),
        "no_limiter": bool(args.no_limiter),
        "plots_mode": str(args.plots_mode),
        "steps_list": steps_list,
        "kappas": kappas,
        "compressmax_list": compress_list,
        "skewmax_list": skew_list,
        "smooth_steps_list": smooth_list,
        "xhigh_list": xhigh_list,
        "relax_steps_list": relax_list,
        "hard_strength_list": hard_list,
        "limiter_smooth_tmp_list": smooth_tmp_list,
        "limiter_smooth_hard_list": smooth_hard_list,
    }
    _save_json(outroot / "sweep_config.json", sweep_config)

    combos = list(
        itertools.product(
            steps_list,
            kappas,
            compress_list,
            skew_list,
            smooth_list,
            xhigh_list,
            relax_list,
            hard_list,
            smooth_tmp_list,
            smooth_hard_list,
        )
    )
    if int(args.max_runs) > 0:
        combos = combos[: int(args.max_runs)]

    results_jsonl = outroot / "results.jsonl"
    results: list[dict] = []
    t0 = time.perf_counter()

    for idx, (steps, kappa, compressmax, skewmax, smooth_steps, xhigh, relax_steps, hard_strength, sm_tmp, sm_hard) in enumerate(combos, start=1):
        run_name = (
            f"st{int(steps)}"
            f"_k{_fmt_float_for_path(float(kappa))}"
            f"_cmp{_fmt_float_for_path(float(compressmax))}"
            f"_sk{_fmt_float_for_path(float(skewmax))}"
            f"_sm{int(smooth_steps)}"
            f"_xh{_fmt_float_for_path(float(xhigh))}"
            f"_rs{_fmt_float_for_path(float(relax_steps))}"
            f"_hs{_fmt_float_for_path(float(hard_strength))}"
            f"_tmp{int(sm_tmp)}"
            f"_hard{int(sm_hard)}"
        )
        run_dir = outroot / run_name
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists() and not bool(args.overwrite):
            m = _add_gain_metrics(_load_json(metrics_path))
            results.append(m)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(args.script),
            "--simdir",
            str(args.simdir),
            "--box",
            str(float(args.box)),
            "--z0",
            str(float(args.z0)),
            "--zfinal",
            str(float(args.zfinal)),
            "--ng-force",
            str(int(args.ng_force)),
            "--ng-field",
            str(int(args.ng_field)),
            "--steps",
            str(int(steps)),
            "--kappa",
            str(float(kappa)),
            "--mg-cycles",
            str(int(args.mg_cycles)),
            "--mg-v1",
            str(int(args.mg_v1)),
            "--mg-v2",
            str(int(args.mg_v2)),
            "--mg-mu",
            str(int(args.mg_mu)),
            "--compressmax",
            str(float(compressmax)),
            "--skewmax",
            str(float(skewmax)),
            "--smooth-steps",
            str(int(smooth_steps)),
            "--xhigh",
            str(float(xhigh)),
            "--relax-steps",
            str(float(relax_steps)),
            "--hard-strength",
            str(float(hard_strength)),
            "--limiter-smooth-tmp",
            str(int(sm_tmp)),
            "--limiter-smooth-hard",
            str(int(sm_hard)),
            "--limiter-mode",
            str(args.limiter_mode),
            "--outdir",
            str(run_dir),
        ]
        if str(args.plots_mode) != "all":
            cmd += ["--no-plots"]
        if int(args.ng_in) > 0:
            cmd += ["--ng-in", str(int(args.ng_in))]
        if bool(args.no_limiter):
            cmd += ["--no-limiter"]
        if bool(args.local_limit):
            cmd += ["--local-limit"]
        if bool(args.global_limit):
            cmd += ["--global-limit"]

        t_case = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")

        if proc.returncode != 0:
            fail = {
                "run_dir": str(run_dir),
                "returncode": int(proc.returncode),
                "cmd": cmd,
            }
            _save_json(run_dir / "FAILED.json", fail)
            print(f"[sweep] {idx:4d}/{len(combos)} FAILED: {run_name}")
            continue

        m = _add_gain_metrics(_load_json(metrics_path))
        m["run_dir"] = str(run_dir)
        m["sweep_case_sec"] = float(time.perf_counter() - t_case)
        _append_jsonl(results_jsonl, m)
        results.append(m)

        if idx % 5 == 0 or idx == 1 or idx == len(combos):
            key = str(args.rerun_metric)
            print(f"[sweep] {idx:4d}/{len(combos)} ok: {key}={m.get(key)}  ({run_name})")

    _save_json(outroot / "results.json", results)

    # Summaries
    summary: dict = {
        "n_results": len(results),
        "elapsed_sec": float(time.perf_counter() - t0),
        "rankings": {},
    }
    summary["rankings"]["min_moving_pk_logrmse_hi"] = _rank(results, key="moving_vs_ref_pk_logrmse_hi", mode="min", topk=10)
    summary["rankings"]["min_moving_delta_rel_l2"] = _rank(results, key="moving_vs_ref_delta_rel_l2", mode="min", topk=10)
    summary["rankings"]["max_moving_corr_hi_mean"] = _rank(results, key="moving_vs_ref_corr_hi_mean", mode="max", topk=10)

    summary["rankings"]["max_gain_static_minus_moving_pk_logrmse_hi"] = _rank(
        results, key="gain_static_minus_moving_pk_logrmse_hi", mode="max", topk=10
    )
    summary["rankings"]["max_gain_static_minus_moving_delta_rel_l2"] = _rank(
        results, key="gain_static_minus_moving_delta_rel_l2", mode="max", topk=10
    )
    _save_json(outroot / "summary.json", summary)

    # Rerun top-K with plots enabled (optional)
    topk = int(args.rerun_topk_plots)
    metric = str(args.rerun_metric)
    if str(args.plots_mode) == "topk" and topk > 0 and metric:
        ranked = _rank(results, key=metric, mode="min", topk=topk)
        for r in ranked:
            run_dir = Path(str(r.get("run_dir", "")))
            if not run_dir.exists():
                continue
            # Skip if plots already exist
            if (run_dir / "pk_and_crosscorr_vs_ref.png").exists() and (run_dir / "projected_density_grid.png").exists():
                continue
            m = _load_json(run_dir / "metrics.json")
            cmd = [
                sys.executable,
                str(args.script),
                "--simdir",
                str(args.simdir),
                "--box",
                str(float(args.box)),
                "--z0",
                str(float(args.z0)),
                "--zfinal",
                str(float(args.zfinal)),
                "--ng-force",
                str(int(args.ng_force)),
                "--ng-field",
                str(int(args.ng_field)),
                "--steps",
                str(int(m["steps"])),
                "--kappa",
                str(float(m["kappa"])),
                "--mg-cycles",
                str(int(m.get("mg_cycles", args.mg_cycles))),
                "--mg-v1",
                str(int(m.get("mg_v1", args.mg_v1))),
                "--mg-v2",
                str(int(m.get("mg_v2", args.mg_v2))),
                "--mg-mu",
                str(int(m.get("mg_mu", args.mg_mu))),
                "--compressmax",
                str(float(m.get("compressmax", 80.0))),
                "--skewmax",
                str(float(m.get("skewmax", 60.0))),
                "--smooth-steps",
                str(int(m.get("smooth_steps", 1))),
                "--xhigh",
                str(float(m.get("xhigh", 2.0))),
                "--relax-steps",
                str(float(m.get("relax_steps", 30.0))),
                "--hard-strength",
                str(float(m.get("hard_strength", 1.0))),
                "--limiter-smooth-tmp",
                str(int(m.get("limiter_smooth_tmp", 3))),
                "--limiter-smooth-hard",
                str(int(m.get("limiter_smooth_hard", 1))),
                "--limiter-mode",
                str(m.get("limiter_mode", args.limiter_mode)),
                "--outdir",
                str(run_dir),
            ]
            if int(args.ng_in) > 0:
                cmd += ["--ng-in", str(int(args.ng_in))]
            if bool(m.get("limiter", True)) is False:
                cmd += ["--no-limiter"]
            if bool(m.get("local_limit", True)):
                cmd += ["--local-limit"]
            else:
                cmd += ["--global-limit"]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (run_dir / "stdout_plots.txt").write_text(proc.stdout, encoding="utf-8")

    print(f"[sweep] wrote {outroot / 'summary.json'}")


if __name__ == "__main__":
    main()
