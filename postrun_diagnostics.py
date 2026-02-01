#!/usr/bin/env python3
"""Post-run diagnostics for moving-mesh vs static PM runs.

This script consumes the saved `.npz` bundles produced by `test_configurations.py`
(`moving_state.npz`, `static_state.npz`, and optionally a high-res reference),
and generates the following comparisons:

  - rho(r) with a convergence-radius criterion
  - f(>rho) high-density tail curve
  - sigma(r) + Q(r) (phase-space fidelity proxy)
  - DeltaE/E and COM drift
  - accuracy vs cost curve relative to a high-res reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np


def _load_npz(path: Path) -> Dict[str, object]:
    d = dict(np.load(path, allow_pickle=True))
    meta_json = d.get("meta_json", None)
    if meta_json is not None:
        try:
            meta = json.loads(str(meta_json.tolist()))
        except Exception:
            meta = {}
    else:
        meta = {}
    d["meta"] = meta
    return d


def _minimal_image(dx: np.ndarray, box: float) -> np.ndarray:
    return (dx + 0.5 * box) % box - 0.5 * box


def _radii(x: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    d = _minimal_image(x - center[None, :], box)
    return np.sqrt(np.sum(d * d, axis=1))


def radial_profiles(
    x: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    *,
    center: Tuple[float, float, float],
    box: float,
    nbins: int,
    rmin: float,
    rmax: float,
) -> Dict[str, np.ndarray]:
    """Compute rho(r), sigma(r), Q(r) on spherical shells."""
    c = np.array(center, dtype=np.float64)
    r = _radii(x, c, float(box))

    edges = np.geomspace(max(rmin, 1e-6), rmax, int(nbins) + 1)
    rc = np.sqrt(edges[:-1] * edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    rho = np.zeros_like(rc)
    sig = np.zeros_like(rc)
    Q = np.zeros_like(rc)
    npart = np.zeros_like(rc)

    for i in range(rc.size):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        npart[i] = float(np.sum(sel))
        if not np.any(sel):
            rho[i] = np.nan
            sig[i] = np.nan
            Q[i] = np.nan
            continue
        mi = m[sel]
        vi = v[sel]
        mtot = float(np.sum(mi))
        rho[i] = mtot / shell_vol[i]

        vmean = np.sum(vi * mi[:, None], axis=0) / (mtot + 1e-30)
        dv = vi - vmean[None, :]
        # mass-weighted 3D dispersion
        sig2 = np.sum(mi * np.sum(dv * dv, axis=1)) / (mtot + 1e-30)
        sig[i] = np.sqrt(max(sig2, 0.0))

        sig1 = sig[i] / np.sqrt(3.0)
        Q[i] = rho[i] / (sig1 ** 3 + 1e-30)

    return {"r": rc, "rho": rho, "sigma3": sig, "Q": Q, "npart": npart, "edges": edges}


def convergence_radius(
    prof: Dict[str, np.ndarray],
    prof_ref: Dict[str, np.ndarray],
    *,
    eps_rel: float = 0.1,
    nmin_shell: int = 50,
    rmin_floor: float = 0.0,
) -> float:
    """Empirical convergence radius: smallest r such that rho(r) agrees with reference.

    Criterion:
      - for all bins >= i, |rho-rho_ref|/|rho_ref| < eps_rel
      - and each bin has at least nmin_shell particles
      - and r >= rmin_floor
    """
    r = prof["r"]
    rho = prof["rho"]
    rho_ref = prof_ref["rho"]
    n = prof["npart"]

    ok = np.isfinite(rho) & np.isfinite(rho_ref) & (np.abs(rho_ref) > 0) & (n >= float(nmin_shell)) & (r >= float(rmin_floor))
    rel = np.full_like(rho, np.inf, dtype=np.float64)
    rel[ok] = np.abs(rho[ok] - rho_ref[ok]) / np.abs(rho_ref[ok])

    for i in range(r.size):
        if not ok[i]:
            continue
        tail_ok = ok[i:] & (rel[i:] < float(eps_rel))
        if np.all(tail_ok):
            return float(r[i])
    return float("nan")


def cdf_tail(field: np.ndarray, *, nbins: int = 200) -> Dict[str, np.ndarray]:
    """Return x (threshold) and f(>x) for a field (flattened)."""
    x = np.asarray(field, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"x": np.array([]), "fgt": np.array([])}

    # Work in overdensity to compare across resolutions.
    mean = np.mean(x)
    if mean == 0:
        mean = 1.0
    od = x / mean

    # thresholds in logspace
    lo = max(1e-3, float(np.percentile(od, 50)))
    hi = float(np.percentile(od, 99.9))
    if hi <= lo:
        hi = lo * 10.0
    thr = np.geomspace(lo, hi, int(nbins))
    fgt = np.array([(od >= t).mean() for t in thr], dtype=np.float64)
    return {"x": thr, "fgt": fgt}


def energy_and_com(npz: Dict[str, object], *, center: Tuple[float, float, float], box: float) -> Dict[str, np.ndarray]:
    """Return time arrays and drift curves if histories exist.

    Notes
    -----
    Older saved bundles may omit E_hist/com_hist. Additionally, dt_hist length
    can differ from the length of E/com histories (e.g. partial writes); we
    therefore return per-curve time axes clipped to matching lengths.
    """
    dt = np.array(npz.get("dt_hist", np.array([])), dtype=np.float64).ravel()

    E = np.array(npz.get("E_hist", np.array([])), dtype=np.float64).ravel()
    nE = int(min(dt.size, E.size))
    if nE > 0:
        tE = np.cumsum(dt[:nE])
        E0 = float(E[0])
        denom = abs(E0) if abs(E0) > 0 else 1.0
        dE = (E[:nE] - E0) / denom
    else:
        tE = np.array([])
        dE = np.array([])

    com = np.array(npz.get("com_hist", np.zeros((0, 3))), dtype=np.float64)
    nC = int(min(dt.size, com.shape[0] if com.ndim == 2 else 0))
    if nC > 0:
        tC = np.cumsum(dt[:nC])
        c = np.array(center, dtype=np.float64)
        drift = np.linalg.norm(_minimal_image(com[:nC] - c[None, :], float(box)), axis=1)
    else:
        tC = np.array([])
        drift = np.array([])

    return {"tE": tE, "dE": dE, "tC": tC, "com_drift": drift}


def cost_proxy(meta: Dict[str, object], mode: str) -> float:
    """Very rough cost proxy for accuracy-vs-cost.

    Prefer measured walltime if present; otherwise approximate:
      - moving: ntotal * ng^3 * (2*mg_cycles)
      - static: ntotal * ng^3 * log2(ng)
    """
    if "walltime_s" in meta:
        try:
            return float(meta["walltime_s"])
        except Exception:
            pass

    cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    ng = int(meta.get("ng", cfg.get("ng", 0)) or 0)
    ntotal = int(cfg.get("ntotal", 0) or 0)
    mg_cycles = int(cfg.get("mg_cycles", 0) or 0)
    if ng <= 0 or ntotal <= 0:
        return float("nan")
    if mode == "moving":
        return float(ntotal) * float(ng ** 3) * float(max(1, 2 * mg_cycles))
    return float(ntotal) * float(ng ** 3) * float(max(1.0, np.log2(ng)))


def error_metric_rho(prof: Dict[str, np.ndarray], prof_ref: Dict[str, np.ndarray], *, rmin: float) -> float:
    r = prof["r"]
    rho = prof["rho"]
    rr = prof_ref["r"]
    rho_ref = prof_ref["rho"]

    ok = np.isfinite(rho) & (r >= float(rmin))
    if not np.any(ok):
        return float("nan")

    # interpolate reference to run r-bins
    rho_ref_i = np.interp(r[ok], rr[np.isfinite(rho_ref)], rho_ref[np.isfinite(rho_ref)])
    y = np.log10(np.maximum(rho[ok], 1e-30))
    yref = np.log10(np.maximum(rho_ref_i, 1e-30))
    return float(np.sqrt(np.mean((y - yref) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moving", type=str, required=True, help="Path to moving_state.npz")
    ap.add_argument("--static", type=str, required=True, help="Path to static_state.npz")
    ap.add_argument("--ref", type=str, default=None, help="Optional high-res reference .npz (moving or static)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: alongside moving file)")
    ap.add_argument("--nbins", type=int, default=40, help="Radial bins")
    ap.add_argument("--eps-rel", type=float, default=0.1, help="Relative error threshold for r_conv")
    ap.add_argument("--nmin-shell", type=int, default=50, help="Minimum particles per shell for r_conv")
    ap.add_argument("--rmin-dx", type=float, default=2.0, help="Minimum r (in units of dx) to consider converged")
    args = ap.parse_args()

    mov = _load_npz(Path(args.moving))
    sta = _load_npz(Path(args.static))
    ref = _load_npz(Path(args.ref)) if args.ref is not None else None

    meta_m = mov.get("meta", {})
    box = float(meta_m.get("box", 1.0))
    ng = int(meta_m.get("ng", 0))
    dx = float(meta_m.get("dx", box / max(1, ng)))
    center = tuple(meta_m.get("center", (0.5 * box, 0.5 * box, 0.5 * box)))

    outdir = Path(args.outdir) if args.outdir is not None else Path(args.moving).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Radial profiles
    rmax = 0.5 * box
    rmin = float(args.rmin_dx) * dx
    x_m = np.array(mov.get("x_phys", np.zeros((0, 3))), dtype=np.float64)
    v_m = np.array(mov.get("v", np.zeros((x_m.shape[0], 3))), dtype=np.float64)
    m_m = np.array(mov.get("pmass", np.ones((x_m.shape[0],))), dtype=np.float64)
    prof_m = radial_profiles(
        x_m,
        v_m,
        m_m,
        center=center,
        box=box,
        nbins=int(args.nbins),
        rmin=rmin,
        rmax=rmax,
    )
    x_s = np.array(sta.get("x_phys", np.zeros((0, 3))), dtype=np.float64)
    v_s = np.array(sta.get("v", np.zeros((x_s.shape[0], 3))), dtype=np.float64)
    m_s = np.array(sta.get("pmass", np.ones((x_s.shape[0],))), dtype=np.float64)
    prof_s = radial_profiles(
        x_s,
        v_s,
        m_s,
        center=center,
        box=box,
        nbins=int(args.nbins),
        rmin=rmin,
        rmax=rmax,
    )
    if ref is not None:
        meta_r = ref.get("meta", {})
        center_r = tuple(meta_r.get("center", center))
        box_r = float(meta_r.get("box", box))
        x_r = np.array(ref.get("x_phys", np.zeros((0, 3))), dtype=np.float64)
        v_r = np.array(ref.get("v", np.zeros((x_r.shape[0], 3))), dtype=np.float64)
        m_r = np.array(ref.get("pmass", np.ones((x_r.shape[0],))), dtype=np.float64)
        prof_r = radial_profiles(
            x_r,
            v_r,
            m_r,
            center=center_r,
            box=box_r,
            nbins=int(args.nbins),
            rmin=max(1e-6, rmin),
            rmax=0.5 * box_r,
        )
        rconv_m = convergence_radius(prof_m, prof_r, eps_rel=float(args.eps_rel), nmin_shell=int(args.nmin_shell), rmin_floor=rmin)
        rconv_s = convergence_radius(prof_s, prof_r, eps_rel=float(args.eps_rel), nmin_shell=int(args.nmin_shell), rmin_floor=rmin)
    else:
        prof_r = None
        rconv_m = float("nan")
        rconv_s = float("nan")

    # f(>rho) tail curves (prefer physical-grid CIC density if stored in npz)
    if "rho_phys" in mov and "rho_phys" in sta:
        tail_m = cdf_tail(np.array(mov["rho_phys"], dtype=np.float64))
        tail_s = cdf_tail(np.array(sta["rho_phys"], dtype=np.float64))
        tail_r = cdf_tail(np.array(ref["rho_phys"], dtype=np.float64)) if (ref is not None and "rho_phys" in ref) else None
    else:
        tail_m = {"x": np.array([]), "fgt": np.array([])}
        tail_s = {"x": np.array([]), "fgt": np.array([])}
        tail_r = None

    # Energy + COM drift
    ec_m = energy_and_com(mov, center=center, box=box)
    ec_s = energy_and_com(sta, center=center, box=box)
    ec_r = energy_and_com(ref, center=center, box=box) if ref is not None else None

    # Plots
    import matplotlib.pyplot as plt

    # rho(r) + r_conv
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(prof_m["r"], prof_m["rho"], label="moving")
    ax.plot(prof_s["r"], prof_s["rho"], label="static")
    if prof_r is not None:
        ax.plot(prof_r["r"], prof_r["rho"], label="ref", lw=2.2, alpha=0.8)
        if np.isfinite(rconv_m):
            ax.axvline(rconv_m, color="C0", ls="--", alpha=0.7, label=f"r_conv moving={rconv_m:.3g}")
        if np.isfinite(rconv_s):
            ax.axvline(rconv_s, color="C1", ls="--", alpha=0.7, label=f"r_conv static={rconv_s:.3g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("rho(r)")
    ax.set_title("Radial density profile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "rho_profile.png", dpi=180)
    plt.close(fig)

    # f(>rho)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(tail_m["x"], tail_m["fgt"], label="moving")
    ax.plot(tail_s["x"], tail_s["fgt"], label="static")
    if tail_r is not None:
        ax.plot(tail_r["x"], tail_r["fgt"], label="ref", lw=2.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rho / <rho> threshold")
    ax.set_ylabel("f(>rho)")
    ax.set_title("High-density tail (volume fraction)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "f_gt_rho.png", dpi=180)
    plt.close(fig)

    # sigma(r) and Q(r)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.0, 5.2))
    ax0, ax1 = axes
    ax0.plot(prof_m["r"], prof_m["sigma3"], label="moving")
    ax0.plot(prof_s["r"], prof_s["sigma3"], label="static")
    if prof_r is not None:
        ax0.plot(prof_r["r"], prof_r["sigma3"], label="ref", lw=2.2, alpha=0.8)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("r")
    ax0.set_ylabel("sigma_3D(r)")
    ax0.set_title("Velocity dispersion")
    ax0.legend(frameon=False)

    ax1.plot(prof_m["r"], prof_m["Q"], label="moving")
    ax1.plot(prof_s["r"], prof_s["Q"], label="static")
    if prof_r is not None:
        ax1.plot(prof_r["r"], prof_r["Q"], label="ref", lw=2.2, alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("r")
    ax1.set_ylabel("Q(r) = rho/sigma_1D^3")
    ax1.set_title("Phase-space density proxy")
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "sigma_Q.png", dpi=180)
    plt.close(fig)

    # Energy drift + COM drift
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.0, 5.2))
    ax0, ax1 = axes
    if ec_m["tE"].size and ec_m["dE"].size:
        ax0.plot(ec_m["tE"], ec_m["dE"], label="moving")
    if ec_s["tE"].size and ec_s["dE"].size:
        ax0.plot(ec_s["tE"], ec_s["dE"], label="static")
    if ec_r is not None and ec_r["tE"].size and ec_r["dE"].size:
        ax0.plot(ec_r["tE"], ec_r["dE"], label="ref", lw=2.2, alpha=0.8)
    ax0.set_xlabel("t")
    ax0.set_ylabel("DeltaE/|E0|")
    ax0.set_title("Energy drift")
    ax0.grid(True, alpha=0.2)
    if ax0.lines:
        ax0.legend(frameon=False)

    if ec_m["tC"].size and ec_m["com_drift"].size:
        ax1.plot(ec_m["tC"], ec_m["com_drift"], label="moving")
    if ec_s["tC"].size and ec_s["com_drift"].size:
        ax1.plot(ec_s["tC"], ec_s["com_drift"], label="static")
    if ec_r is not None and ec_r["tC"].size and ec_r["com_drift"].size:
        ax1.plot(ec_r["tC"], ec_r["com_drift"], label="ref", lw=2.2, alpha=0.8)
    ax1.set_xlabel("t")
    ax1.set_ylabel("|COM - center|")
    ax1.set_title("COM drift")
    ax1.grid(True, alpha=0.2)
    if ax1.lines:
        ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "energy_com.png", dpi=180)
    plt.close(fig)

    # Accuracy vs cost (rho-profile error vs reference)
    if prof_r is not None:
        err_m = error_metric_rho(prof_m, prof_r, rmin=float(rconv_m) if np.isfinite(rconv_m) else rmin)
        err_s = error_metric_rho(prof_s, prof_r, rmin=float(rconv_s) if np.isfinite(rconv_s) else rmin)

        cm = cost_proxy(mov.get("meta", {}), "moving")
        cs = cost_proxy(sta.get("meta", {}), "static")
        cr = cost_proxy(ref.get("meta", {}), str(ref.get("meta", {}).get("mode", "ref"))) if ref is not None else float("nan")

        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        ax.scatter([cs], [err_s], label="static", s=70)
        ax.scatter([cm], [err_m], label="moving", s=70)
        ax.scatter([cr], [0.0], label="ref (0 error)", s=70, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log" if err_m > 0 and err_s > 0 else "linear")
        ax.set_xlabel("cost proxy (or walltime_s)")
        ax.set_ylabel("rho-profile RMSE (log10)")
        ax.set_title("Accuracy vs cost (relative to reference)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "accuracy_vs_cost.png", dpi=180)
        plt.close(fig)

        summary = {
            "rconv_moving": rconv_m,
            "rconv_static": rconv_s,
            "err_rho_moving": err_m,
            "err_rho_static": err_s,
            "cost_moving": cm,
            "cost_static": cs,
            "cost_ref": cr,
        }
    else:
        summary = {"note": "No --ref provided; skipping convergence and accuracy-vs-cost."}

    (outdir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[postrun_diagnostics] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
