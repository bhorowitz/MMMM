"""Config file I/O + reproducibility helpers for diffAPM_new scripts.

Supported config formats:
  - TOML (recommended; Python 3.11+ stdlib `tomllib`)
  - JSON

The intent is:
  - load a parameter file to avoid long command lines
  - write the *resolved* configuration into the output directory so runs are
    easy to reproduce.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import datetime as _dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional


def _normalize_keys(obj: Any) -> Any:
    """Recursively replace '-' with '_' in dict keys for argparse compatibility."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            kk = str(k).replace("-", "_")
            out[kk] = _normalize_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(v) for v in obj]
    return obj


def load_config_file(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suf = p.suffix.lower()
    if suf == ".json":
        cfg = json.loads(p.read_text())
    elif suf in (".toml", ".tml"):
        # Python 3.11+ stdlib
        import tomllib  # type: ignore

        cfg = tomllib.loads(p.read_text())
    else:
        raise ValueError(f"Unsupported config format {p.suffix!r}; use .toml or .json")

    cfg = _normalize_keys(cfg)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must parse to a dict/table at the top level.")
    return cfg


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _git_info(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility (doesn't raise)."""
    out: Dict[str, Any] = {}
    try:
        kw = {"cwd": str(cwd) if cwd is not None else None, "text": True, "stderr": subprocess.DEVNULL}
        # NOTE: subprocess doesn't accept None for cwd.
        if kw["cwd"] is None:
            kw.pop("cwd")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], **kw).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], **kw).strip()
        out["commit"] = sha
        out["dirty"] = bool(status)
        out["status_porcelain"] = status.splitlines()[:200]  # keep it bounded
    except Exception:
        return {}
    return out


def write_repro_bundle(
    outdir: Path,
    *,
    argv: list[str],
    resolved_args: Dict[str, Any],
    scenario: str,
    cfg_obj: Any = None,
    config_path: Optional[str | Path] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Write run configuration files into outdir for easy reproduction."""
    outdir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "scenario": scenario,
        "argv": list(argv),
        "resolved_args": _jsonable(resolved_args),
    }
    if cfg_obj is not None:
        payload["run_config"] = _jsonable(cfg_obj)
    if config_path is not None:
        payload["config_path"] = str(config_path)
    if config_dict is not None:
        payload["config_dict"] = _jsonable(config_dict)

    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import numpy as np  # type: ignore

        versions["numpy"] = getattr(np, "__version__", "unknown")
    except Exception:
        pass
    try:
        import jax  # type: ignore

        versions["jax"] = getattr(jax, "__version__", "unknown")
    except Exception:
        pass
    payload["versions"] = versions

    payload["git"] = _git_info(outdir)

    (outdir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (outdir / "command.txt").write_text(" ".join(argv) + "\n")

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            (outdir / f"config_input{p.suffix.lower()}").write_text(p.read_text())
