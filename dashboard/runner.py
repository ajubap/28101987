"""Shared helpers for the dashboard: project paths, config discovery, and a
single place that shells out to the project's CLI scripts.

The dashboard is a thin control surface over the existing ``scripts/*.py``
command-line tools. Keeping all subprocess handling here means the page modules
stay declarative and never build commands by hand.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = "configs/generic_cfd_template.yaml"

# Make the backend package importable for the few pages that register cases
# directly (uploads) rather than via a CLI script.
_SRC = str(PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def list_configs() -> list[str]:
    """Return repo-relative paths of every YAML config, generic template first."""
    configs = sorted(str(p.relative_to(PROJECT_ROOT)) for p in CONFIG_DIR.glob("*.y*ml"))
    configs.sort(key=lambda p: (DEFAULT_CONFIG not in p, p))
    return configs


def load_cfg(config_path: str) -> dict[str, Any]:
    path = (PROJECT_ROOT / config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cfg_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve the config's declared paths against the project root."""
    paths = cfg.get("paths", {})
    return {key: (PROJECT_ROOT / value) for key, value in paths.items()}


def ensure_data_dirs(cfg: dict[str, Any]) -> None:
    paths = cfg_paths(cfg)
    for key in ("raw_dir", "processed_dir", "output_dir"):
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)


def run_script(args: list[str], timeout: int = 60 * 60) -> dict[str, Any]:
    """Run ``python <args...>`` from the project root and capture output.

    Returns a dict with ``cmd``, ``returncode``, ``stdout``, ``stderr`` so the
    UI can render results uniformly.
    """
    cmd = [sys.executable, *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s.",
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": str(exc)}


def save_upload(contents: str, filename: str, dest_dir: Path) -> Path:
    """Persist a ``dcc.Upload`` payload (``data:...;base64,XXXX``) to disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    _, _, b64 = contents.partition(",")
    data = base64.b64decode(b64)
    out_path = dest_dir / filename
    out_path.write_bytes(data)
    return out_path


def global_cli_args(values: dict[str, float]) -> list[str]:
    """Expand {name: value} into repeated ``--global name=value`` CLI flags."""
    args: list[str] = []
    for name, value in values.items():
        if value is None or value == "":
            continue
        args.extend(["--global", f"{name}={value}"])
    return args
