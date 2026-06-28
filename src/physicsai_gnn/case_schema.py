from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import meshio
except Exception:
    meshio = None


def data_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("data", {})


def global_feature_columns(cfg: dict[str, Any]) -> list[str]:
    return list(data_cfg(cfg).get("global_feature_columns", []))


def point_feature_arrays(cfg: dict[str, Any]) -> list[str]:
    return list(data_cfg(cfg).get("point_feature_arrays", []))


def target_field_arrays(cfg: dict[str, Any]) -> list[str]:
    return list(data_cfg(cfg).get("target_field_arrays", []))


def target_scalar_columns(cfg: dict[str, Any]) -> list[str]:
    return list(data_cfg(cfg).get("target_scalar_columns", []))


def case_id_column(cfg: dict[str, Any]) -> str:
    return str(data_cfg(cfg).get("case_id_column", "case_id"))


def mesh_file_column(cfg: dict[str, Any]) -> str:
    return str(data_cfg(cfg).get("mesh_file_column", "mesh_file"))


def parse_key_value_pairs(pairs: list[str] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Expected name=value, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty metadata name in: {item}")
        values[name] = to_finite_float(value, name)
    return values


def load_json_values(path_or_json: str | None) -> dict[str, float]:
    if not path_or_json:
        return {}
    stripped = path_or_json.strip()
    text = stripped
    if not stripped.startswith("{"):
        candidate = Path(stripped)
        try:
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
        except OSError:
            text = stripped
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--globals_json must be a JSON object or a path to one.")
    return {str(k): to_finite_float(v, str(k)) for k, v in payload.items()}


def to_finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if pd.isna(value):
        return None
    return to_finite_float(value, name)


def build_global_features(cfg: dict[str, Any], values: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in global_feature_columns(cfg) if name not in values or values[name] in (None, "")]
    if missing:
        example = " ".join(f"--global {name}=0.0" for name in missing)
        raise ValueError(f"Missing required global feature(s): {missing}. Example: {example}")
    return {name: to_finite_float(values[name], name) for name in global_feature_columns(cfg)}


def row_to_global_features(cfg: dict[str, Any], row: Any) -> dict[str, float]:
    return build_global_features(cfg, {name: row[name] for name in global_feature_columns(cfg) if name in row})


def row_to_scalar_targets(cfg: dict[str, Any], row: Any, require: bool = True) -> dict[str, float] | None:
    scalars: dict[str, float] = {}
    missing: list[str] = []
    for name in target_scalar_columns(cfg):
        if name not in row:
            missing.append(name)
            continue
        value = optional_float(row[name], name)
        if value is None:
            missing.append(name)
            continue
        scalars[name] = value
    if missing and require:
        raise ValueError(f"Missing required scalar target(s): {missing}")
    return scalars or None


def append_case_to_metadata(
    metadata_csv: str | Path,
    cfg: dict[str, Any],
    case_id: str,
    mesh_file: str,
    values: dict[str, Any] | None = None,
) -> Path:
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    values = values or {}

    row: dict[str, Any] = {
        case_id_column(cfg): case_id,
        mesh_file_column(cfg): mesh_file,
    }
    for name in global_feature_columns(cfg):
        row[name] = to_finite_float(values.get(name), name)
    for name in target_scalar_columns(cfg):
        value = optional_float(values.get(name), name)
        row[name] = "" if value is None else value

    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        case_col = case_id_column(cfg)
        if case_col not in df.columns:
            raise ValueError(f"{metadata_csv} is missing case id column: {case_col}")
        df = df[df[case_col].astype(str) != str(case_id)]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(metadata_csv, index=False)
    return metadata_csv


def mesh_path_for_row(cfg: dict[str, Any], row: Any) -> Path:
    return Path(cfg["paths"]["raw_dir"]) / str(row[mesh_file_column(cfg)])


def read_first_case_row(csv_path: str | Path, case_id: str | None = None) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if case_id is not None:
        if "case_id" not in df.columns:
            raise ValueError(f"{csv_path} does not contain a case_id column.")
        df = df[df["case_id"].astype(str) == str(case_id)]
        if df.empty:
            raise ValueError(f"No case_id={case_id!r} found in {csv_path}")
    return df.iloc[0].to_dict()


def _mesh_arrays(mesh: Any) -> set[str]:
    names = set(mesh.point_data.keys())
    names.update(getattr(mesh, "cell_data_dict", {}).keys())
    return names


def validate_config(cfg: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for section in ["paths", "data", "model", "training"]:
        if section not in cfg:
            issues.append({"level": "error", "where": "config", "message": f"Missing section: {section}"})
    required_paths = ["raw_dir", "processed_dir", "metadata_csv", "output_dir"]
    for name in required_paths:
        if name not in cfg.get("paths", {}):
            issues.append({"level": "error", "where": "config.paths", "message": f"Missing path: {name}"})
    if not global_feature_columns(cfg):
        issues.append({"level": "warning", "where": "config.data", "message": "No global_feature_columns configured."})
    if not target_field_arrays(cfg):
        issues.append({"level": "error", "where": "config.data", "message": "No target_field_arrays configured."})
    duplicate_groups = {
        "global_feature_columns": global_feature_columns(cfg),
        "point_feature_arrays": point_feature_arrays(cfg),
        "target_field_arrays": target_field_arrays(cfg),
        "target_scalar_columns": target_scalar_columns(cfg),
    }
    for group, names in duplicate_groups.items():
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            issues.append({"level": "error", "where": f"config.data.{group}", "message": f"Duplicate names: {duplicates}"})
    return issues


def validate_metadata(cfg: dict[str, Any], check_mesh_arrays: bool = False) -> list[dict[str, str]]:
    issues = validate_config(cfg)
    metadata_csv = Path(cfg["paths"].get("metadata_csv", ""))
    if not metadata_csv.exists():
        issues.append({"level": "error", "where": "metadata", "message": f"Missing metadata CSV: {metadata_csv}"})
        return issues

    df = pd.read_csv(metadata_csv)
    case_col = case_id_column(cfg)
    mesh_col = mesh_file_column(cfg)
    required_cols = [case_col, mesh_col] + global_feature_columns(cfg) + target_scalar_columns(cfg)
    for col in required_cols:
        if col not in df.columns:
            issues.append({"level": "error", "where": "metadata", "message": f"Missing column: {col}"})
    if issues and any(issue["level"] == "error" and issue["where"] == "metadata" for issue in issues):
        return issues

    if df.empty:
        issues.append({"level": "error", "where": "metadata", "message": "No cases are registered."})
        return issues

    duplicates = df[case_col][df[case_col].astype(str).duplicated()].astype(str).tolist()
    if duplicates:
        issues.append({"level": "error", "where": "metadata", "message": f"Duplicate case ids: {duplicates}"})

    raw_dir = Path(cfg["paths"]["raw_dir"])
    for idx, row in df.iterrows():
        label = f"row {idx + 2} ({row.get(case_col, '<missing>')})"
        mesh_path = raw_dir / str(row[mesh_col])
        if not mesh_path.exists():
            issues.append({"level": "error", "where": label, "message": f"Missing mesh file: {mesh_path}"})
            continue
        for name in global_feature_columns(cfg):
            try:
                to_finite_float(row[name], name)
            except Exception as exc:
                issues.append({"level": "error", "where": label, "message": str(exc)})
        for name in target_scalar_columns(cfg):
            try:
                to_finite_float(row[name], name)
            except Exception as exc:
                issues.append({"level": "error", "where": label, "message": str(exc)})
        if check_mesh_arrays:
            if meshio is None:
                issues.append({"level": "warning", "where": label, "message": "meshio is not installed; skipped mesh array checks."})
                continue
            try:
                mesh = meshio.read(mesh_path)
            except Exception as exc:
                issues.append({"level": "error", "where": label, "message": f"Mesh read failed: {exc}"})
                continue
            names = _mesh_arrays(mesh)
            point_names = set(mesh.point_data.keys())
            for name in point_feature_arrays(cfg):
                if name not in point_names:
                    issues.append({"level": "error", "where": label, "message": f"Missing point feature array: {name}"})
            for name in target_field_arrays(cfg):
                if name not in point_names:
                    issues.append({"level": "error", "where": label, "message": f"Missing target point array: {name}"})
            if not names:
                issues.append({"level": "warning", "where": label, "message": "Mesh contains no named arrays."})
    return issues


def summarize_issues(issues: list[dict[str, str]]) -> dict[str, int]:
    return {
        "errors": sum(1 for issue in issues if issue.get("level") == "error"),
        "warnings": sum(1 for issue in issues if issue.get("level") == "warning"),
        "total": len(issues),
    }
