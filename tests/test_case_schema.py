from pathlib import Path

import pandas as pd

from physicsai_gnn.case_schema import (
    append_case_to_metadata,
    build_global_features,
    parse_key_value_pairs,
    validate_metadata,
)


def base_cfg(tmp_path: Path):
    return {
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "metadata_csv": str(tmp_path / "raw" / "cases.csv"),
            "output_dir": str(tmp_path / "outputs"),
        },
        "data": {
            "case_id_column": "case_id",
            "mesh_file_column": "mesh_file",
            "global_feature_columns": ["reynolds", "mach"],
            "point_feature_arrays": ["wall_mask"],
            "target_field_arrays": ["velocity"],
            "target_scalar_columns": ["drag"],
        },
        "model": {},
        "training": {},
    }


def test_parse_key_value_pairs():
    assert parse_key_value_pairs(["reynolds=1200", "mach=0.2"]) == {
        "reynolds": 1200.0,
        "mach": 0.2,
    }


def test_build_global_features_requires_configured_values(tmp_path):
    cfg = base_cfg(tmp_path)
    assert build_global_features(cfg, {"reynolds": 1000, "mach": "0.1"}) == {
        "reynolds": 1000.0,
        "mach": 0.1,
    }


def test_append_case_to_metadata_uses_config_columns(tmp_path):
    cfg = base_cfg(tmp_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True)
    mesh = raw_dir / "case_001.vtu"
    mesh.write_text("placeholder", encoding="utf-8")

    append_case_to_metadata(
        cfg["paths"]["metadata_csv"],
        cfg,
        "case_001",
        "case_001.vtu",
        {"reynolds": 1200, "mach": 0.2, "drag": 3.5},
    )

    df = pd.read_csv(cfg["paths"]["metadata_csv"])
    assert list(df.columns) == ["case_id", "mesh_file", "reynolds", "mach", "drag"]
    assert df.loc[0, "reynolds"] == 1200
    assert df.loc[0, "drag"] == 3.5


def test_validate_metadata_reports_missing_mesh(tmp_path):
    cfg = base_cfg(tmp_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True)
    pd.DataFrame([
        {"case_id": "case_001", "mesh_file": "missing.vtu", "reynolds": 1000, "mach": 0.1, "drag": 2.0}
    ]).to_csv(cfg["paths"]["metadata_csv"], index=False)

    issues = validate_metadata(cfg)

    assert any(issue["level"] == "error" and "Missing mesh file" in issue["message"] for issue in issues)
