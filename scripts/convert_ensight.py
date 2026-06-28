import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config, ensure_dirs
from physicsai_gnn.ensight_convert import (
    prepare_ensight_input,
    convert_ensight_case_to_vtu,
    guess_array_mapping,
    append_case_to_metadata_from_config,
)
from physicsai_gnn.case_schema import (
    build_global_features,
    load_json_values,
    parse_key_value_pairs,
    target_scalar_columns,
)


def main():
    parser = argparse.ArgumentParser(description="Convert EnSight .case or ZIP to VTU for MeshGNN workflow.")
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--input", required=True, help=".case file, directory, or ZIP containing full EnSight folder")
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--work_dir", default="data/raw/_ensight_uploads")
    parser.add_argument("--time_index", type=int, default=None)
    parser.add_argument("--no_combine", action="store_true", help="Save one VTU per EnSight part instead of one combined VTU")

    parser.add_argument("--register", action="store_true", help="Append converted case to metadata CSV")
    parser.add_argument("--global", dest="global_pairs", action="append", default=[], help="Global input as name=value. Repeat for each config global feature.")
    parser.add_argument("--scalar-target", dest="scalar_pairs", action="append", default=[], help="Scalar target as name=value. Repeat for each configured scalar target.")
    parser.add_argument("--metadata_json", default=None, help="JSON object or path with global inputs and optional scalar targets.")
    parser.add_argument("--flow_rate_lpm", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--inlet_temp_c", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--power_w", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--plate_thickness_mm", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--target_max_temp_c", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--target_pressure_drop_pa", type=float, default=None, help="Backward-compatible coldplate shortcut.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else raw_dir

    case_files = prepare_ensight_input(args.input, args.work_dir)

    results = []
    for i, case_file in enumerate(case_files):
        case_id = args.case_id or case_file.stem
        if len(case_files) > 1 and args.case_id:
            case_id = f"{args.case_id}_{i:03d}"

        result = convert_ensight_case_to_vtu(
            case_file=case_file,
            output_dir=out_dir,
            case_id=case_id,
            combine_blocks=not args.no_combine,
            time_index=args.time_index,
        )

        mapping = guess_array_mapping(result.point_arrays)
        record = {
            "case_id": result.case_id,
            "source_case_file": result.source_case_file,
            "output_files": result.output_files,
            "combined_file": result.combined_file,
            "n_points": result.n_points,
            "n_cells": result.n_cells,
            "point_arrays": result.point_arrays,
            "cell_arrays": result.cell_arrays,
            "suggested_array_mapping": mapping,
        }

        if args.register:
            values = load_json_values(args.metadata_json)
            values.update(parse_key_value_pairs(args.global_pairs))
            values.update(parse_key_value_pairs(args.scalar_pairs))
            for name in [
                "flow_rate_lpm",
                "inlet_temp_c",
                "power_w",
                "plate_thickness_mm",
                "target_max_temp_c",
                "target_pressure_drop_pa",
            ]:
                value = getattr(args, name)
                if value is not None:
                    values[name] = value
            build_global_features(cfg, values)
            missing_scalars = [
                name for name in target_scalar_columns(cfg)
                if name not in values or values[name] in (None, "")
            ]
            if missing_scalars:
                raise ValueError(f"--register requires configured scalar target value(s): {missing_scalars}")

            # Store mesh path relative to raw_dir if possible.
            mesh_file = result.combined_file or result.output_files[0]
            mesh_path = Path(mesh_file)
            try:
                mesh_rel = str(mesh_path.relative_to(raw_dir))
            except Exception:
                mesh_rel = str(mesh_path)

            append_case_to_metadata_from_config(
                metadata_csv=cfg["paths"]["metadata_csv"],
                cfg=cfg,
                case_id=result.case_id,
                mesh_file=mesh_rel,
                values=values,
            )
            record["registered_to_metadata"] = cfg["paths"]["metadata_csv"]

        results.append(record)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
