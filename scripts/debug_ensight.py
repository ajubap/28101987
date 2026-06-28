import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.ensight_convert import (
    prepare_ensight_input,
    read_ensight_case,
    dataset_tree_summary,
    case_file_references,
    convert_ensight_case_to_vtu,
)


def main():
    parser = argparse.ArgumentParser(description="Debug an EnSight Gold .case/ZIP before conversion.")
    parser.add_argument("--input", required=True, help=".case file, folder, or ZIP containing a full EnSight export")
    parser.add_argument("--work_dir", default="data/raw/_ensight_debug")
    parser.add_argument("--time_index", type=int, default=None)
    parser.add_argument("--try_convert", action="store_true")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    try:
        import pyvista as pv
        print(f"PyVista: {pv.__version__}")
        try:
            import vtk
            print(f"VTK: {vtk.vtkVersion.GetVTKVersion()}")
        except Exception:
            pass
    except Exception as e:
        print(f"PyVista import failed: {e}")
        raise

    case_files = prepare_ensight_input(args.input, args.work_dir)
    print(f"Found {len(case_files)} case file(s):")
    for c in case_files:
        print(f"  - {c}")

    all_reports = []
    for case_file in case_files:
        print("\n" + "=" * 90)
        print(f"CASE: {case_file}")
        print(f"CASE EXISTS: {Path(case_file).exists()}")
        print(f"CASE FOLDER: {Path(case_file).parent}")

        refs = case_file_references(case_file)
        missing = [r for r in refs if not r["exists"]]
        print(f"Referenced files detected: {len(refs)}")
        print(f"Missing references detected: {len(missing)}")
        if missing:
            print("First missing references:")
            for r in missing[:20]:
                print(f"  line {r['line']}: {r['reference']} -> {r['resolved']}")

        data = read_ensight_case(case_file, time_index=args.time_index)
        tree = dataset_tree_summary(data, max_depth=8)
        print("\nReader tree summary:")
        for row in tree:
            print(
                f"{row['path']} | {row['type']} | "
                f"points={row['n_points']} cells={row['n_cells']} | "
                f"point_arrays={row['point_arrays']} | cell_arrays={row['cell_arrays']}"
            )

        report = {"case_file": str(case_file), "references": refs, "missing": missing, "tree": tree}

        if args.try_convert:
            print("\nTrying conversion...")
            converted = convert_ensight_case_to_vtu(
                case_file=case_file,
                output_dir=args.out_dir,
                case_id=Path(case_file).stem + "_debug",
                combine_blocks=True,
                time_index=args.time_index,
            )
            print(f"Converted: {converted.combined_file or converted.output_files}")
            report["converted"] = converted.__dict__

        all_reports.append(report)

    out = Path(args.work_dir) / "ensight_debug_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nSaved debug report: {out}")


if __name__ == "__main__":
    main()
