import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config
from physicsai_gnn.case_schema import validate_metadata, summarize_issues


def main():
    parser = argparse.ArgumentParser(description="Validate CFD dataset metadata and optional mesh arrays.")
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--check_mesh_arrays", action="store_true", help="Read meshes and check configured point arrays.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    issues = validate_metadata(cfg, check_mesh_arrays=args.check_mesh_arrays)
    summary = summarize_issues(issues)

    if args.json:
        print(json.dumps({"summary": summary, "issues": issues}, indent=2))
    else:
        print(f"Dataset validation: {summary['errors']} error(s), {summary['warnings']} warning(s)")
        for issue in issues:
            print(f"[{issue['level'].upper()}] {issue['where']}: {issue['message']}")

    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
